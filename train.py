"""Train the two-stage brain tumour classifier on the training split of dataset/combined_dataset.csv.

    uv run python train.py

Stage 1 (benign vs malignant) and Stage 2 (glioma subtype) are each tuned the same way as in the notebook:
four model families (L1/L2/Elastic-Net logistic regression, SVM, random forest, gradient boosting) plus a
soft-voting ensemble, 5-fold stratified CV on the training patients, macro-F1. The model with the best CV
score in each stage is refitted on all training patients and saved to models/stage{1,2}.joblib.
The test patients are never read here - run test.py for that.
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "combined_dataset.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
STAGE1_CLASSES = ["Benign", "Malignant"]
STAGE2_CLASSES = ["Glioblastoma", "Astrocytoma IDH-mut", "Oligodendroglioma"]


def load(split=None):
    df = pd.read_csv(DATASET, float_precision="round_trip")
    return df if split is None else df[df.split == split].reset_index(drop=True)


def stage_data(df, stage):
    """Feature matrix and labels for one stage; Stage 2 keeps only the three subtype classes."""
    if stage == 2:
        df = df[df.stage2_label.notna()]
    cols = [c for c in df.columns if c.startswith(f"s{stage}_")]
    return df[cols], df[f"stage{stage}_label"]


def pipe(clf):
    # every step is re-fitted inside each CV fold, so nothing leaks from validation folds
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-8)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k="all")),
        ("clf", clf),
    ])


K = [10, 25, 50, "all"]
C = np.logspace(-3, 2, 6)
SEARCHES = {
    # l1_ratio: 0 = pure L2 (ridge), 1 = pure L1 (lasso), in between = Elastic-Net
    "LogReg (L1/L2/EN)": (pipe(LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", random_state=SEED)), {
        "select__k": K, "clf__C": C, "clf__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    }, "grid"),
    "SVM": (pipe(SVC(class_weight="balanced", probability=True, random_state=SEED)), [
        {"select__k": K, "clf__kernel": ["linear"], "clf__C": C},
        {"select__k": K, "clf__kernel": ["rbf"], "clf__C": C, "clf__gamma": ["scale", 1e-3, 1e-2, 1e-1]},
    ], "grid"),
    "Random Forest": (pipe(RandomForestClassifier(class_weight="balanced", random_state=SEED)), {
        "select__k": K, "clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.3],
    }, "random"),
    "Gradient Boosting": (pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)), {
        "select__k": K, "clf__learning_rate": [0.03, 0.1, 0.3], "clf__max_depth": [2, 3, None],
        "clf__l2_regularization": [0.0, 1.0, 10.0], "clf__max_iter": [100, 300],
    }, "random"),
}


def tune(X, y):
    fitted, rows = {}, []
    for name, (est, grid, kind) in SEARCHES.items():
        t = time.time()
        if kind == "grid":
            s = GridSearchCV(est, grid, scoring="f1_macro", cv=CV, n_jobs=-1)
        else:
            s = RandomizedSearchCV(est, grid, n_iter=25, scoring="f1_macro", cv=CV, n_jobs=-1, random_state=SEED)
        s.fit(X, y)
        fitted[name] = s.best_estimator_
        params = {k.replace("clf__", ""): v for k, v in s.best_params_.items()}
        rows.append({"model": name, "CV macro-F1": s.best_score_,
                     "CV sd": s.cv_results_["std_test_score"][s.best_index_], "best params": params})
        print(f"  {name:<18} CV macro-F1 {s.best_score_:.3f}  ({time.time() - t:.0f}s)  {params}", flush=True)
    vote = VotingClassifier([(n, clone(fitted[n])) for n in ["LogReg (L1/L2/EN)", "SVM", "Random Forest"]],
                            voting="soft")
    sc = cross_val_score(vote, X, y, cv=CV, scoring="f1_macro", n_jobs=-1)
    fitted["Soft Voting"] = vote.fit(X, y)
    rows.append({"model": "Soft Voting", "CV macro-F1": sc.mean(), "CV sd": sc.std(), "best params": "LR+SVM+RF"})
    print(f"  {'Soft Voting':<18} CV macro-F1 {sc.mean():.3f}", flush=True)
    return fitted, pd.DataFrame(rows).set_index("model")


def main():
    train = load("train")
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for stage, title in [(1, "benign vs malignant"), (2, "malignant glioma subtype")]:
        X, y = stage_data(train, stage)
        print(f"\nStage {stage} ({title}): {len(y)} training patients, {X.shape[1]} features, "
              f"{y.value_counts().to_dict()}", flush=True)
        fitted, cv = tune(X, y)
        best = cv["CV macro-F1"].idxmax()  # chosen on training CV only, never on the test set
        print(f"  -> selected: {best}")
        joblib.dump({"model": fitted[best], "name": best, "features": list(X.columns),
                     "cv": cv, "all_models": fitted}, MODELS / f"stage{stage}.joblib")
        cv.round(3).to_csv(RESULTS / f"train_cv_stage{stage}.csv")
    print(f"\nsaved {MODELS / 'stage1.joblib'} and {MODELS / 'stage2.joblib'}")


if __name__ == "__main__":
    main()
