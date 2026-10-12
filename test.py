"""Evaluate the trained two-stage classifier on the held-out test split.

    uv run python test.py                         # Stage 1, Stage 2 and end-to-end metrics
    uv run python test.py --patient UCSF-PDGM-0004   # prediction for one patient

Needs models/stage{1,2}.joblib from train.py. Writes results/test_metrics.csv and results/test_predictions.csv.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

from train import MODELS, RESULTS, SEED, STAGE1_CLASSES, STAGE2_CLASSES, load, stage_data

E2E_CLASSES = ["Benign"] + STAGE2_CLASSES


def bootstrap_ci(y, p, n=2000, seed=SEED):
    """95% percentile CI of macro-F1 by resampling test patients with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(stats, [2.5, 97.5])


def print_cm(y, p, labels):
    cm = pd.DataFrame(confusion_matrix(y, p, labels=labels), index=[f"true {l}" for l in labels], columns=labels)
    print(cm.to_string())


def report(name, y, p, labels, proba=None, classes=None):
    lo, hi = bootstrap_ci(y, p)
    row = {"stage": name, "patients": len(y), "macro-F1": f1_score(y, p, labels=labels, average="macro"),
           "95% CI": f"{lo:.2f}-{hi:.2f}", "accuracy": accuracy_score(y, p),
           "balanced acc": balanced_accuracy_score(y, p)}
    if proba is not None:
        row["ROC-AUC"] = (roc_auc_score(y, proba[:, 1]) if len(classes) == 2 else
                          roc_auc_score(y, proba, multi_class="ovr", labels=classes))
    print(f"\n=== {name}: {len(y)} test patients ===")
    print(f"macro-F1 {row['macro-F1']:.3f} (95% CI {row['95% CI']})  accuracy {row['accuracy']:.3f}"
          + (f"  ROC-AUC {row['ROC-AUC']:.3f}" if "ROC-AUC" in row else ""))
    print(classification_report(y, p, labels=labels, zero_division=0, digits=3))
    print_cm(y, p, labels)
    return row


def predict(df, m1, m2):
    """Run the two-stage pipeline: Stage 2 only for patients Stage 1 calls malignant."""
    p1 = m1["model"].predict(df[m1["features"]])
    p_mal = m1["model"].predict_proba(df[m1["features"]])[:, list(m1["model"].classes_).index("Malignant")]
    final = np.where(p1 == "Benign", "Benign", "").astype(object)
    mal = p1 == "Malignant"
    if mal.any():
        final[mal] = m2["model"].predict(df.loc[mal, m2["features"]])
    return pd.DataFrame({"patient_id": df.patient_id, "true": df.final_label, "split": df.split,
                         "stage1_pred": p1, "p_malignant": p_mal.round(3), "final_pred": final})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", help="predict one patient ID instead of the full test-set evaluation")
    args = ap.parse_args()
    m1, m2 = joblib.load(MODELS / "stage1.joblib"), joblib.load(MODELS / "stage2.joblib")

    if args.patient:
        df = load()
        df = df[df.patient_id == args.patient].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"unknown patient {args.patient}")
        r = predict(df, m1, m2).iloc[0]
        print(f"{r.patient_id} ({r.split} set)\n  Stage 1: {r.stage1_pred}  (P(malignant) = {r.p_malignant:.3f})"
              f"\n  Final  : {r.final_pred}\n  Truth  : {r.true}  [{df.diagnosis[0]}]")
        if r.split == "train":
            print("  note: this patient was used for training, so this is not an unbiased test")
        return

    test = load("test")
    print(f"Stage 1 model: {m1['name']}   Stage 2 model: {m2['name']}")

    X1, y1 = stage_data(test, 1)
    rows = [report("Stage 1 (benign vs malignant)", y1, m1["model"].predict(X1), STAGE1_CLASSES,
                   m1["model"].predict_proba(X1), list(m1["model"].classes_))]
    X2, y2 = stage_data(test, 2)
    rows.append(report("Stage 2 (subtype, true malignant cases)", y2, m2["model"].predict(X2), STAGE2_CLASSES,
                       m2["model"].predict_proba(X2), list(m2["model"].classes_)))

    pred = predict(test, m1, m2)
    ev = pred[pred.true != "Astrocytoma IDH-wt"]  # no Stage 2 class for these; reported below
    rows.append(report("End-to-end (Stage 1 -> Stage 2)", ev.true, ev.final_pred, E2E_CLASSES))
    idh_wt = pred[pred.true == "Astrocytoma IDH-wt"]
    print(f"\nAstrocytoma IDH-wt test patients (Stage 1 only): {(idh_wt.stage1_pred == 'Malignant').sum()}/"
          f"{len(idh_wt)} correctly called malignant")

    pd.DataFrame(rows).round(3).to_csv(RESULTS / "test_metrics.csv", index=False)
    pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    print(f"\nsaved {RESULTS / 'test_metrics.csv'} and {RESULTS / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
