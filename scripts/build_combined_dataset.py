"""Merge all four collections into one labelled table.

    uv run python scripts/build_combined_dataset.py

Output: dataset/combined_dataset.csv - one row per patient, with
  metadata  : patient_id, source, diagnosis, who_grade, age, sex
  labels    : stage1_label (Benign/Malignant), stage2_label (3 glioma subtypes; empty if not in Stage 2),
              final_label, split (train/test)
  features  : s1_* (102 Stage 1 features, T1c in solid tumour) and s2_* (374 Stage 2 features, gliomas only)
Age/sex/grade are provenance only; the models never use them.

Split: the original 735 patients (UCSF-PDGM + VS-SEG) keep the exact seed-42 stratified 80/20 split used since
the first version; the patients added later (BraTS-TCGA-LGG + VS-MC-RC) get their own stratified 80/20 split.
So no earlier test patient moves into training, and every class keeps ~20% in test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
SEED = 42
SUBTYPE = {
    "Glioblastoma, IDH-wildtype": "Glioblastoma",
    "Astrocytoma, IDH-mutant": "Astrocytoma IDH-mut",
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": "Oligodendroglioma",
    "Astrocytoma, IDH-wildtype": "Astrocytoma IDH-wt",
}
STAGE2_CLASSES = ["Glioblastoma", "Astrocytoma IDH-mut", "Oligodendroglioma"]
ORIGINAL = ["UCSF-PDGM", "Vestibular-Schwannoma-SEG"]
VS = "Vestibular schwannoma (ICD-O-3 9560/0)"


def clinical():
    pdgm = pd.read_csv(RAW / "UCSF-PDGM-metadata_v5.csv")
    pdgm = pdgm[~pdgm.ID.str.contains("_FU")].copy()
    pdgm["patient_id"] = pdgm.ID.str.replace(r"UCSF-PDGM-(\d+)", lambda m: f"UCSF-PDGM-{int(m.group(1)):04d}",
                                            regex=True)
    pdgm = pdgm.rename(columns={"Final pathologic diagnosis (WHO 2021)": "diagnosis", "WHO CNS Grade": "who_grade",
                                "Age at MRI": "age", "Sex": "sex"})
    tcga = pd.read_csv(RAW / "brats_tcga_lgg" / "tcga_labels.csv")
    tcga = tcga.assign(who_grade=tcga.GRADE.str[1:].astype(float), age=tcga.AGE,
                       sex=tcga.SEX.map({"Male": "M", "Female": "F"}))
    cols = ["patient_id", "diagnosis", "who_grade", "age", "sex"]
    return pd.concat([pdgm[cols], tcga[cols]], ignore_index=True)


def main():
    s1 = pd.read_csv(ROOT / "dataset" / "features" / "stage1_features.csv")
    s2 = pd.read_csv(ROOT / "dataset" / "features" / "stage2_features.csv")
    s1 = s1[~s1.patient_id.str.contains("_FU")]
    s2 = s2[~s2.patient_id.str.contains("_FU")]

    df = s1[["patient_id", "source"]].merge(clinical(), on="patient_id", how="left")
    benign = df.source.str.startswith("Vestibular-Schwannoma")
    df.loc[benign, "diagnosis"] = VS
    dropped = df[df.diagnosis.isna()]
    df = df[df.diagnosis.notna()].reset_index(drop=True)  # TCGA patients without a molecular result
    df["stage1_label"] = np.where(df.diagnosis == VS, "Benign", "Malignant")  # ICD-O-3 /0 vs /3
    df["final_label"] = df.diagnosis.map(SUBTYPE).fillna("Benign")
    df["stage2_label"] = df.final_label.where(df.final_label.isin(STAGE2_CLASSES))

    df["split"] = "train"
    for part in (df.source.isin(ORIGINAL), ~df.source.isin(ORIGINAL)):
        sub = df[part]
        _, test_ids = train_test_split(sub.patient_id, test_size=0.2, stratify=sub.final_label, random_state=SEED)
        df.loc[df.patient_id.isin(set(test_ids)), "split"] = "test"

    feats1 = s1.drop(columns="source").set_index("patient_id").add_prefix("s1_")
    feats2 = s2.set_index("patient_id").add_prefix("s2_")
    df = df.join(feats1, on="patient_id").join(feats2, on="patient_id")
    df = df[["patient_id", "source", "diagnosis", "who_grade", "age", "sex", "stage1_label", "stage2_label",
             "final_label", "split"] + list(feats1.columns) + list(feats2.columns)]

    out = ROOT / "dataset" / "combined_dataset.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {df.shape[0]} patients x {df.shape[1]} columns"
          f" (excluded, no molecular label: {dropped.patient_id.tolist()})")
    print(pd.crosstab([df.final_label], [df.split], margins=True))
    print(pd.crosstab(df.source, df.final_label))


if __name__ == "__main__":
    main()
