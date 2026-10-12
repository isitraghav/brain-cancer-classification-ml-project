"""Fetch the molecular labels of the BraTS-TCGA-LGG patients and map them to WHO 2021 diagnoses.

    uv run python scripts/fetch_tcga_labels.py

Source: TCGA merged LGG/GBM study (Ceccarelli et al., Cell 2016) via the cBioPortal public API, study
`lgggbm_tcga_pub`: IDH mutation + 1p/19q co-deletion status measured on the tumour tissue. WHO 2021 defines the
diffuse glioma types by exactly these markers, so the mapping is by definition, not by guess:
    IDHmut-codel     -> Oligodendroglioma, IDH-mutant, 1p/19q-codeleted
    IDHmut-non-codel -> Astrocytoma, IDH-mutant
    IDHwt            -> Astrocytoma, IDH-wildtype   (Stage 1 only, as for UCSF-PDGM)
Patients without a molecular result are written with an empty diagnosis and excluded downstream.
Output: dataset/raw/brats_tcga_lgg/tcga_labels.csv
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TCGA_DIR = ROOT / "dataset" / "raw" / "brats_tcga_lgg"
API = "https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data/fetch?clinicalDataType="
WHO2021 = {
    "IDHmut-codel": "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted",
    "IDHmut-non-codel": "Astrocytoma, IDH-mutant",
    "IDHwt": "Astrocytoma, IDH-wildtype",
}


def fetch(kind, attrs):
    req = urllib.request.Request(API + kind, data=json.dumps({"attributeIds": attrs}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "curl/8"})
    rows = json.load(urllib.request.urlopen(req, timeout=300))
    return pd.DataFrame(rows).pivot_table(index="patientId", columns="clinicalAttributeId", values="value",
                                          aggfunc="first")


def main():
    sample = fetch("SAMPLE", ["IDH_CODEL_SUBTYPE", "IDH_STATUS", "GRADE"])
    patient = fetch("PATIENT", ["HISTOLOGICAL_DIAGNOSIS", "AGE", "SEX"])
    lab = sample.join(patient, how="outer")
    ids = sorted(p.name for p in (TCGA_DIR / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations").glob("TCGA-*")
                 if p.is_dir())
    out = lab.reindex(ids).rename_axis("patient_id").reset_index()
    out["diagnosis"] = out.IDH_CODEL_SUBTYPE.map(WHO2021)
    out.to_csv(TCGA_DIR / "tcga_labels.csv", index=False)
    print(f"{len(out)} BraTS-TCGA-LGG patients:")
    print(out.diagnosis.fillna("no molecular result -> excluded").value_counts().to_string())


if __name__ == "__main__":
    main()
