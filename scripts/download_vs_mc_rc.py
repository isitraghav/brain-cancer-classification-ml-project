"""Download the extra benign cases from Vestibular-Schwannoma-MC-RC (TCIA, DOI 10.7937/hrzh-2n82, CC BY 4.0).

    uv run python scripts/download_vs_mc_rc.py

VS-MC-RC is routine clinical surveillance MRI from 10 UK hospitals. 58 of its 160 patients are VS-SEG patients
(IDs "VS-SEG-xxx") - they are skipped so no patient appears twice. For each new patient ("VS-MC-RC-xxx") with a
segmented contrast T1 scan, the earliest such timepoint is downloaded (DICOM via NCI IDC) with its NIfTI mask:
    dataset/raw/vs_mc_rc/<PatientID>/MR_<SeriesUID>/*.dcm
    dataset/raw/vs_mc_rc/<PatientID>/seg_T1.nii.gz
"""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "raw" / "vs_mc_rc"
SEG_ZIP = "https://www.cancerimagingarchive.net/wp-content/uploads/VS-MC-RC-Segmentations_with_pathname_spreadsheet-Release2023.zip"
INFO = "Vestibular-Schwannoma-MC-RC_NIfTI-segmentation-info3.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(SEG_ZIP).read()))
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
    info = pd.read_csv(zf.open(next(n for n in names if n.endswith(INFO))))
    info.to_csv(OUT / INFO, index=False)

    t1 = info[info.patient_id.str.startswith("VS-MC-RC") & info.series_description.str.startswith("_T1")
              & info.SegmentationPath.notna()].copy()
    t1["date"] = t1.SegmentationPath.str.split("/").str[1]  # study folder, YYYY-MM-DD (date-shifted)
    first = t1.sort_values(["patient_id", "date"]).groupby("patient_id").head(1)
    first[["patient_id", "date", "series_description", "series_instance_uid", "SegmentationPath"]].to_csv(
        OUT / "selected_series.csv", index=False)
    print(f"{info.patient_id.nunique()} patients in VS-MC-RC, {len(first)} new patients with a contrast T1 mask")

    for r in first.itertuples():
        (OUT / r.patient_id).mkdir(exist_ok=True)
        seg_member = next(n for n in names if n.endswith(r.SegmentationPath))
        with zf.open(seg_member) as src, open(OUT / r.patient_id / "seg_T1.nii.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    # one parallel s5cmd transfer for all series (per-series calls are ~30 s each)
    IDCClient().download_from_selection(seriesInstanceUID=first.series_instance_uid.tolist(), downloadDir=str(OUT),
                                        dirTemplate="%PatientID/%Modality_%SeriesInstanceUID", quiet=True)
    print("done:", OUT)


if __name__ == "__main__":
    main()
