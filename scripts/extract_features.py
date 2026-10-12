"""Radiomic feature extraction for the two-stage brain tumour pipeline.

Outputs (in dataset/features/):
  stage1_features.csv  - one row per patient from BOTH sources, T1-post-contrast only,
                          computed inside the solid tumour mask (the only sequence/mask
                          both collections share). Used for Stage 1 benign vs malignant.
  stage2_features.csv  - one row per UCSF-PDGM glioma, all four structural sequences
                          plus tumour-compartment features. Used for Stage 2 subtype.

Sources
  UCSF-PDGM v5 (TCIA, CC BY 4.0): 1 mm isotropic, skull-stripped, bias-corrected NIfTI,
      BraTS labels 1 = necrotic core, 2 = oedema, 4 = enhancing tumour.
  BraTS-TCGA-LGG (TCIA analysis result, CC BY 3.0): same 1 mm skull-stripped BraTS space and labels as
      UCSF-PDGM but not bias-corrected, so N4 is applied here to match.
  Vestibular-Schwannoma-SEG (TCIA via NCI IDC, CC BY 4.0): DICOM contrast T1 + RTSTRUCT
      tumour contour drawn by the treating clinicians.
  Vestibular-Schwannoma-MC-RC (TCIA via NCI IDC, CC BY 4.0): routine clinical contrast T1 from 10 UK
      hospitals + NIfTI tumour mask (mapped onto the DICOM grid in physical space).

Run:  uv run python scripts/extract_features.py [--workers 3] [--limit N] [--ids ID ...] [--retry-failed]
"""
import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

ROOT = Path(__file__).resolve().parents[1]
PDGM_DIR = ROOT / "dataset" / "raw" / "ucsf_pdgm"
VS_DIR = ROOT / "dataset" / "raw" / "vs_seg"
TCGA_DIR = ROOT / "dataset" / "raw" / "brats_tcga_lgg" / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
VSMC_DIR = ROOT / "dataset" / "raw" / "vs_mc_rc"
TCGA_SUFFIX = {"T1": "t1", "T1c": "t1Gd", "T2": "t2", "FLAIR": "flair"}
OUT_DIR = ROOT / "dataset" / "features"

SEQUENCES = ["T1", "T1c", "T2", "FLAIR"]
FEATURE_CLASSES = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm"]
MIN_CORE_VOXELS = 100  # 0.1 cm^3 at 1 mm isotropic
BIN_WIDTH = 0.25  # images are z-scored first, so a fixed bin width in SD units is comparable across scans

warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)


def make_extractor(with_shape=True):
    from radiomics import featureextractor

    ext = featureextractor.RadiomicsFeatureExtractor(binWidth=BIN_WIDTH, normalize=False, force2D=False)
    ext.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        if fc != "shape" or with_shape:
            ext.enableFeatureClassByName(fc)
    return ext


def zscore_foreground(img, fg_mask_arr):
    """Z-score intensities using foreground voxels only (brain for PDGM, head for VS)."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    fg = arr[fg_mask_arr > 0]
    arr = (arr - fg.mean()) / (fg.std() + 1e-6)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def radiomics_row(extractor, img, mask, prefix):
    res = extractor.execute(img, mask, label=1)
    return {f"{prefix}{k.replace('original_', '')}": float(v) for k, v in res.items() if k.startswith("original_")}


def binary(mask_arr, ref, labels):
    m = sitk.GetImageFromArray(np.isin(mask_arr, labels).astype(np.uint8))
    m.CopyInformation(ref)
    return m


# ---------------------------------------------------------------- UCSF-PDGM (malignant)
def pdgm_case(pid):
    seg_img = sitk.ReadImage(str(PDGM_DIR / f"{pid}_tumor_segmentation.nii.gz"))
    imgs = {s: sitk.ReadImage(str(PDGM_DIR / f"{pid}_{s}_bias.nii.gz"), sitk.sitkFloat32) for s in SEQUENCES}
    return glioma_case(pid, "UCSF-PDGM", imgs, seg_img)


def n4(img, brain_arr):
    """N4 bias-field correction inside the brain (UCSF-PDGM ships N4-corrected '_bias' images)."""
    mask = sitk.GetImageFromArray(brain_arr.astype(np.uint8))
    mask.CopyInformation(img)
    shrink = 4
    small, small_mask = sitk.Shrink(img, [shrink] * 3), sitk.Shrink(mask, [shrink] * 3)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(small, small_mask)
    log_field = corrector.GetLogBiasFieldAsImage(img)
    out = img / sitk.Exp(log_field)
    return sitk.Mask(out, mask)


def tcga_case(pid):
    pdir = TCGA_DIR / pid
    seg_file = next(iter(pdir.glob("*_GlistrBoost_ManuallyCorrected.nii.gz")), None) or next(pdir.glob("*_GlistrBoost.nii.gz"))
    seg_img = sitk.ReadImage(str(seg_file))
    imgs = {s: sitk.ReadImage(str(next(pdir.glob(f"*_{suf}.nii.gz"))), sitk.sitkFloat32) for s, suf in TCGA_SUFFIX.items()}
    brain = sitk.GetArrayFromImage(imgs["T1c"]) > 0
    imgs = {s: n4(im, brain) for s, im in imgs.items()}
    return glioma_case(pid, "BraTS-TCGA-LGG", imgs, seg_img)


def glioma_case(pid, source, imgs, seg_img):
    seg = sitk.GetArrayFromImage(seg_img)
    brain = sitk.GetArrayFromImage(imgs["T1c"]) > 0
    imgs = {s: zscore_foreground(im, brain) for s, im in imgs.items()}
    # Solid tumour = necrosis + enhancing. Non-enhancing (mostly lower-grade) gliomas have no such region, or
    # only a few stray voxels; there the T2/FLAIR-abnormal region *is* the tumour, so the whole tumour is used
    # instead of dropping those patients or describing a 3-voxel "core".
    core_labels = [1, 4] if np.isin(seg, [1, 4]).sum() >= MIN_CORE_VOXELS else [1, 2, 4]
    core = binary(seg, seg_img, core_labels)
    whole = binary(seg, seg_img, [1, 2, 4])

    # Stage 1: T1c inside solid tumour (same definition as the schwannoma contour)
    s1 = {"patient_id": pid, "source": source}
    s1.update(radiomics_row(make_extractor(), imgs["T1c"], core, "t1c_"))

    # Stage 2: compartment volumes + per-sequence radiomics over the whole tumour
    v = {lab: int((seg == lab).sum()) for lab in (1, 2, 4)}
    wt = max(v[1] + v[2] + v[4], 1)
    tc = max(v[1] + v[4], 1)
    s2 = {
        "patient_id": pid,
        "vol_necrosis_cc": v[1] / 1000, "vol_oedema_cc": v[2] / 1000, "vol_enhancing_cc": v[4] / 1000,
        "vol_whole_cc": wt / 1000, "frac_enhancing_of_whole": v[4] / wt, "frac_necrosis_of_core": v[1] / tc,
        "frac_oedema_of_whole": v[2] / wt, "has_enhancement": int(v[4] > 0),
    }
    ext_shape, ext_tex = make_extractor(), make_extractor(with_shape=False)
    s2.update({k: val for k, val in radiomics_row(ext_shape, imgs["T1c"], whole, "wt_").items() if "shape_" in k})
    for s in SEQUENCES:
        s2.update(radiomics_row(ext_tex, imgs[s], whole, f"{s.lower()}_"))
    return s1, s2


# ------------------------------------------------------- Vestibular-Schwannoma-SEG (benign)
def read_dicom_series(folder):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder)))
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


# Names used for the treated schwannoma across years of clinical contouring ("AN" = acoustic neuroma).
# "Vol 2y" etc. are other time points, "Men" a meningioma - neither is used.
TUMOUR_ROI_NAMES = {"TV", "AN", "RT AN", "TUMOUR", "TVT1"}


def rtstruct_mask(rt_path, ref_img):
    """Rasterise RTSTRUCT contour polygons (patient mm coords) onto the reference image grid."""
    ds = pydicom.dcmread(rt_path)
    roi_num = next((r.ROINumber for r in ds.StructureSetROISequence
                    if r.ROIName.strip().upper() in TUMOUR_ROI_NAMES), None)
    if roi_num is None:
        raise RuntimeError("no vestibular schwannoma ROI (TV/AN) in RTSTRUCT")
    contours = next(c for c in ds.ROIContourSequence if c.ReferencedROINumber == roi_num).ContourSequence
    size = ref_img.GetSize()
    mask = np.zeros(size[::-1], dtype=np.uint8)  # (z, y, x)
    for c in contours:
        pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
        idx = np.array([ref_img.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in pts])
        z = int(round(idx[:, 2].mean()))
        if 0 <= z < size[2]:
            rr, cc = polygon(idx[:, 1], idx[:, 0], shape=(size[1], size[0]))
            mask[z, rr, cc] ^= 1  # XOR handles holes / multiple polygons per slice
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(ref_img)
    return out


def matching_rtstruct(patient_dir, series_uid):
    for d in patient_dir.glob("RTSTRUCT_*"):
        f = next(d.glob("*.dcm"))
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        refs = {s.SeriesInstanceUID for fr in ds.ReferencedFrameOfReferenceSequence
                for st in fr.RTReferencedStudySequence for s in st.RTReferencedSeriesSequence}
        if series_uid in refs:
            return f
    return None


def resample_iso(img, is_mask, spacing=(1.0, 1.0, 1.0)):
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), spacing)]
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), spacing, img.GetDirection(), 0,
                         img.GetPixelID())


def vs_case(pid):
    pdir = VS_DIR / pid
    mr_dir = next(pdir.glob("MR_*"))
    series_uid = mr_dir.name[3:]
    rt = matching_rtstruct(pdir, series_uid)
    if rt is None:
        raise RuntimeError("no RTSTRUCT references the T1 series")
    img = read_dicom_series(mr_dir)
    mask = rtstruct_mask(rt, img)
    if sitk.GetArrayViewFromImage(mask).sum() == 0:
        raise RuntimeError("empty tumour mask")
    return schwannoma_features(pid, "Vestibular-Schwannoma-SEG", img, mask)


def vsmc_case(pid):
    pdir = VSMC_DIR / pid
    img = read_dicom_series(next(pdir.glob("MR_*")))
    # the NIfTI mask uses a different voxel order/origin than the DICOM series; both headers are correct in
    # physical space, so nearest-neighbour resampling with an identity transform puts it on the image grid
    seg = sitk.ReadImage(str(pdir / "seg_T1.nii.gz"), sitk.sitkUInt8)
    mask = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    if sitk.GetArrayViewFromImage(mask).sum() == 0:
        raise RuntimeError("empty tumour mask")
    return schwannoma_features(pid, "Vestibular-Schwannoma-MC-RC", img, mask)


def schwannoma_features(pid, source, img, mask):
    img, mask = resample_iso(img, False), resample_iso(mask, True)  # match the 1 mm glioma grid
    head = sitk.GetArrayFromImage(sitk.OtsuThreshold(img, 0, 1))  # head foreground (scans are not skull-stripped)
    img = zscore_foreground(img, head)
    s1 = {"patient_id": pid, "source": source}
    s1.update(radiomics_row(make_extractor(), img, mask, "t1c_"))
    return s1, None


CASES = {"pdgm": pdgm_case, "tcga": tcga_case, "vs": vs_case, "vsmc": vsmc_case}


def run_case(kind, pid):
    try:
        return kind, pid, CASES[kind](pid), None
    except Exception as e:  # noqa: BLE001 - keep the batch going, log failures
        return kind, pid, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="process only N cases per source (for testing)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run only cases in extraction_failures.csv and merge into the existing CSVs")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="re-run only these patient IDs and merge into the existing CSVs")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    # "_FU" folders are follow-up scans of patients already in the cohort (TCIA v3+ note) -> excluded,
    # so each patient contributes exactly one scan and cannot leak across train/test.
    pdgm_ids = sorted({p.name[:14] for p in PDGM_DIR.glob("*_tumor_segmentation.nii.gz")
                       if "_FU" not in p.name
                       and all((PDGM_DIR / f"{p.name[:14]}_{s}_bias.nii.gz").exists() for s in SEQUENCES)})
    vs_ids = sorted(p.name for p in VS_DIR.glob("VS-SEG-*") if any(p.glob("MR_*")))
    tcga_ids = sorted(p.name for p in TCGA_DIR.glob("TCGA-*") if p.is_dir())
    vsmc_ids = sorted(p.name for p in VSMC_DIR.glob("VS-MC-RC-*") if any(p.glob("MR_*")))
    ids = {"pdgm": pdgm_ids, "tcga": tcga_ids, "vs": vs_ids, "vsmc": vsmc_ids}
    if args.limit:
        ids = {k: v[: args.limit] for k, v in ids.items()}
    jobs = [(k, p) for k, v in ids.items() for p in v]
    old, prev_fail = {}, pd.DataFrame(columns=["source", "patient_id", "error"])
    if args.retry_failed or args.ids:
        prev_fail = pd.read_csv(OUT_DIR / "extraction_failures.csv")
        if args.retry_failed:
            jobs = list(prev_fail[["source", "patient_id"]].itertuples(index=False, name=None))
        else:
            jobs = [(k, p) for k, p in jobs if p in set(args.ids)]
        prev_fail = prev_fail[~prev_fail.patient_id.isin({p for _, p in jobs})]
        old = {n: pd.read_csv(OUT_DIR / f"{n}_features.csv") for n in ("stage1", "stage2")}
    print("cases:", {k: len(v) for k, v in ids.items()}, "| running:", len(jobs), flush=True)

    s1_rows, s2_rows, failures = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_case, k, p) for k, p in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            kind, pid, res, err = fut.result()
            if err:
                failures.append({"source": kind, "patient_id": pid, "error": err})
            else:
                s1_rows.append(res[0])
                if res[1] is not None:
                    s2_rows.append(res[1])
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {len(failures)} failed", flush=True)

    for name, rows in (("stage1", s1_rows), ("stage2", s2_rows)):
        df = pd.concat([old.get(name), pd.DataFrame(rows)], ignore_index=True)
        df.drop_duplicates("patient_id", keep="last").sort_values("patient_id").to_csv(
            OUT_DIR / f"{name}_features.csv", index=False)
    pd.concat([prev_fail, pd.DataFrame(failures, columns=["source", "patient_id", "error"])]).to_csv(
        OUT_DIR / "extraction_failures.csv", index=False)
    print(f"new stage1 rows {len(s1_rows)}, new stage2 rows {len(s2_rows)}, failures {len(failures)}")


if __name__ == "__main__":
    main()
