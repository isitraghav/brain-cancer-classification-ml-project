"""Export PNG previews of the raw MRI (NIfTI / DICOM) so the dataset can be browsed in any image viewer.

    uv run python scripts/export_previews.py                  # 5 patients per class
    uv run python scripts/export_previews.py --per-class 0    # every patient

Writes dataset/previews/<class>/<patient>.png (slice of largest tumour area, tumour outlined) and
dataset/previews/overview_<class>.png (all rendered patients of that class in one grid).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import (PDGM_DIR, ROOT, SEQUENCES, VS_DIR, matching_rtstruct, read_dicom_series,
                              resample_iso, rtstruct_mask)

OUT = ROOT / "dataset" / "previews"
SUBTYPE = {
    "Glioblastoma, IDH-wildtype": "malignant_glioblastoma",
    "Astrocytoma, IDH-mutant": "malignant_astrocytoma_IDH-mut",
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": "malignant_oligodendroglioma",
    "Astrocytoma, IDH-wildtype": "malignant_astrocytoma_IDH-wt",
}
BENIGN = "benign_vestibular_schwannoma"
# BraTS labels: 1 necrosis (red), 2 oedema (green), 4 enhancing (yellow)
SEG_CMAP = ListedColormap([(0, 0, 0, 0), (1, 0, 0, .45), (0, .8, 0, .35), (0, 0, 0, 0), (1, .9, 0, .5)])


def window(a, fg=None):
    vals = a[fg] if fg is not None and fg.any() else a
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def largest_slice(mask):
    return int(np.argmax(mask.reshape(mask.shape[0], -1).sum(1)))


def cases():
    clin = pd.read_csv(ROOT / "dataset" / "raw" / "UCSF-PDGM-metadata_v5.csv")
    clin = clin[~clin.ID.str.contains("_FU")]
    out = []
    for _, r in clin.iterrows():
        pid = f"UCSF-PDGM-{int(r.ID.split('-')[-1]):04d}"
        dx = r["Final pathologic diagnosis (WHO 2021)"]
        info = f"{dx} | WHO grade {r['WHO CNS Grade']} | {r.Sex}, {r['Age at MRI']} y"
        out.append(("pdgm", pid, SUBTYPE[dx], info))
    feats = set(pd.read_csv(ROOT / "dataset" / "features" / "stage1_features.csv").patient_id)  # VS cases with a usable mask
    for d in sorted(VS_DIR.glob("VS-SEG-*")):
        if d.name in feats:
            out.append(("vs", d.name, BENIGN, "Vestibular schwannoma (ICD-O-3 9560/0, benign)"))
    return out


def load(kind, pid):
    """Return (arrays dict name->3D, segmentation 3D) on a 1 mm grid."""
    if kind == "pdgm":
        imgs = {s: sitk.GetArrayFromImage(sitk.ReadImage(str(PDGM_DIR / f"{pid}_{s}_bias.nii.gz"))) for s in SEQUENCES}
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(PDGM_DIR / f"{pid}_tumor_segmentation.nii.gz")))
        return imgs, seg
    mr_dir = next((VS_DIR / pid).glob("MR_*"))
    img = read_dicom_series(mr_dir)
    mask = rtstruct_mask(matching_rtstruct(VS_DIR / pid, mr_dir.name[3:]), img)
    return ({"T1c": sitk.GetArrayFromImage(resample_iso(img, False))},
            sitk.GetArrayFromImage(resample_iso(mask, True)).astype(np.uint8))


def render(kind, pid, cls, info):
    imgs, seg = load(kind, pid)
    z = largest_slice(seg > 0)
    fg = imgs["T1c"] > (0 if kind == "pdgm" else np.percentile(imgs["T1c"], 50))
    ys, xs = np.nonzero(fg[z]) if kind == "pdgm" else (np.arange(seg.shape[1]), np.arange(seg.shape[2]))
    crop = np.s_[max(ys.min() - 5, 0):ys.max() + 6, max(xs.min() - 5, 0):xs.max() + 6]  # PDGM: trim empty border
    panels = [(s, window(a, fg)[z][crop], None) for s, a in imgs.items()]
    panels.append(("T1c + tumour mask", window(imgs["T1c"], fg)[z][crop], seg[z][crop]))
    if kind == "vs":  # coronal view too: the schwannoma sits in the internal auditory canal
        y = int(np.argmax((seg > 0).sum(axis=(0, 2))))
        panels.append(("coronal + mask", window(imgs["T1c"], fg)[::-1, y, :], seg[::-1, y, :]))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), facecolor="black")
    for ax, (title, im, m) in zip(np.atleast_1d(axes), panels):
        ax.imshow(im, cmap="gray", interpolation="none")
        if m is not None:
            if kind == "pdgm":
                ax.imshow(m, cmap=SEG_CMAP, vmin=0, vmax=4, interpolation="none")
            ax.contour(m > 0, levels=[.5], colors="red", linewidths=.8)
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{pid}  —  {info}", color="white", fontsize=10)
    legend = "  (red = necrosis, green = oedema, yellow = enhancing)" if kind == "pdgm" else ""
    fig.text(.5, .01, f"axial slice {z} of largest tumour area{legend}", color="#aaa", ha="center", fontsize=8)
    (OUT / cls).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / cls / f"{pid}.png", dpi=100, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return cls, pid, panels[-1][1], panels[-1][2] > 0


def overview(cls, items):
    items = sorted(items)[:24]
    cols = min(6, len(items))
    rows = -(-len(items) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), facecolor="black")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (pid, im, m) in zip(np.ravel(axes), items):
        ax.imshow(im, cmap="gray", interpolation="none")
        ax.contour(m, levels=[.5], colors="red", linewidths=.6)
        ax.set_title(pid, color="white", fontsize=7)
    fig.suptitle(f"{cls.replace('_', ' ')} — contrast T1, tumour outlined (red)", color="white", y=1.0 + .12 / rows)
    fig.savefig(OUT / f"overview_{cls}.png", dpi=100, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=5, help="patients per class; 0 = all")
    args = ap.parse_args()

    todo = cases()
    if args.per_class:
        todo = pd.DataFrame(todo).groupby(2).head(args.per_class).values.tolist()
    thumbs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(render, *c) for c in todo]
        for i, f in enumerate(as_completed(futs), 1):
            cls, pid, im, m = f.result()
            thumbs.setdefault(cls, []).append((pid, im, m))
            if i % 50 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)
    for cls, items in thumbs.items():
        overview(cls, items)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
