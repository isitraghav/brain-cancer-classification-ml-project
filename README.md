# Two-Stage Brain Tumour Classification from MRI Radiomics

Stage 1: benign vs malignant brain tumour. Stage 2: pathology-confirmed WHO 2021 subtype of malignant tumours.

## Dataset

All imaging comes from **The Cancer Imaging Archive (TCIA)** [4], an NCI-funded research repository; no
Kaggle/GitHub data are used. Every label is taken from documented pathology, molecular or clinical metadata,
never inferred from file or tumour names.

**821 patients from four collections:** 559 malignant gliomas and 262 benign schwannomas.

### Collections

| | Malignant: **UCSF-PDGM** v5 [1, 2] | Malignant: **BraTS-TCGA-LGG** [17, 18] | Benign: **Vestibular-Schwannoma-SEG** [5, 6] | Benign: **Vestibular-Schwannoma-MC-RC** [21, 22] |
|---|---|---|---|---|
| Data DOI | [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37) | [10.7937/K9/TCIA.2017.GJQ7R0EF](https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF) | [10.7937/TCIA.9YTJ-5Q73](https://doi.org/10.7937/TCIA.9YTJ-5Q73) | [10.7937/hrzh-2n82](https://doi.org/10.7937/hrzh-2n82) |
| Licence | CC BY 4.0 | CC BY 3.0 | CC BY 4.0 | CC BY 4.0 |
| Institution | UCSF | TCGA multi-institution (US) | one UK neurosurgical centre (London) | 10 UK hospitals |
| Patients used | 495 (6 follow-up scans excluded) | 64 of 65 (1 has no molecular result) | 240 of 242 (2 contours are a meningioma "Men") | 22 (58 patients shared with VS-SEG excluded; the rest have no segmented contrast T1) |
| Tumours | Adult diffuse gliomas, WHO grades 2-4 | Lower-grade gliomas, grades 2-3 | Schwannoma before Gamma Knife radiosurgery | Schwannoma under routine surveillance |
| Ground truth | Histopathology + molecular (IDH, 1p/19q), WHO 2021 integrated diagnosis | TCGA IDH + 1p/19q status [19, 20] mapped by the WHO 2021 definition | Clinical diagnosis; ICD-O-3 9560/0 | Clinical diagnosis; ICD-O-3 9560/0 |
| MRI used | T1, T1c, T2, FLAIR (3 T) | T1, T1c, T2, FLAIR | T1c (Siemens Avanto 1.5 T) | T1c (Siemens, Philips, GE; 1-3 T) |
| Format | NIfTI, skull-stripped, N4-corrected, co-registered, 1 mm, 240x240x155 | Same BraTS space as UCSF-PDGM; N4 applied by us [23] | DICOM, 0.41 mm in-plane, 1-1.5 mm slices | DICOM, 0.3-1 mm in-plane, 0.9-3.5 mm slices |
| Tumour mask | BraTS labels [9]: 1 necrosis, 2 oedema, 4 enhancing | BraTS labels, neuroradiologist-corrected | RTSTRUCT treatment-planning contour (`TV`/`AN`) | NIfTI mask, radiologist-reviewed |
| Access | TCIA Aspera (selective, ~6 GB of 142 GB) | TCIA Aspera (546 MB) | NCI IDC [7] via `idc-index` (~15 GB) | NCI IDC public bucket (22 series, 361 MB) |

### Dataset links and download scripts

| Collection | Data DOI | Collection page | Download script | Other source used |
|---|---|---|---|---|
| UCSF-PDGM v5 | https://doi.org/10.7937/tcia.bdgf-8v37 | https://www.cancerimagingarchive.net/collection/ucsf-pdgm/ | [`scripts/download_pdgm.sh`](scripts/download_pdgm.sh) (Docker + Aspera, resumable; `pdgm_ts.json` selects the files) | Metadata CSV `UCSF-PDGM-metadata_v5.csv` from the collection page |
| BraTS-TCGA-LGG | https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF | https://www.cancerimagingarchive.net/analysis-result/brats-tcga-lgg/ | [`scripts/download_tcga_lgg.sh`](scripts/download_tcga_lgg.sh) (Docker + Aspera, 546 MB) | Labels: [`scripts/fetch_tcga_labels.py`](scripts/fetch_tcga_labels.py) via https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub |
| Vestibular-Schwannoma-SEG | https://doi.org/10.7937/TCIA.9YTJ-5Q73 | https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-seg/ | [`scripts/download_vs.py`](scripts/download_vs.py) (`idc-index`, ~15 GB) | NCI Imaging Data Commons: https://portal.imaging.datacommons.cancer.gov/ |
| Vestibular-Schwannoma-MC-RC | https://doi.org/10.7937/hrzh-2n82 | https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-mc-rc/ | [`scripts/download_vs_mc_rc.py`](scripts/download_vs_mc_rc.py) (IDC + TCIA segmentation zip) | Masks: https://www.cancerimagingarchive.net/wp-content/uploads/VS-MC-RC-Segmentations_with_pathname_spreadsheet-Release2023.zip |

Other links: [TCIA](https://www.cancerimagingarchive.net/) ·
[Aspera download help](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) ·
[Meningioma-SEG-CLASS, not downloaded (restricted licence)](https://doi.org/10.7937/0TKV-1A36).
The full command sequence is under [Rebuild the dataset from scratch](#rebuild-the-dataset-from-scratch).

### Labels

**Stage 1, benign vs malignant**, uses the ICD-O-3 behaviour code [8] of each diagnosis (`/0` benign,
`/3` malignant), the coding used by cancer registries:

| Diagnosis (WHO 2021 [3]) | ICD-O-3 | Stage 1 | Stage 2 | UCSF-PDGM | TCGA-LGG | VS-SEG | VS-MC-RC | Total | Train / test |
|---|---|---|---|---|---|---|---|---|---|
| Vestibular schwannoma | 9560/0 | Benign | - | | | 240 | 22 | **262** | 209 / 53 |
| Glioblastoma, IDH-wildtype | 9440/3 | Malignant | Glioblastoma | 368 | | | | **368** | 295 / 73 |
| Astrocytoma, IDH-mutant | 9400/3 | Malignant | Astrocytoma IDH-mut | 90 | 40 | | | **130** | 104 / 26 |
| Oligodendroglioma, IDH-mutant, 1p/19q-codeleted | 9450/3 | Malignant | Oligodendroglioma | 13 | 13 | | | **26** | 20 / 6 |
| Astrocytoma, IDH-wildtype | 9400/3 | Malignant | not used* | 24 | 11 | | | **35** | 28 / 7 |
| **Total** | | | | 495 | 64 | 240 | 22 | **821** | **656 / 165** |

\*Too few cases for a Stage 2 class, and under WHO 2021 most IDH-wildtype diffuse astrocytomas with GBM
molecular features are glioblastomas. They stay in Stage 1 as malignant.

- **UCSF-PDGM labels** come from the column `Final pathologic diagnosis (WHO 2021)` of
  `UCSF-PDGM-metadata_v5.csv`.
- **TCGA-LGG labels** come from the IDH/1p19q subtype measured by TCGA (`scripts/fetch_tcga_labels.py`, saved to
  `dataset/raw/brats_tcga_lgg/tcga_labels.csv`). WHO 2021 *defines* the types by these markers:
  - IDH-mutant + 1p/19q-codeleted → oligodendroglioma
  - IDH-mutant without codeletion → astrocytoma IDH-mutant
  - IDH-wildtype → astrocytoma IDH-wildtype
- **Molecular columns are never model inputs.** IDH, 1p/19q and MGMT partly *define* the subtype, so using them
  would leak the label.
- **Glioma patients:** median age 57 (17-94), 322 M / 237 F.

**Class balance.** The extra collections doubled oligodendroglioma (13 → 26) and grew IDH-mutant astrocytoma
(90 → 130) and benign (240 → 262). The worst Stage 2 imbalance improves from 28:1 to 14:1. No further verified
public data without a licence exist, so the remaining imbalance is handled in training:
- `class_weight="balanced"` in every model
- stratified splits and CV folds
- macro-F1 as the main metric

**Train/test split** (column `split`):
- The original 735 patients keep the seed-42 stratified 80/20 split of the first version.
- The 86 added patients get their own stratified 80/20 split, so no earlier test patient moved into training.
- Each patient has exactly one scan, so no patient is in both sets.

### Features (`dataset/combined_dataset.csv`)

Images are N4-corrected (gliomas), resampled to 1 mm and z-scored over the brain or head. Then PyRadiomics [10]
extracts features inside the tumour mask.

- `s1_*` **Stage 1, 102 features:** contrast-T1 inside the solid tumour (necrosis + enhancing; whole tumour if
  that is under 100 voxels). The groups are shape 14, first-order 18, GLCM 24, GLRLM 16, GLSZM 16 and GLDM 14.
- `s2_*` **Stage 2, 374 features:**
  - first-order and texture features on each of T1, T1c, T2 and FLAIR over the whole tumour
  - whole-tumour shape
  - necrosis, oedema and enhancing volumes and their ratios
  - a `has_enhancement` flag

### Known limitations of the data

- **Site and size confound.**
  - All benign tumours are schwannomas scanned as clinical MRI with skull. All malignant tumours are gliomas
    in skull-stripped research format.
  - The tumours differ in location and size: median 1.4-1.5 cm³ for schwannomas vs 27-52 cm³ for gliomas.
  - So Stage 1 partly learns acquisition and size, not only malignancy. The notebook (section 7) measures this,
    including leave-one-collection-out tests on hospitals the model never saw.
- **Stage 2 site mix.** Glioblastomas come only from UCSF, while astrocytomas and oligodendrogliomas come from
  UCSF and TCGA. The notebook reports Stage 2 results per collection.
- **Rare class.** Oligodendroglioma still has only 26 patients (6 in the test set).
- **Better benign class.** Meningioma-SEG-CLASS [11] (TCIA, DOI 10.7937/0TKV-1A36; 55 grade-1 benign
  meningiomas) would be a better-matched benign class. It needs a signed TCIA Restricted License Agreement, so it
  is not downloaded.

## Layout

```
train.py                      trains Stage 1 + Stage 2 on the training split -> models/stage1.joblib, stage2.joblib
test.py                       evaluates the saved models on the 165 test patients (or --patient <ID>)
dataset/
  combined_dataset.csv        THE dataset: 821 patients x (labels, train/test split, 102 + 374 features)
  features/                   per-stage feature tables written by extract_features.py
  raw/                        original MRI + label files of the four collections:
                                ucsf_pdgm/ (NIfTI) + UCSF-PDGM-metadata_v5.csv
                                brats_tcga_lgg/ (NIfTI) + tcga_labels.csv
                                vs_seg/ (DICOM + RTSTRUCT)
                                vs_mc_rc/ (DICOM + NIfTI masks, download links)
  previews/                   PNG slices of the raw MRI for browsing
models/                       trained models (from train.py)
results/                      metrics CSVs + figures/ (notebook plots)
notebooks/brain_tumour_pipeline.ipynb   full report: EDA, models, tuning, ensembles, nested CV, evaluation
scripts/                      data preparation (download, feature extraction, combined CSV, previews)
```

Raw data are CC BY 3.0/4.0: cite references [1], [2], [5], [6], [17], [18], [21] and [22] when using or
redistributing them.

## Train and test

```bash
uv sync                          # Python 3.11 + all deps from pyproject.toml / uv.lock
uv run python train.py           # ~10 min: tunes 4 models + voting per stage, keeps the best by CV
uv run python test.py            # Stage 1, Stage 2 and end-to-end metrics on the held-out test set
uv run python test.py --patient UCSF-PDGM-0004
```

## Rebuild the dataset from scratch

```bash
./scripts/download_pdgm.sh                                 # needs Docker; re-run if it times out (it resumes)
./scripts/download_tcga_lgg.sh                             # needs Docker, 546 MB
uv run python scripts/fetch_tcga_labels.py                 # TCGA IDH/1p19q -> WHO 2021 labels
uv run python scripts/download_vs.py
uv run python scripts/download_vs_mc_rc.py                 # or: aria2c -i dataset/raw/vs_mc_rc/aria2_input.txt -j 16
uv run python scripts/extract_features.py --workers 4      # ~40 min on 8 cores
uv run python scripts/build_combined_dataset.py            # -> dataset/combined_dataset.csv
uv run python scripts/export_previews.py                   # optional PNG previews
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/brain_tumour_pipeline.ipynb   # ~30 min
```

## References

All DOIs and links below were checked against Crossref, DataCite and the publisher pages.

**Datasets**

1. Calabrese E, Villanueva-Meyer JE, Rudie JD, Rauschecker AM, Baid U, Bakas S, Cha S, Mongan JT, Hess CP.
   *The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM)* (Version 5)
   [Dataset]. The Cancer Imaging Archive, 2022.
   Data: https://doi.org/10.7937/tcia.bdgf-8v37 ·
   Collection page: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
2. Calabrese E, et al. *The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset.*
   Radiology: Artificial Intelligence 2022;4(6):e220058.
   Paper: https://doi.org/10.1148/ryai.220058
3. Louis DN, Perry A, Wesseling P, et al. *The 2021 WHO Classification of Tumors of the Central Nervous System:
   a summary.* Neuro-Oncology 2021;23(8):1231-1251.
   Paper: https://doi.org/10.1093/neuonc/noab106
4. Clark K, Vendt B, Smith K, et al. *The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
   Information Repository.* Journal of Digital Imaging 2013;26(6):1045-1057.
   Paper: https://doi.org/10.1007/s10278-013-9622-7 · Archive: https://www.cancerimagingarchive.net/
5. Shapey J, Kujawa A, Dorent R, et al. *Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging:
   An Open Annotated Dataset and Baseline Algorithm* (Version 2) [Dataset]. The Cancer Imaging Archive, 2021.
   Data: https://doi.org/10.7937/TCIA.9YTJ-5Q73 ·
   Collection page: https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-seg/
6. Shapey J, Kujawa A, Dorent R, et al. *Segmentation of vestibular schwannoma from MRI, an open annotated dataset
   and baseline algorithm.* Scientific Data 2021;8:286.
   Paper: https://doi.org/10.1038/s41597-021-01064-w
7. Fedorov A, Longabaugh WJR, Pot D, et al. *National Cancer Institute Imaging Data Commons: Toward Transparency,
   Reproducibility, and Scalability in Imaging Artificial Intelligence.* RadioGraphics 2023;43(12):e230180.
   Paper: https://doi.org/10.1148/rg.230180 · Portal: https://portal.imaging.datacommons.cancer.gov/
8. Fritz A, Percy C, Jack A, et al. *International Classification of Diseases for Oncology, 3rd edition
   (ICD-O-3).* World Health Organization, 2000.
   Link: https://iris.who.int/handle/10665/42344
9. Baid U, Ghodasara S, Mohan S, et al. *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation
   and Radiogenomic Classification.* arXiv:2107.02314, 2021.
   Paper: https://arxiv.org/abs/2107.02314
10. van Griethuysen JJM, Fedorov A, Parmar C, et al. *Computational Radiomics System to Decode the Radiographic
    Phenotype.* Cancer Research 2017;77(21):e104-e107.
    Paper: https://doi.org/10.1158/0008-5472.CAN-17-0339 · Docs: https://pyradiomics.readthedocs.io/
11. Vassantachart A, Cao Y, Gribble M, et al. *Automatic differentiation of Grade I and II meningiomas on magnetic
    resonance image using an asymmetric convolutional neural network.* Scientific Reports 2022;12:3806.
    Paper: https://doi.org/10.1038/s41598-022-07859-0 · Data: https://doi.org/10.7937/0TKV-1A36

**Methods**

12. Pedregosa F, Varoquaux G, Gramfort A, et al. *Scikit-learn: Machine Learning in Python.* Journal of Machine
    Learning Research 2011;12:2825-2830.
    Paper: https://jmlr.org/papers/v12/pedregosa11a.html
13. Zou H, Hastie T. *Regularization and variable selection via the elastic net.* Journal of the Royal
    Statistical Society B 2005;67(2):301-320.
    Paper: https://doi.org/10.1111/j.1467-9868.2005.00503.x
14. Varma S, Simon R. *Bias in error estimation when using cross-validation for model selection.* BMC
    Bioinformatics 2006;7:91. (nested cross-validation)
    Paper: https://doi.org/10.1186/1471-2105-7-91
15. Cawley GC, Talbot NLC. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance
    Evaluation.* Journal of Machine Learning Research 2010;11:2079-2107.
    Paper: https://jmlr.org/papers/v11/cawley10a.html
16. Zwanenburg A, Vallières M, Abdalah MA, et al. *The Image Biomarker Standardization Initiative: Standardized
    Quantitative Radiomics for High-Throughput Image-based Phenotyping.* Radiology 2020;295(2):328-338.
    Paper: https://doi.org/10.1148/radiol.2020191145

**Collections added to balance the classes**

17. Bakas S, Akbari H, Sotiras A, et al. *Segmentation Labels and Radiomic Features for the Pre-operative Scans
    of the TCGA-LGG collection* [Dataset]. The Cancer Imaging Archive, 2017.
    Data: https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF ·
    Page: https://www.cancerimagingarchive.net/analysis-result/brats-tcga-lgg/
18. Bakas S, Akbari H, Sotiras A, et al. *Advancing The Cancer Genome Atlas glioma MRI collections with expert
    segmentation labels and radiomic features.* Scientific Data 2017;4:170117.
    Paper: https://doi.org/10.1038/sdata.2017.117
19. Ceccarelli M, Barthel FP, Malta TM, et al. *Molecular Profiling Reveals Biologically Discrete Subsets and
    Pathways of Progression in Diffuse Glioma.* Cell 2016;164(3):550-563. (source of the TCGA IDH/1p19q labels)
    Paper: https://doi.org/10.1016/j.cell.2015.12.028 · Data: https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub
20. Cerami E, Gao J, Dogrusoz U, et al. *The cBio Cancer Genomics Portal: An Open Platform for Exploring
    Multidimensional Cancer Genomics Data.* Cancer Discovery 2012;2(5):401-404.
    Paper: https://doi.org/10.1158/2159-8290.CD-12-0095
21. Kujawa A, Dorent R, Wijethilake N, et al. *Segmentation of Vestibular Schwannoma from Magnetic Resonance
    Imaging: An Annotated Multi-Center Routine Clinical Dataset* (Version 2) [Dataset]. The Cancer Imaging
    Archive, 2023. Data: https://doi.org/10.7937/hrzh-2n82 ·
    Page: https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-mc-rc/
22. Kujawa A, Dorent R, Connor S, et al. *Deep learning for automatic segmentation of vestibular schwannoma: a
    retrospective study from multi-center routine MRI.* Frontiers in Computational Neuroscience 2024;18:1365727.
    Paper: https://doi.org/10.3389/fncom.2024.1365727
23. Tustison NJ, Avants BB, Cook PA, et al. *N4ITK: Improved N3 Bias Correction.* IEEE Transactions on Medical
    Imaging 2010;29(6):1310-1320.
    Paper: https://doi.org/10.1109/TMI.2010.2046908
