# Machine Learning Project Synopsis

## 1. Project Title
**Two-Stage Brain Tumour Classification from MRI Radiomics: Benign vs Malignant, then Malignant Glioma Subtype**

## 2. Team Members
| Name | Roll No. | Contribution |
|---|---|---|
| _Name_ | _Roll_ | Data acquisition & label audit (TCIA/IDC download, ICD-O-3 label mapping, follow-up removal) |
| _Name_ | _Roll_ | Feature derivation (PyRadiomics pipeline, masks, intensity normalisation) & EDA |
| _Name_ | _Roll_ | Stage 1 modelling, regularisation & hyper-parameter tuning |
| _Name_ | _Roll_ | Stage 2 modelling, ensembles, performance analysis & report |

## 3. Problem Statement
Brain tumours range from benign, slow-growing lesions that may only need monitoring or focused radiosurgery to
malignant gliomas that need urgent surgery, radiotherapy and chemotherapy. The definitive diagnosis comes from
histopathology and molecular testing after surgery, which is invasive and takes days to weeks, while MRI is
acquired for almost every patient before treatment. We use machine learning on quantitative features extracted
from routine MRI to (1) separate benign from malignant tumours and (2) predict the WHO 2021 subtype of malignant
gliomas, as a non-invasive decision-support tool. The problem is hard because the data are high-dimensional
(hundreds of correlated radiomic features), imbalanced (one rare subtype has only 13 patients) and multi-centre.

## 4. Objective
- Build a reproducible two-stage pipeline: Stage 1 benign vs malignant; Stage 2 glioblastoma vs IDH-mutant
  astrocytoma vs oligodendroglioma — with labels taken only from documented pathology/clinical metadata.
- Derive and justify interpretable MRI features (tumour shape, intensity, texture, tumour-compartment ratios) and
  compare regularised linear models, SVMs and ensembles under leakage-free cross-validation.
- Quantify per-stage and end-to-end performance with macro-F1, per-class recall and confusion matrices, and
  analyse confounding and failure modes (false negatives, rare classes).

## 5. Dataset
- **Dataset name/source:** four public collections from The Cancer Imaging Archive (TCIA), CC BY 3.0/4.0
  - *Malignant:* **UCSF-PDGM v5** — preoperative MRI of diffuse gliomas with pathology-confirmed WHO 2021
    diagnosis, grade, IDH and 1p/19q status, plus tumour segmentations.
  - *Malignant:* **BraTS-TCGA-LGG** — preprocessed MRI + expert segmentations of TCGA lower-grade gliomas; WHO 2021
    type from TCGA's measured IDH and 1p/19q status (Ceccarelli et al., Cell 2016).
  - *Benign:* **Vestibular-Schwannoma-SEG** and **Vestibular-Schwannoma-MC-RC** — contrast T1 MRI of vestibular
    schwannomas (ICD-O-3 9560/0, benign) with clinician-drawn tumour contours, from 1 + 10 UK hospitals, accessed
    via NCI Imaging Data Commons.
- **Link:** https://doi.org/10.7937/tcia.bdgf-8v37 · https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF ·
  https://doi.org/10.7937/TCIA.9YTJ-5Q73 · https://doi.org/10.7937/hrzh-2n82
- **Approx. number of samples:** 821 patients (one 3-D MRI study each): 262 benign + 559 malignant
  (368 glioblastoma, 130 IDH-mutant astrocytoma, 26 oligodendroglioma, 35 IDH-wildtype astrocytoma).
  The two extra collections were added to reduce class imbalance (oligodendroglioma 13 → 26) and to give each
  class more than one hospital.
- **Key features:** 102 radiomic features from contrast-T1 for Stage 1 (shape, first-order intensity, GLCM, GLRLM,
  GLSZM, GLDM texture); 374 features for Stage 2 (texture/intensity on T1, T1c, T2, FLAIR + tumour shape +
  necrosis/enhancing/oedema volumes and ratios).
- **Target variable:** Stage 1 — Benign / Malignant (from ICD-O-3 behaviour code `/0` vs `/3`);
  Stage 2 — WHO 2021 integrated diagnosis (glioblastoma / IDH-mutant astrocytoma / oligodendroglioma).

## 6. ML Task & Proposed Approach
- **ML task:** Supervised classification — binary (Stage 1) followed by multi-class (Stage 2), applied hierarchically.
- **Proposed ML algorithms/models:** Logistic Regression with L1 / L2 / Elastic-Net regularisation; Support Vector
  Machine (linear and RBF, tuned C and γ); Random Forest (bagging ensemble); Histogram Gradient Boosting (boosting
  ensemble); soft-voting ensemble of LR + SVM + RF.
- **How ML is applied:** MRI volumes are resampled to 1 mm, intensity-normalised, and radiomic features are
  extracted inside the tumour masks (PyRadiomics). A patient-level stratified 80/20 split is made once; every model
  is a scikit-learn Pipeline (imputation → variance filter → scaling → ANOVA SelectKBest → classifier) tuned with
  GridSearchCV / RandomizedSearchCV under 5-fold stratified CV and class-balanced weights, so no information from
  validation or test folds leaks into preprocessing or feature selection. Test patients pass through Stage 1, and
  those predicted malignant pass to Stage 2.

## 7. Evaluation Metrics
Macro-F1 (primary, because classes are imbalanced), per-class precision and recall (with emphasis on malignant
recall = avoiding false negatives), confusion matrices, accuracy, balanced accuracy, ROC-AUC for Stage 1, and
repeated stratified cross-validation (mean ± SD) for Stage 2 stability.

## 8. Expected Outcome
The system outputs, for each patient's MRI, a benign/malignant decision and (if malignant) a predicted glioma
subtype with class probabilities. Preliminary results on a held-out test set of 147 patients: Stage 1 macro-F1
0.99 (95% CI 0.97–1.00) with no malignant tumour missed — expected to be partly inflated by scanner/site
differences between the two collections, which we analyse explicitly; Stage 2 macro-F1 0.72 under nested
cross-validation (best: regularised logistic regression, ROC-AUC 0.95; glioblastoma recall 0.99, the rare
oligodendroglioma is the main error source); end-to-end two-stage macro-F1 0.80 and accuracy 0.94.

## 9. References / Dataset Link
1. Calabrese E. et al. *The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset.*
   Radiology: Artificial Intelligence 2022. Data: https://doi.org/10.7937/tcia.bdgf-8v37
2. Shapey J. et al. *Segmentation of vestibular schwannoma from MRI, an open annotated dataset and baseline
   algorithm.* Scientific Data 2021. Data: https://doi.org/10.7937/TCIA.9YTJ-5Q73
3. van Griethuysen J.J.M. et al. *Computational Radiomics System to Decode the Radiographic Phenotype.*
   Cancer Research 2017 (PyRadiomics).
4. Bakas S. et al. *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and
   radiomic features.* Scientific Data 2017. Data: https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF
5. Ceccarelli M. et al. *Molecular Profiling Reveals Biologically Discrete Subsets and Pathways of Progression in
   Diffuse Glioma.* Cell 2016 (TCGA IDH/1p19q labels).
6. Kujawa A. et al. *Deep learning for automatic segmentation of vestibular schwannoma: a retrospective study
   from multi-center routine MRI.* Frontiers in Computational Neuroscience 2024. Data: https://doi.org/10.7937/hrzh-2n82
