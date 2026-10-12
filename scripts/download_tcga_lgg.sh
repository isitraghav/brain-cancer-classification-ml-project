#!/usr/bin/env bash
set -euo pipefail
# BraTS-TCGA-LGG (TCIA analysis result, DOI 10.7937/K9/TCIA.2017.GJQ7R0EF, CC BY 3.0): 65 pre-operative TCGA-LGG
# patients, skull-stripped + co-registered T1/T1Gd/T2/FLAIR NIfTI with neuroradiologist-corrected BraTS labels.
URL='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjYwMCIsInBhc3Njb2RlIjoiNmRkMWUzNTlhMWUwMDY0ZTZmNGI3NmE3YTczZDljZTZlMzJkMjA4OSIsInBhY2thZ2VfaWQiOiI2MDAiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0='
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dataset/raw/brats_tcga_lgg"
mkdir -p "$OUT"
docker run --rm -v "$OUT:/data" martinlaurent/ascli:latest \
  faspex5 packages receive --url="$URL" --to-folder=/data --progress-bar=no \
  --sources=@ts --ts=@json:'{"paths":[{"source":"/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations.zip"}]}' \
  --transfer-info=@json:'{"resume":{"iter_max":20}}'
cd "$OUT" && unzip -q -o Pre-operative_TCGA_LGG_NIfTI_and_Segmentations.zip && rm Pre-operative_TCGA_LGG_NIfTI_and_Segmentations.zip
