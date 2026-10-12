#!/usr/bin/env bash
# Selective Aspera download of UCSF-PDGM v5 (TCIA): structural MRI + tumour masks only.
URL='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEwNjUiLCJwYXNzY29kZSI6IjYxZmRlZDkzYzYzM2JhODIxNWYwZjBjMDU2YWRiNjk3Y2IwOTM4MDIiLCJwYWNrYWdlX2lkIjoiMTA2NSIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=='
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm -v "$ROOT/dataset/raw/ucsf_pdgm:/data" -v "$ROOT/scripts:/cfg:ro" martinlaurent/ascli:latest \
  faspex5 packages receive --url="$URL" --to-folder=/data --progress-bar=no --sources=@ts \
  --ts=@json:@file:/cfg/pdgm_ts.json --transfer-info=@json:'{"resume":{"iter_max":20}}'
