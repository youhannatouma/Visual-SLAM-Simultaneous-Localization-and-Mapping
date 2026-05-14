#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=".venv311/bin/python"
MEDIA_E="/Users/guenayfer/SLAM/images_batch_E"
MEDIA_F="/Users/guenayfer/SLAM/images_batch_F"
STRIDE=8

if [[ ! -d "$MEDIA_E" || ! -d "$MEDIA_F" ]]; then
  echo "Missing media directories. Expected: $MEDIA_E and $MEDIA_F" >&2
  exit 2
fi

if [[ ! -f reports/corrections_batch_E_real.csv || ! -f reports/corrections_batch_F_real.csv ]]; then
  echo "Missing corrections files reports/corrections_batch_E_real.csv and/or reports/corrections_batch_F_real.csv" >&2
  exit 2
fi

echo "[1/3] Applying corrected ingest for batch_E_real"
"$PYTHON_BIN" scripts/build_reasoning_data_from_media.py \
  --media-dir "$MEDIA_E" \
  --batch-id batch_E_real \
  --scenario mixed \
  --video-stride "$STRIDE" \
  --out-csv data/raw/media_labeled_batch_E_real.csv \
  --review-out reports/media_review_batch_E_real.csv \
  --review-corrections reports/corrections_batch_E_real.csv

echo "[2/3] Applying corrected ingest for batch_F_real"
"$PYTHON_BIN" scripts/build_reasoning_data_from_media.py \
  --media-dir "$MEDIA_F" \
  --batch-id batch_F_real \
  --scenario occlusion \
  --video-stride "$STRIDE" \
  --out-csv data/raw/media_labeled_batch_F_real.csv \
  --review-out reports/media_review_batch_F_real.csv \
  --review-corrections reports/corrections_batch_F_real.csv

echo "[3/3] Running strict guarded pipeline"
scripts/run_reasoning_training_pipeline.sh \
  --python-bin "$PYTHON_BIN" \
  --disk-min-free-gb 0.90

echo "Done. Check reports/promotion_summary.json"
