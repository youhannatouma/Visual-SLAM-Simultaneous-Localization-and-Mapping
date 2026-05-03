# Targeted Capture Runbook (Option 1)

Goal: recover promotion non-regression for `MOVE_TO_CHAIR` and `CHECK_TABLE` while preserving fresh-real gains.

## 1) Capture quotas (minimum per batch)

Use two independent batches:
- `batch_E_real` (scenario focus: `mixed`, `low_light`)
- `batch_F_real` (scenario focus: `occlusion`, `long_range`, `clutter`)

Per batch class targets (post-review kept rows):
- `MOVE_TO_CHAIR`: 120
- `CHECK_TABLE`: 120
- `AVOID_PERSON`: 80
- `EXPLORE`: 80

Total per batch target kept rows: 400

## 2) Shot design (high-signal)

`MOVE_TO_CHAIR` hard cases:
- Chair partially occluded by table or person
- Chair at far range (>4m equivalent framing)
- Chair near edge/corner of frame
- Multiple chairs with one dominant target

`CHECK_TABLE` hard cases:
- Dining/meeting table without prominent person centered
- Table with clutter (laptop/cup/books) and varying viewpoints
- Table under low-light and side-light
- Table partially occluded by chairs/objects

`EXPLORE` hard cases:
- Empty corridor/wall/floor scenes
- Non-chair/non-table dominant scenes
- Low-detail backgrounds and long-range scenes

`AVOID_PERSON` hard cases:
- Person centered and near
- Person off-center but large area
- Multiple people crossing frame

## 3) Directory structure

Create two directories before ingest:
- `/Users/guenayfer/SLAM/images_batch_E`
- `/Users/guenayfer/SLAM/images_batch_F`

Put only that batch's media in each directory.

## 4) Ingest + review export

```bash
.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_E \
  --batch-id batch_E_real \
  --scenario mixed \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_E_real.csv \
  --review-out reports/media_review_batch_E_real.csv

.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_F \
  --batch-id batch_F_real \
  --scenario occlusion \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_F_real.csv \
  --review-out reports/media_review_batch_F_real.csv
```

## 5) Review corrections

Create corrections files from reviews:
- `reports/corrections_batch_E_real.csv`
- `reports/corrections_batch_F_real.csv`

Required columns:
- `row_id,final_label,drop_row`

Review policy:
- Keep only high-signal rows for target classes.
- Drop ambiguous rows (`drop_row=1`).
- Use reviewer override where needed, but keep label intent strong.

## 6) Apply corrections (required for gate)

```bash
.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_E \
  --batch-id batch_E_real \
  --scenario mixed \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_E_real.csv \
  --review-out reports/media_review_batch_E_real.csv \
  --review-corrections reports/corrections_batch_E_real.csv

.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_F \
  --batch-id batch_F_real \
  --scenario occlusion \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_F_real.csv \
  --review-out reports/media_review_batch_F_real.csv \
  --review-corrections reports/corrections_batch_F_real.csv
```

## 7) Strict guarded pipeline (unchanged policy)

```bash
scripts/run_reasoning_training_pipeline.sh \
  --python-bin .venv311/bin/python \
  --disk-min-free-gb 0.90
```

Success condition:
- `reports/promotion_summary.json` has `"promotable": true`.

## 8) If still blocked

First check in `reports/promotion_summary.json`:
- `test_accuracy_non_regression`
- `key_class_f1_non_regression` for `CHECK_TABLE`, `MOVE_TO_CHAIR`

Then increase only weak-class captures in next micro-cycle (batch_G_real):
- Add +80 `MOVE_TO_CHAIR` and +80 `CHECK_TABLE` high-signal rows.
