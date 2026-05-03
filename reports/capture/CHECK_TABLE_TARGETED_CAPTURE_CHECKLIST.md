# CHECK_TABLE Targeted Capture Checklist (2 New Real Batches)

Date: 2026-05-03
Goal: collect two new real batches that improve CHECK_TABLE robustness without harming MOVE_TO_CHAIR.

## Batch Plan

1. Batch G (`batch_G_real`, scenario: `table_focus_clutter`)
- Primary target: CHECK_TABLE true positives in realistic desk/table scenes.
- Duration target: 3 to 6 minutes total video.
- Frame diversity target: at least 120 usable rows after review.
- Environment mix:
  - 40% clean table
  - 40% cluttered table (books, laptop, keyboard, mouse, bottles)
  - 20% partial occlusion of table surface

2. Batch H (`batch_H_real`, scenario: `table_chair_transition`)
- Primary target: hard boundary between CHECK_TABLE and MOVE_TO_CHAIR.
- Duration target: 3 to 6 minutes total video.
- Frame diversity target: at least 120 usable rows after review.
- Environment mix:
  - 50% seated/approaching chair near table
  - 30% standing around table with visible tabletop objects
  - 20% motion transitions (walk-to-chair, turn-to-table, lean-to-table)

## Mandatory Capture Requirements

- Use different rooms/angles between G and H (independent real sources).
- Record at least 3 camera viewpoints per batch:
  - wide shot (full table/chair context)
  - medium shot (torso + table/chair)
  - off-axis shot (side angle)
- Include lighting variation:
  - bright daylight
  - warm indoor/low-light
- Include actor/count variation:
  - single person
  - 2+ people passing/interacting in background
- Keep clip segments short (10 to 25 seconds) to reduce repeated near-identical frames.

## Labeling Rules (CHECK_TABLE vs MOVE_TO_CHAIR)

Apply these during review corrections for both batches:

- Label as `CHECK_TABLE` when most evidence is table-object interaction or tabletop inspection:
  - visible tabletop objects dominate (laptop/keyboard/mouse/book/etc.)
  - posture oriented to table (leaning/looking/reaching toward table)
  - chair may be present but not primary interaction target

- Label as `MOVE_TO_CHAIR` when motion/intent is clearly chair-directed:
  - body trajectory toward chair or sitting transition
  - chair occupancy/approach is primary action
  - table objects are incidental

- Use `drop_row=1` for low-confidence ambiguous frames:
  - strong occlusion of person/table/chair
  - severe motion blur
  - frame where action intent is not inferable

## Quality Gates Before Training

- Per-batch manual review complete (no pending statuses).
- Correction audit generated for each new batch.
- Combined raw distribution must still satisfy strict audit:
  - imbalance <= 1.3
  - min per class gate
  - real share / synthetic share gates
  - two-independent-real-batches holdout requirement

## Data Landing Paths

- Media folders to prepare:
  - `/Users/guenayfer/SLAM/images_batch_G`
  - `/Users/guenayfer/SLAM/images_batch_H`
- Corrections files to prepare:
  - `reports/corrections_batch_G_real.csv`
  - `reports/corrections_batch_H_real.csv`

## Runbook Once Data Lands

1. Ingest batch G with corrections:
```bash
.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_G \
  --batch-id batch_G_real \
  --scenario table_focus_clutter \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_G_real.csv \
  --review-out reports/media_review_batch_G_real.csv \
  --review-corrections reports/corrections_batch_G_real.csv
```

2. Ingest batch H with corrections:
```bash
.venv311/bin/python scripts/build_reasoning_data_from_media.py \
  --media-dir /Users/guenayfer/SLAM/images_batch_H \
  --batch-id batch_H_real \
  --scenario table_chair_transition \
  --video-stride 8 \
  --out-csv data/raw/media_labeled_batch_H_real.csv \
  --review-out reports/media_review_batch_H_real.csv \
  --review-corrections reports/corrections_batch_H_real.csv
```

3. Run guarded train + promotion checks:
```bash
scripts/run_reasoning_training_pipeline.sh \
  --python-bin .venv311/bin/python \
  --disk-min-free-gb 0.7
```

## Exit Criteria For This Capture Round

- CHECK_TABLE F1 no longer regresses vs `reports/metrics_promoted_baseline.json`.
- Fresh-real gates pass:
  - accuracy delta >= +0.10
  - macro F1 delta >= +0.10
- Promotion summary status becomes `promoted`.
