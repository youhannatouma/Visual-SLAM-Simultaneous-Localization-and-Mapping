# Training And Data Workflow

Use `.venv311/bin/python` for project commands unless you have activated an equivalent environment.

## 1. Collect Data

Capture diverse videos with chairs, tables, people, obstacles, empty rooms, close/far objects, camera motion, and lighting changes.

```bash
.venv311/bin/python video_processor.py label --video videos/room1.mp4 --session room1_fixed
```

Controls:

| Key | Label |
| --- | --- |
| A | `AVOID_PERSON` |
| C | `MOVE_TO_CHAIR` |
| T | `CHECK_TABLE` |
| E | `EXPLORE` |
| SPACE | pause/resume |
| ESC | quit |

You can bootstrap labels with rules, then manually review/correct them:

```bash
.venv311/bin/python video_processor.py autolabel --video videos/room1.mp4 --output data/raw/room1_auto.csv
```

## 2. Run The Guarded Pipeline

```bash
scripts/run_reasoning_training_pipeline.sh --python-bin .venv311/bin/python
```

The pipeline audits data, prepares train/validation/test/fresh-real splits, trains the MLP policy, writes metrics, and runs promotion gates.

Key outputs:

- `models/reasoning_model.pt`
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `reports/promotion_summary.json`
- `data/manifest/dataset_manifest.json`

## 3. Interpret Promotion Results

Check:

```bash
cat reports/promotion_summary.json
```

Only treat a model as promoted when `promotable` is `true`. If the run is `not_promoted`, follow the targeted recommendations in the summary before retraining.

## 4. Runtime Validation

Run a benchmark video with annotations:

```bash
.venv311/bin/python main.py \
  --benchmark-video videos/benchmark.mp4 \
  --run-annotations data/annotations/mapping_benchmark_gt.json \
  --mapping-backend depth \
  --camera-calibration calibration/camera.json \
  --run-report-out reports/runtime/benchmark_report.json
```

The current `mapping_benchmark_gt.json` is a starter template. Replace it with real per-frame annotations before using mapping promotion results seriously.

## 5. Important Limit

The reasoning model is an action policy. It is not the SLAM backend. Improving training can improve decisions, but real VSLAM quality requires better localization/mapping: calibration, depth/scale handling, keyframes, landmarks, pose graph optimization, and relocalization.
