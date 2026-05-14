# Single-Machine CLI Deployment Runbook

This project currently supports a first deployment target as a **single-machine CLI runtime**.
It is not a service/API deployment, and it is not yet a full production VSLAM stack.

## Release Contract

The canonical release contract is:

- [release_contract.json](/Users/guenayfer/SLAM/Visual-SLAM-Simultaneous-Localization-and-Mapping/deployment/release_contract.json)

That file defines:

- the deployed reasoning checkpoint
- the required metrics and promotion artifacts
- the runtime smoke-validation profile
- the temporary promotion policy used for this release channel

## Environment Setup

Use Python through `.venv311` when available.

```bash
python3 -m venv .venv311
. .venv311/bin/activate
.venv311/bin/python -m pip install --upgrade pip
.venv311/bin/python -m pip install -r requirements.txt
```

## Required Assets

Required:

- `models/reasoning_model.pt`
- `yolov8n.pt`
- `reports/metrics.json`
- `reports/promotion_summary.json`

Optional:

- `calibration/*.json` for calibrated depth-aware mapping
- `data/annotations/*.json` for benchmark validation

## Supported Runtime Profile

The supported release smoke profile is:

- headless benchmark run
- `--no-depth`
- `--mapping-backend heuristic`
- `--det-confidence 0.1`
- `--det-imgsz 640`
- `--map-benchmark-obstacle-metric object`
- `--map-obstacle-footprint-radius-cells 1`
- `--map-obstacle-footprint-shape class_aware`

Canonical smoke command:

```bash
.venv311/bin/python main.py \
  --benchmark-video videos/benchmark_room1.MOV \
  --run-annotations data/annotations/benchmark_room1_gt.json \
  --run-report-out reports/runtime/release_smoke_report.json \
  --headless \
  --max-frames 90 \
  --no-depth \
  --mapping-backend heuristic \
  --det-confidence 0.1 \
  --det-imgsz 640 \
  --map-benchmark-obstacle-metric object \
  --map-obstacle-footprint-radius-cells 1 \
  --map-obstacle-footprint-shape class_aware
```

## Canonical Validation

Run the release validator:

```bash
.venv311/bin/python scripts/validate_release.py --python-bin .venv311/bin/python
```

The validator checks:

- `cv2`, `torch`, and `ultralytics` imports
- promoted checkpoint presence and load compatibility
- presence of metrics and promotion summary artifacts
- promoted status in `reports/promotion_summary.json`
- headless runtime smoke run completion
- required runtime report sections

## Expected Outputs

After a successful validation run, expect:

- `reports/runtime/release_smoke_report.json`
- `reports/metrics.json`
- `reports/promotion_summary.json`

Treat the release as valid only if:

- the promotion summary says `promotable: true`
- the release validator exits successfully
- the runtime smoke report contains:
  - `label_metrics`
  - `map_metrics`
  - `pose_stats`
  - `config`
  - `timing`

## Temporary Promotion Policy

This release channel intentionally uses a temporary operational policy:

- max dataset imbalance: `1.5`
- min real share: `0.15`
- fresh-real accuracy delta floor: `-0.06`
- fresh-real macro F1 delta floor: `-0.03`
- fresh-real absolute accuracy floor: `0.50`
- fresh-real absolute macro F1 floor: `0.55`
- fresh-real per-class regression is allowed

This is acceptable for first single-machine deployment, but it is not the long-term research-quality bar.

## Rollback

If the deployed model or policy needs to be rolled back:

1. Restore the previous promoted baseline artifact from git history or a release tag.
2. Restore the previous model checkpoint from git history, release storage, or a tagged artifact set.
3. Re-run the same validation command:

```bash
.venv311/bin/python scripts/validate_release.py --python-bin .venv311/bin/python
```

Rollback is complete only when the restored artifact set passes the same validation flow.
