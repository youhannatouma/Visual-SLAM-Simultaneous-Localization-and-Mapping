# AI Visual Navigation With VSLAM-Oriented Mapping

This repository is an AI visual navigation demo with VSLAM-oriented mapping hooks. It is not a full production Visual SLAM stack yet. The current system combines YOLO object detection, ORB-based visual motion, a learned/rule action policy, semantic occupancy-grid mapping, runtime evaluation reports, and dataset/training utilities.

## Current Architecture

```text
Camera or Video
  -> YOLO object detection
  -> ORB visual motion estimate
  -> optional MiDaS relative depth
  -> action policy: MLP fallback to rules
  -> semantic occupancy-grid mapper
  -> live HUD, map window, and JSON report
```

The mapper supports three backend modes:

- `heuristic`: default, uses bounding-box size plus bearing to estimate obstacle range.
- `depth`: uses depth-map values for obstacle projection when depth is enabled.
- `orb_slam_like`: records an external-backend-required contract while preserving the current runtime fallback. This is a placeholder for a real SLAM backend, not a bundled ORB-SLAM implementation.

## Setup

Use the project virtualenv when available:

```bash
.venv311/bin/python -m pip install -r requirements.txt
```

If `.venv311` does not exist, create a compatible environment first. The global `python` command may not exist on macOS, and global `python3` may not include OpenCV.

## Run Live Or Video

```bash
.venv311/bin/python main.py
```

```bash
.venv311/bin/python main.py --video videos/test.mp4 --no-depth
```

Depth-aware mapping with calibration:

```bash
.venv311/bin/python main.py \
  --video videos/test.mp4 \
  --mapping-backend depth \
  --camera-calibration calibration/camera.json \
  --run-annotations data/annotations/mapping_benchmark_gt.json \
  --run-report-out reports/runtime/run_report_test.json
```

Calibration JSON may use either:

```json
{"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480}
```

or:

```json
{"camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]]}
```

## Runtime Flags

Important mapping/evaluation flags:

- `--no-mapping`
- `--mapping-backend heuristic|depth|orb_slam_like`
- `--camera-calibration <path>`
- `--benchmark-video <path>`
- `--run-annotations <path>`
- `--run-report-out <path>`
- `--headless`
- `--max-frames <n>`
- `--det-confidence <float>`
- `--det-imgsz <n>`
- `--map-grid-size`
- `--map-meters-per-cell`
- `--pose-motion-to-meter-scale`
- `--map-require-benchmark-for-promotion`
- `--map-obstacle-match-radius-cells <n>`
- `--map-benchmark-obstacle-metric cell|object`
- `--map-obstacle-footprint-radius-cells <n>`
- `--map-obstacle-footprint-shape square|horizontal|vertical|cross|class_aware`
- `--map-obstacle-temporal-persistence-frames <n>`

Reports keep the top-level fields `label_metrics`, `map_metrics`, `pose_stats`, `config`, `timing`, `warnings`, and `failures`.
Obstacle reports include cell-level metrics plus object-component metrics. Use `--map-benchmark-obstacle-metric object` when annotations describe obstacle blobs but the mapper emits object center cells.

## Offline Video Tools

Interactive labeling:

```bash
.venv311/bin/python video_processor.py label --video videos/room1.mp4 --session room1_fixed
```

Automatic rule-based labeling:

```bash
.venv311/bin/python video_processor.py autolabel --video videos/room1.mp4 --output data/raw/room1.csv
```

Offline inference with map inset:

```bash
.venv311/bin/python video_processor.py infer \
  --video videos/test.mp4 \
  --output out/test_annotated.mp4 \
  --mapping \
  --mapping-backend heuristic
```

## Training And Evaluation

The reasoning model is an action policy, not a SLAM model. It predicts one of:

- `AVOID_PERSON`
- `MOVE_TO_CHAIR`
- `CHECK_TABLE`
- `EXPLORE`

Run the guarded training pipeline:

```bash
scripts/run_reasoning_training_pipeline.sh --python-bin .venv311/bin/python
```

Current promotion status is tracked in `reports/promotion_summary.json`. If it says `not_promoted`, use the recommended actions there before replacing the promoted model.

## Deployment

The first supported deployment target is a single-machine CLI release.

- Operator runbook: `deployment/DEPLOYMENT_RUNBOOK.md`
- Release contract: `deployment/release_contract.json`
- Pre-release validation: `.venv311/bin/python scripts/validate_release.py --python-bin .venv311/bin/python`

The current promoted model is shipped under a temporary operational promotion policy. Treat that policy as release governance for this channel, not as the long-term research-quality bar.

## What Is Still Not Full VSLAM

To become real VSLAM, the project still needs calibrated camera motion, keyframes, persistent landmarks, triangulation/depth-scale grounding, pose graph optimization, relocalization, and true loop closure. The current map is a useful semantic occupancy grid for navigation experiments, but it should be described as heuristic/depth-aided mapping unless an external SLAM backend is integrated.

## Tests

```bash
.venv311/bin/python -m unittest discover -s tests -p 'test_*.py'
```
