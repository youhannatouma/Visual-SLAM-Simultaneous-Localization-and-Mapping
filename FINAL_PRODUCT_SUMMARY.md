# Final Product Summary

This repository implements an AI visual navigation prototype with VSLAM-oriented mapping. The system combines object detection, visual motion estimation, a learned reasoning/action policy, semantic occupancy-grid mapping, and guarded data/training pipelines.
The current product is best described as an AI navigation and mapping demo, not a complete production Visual SLAM stack. It has useful SLAM-like runtime behavior: pose tracking from visual motion, semantic obstacle projection, a live map, loop-closure-style correction, benchmark reports, and promotion gates. It does not yet include full keyframe SLAM, triangulated landmarks, bundle adjustment, pose graph optimization, relocalization, or true loop closure from a dedicated SLAM backend.

## How The Final Product Works

At runtime, video frames come from a webcam or video file. The processing flow is:

```text
Camera or video
  -> YOLO object detection
  -> object tracking and feature extraction
  -> ORB visual motion estimation
  -> optional MiDaS relative depth
  -> reasoning policy: MLP model with rule fallback
  -> semantic occupancy-grid mapper
  -> HUD, map window, and JSON run report
```

The main runtime entry point is `main.py`.

1. `main.py` opens the camera or video source and resizes frames to `640x480`.
2. A detection worker loads `yolov8n.pt` through Ultralytics YOLO and detects objects such as `person`, `chair`, `table`, `sofa`, and `tv`.
3. A reasoning worker runs `ReasoningEngine` from `reasoning.py`.
4. The reasoning engine tracks objects by frame-to-frame center distance, smooths boxes, extracts a fixed 43-feature vector, and predicts one of four navigation actions:
   - `AVOID_PERSON`
   - `MOVE_TO_CHAIR`
   - `CHECK_TABLE`
   - `EXPLORE`
5. The runtime estimates visual motion with ORB features and robust affine matching.
6. `LiveMapper` in `mapping_runtime.py` updates a semantic occupancy grid using the estimated pose and object detections.
7. The UI overlays bounding boxes, current action, mode, motion, detection confidence, and an occupancy map.
8. At the end of a run, the system writes a JSON report with label metrics, mapping metrics, pose stats, timing, warnings, action samples, and map events.

## Main Runtime Components

### YOLO Object Detection

The detector is loaded with:

```python
model = YOLO("yolov8n.pt")
```

The repository uses YOLO for inference and for generating reasoning datasets from media. The detector returns bounding boxes, class names, confidence values, centers, and areas. Those detections become the input to the reasoning model and mapper.

Important note: this repository does not contain a YOLO fine-tuning script, YOLO dataset YAML, YOLO annotation conversion pipeline, or custom YOLO training command. In the checked-in code, `yolov8n.pt` is treated as an existing detector weight file. So the YOLO model was not trained inside this codebase; it is used as a pretrained object detector.

### Reasoning Model

The reasoning model is an MLP action policy, not a SLAM model. It predicts the robot/navigation intent from object detections and motion context.

The model input is a sequence of 10 frames. Each frame has 43 features:

- 3 object slots.
- Each object slot contains a one-hot object label plus confidence, area ratio, horizontal offset, and vertical offset.
- Motion one-hot vector for `NONE`, `LEFT`, `RIGHT`, `UP`, `DOWN`.
- A motion-present flag.
- Normalized object count.
- Short-term temporal memory showing which object labels were seen in the previous frame.

The model architecture in `reasoning.py` is:

```text
input sequence -> flatten
  -> Linear(430, 256) + BatchNorm + ReLU + Dropout
  -> Linear(256, 128) + BatchNorm + ReLU + Dropout
  -> Linear(128, 64) + ReLU + Dropout
  -> Linear(64, 4 action classes)
```

At runtime, the model is used only when:

- `models/reasoning_model.pt` exists,
- the checkpoint feature schema matches the runtime feature schema,
- enough sequence history is available,
- prediction confidence is at least the configured threshold.

If any of those fail, the system falls back to deterministic rules. The rule fallback prioritizes nearby/central people, then chairs, then tables, then exploration.

### Mapping

`mapping_runtime.py` provides the live semantic mapper. It stores:

- `PoseSample`: estimated pose `(x, y, theta, timestamp)`.
- `ActionSample`: action label, confidence, source mode, timestamp.
- `MapEvent`: obstacle and free-space updates in grid/world coordinates.

The mapper supports three backend modes:

- `heuristic`: estimates obstacle range from bounding-box area and object class.
- `depth`: uses depth-map values when MiDaS depth is enabled.
- `orb_slam_like`: records an external-backend-required contract, but still uses the local fallback path. It is not a bundled ORB-SLAM implementation.

Detected objects are projected into world/grid space using bearing and estimated range. Obstacle cells are reinforced, and free-space rays are carved between the camera pose and the obstacle. The mapper also supports confidence weighting, temporal persistence, object footprints, and a lightweight loop-closure-style correction when the path revisits a nearby previous pose.

### Evaluation And Reports

Runtime reports can compare predictions against benchmark annotations. Annotation JSON files may include:

- `frame_labels`: per-frame action labels.
- `obstacles`: obstacle grid cells, grid-cell groups, or component cells.

The report includes:

- action classification accuracy and macro F1,
- per-class precision/recall/F1,
- pose jitter,
- loop-closure drift,
- map consistency,
- obstacle persistence,
- occupancy confidence concentration,
- cell-level and object-component obstacle precision/recall,
- mapping quality gate summary.

## How We Trained The Reasoning Model

The canonical reasoning training path is:

```bash
scripts/run_reasoning_training_pipeline.sh --python-bin .venv311/bin/python
```

That guarded pipeline performs:

1. Disk preflight and artifact retention with `scripts/manage_artifacts.py`.
2. Raw dataset audit with `scripts/audit_reasoning_data.py`.
3. Dataset preparation with `scripts/prepare_reasoning_data.py`.
4. Dataset manifest/changelog creation with `scripts/create_dataset_manifest.py`.
5. MLP training with `train_reasoning.py`.
6. Promotion evaluation with `scripts/run_promotion_summary.py`.

The model is trained from CSV rows containing `f0..f42` feature columns plus a `label` column. Labels are the four action classes:

- `AVOID_PERSON`
- `MOVE_TO_CHAIR`
- `CHECK_TABLE`
- `EXPLORE`

`train_reasoning.py` builds sequence windows of length 10. The label for a sequence is the label of the last row in that sequence. Training uses PyTorch, Adam, cross-entropy loss, optional label smoothing, optional weighted sampling, and deterministic seeds.

The training script saves:

- `models/reasoning_model.pt`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

Metrics include test accuracy, macro F1, per-class precision/recall/F1, confusion matrix, and optional fresh-real evaluation metrics.

## How We Processed Data And Annotations

### Raw Data Sources

The reasoning dataset is built from several sources:

- manual/live labeling sessions,
- interactive video labeling,
- automatic rule-based video labeling,
- image/video media ingestion,
- curated real batches.

Raw CSV files live mainly under `data/raw/`. Processed train/validation/test splits live under `data/processed/`.

### Manual And Automatic Labeling

`video_processor.py` supports:

- `label`: interactive video labeling with keyboard labels.
- `autolabel`: automatic rule-based labels from YOLO detections.
- `infer`: offline inference and annotated video export.

Interactive labels use:

| Key | Label |
| --- | --- |
| `A` | `AVOID_PERSON` |
| `C` | `MOVE_TO_CHAIR` |
| `T` | `CHECK_TABLE` |
| `E` | `EXPLORE` |

The `ReasoningEngine.log_example()` method writes accepted examples to CSV after quality gates. It rejects samples with too many objects, tiny objects, or low average detection confidence.

### Media-Based Data Generation

`scripts/build_reasoning_data_from_media.py` builds reasoning CSVs from image and video folders. It:

- loads images/videos,
- runs YOLO detection,
- samples video frames by stride,
- infers simple optical-flow motion,
- assigns an automatic action label from detections,
- extracts the same 43-feature schema as runtime,
- writes rows with source metadata,
- exports review CSVs,
- applies optional correction CSVs,
- writes correction audits, review status JSON, and coverage reports.

Rows can be marked `needs_review` when cases are ambiguous, weak, crowded, or mixed chair/table scenes. Correction CSVs use:

```text
row_id,final_label,drop_row
```

This lets a reviewer relabel or drop generated rows before training.

### Dataset Audit

`scripts/audit_reasoning_data.py` checks raw CSV quality before training:

- required label and feature columns,
- finite numeric features,
- valid action labels,
- duplicate rows,
- class distribution,
- class imbalance ratio,
- real-data share,
- independent real batch/source count,
- optional scenario quotas.

The audit writes `reports/dataset_audit.json` and sets `ready_for_training`.

### Dataset Preparation

`scripts/prepare_reasoning_data.py` turns raw CSVs into model-ready splits. It:

- merges matching raw CSVs,
- excludes known holdout/archive/experimental files by policy,
- validates and cleans rows,
- infers `source_type`,
- enforces minimum rows per class,
- enforces real-data share gates,
- enforces applied review status,
- optionally builds a deterministic multi-source fresh-real holdout,
- balances classes by `none`, `cap`, or `oversample`,
- writes `train.csv`, `val.csv`, `test.csv`, `fresh_real_eval.csv`, and `metadata.json`.

The fresh-real holdout is important because it tests the model on real rows from independent sources rather than only the training distribution.

### Benchmark Mapping Annotations

Mapping benchmark annotations live in JSON files under `data/annotations/`. The loader accepts:

- frame-level action labels,
- obstacle cells through `grid_xy`,
- obstacle lists through `grid_cells`,
- obstacle components through `component_cells`.

These annotations are used by `main.py` and `mapping_runtime.py` to calculate label metrics and obstacle precision/recall.

## How The YOLO Model Was Trained

Inside this repository, YOLO is not trained. The code uses `yolov8n.pt` as an existing Ultralytics YOLO weight file.

YOLO is used for:

- runtime object detection in `main.py`,
- offline labeling/inference in `video_processor.py`,
- media ingestion in `scripts/build_reasoning_data_from_media.py`.

The repository does not include:

- custom YOLO image annotations,
- YOLO label text files,
- a `data.yaml`,
- a YOLO training command,
- a fine-tuning script,
- model evaluation reports for custom YOLO training.

So the correct codebase-level explanation is: the YOLO detector was not trained here; the project relies on a pretrained YOLO model for detecting semantic objects, then trains a separate reasoning MLP on top of those detections.

If a custom YOLO model is trained later, the runtime can use it by replacing the model path, because both `main.py` and `video_processor.py` accept YOLO model paths.

## What Each Script Does

### `main.py`

Main live/video runtime. It loads YOLO, starts detection and reasoning workers, estimates ORB motion, optionally loads MiDaS depth, updates `LiveMapper`, displays HUD/map/depth windows, supports keyboard labeling, and writes runtime JSON reports.

### `reasoning.py`

Defines the reasoning model and decision engine. It includes constants for labels/actions, checkpoint loading, the MLP architecture, object tracking, bbox smoothing, temporal feature extraction, model inference, confidence smoothing, rule fallback, action formatting, and CSV logging for labeled examples.

### `mapping_runtime.py`

Implements semantic occupancy-grid mapping and mapping evaluation. It defines pose/action/map dataclasses, camera calibration loading, pose updates from visual flow, depth/heuristic object projection, obstacle/free-space grid updates, loop-closure-style correction, map rendering, label metrics, pose/map quality metrics, obstacle precision/recall, annotation loading, and report writing.

### `video_processor.py`

Offline video utility with three subcommands:

- `label`: interactively label frames and save feature rows.
- `autolabel`: sample video frames and label them with rules.
- `infer`: run inference on a video and save an annotated MP4, optionally with a map inset.

### `train_reasoning.py`

Trains the MLP reasoning policy. It loads cleaned feature CSVs, creates 10-frame sequence windows, optionally weights reviewed real rows and hard negatives, trains with Adam and cross-entropy, saves the best validation checkpoint, evaluates on test and fresh-real sets, and writes metrics plus a confusion matrix.

### `merge.py`

Legacy/simple utility that concatenates all `data/raw/*.csv` files into `data/processed/full_dataset.csv`.

### `split_dataset.py`

Legacy/simple utility that stratifies `data/processed/full_dataset.csv` into `train.csv`, `val.csv`, and `test.csv`.

### `analyze_dataset.py`

Small dataset inspection script. It reads `data/processed/full_dataset.csv`, drops duplicates, and prints label counts and percentages.

### `scripts/run_reasoning_training_pipeline.sh`

Canonical guarded training pipeline. It runs disk checks, dataset audit, preprocessing, manifest creation, training, and promotion summary generation with configurable gates and holdout settings.

### `scripts/manage_artifacts.py`

Checks free disk space, optionally prunes old images/reports/raw archives, and writes an artifact budget report.

### `scripts/audit_reasoning_data.py`

Audits raw reasoning CSVs before training. It checks schema, numeric validity, labels, duplicates, class balance, source quality, real-batch diversity, and optional scenario quotas. Writes `reports/dataset_audit.json`.

### `scripts/prepare_reasoning_data.py`

Cleans, validates, balances, and splits raw reasoning data. Also builds a fresh-real holdout and enforces review-status gates.

### `scripts/create_dataset_manifest.py`

Creates a reproducible dataset manifest with file hashes, row counts, source/class distributions, processed file hashes, and a combined fingerprint. Also appends to a dataset changelog.

### `scripts/build_reasoning_data_from_media.py`

Builds reasoning training CSVs from image/video folders using YOLO detections. It writes review exports, applies reviewer corrections, writes correction audits, records review status, and reports batch coverage.

### `scripts/build_stage2_real_sets.py`

Builds stage-2 real holdout and training augmentation CSVs from archive/staging candidates while avoiding duplicates already present in active raw data. It can focus on hard negatives for a target class inferred from fresh-real metrics.

### `scripts/run_promotion_summary.py`

Compares current training metrics against the promoted baseline. It checks test accuracy, key-class F1, fresh-real aggregate metrics, fresh-real minimum rows, optional absolute floors, and optional per-class regression. Writes `reports/promotion_summary.json`.

### `scripts/run_reasoning_seed_sweep.py`

Runs the guarded training pipeline across multiple random seeds. It stores per-seed processed data, models, metrics, and promotion summaries, ranks runs by fresh-real and test metrics, and can promote the best promotable seed to canonical outputs.

### `scripts/run_track1_reasoning_loop.py`

Higher-level Track 1 orchestration loop. It alternates comparable and real-recovery seed sweeps, evaluates fresh-real regressions, records recovery summaries, recommends targeted data refreshes, and enforces architecture-freeze constraints.

### `scripts/triage_reasoning_confusions.py`

Loads a trained reasoning model and a CSV split/holdout, runs sequence predictions, and exports rows matching a target confusion pair such as `CHECK_TABLE -> MOVE_TO_CHAIR`. Useful for manual review and targeted correction.

## Tests

The test suite is under `tests/` and focuses on:

- mapping runtime behavior,
- reasoning contract compatibility,
- fresh-real recovery and seed-sweep ranking behavior.

Run tests with:

```bash
.venv311/bin/python -m unittest discover -s tests -p 'test_*.py'
```
## Current Limitations

- YOLO is used as a pretrained detector in this repo; custom YOLO training is not present.
- The reasoning model predicts actions, not camera pose or SLAM state.
- Mapping is semantic occupancy-grid mapping with heuristic/depth projection, not full Visual SLAM.
- The `orb_slam_like` mode is a contract placeholder requiring an external backend.
- Mapping quality depends heavily on camera calibration, depth quality, object detector quality, and benchmark annotations.
- Some legacy scripts (`merge.py`, `split_dataset.py`, `analyze_dataset.py`) are simpler than the guarded pipeline and should be treated as exploratory utilities.
