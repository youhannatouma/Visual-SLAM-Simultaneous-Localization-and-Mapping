# 🧠 AI Visual Navigation System

### Visual Odometry + Object Detection + AI Reasoning

---

## 📌 Overview

This project implements a **real-time AI-driven visual navigation system** that combines:

* Computer Vision (feature tracking)
* Deep Learning (object detection)
* AI Logic (rule-based reasoning)

The system uses a camera to **perceive its environment**, estimate motion, and **make decisions based on detected objects**.

---

## 🎯 Objectives

* Detect objects in real time using AI
* Estimate camera motion using visual features
* Build a simple understanding of the environment
* Apply AI reasoning to make decisions

---

## 🧠 System Architecture

```
Camera Input
    ↓
Object Detection (YOLOv8)
    ↓
Feature Tracking (ORB - Visual Odometry)
    ↓
State Interpretation
    ↓
AI Reasoning Engine
    ↓
Decision Output
    ↓
Visualization
```

---

## ⚙️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy

---

## 🔍 Features

* ✅ Real-time object detection
* ✅ ORB feature tracking (motion awareness)
* ✅ Rule-based AI reasoning system
* ✅ Live decision overlay
* ✅ Lightweight and real-time performance

---

## 🧠 AI Reasoning System

The system includes a **rule-based decision engine**:

* IF a *chair* is detected → **Target the chair**
* IF a *person* is detected → **Avoid**
* ELSE → **Explore the environment**

This demonstrates **symbolic AI logic and decision-making** based on perception.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the project

```bash
python main.py
```

---

## 📷 Demo

The system will:

* Open your webcam
* Detect objects in real time
* Track visual features
* Display decisions such as:

  * TARGET CHAIR
  * AVOID PERSON
  * EXPLORE

---

## 📁 Project Structure

```
project/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👥 Team Roles

* **Vision & Detection** → Object detection (YOLO)
* **Motion Estimation** → Feature tracking (ORB)
* **AI Logic** → Reasoning system and decision-making

---

## ⏱️ Development Timeline

* Week 1 → Setup + detection + motion tracking
* Week 2 → Reasoning system + integration + demo

---

## 🎓 Academic Context

This project was developed for a **3rd-year AI course**, focusing on:

* Perception → reasoning → action pipeline
* Integration of deep learning and symbolic AI
* Real-time intelligent systems

---

## ⚠️ Limitations

* No full 3D mapping (uses simplified visual odometry)
* Basic rule-based reasoning (not learning-based)
* Approximate motion estimation

---

## 🚀 Future Improvements

* Add object tracking across frames
* Improve motion estimation accuracy
* Implement path planning (A*)
* Upgrade reasoning with probabilistic models
* Integrate with robotics (ROS)

---

## Training the Reasoning Model

This project includes a learnable reasoning engine with a pandas/numpy preprocessing workflow.

1. Run the app with `python main.py`.
2. Use the keyboard to label examples while the camera is running:
   * `a` → save `AVOID_PERSON`
   * `c` → save `MOVE_TO_CHAIR`
   * `t` → save `CHECK_TABLE`
   * `e` → save `EXPLORE`
3. Move or copy raw CSV logs to `data/raw/`.
4. Prepare train/val/test splits:

```bash
python scripts/prepare_reasoning_data.py --input-glob "data/raw/*.csv" --out-dir data/processed --balance cap --seed 42
```

5. Train the model:

```bash
python train_reasoning.py --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv --model models/reasoning_model.pt
```

6. Restart `main.py`.

The app auto-loads `models/reasoning_model.pt` when present.

### Local training guardrail

If `train + val + test` has **50,000+ rows**, `train_reasoning.py` exits and asks you to train remotely (Colab/server), then copy the trained `.pt` back into `models/`.

---

## Build a High-Quality Reasoning Dataset

Use this checklist before training:

1. Collect diverse scenes with `main.py`:
   * person close/centered and person off-center
   * chair near/far and partially occluded
   * table-focused scenes
   * empty scenes for `EXPLORE`
2. Keep labels balanced across all 4 actions (`a/c/t/e`) instead of over-logging one key.
3. Save session CSVs in `data/raw/` (multiple files are supported).
4. Audit quality first (real-data-first gates):

```bash
python scripts/audit_reasoning_data.py --input-glob "data/raw/*.csv" --min-per-class 50 --max-class-imbalance-ratio 1.3 --min-real-share 0.6 --max-synthetic-share 0.4 --report reports/dataset_audit.json
```

5. Prepare processed splits with a quality gate:

```bash
python scripts/prepare_reasoning_data.py --input-glob "data/raw/*.csv" --out-dir data/processed --balance cap --min-per-class 50 --min-real-share 0.6 --max-synthetic-share 0.4 --holdout-latest-real-source --seed 42
```

If any class is below `--min-per-class`, preprocessing fails and tells you which labels need more data.

6. Run the full guarded pipeline (audit -> preprocess -> train -> promotion checks):

```bash
scripts/run_reasoning_training_pipeline.sh --python-bin .venv311/bin/python
```

### Data source tracking

Each row is tagged with a `source_type`:

* `manual_live` (from `main.py` key labeling)
* `real_media` (from `build_reasoning_data_from_media.py`)
* `synthetic`, `simulated`, `rebalance`

Audit now reports class distribution by source and enforces:

* real share >= `0.6`
* synthetic/simulated/rebalance share <= `0.4`
* class imbalance <= `1.3` before balancing

### Lightweight media label review

When building from media, a review file is exported automatically:

```bash
python scripts/build_reasoning_data_from_media.py --media-dir /path/to/images_or_videos --video-stride 10
```

This creates:

* `data/raw/media_labeled_<timestamp>.csv`
* `reports/media_review_<timestamp>.csv` (sample + hard cases for manual correction)

You can apply review corrections by passing:

```bash
python scripts/build_reasoning_data_from_media.py --media-dir /path/to/images_or_videos --review-corrections reports/your_review_corrections.csv
```

Correction CSV columns: `row_id,final_label,drop_row`.

When corrections are provided, schema/label validation is strict. Invalid labels or `drop_row` values fail the run.
Each ingestion also writes a correction audit artifact:

- `reports/correction_audit_<review_file_stem>.json`

The audit includes relabel/drop/unchanged counts, override summary, QA sample evidence, and machine-checkable gate fields.

### Training algorithm policy

`train_reasoning.py` is locked to **MLP** (`--algorithm mlp` only).  
This project always trains reasoning with a multilayer perceptron.

## Real-Only Governance & Operations

### Default data policy

- Default training source is `data/raw/*.csv` (real-only workflow).
- `data/raw_archive` is excluded by default and only allowed in explicit experiments.
- Synthetic/simulated data should not enter default production training cycles.

### Batch-aware ingestion and coverage

Use `--batch-id` and `--scenario` when ingesting media:

```bash
python scripts/build_reasoning_data_from_media.py \
  --media-dir /path/to/media \
  --batch-id batch_A_real \
  --scenario low_light \
  --video-stride 10
```

Outputs now include:

- `reports/batch_coverage_<timestamp>.json` (scenario x class counts)
- `reports/review_status/<review_file_stem>.json` (`pending`/`applied`)

### Mandatory review gate

Preprocessing can enforce that review corrections are applied before training:

```bash
python scripts/prepare_reasoning_data.py \
  --input-glob "data/raw/*.csv" \
  --out-dir data/processed \
  --enforce-review-applied \
  --holdout-latest-real-source \
  --require-two-real-batches-for-holdout
```

### Dataset manifest & changelog snapshot

Generate immutable dataset metadata for each cycle:

```bash
python scripts/create_dataset_manifest.py \
  --input-glob "data/raw/*.csv" \
  --processed-dir data/processed \
  --manifest-path data/manifest/dataset_manifest.json \
  --changelog-path data/manifest/CHANGELOG.md
```

Capture plan template:

- `data/manifest/capture_plan_template.json`

### Disk guard and aggressive retention

Preflight + prune + budget report:

```bash
python scripts/manage_artifacts.py --min-free-gb 1.0 --prune --budget-report reports/artifact_budget.json
```

### End-to-end guarded run

```bash
scripts/run_reasoning_training_pipeline.sh --python-bin .venv311/bin/python
```

This run now enforces:

- strict audit gates
- two independent real batches
- review applied gate
- fresh real holdout policy
- dataset manifest snapshot
- disk preflight and retention
- MLP-only training with promotion checks

### Promotion baseline + fresh-real hard improvement gate

Promotion now uses a promoted baseline artifact:

- baseline file default: `reports/metrics_promoted_baseline.json`
- summary file default: `reports/promotion_summary.json`

Fresh-real promotion requires:

- `delta(fresh_real_eval.accuracy) >= +0.10`
- `delta(fresh_real_eval.macro_f1) >= +0.10`

Initialize baseline once (non-promotable establishment run):

```bash
scripts/run_reasoning_training_pipeline.sh \
  --python-bin .venv311/bin/python \
  --establish-promotion-baseline
```

Regular cycle run (requires improvement vs baseline):

```bash
scripts/run_reasoning_training_pipeline.sh \
  --python-bin .venv311/bin/python \
  --fresh-real-min-improve-acc 0.10 \
  --fresh-real-min-improve-macro-f1 0.10
```

### Real cycle runner (batch C / batch D workflow)

Use the cycle runner to execute:
ingest -> correction audit -> coverage report -> guarded pipeline.

```bash
scripts/run_real_data_cycle.sh \
  --python-bin .venv311/bin/python \
  --media-dir-a /path/to/batch_C_media \
  --media-dir-b /path/to/batch_D_media \
  --batch-a-id batch_C_real \
  --batch-b-id batch_D_real \
  --scenario-a clutter \
  --scenario-b occlusion \
  --review-corrections-a reports/corrections_batch_C_real.csv \
  --review-corrections-b reports/corrections_batch_D_real.csv \
  --pipeline-arg --fresh-real-min-improve-acc \
  --pipeline-arg 0.10 \
  --pipeline-arg --fresh-real-min-improve-macro-f1 \
  --pipeline-arg 0.10
```

Cycle summary output:

- `reports/cycle_report.json`

This report includes scenario coverage, correction-audit references, and gate fields for CI parsing.

---

## �📄 License

This project is for academic and educational use.

---

## 💬 Acknowledgments

* Open-source computer vision and AI libraries
* YOLOv8 by Ultralytics
* OpenCV community

---
