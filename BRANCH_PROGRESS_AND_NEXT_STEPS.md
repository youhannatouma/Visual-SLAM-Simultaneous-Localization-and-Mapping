# Branch Summary: `cwehbe`

## What Has Been Done

### 1. End-to-end reasoning data pipeline added
- Added raw data audit script with quality gates:
  - `scripts/audit_reasoning_data.py`
- Added preprocessing/splitting script with balancing and validation:
  - `scripts/prepare_reasoning_data.py`
- Added one-command orchestrator:
  - `scripts/run_reasoning_training_pipeline.sh`

### 2. Data quality controls significantly strengthened
- Added minimum-per-class gate.
- Added stricter imbalance checks.
- Added source-aware quality gates:
  - minimum real-data share
  - maximum synthetic-data share
- Added source distribution reporting by class in audit output.

### 3. Data source tracking implemented
- `source_type` is now tracked across ingestion/generation/logging flows.
- Source types used:
  - `manual_live`
  - `real_media`
  - `synthetic`
  - `simulated`
  - `rebalance`

### 4. Media ingestion capability implemented
- Added automatic labeling from image/video folders:
  - `scripts/build_reasoning_data_from_media.py`
- Added lightweight review export for manual correction sampling.
- Added correction ingestion support for reviewed labels.

### 5. Synthetic/simulated generation utilities added
- Added simulation generator for large controlled datasets:
  - `scripts/generate_simulated_reasoning_data.py`
- Added balancing and patching flows used during experiments.

### 6. Training path constrained to MLP
- `train_reasoning.py` now enforces MLP usage (`--algorithm mlp`).
- Added metrics structure improvements and optional fresh-real evaluation support.

### 7. Live collection path aligned with pipeline
- `main.py` + `reasoning.py` updated so manual labeling writes compatible CSV schema.
- Session outputs now align better with raw pipeline expectations.

### 8. Documentation updated
- `README.md` updated with higher-quality dataset workflow, audit/prep commands, and stricter gate usage.

---

## Current State
- Latest strict audit can be made to pass with curated real-only balanced data.
- Current training completed using MLP on curated real set.
- Resulting model metrics are moderate-good but still limited by dataset breadth/variety and small final curated size.

---

## What Is Missing / Gaps

### Data & evaluation gaps
- Real-world data volume is still limited for robust generalization.
- Need broader real scenarios for difficult edge cases:
  - cluttered rooms
  - low light
  - partial occlusions
  - mixed chair/table scenes
  - long-range targets
- Fresh-real holdout workflow needs at least 2+ independent real source batches per cycle.

### Process gaps
- Need consistent dataset versioning (naming, changelog, reproducible snapshots).
- Need stable policy on when archived raw data is reintroduced vs excluded.
- Need regular manual review loop for auto-labeled media outputs.

### Infrastructure gaps
- Disk pressure is a recurring bottleneck during media ingestion.
- Need cleanup/retention policy for large media and intermediate artifacts.

---

## Where To Move Next

### Immediate next iteration (recommended)
1. Collect + ingest a larger **real-only** dataset (target: at least 3k–5k rows).
2. Preserve strict quality gates:
   - imbalance <= 1.3
   - real share >= 0.6
   - synthetic share <= 0.4
3. Build two real batches (Batch A / Batch B), hold out newest batch as fresh real eval.
4. Retrain with MLP and compare against previous metrics using promotion rules.
5. Only promote model if no unacceptable regression on:
   - overall test accuracy
   - `CHECK_TABLE` and `MOVE_TO_CHAIR` F1
   - fresh-real accuracy

### Short-term hardening
- Add dataset version manifest (`dataset_version`, counts, source mix, hashes).
- Add automated cleanup helper for stale media/report artifacts.
- Add dashboard-like report summary (single markdown/json snapshot per run).

---

## Important Constraints To Keep
- Continue training with **MLP only** for reasoning model.
- Keep strict data gates enabled in audit/prep/pipeline.
- Prefer real data growth over synthetic balancing for production-quality behavior.
