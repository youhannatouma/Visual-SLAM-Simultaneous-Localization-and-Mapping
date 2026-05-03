# Branch Summary: `cwehbe`

## Sprint Outcome (As of 2026-05-03)

The branch moved from a blocked holdout workflow to a fully runnable strict pipeline:

- Blocker resolved: the project now has at least 2 independent real batches.
- Strict data audit now passes under policy gates.
- End-to-end guarded pipeline completes (audit -> preprocess -> manifest -> train -> evaluate -> promotion).
- MLP-only training remains enforced.

Latest full pipeline run completed successfully with:

- `ready_for_training=true`
- test accuracy: `0.889`
- test macro F1: `0.889`
- fresh-real holdout accuracy: `0.256`
- fresh-real holdout macro F1: `0.197`

---

## What Was Implemented

### 1. Governance + guardrails now active in pipeline
- Added artifact/disk preflight and retention management:
  - `scripts/manage_artifacts.py`
- Added manifest/changelog snapshot writer:
  - `scripts/create_dataset_manifest.py`
- Integrated into pipeline orchestrator:
  - `scripts/run_reasoning_training_pipeline.sh`

### 2. Dataset quality controls strengthened
- Audit gate coverage includes:
  - per-class minimum checks
  - imbalance threshold checks
  - real/synthetic share checks
  - two-independent-real-batches check
  - scenario/class distribution reporting
- Script:
  - `scripts/audit_reasoning_data.py`

### 3. Review/correction loop is now enforced in flow
- Media ingestion supports review export + correction application:
  - `scripts/build_reasoning_data_from_media.py`
- Review status artifacts are generated under:
  - `reports/review_status/`

### 4. Preprocessing and holdout behavior hardened
- Holdout workflow and quality checks are integrated in preprocessing:
  - `scripts/prepare_reasoning_data.py`
- Fresh real holdout is generated during pipeline when policy prerequisites are satisfied.

### 5. Real batch ingestion completed and made policy-compliant
- Two independent real batches were produced and used.
- Corrected balanced batch files are now present in `data/raw/`:
  - `media_labeled_batch_A_real_corrected.csv`
  - `media_labeled_batch_B_real_corrected.csv`
- Older skewed media-labeled files were moved to `data/raw_archive/`.

---

## Current Data/Training State

### Active raw training inputs (current cycle)
- `data/raw/curated_real_balanced.csv`
- `data/raw/media_labeled_batch_A_real_corrected.csv`
- `data/raw/media_labeled_batch_B_real_corrected.csv`

### Latest strict audit result
- files: `3`
- raw rows: `816`
- clean rows: `648`
- class counts:
  - `AVOID_PERSON`: `160`
  - `MOVE_TO_CHAIR`: `168`
  - `CHECK_TABLE`: `160`
  - `EXPLORE`: `160`
- imbalance ratio: `1.05` (passes `<=1.3`)
- real share: `1.000` (passes `>=0.6`)
- synthetic share: `0.000` (passes `<=0.4`)
- independent real batches/sources: `2` (passes)

### Latest model evaluation
- Test set:
  - accuracy `0.889`
  - macro F1 `0.889`
- Fresh real holdout:
  - accuracy `0.256`
  - macro F1 `0.197`

---

## Main Risk Right Now

The pipeline is operational and policy-compliant, but generalization to fresh-real data is weak.

Interpretation:
- In-split performance is good.
- Out-of-source robustness is currently not good enough.

This is now the primary bottleneck for production confidence.

---

## Next Steps (Priority Order)

1. Expand real capture diversity before next promotion decision.
- Collect at least 2 new independent real batches (`batch_C_real`, `batch_D_real`).
- Focus on underrepresented real-world conditions:
  - low light
  - clutter
  - occlusion
  - mixed chair/table
  - longer distance scenes

2. Improve label quality with targeted review pass.
- Run mandatory review sampling on new media review files.
- Apply corrections before preprocessing each cycle.
- Track correction application in review status artifacts.

3. Keep strict gates unchanged.
- Continue enforcing:
  - imbalance <= `1.3`
  - real share >= `0.6`
  - synthetic share <= `0.4`
  - min-per-class gate
  - two-independent-real-batches gate for holdout

4. Re-run guarded pipeline and compare against previous run.
- Preserve MLP-only constraint.
- Compare:
  - test accuracy and macro F1
  - class-level F1 (`CHECK_TABLE`, `MOVE_TO_CHAIR` especially)
  - fresh-real holdout metrics

5. Promotion policy for upcoming cycle.
- Do not treat this cycle as final generalization success.
- Require fresh-real non-regression and meaningful improvement before final promotion sign-off.

---

## Important Constraints To Keep

- Keep reasoning model training as **MLP only**.
- Keep strict audit/preprocess/pipeline gates enabled.
- Keep default training source as real-first raw data (`data/raw`) with archived data excluded unless explicitly opted in for experiment mode.
