# Branch Summary: `cwehbe`

## Sprint Outcome (As of 2026-05-03)

The branch is now policy-compliant and operational with strict training gates enabled end-to-end.

What is now true:
- Strict audit gates pass reliably on current raw inputs.
- Two+ independent real batches are present and enforced.
- Review/correction status and correction-audit artifacts are integrated.
- Promotion framework with baseline + hard fresh-real improvement gate is active.
- MLP-only training remains enforced.

---

## What Was Implemented

### 1. Pipeline governance + reliability
- Disk preflight and retention integrated in training pipeline.
- Dataset manifest/changelog snapshots written each run.
- Archive usage remains opt-in experiment mode only.

### 2. Data quality and review hardening
- Strict audit gates: min-per-class, imbalance, real/synthetic shares, two-real-batches.
- Review status gate (`pending` vs `applied`) enforced before preprocessing.
- Correction audit JSON now produced for media ingestion with machine-checkable gate fields.

### 3. Promotion policy upgrade
- Promotion now compares against promoted baseline metrics artifact.
- Fresh-real improvement is now a hard gate:
  - accuracy delta >= `+0.10`
  - macro F1 delta >= `+0.10`
- Promotion summary is emitted as `reports/promotion_summary.json`.

### 4. Data-cycle automation
- Added cycle runner script for batch ingest/correction/pipeline execution.
- Batch/scenario coverage and correction audit artifacts are emitted per cycle.

---

## Current Data/Training State

### Active raw inputs
- `data/raw/curated_real_balanced.csv`
- `data/raw/media_labeled_batch_A_real_corrected.csv`
- `data/raw/media_labeled_batch_B_real_corrected.csv`
- `data/raw/media_labeled_batch_C_real.csv`
- `data/raw/media_labeled_batch_D_real.csv`

### Latest strict audit snapshot
- files: `5`
- raw rows: `1296`
- clean rows: `723`
- class counts:
  - `AVOID_PERSON`: `180`
  - `MOVE_TO_CHAIR`: `185`
  - `CHECK_TABLE`: `179`
  - `EXPLORE`: `179`
- imbalance ratio: `1.03` (passes `<=1.3`)
- real share: `1.000` (passes `>=0.6`)
- synthetic share: `0.000` (passes `<=0.4`)
- independent real batches/sources: `3` (passes)
- `ready_for_training=true`

---

## Promotion Readiness (Latest)

Latest run is still **not promoted**.

From `reports/promotion_summary.json`:
- fresh-real hard gate: **PASS**
  - fresh-real accuracy delta: `+0.5709`
  - fresh-real macro F1 delta: `+0.6409`
- test/key-class non-regression gates: **FAIL**
  - test accuracy: `0.8889 -> 0.8113` (regression)
  - `CHECK_TABLE` F1: `0.8485 -> 0.7500` (regression)
  - `MOVE_TO_CHAIR` F1: `0.9189 -> 0.7143` (regression)

Interpretation:
- Out-of-source robustness improved strongly.
- In-split and key-class retention against promoted baseline regressed.
- Promotion is blocked by non-regression policy, not by fresh-real gate anymore.

---

## Main Risk Right Now

Primary risk has shifted from “fresh-real weakness” to a **tradeoff instability**:
- Improving fresh-real robustness is currently reducing test/key-class baseline retention.

This is now the blocker for production promotion.

---

## Next Steps (Priority Order)

1. Recover key-class retention without losing fresh-real gains.
- Focus data expansion on high-quality `MOVE_TO_CHAIR` and `CHECK_TABLE` hard cases.
- Prioritize true positive table/chair contexts with reduced label ambiguity.

2. Keep strict gates unchanged.
- Continue enforcing:
  - imbalance <= `1.3`
  - real share >= `0.6`
  - synthetic share <= `0.4`
  - min-per-class gate
  - two-independent-real-batches gate for holdout
  - fresh-real hard improvement gate

3. Improve review/correction quality on targeted classes.
- Use correction audits to track relabel/drop patterns per class.
- Increase sampled QA scrutiny for `CHECK_TABLE` and `MOVE_TO_CHAIR` rows.

4. Re-run guarded pipeline and compare to promoted baseline.
- Preserve MLP-only constraint.
- Promote only when all gates pass together:
  - test non-regression
  - key-class F1 non-regression
  - fresh-real hard improvement thresholds

---

## Important Constraints To Keep

- Keep reasoning model training as **MLP only**.
- Keep strict audit/preprocess/pipeline/promotion gates enabled.
- Keep default training source as real-first raw data (`data/raw`) with archive excluded unless explicitly opted in for experiments.
