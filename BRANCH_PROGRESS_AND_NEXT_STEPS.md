# Branch Summary: `cwehbe`

## Sprint Outcome (As of 2026-05-03)

The branch is operational with strict gates and repeatable training cycles, but the latest model is still not promotable.

What is currently true:
- Strict audit, preprocessing, review, and training gates are active end-to-end.
- Promotion baseline is initialized at `reports/metrics_promoted_baseline.json`.
- MLP-only training remains enforced.
- Latest normal cycle is **not promoted**.

---

## What Was Implemented

### 1. Pipeline governance
- Disk preflight + retention integrated in the guarded pipeline.
- Dataset manifest/changelog snapshots written each run.
- Review status gate (`pending` vs `applied`) enforced before preprocessing.

### 2. Data-cycle and correction workflow
- Targeted capture checklist created for table/chair edge cases.
- New real batches ingested (`batch_E`, `batch_F`, `batch_G`, `batch_H`) with correction files and audits.
- Targeted correction pass for `MOVE_TO_CHAIR` executed and retrained.

### 3. Promotion policy enforcement
- Promotion compares against promoted baseline artifact.
- Fresh-real hard improvement gate enforced:
  - `delta accuracy >= +0.10`
  - `delta macro_f1 >= +0.10`

---

## Current Data/Training State

### Active raw inputs
- `data/raw/curated_real_balanced.csv`
- `data/raw/media_labeled_batch_A_real_corrected.csv`
- `data/raw/media_labeled_batch_B_real_corrected.csv`
- `data/raw/media_labeled_batch_D_real.csv`
- `data/raw/media_labeled_batch_E_real.csv`
- `data/raw/media_labeled_batch_F_real.csv`
- `data/raw/media_labeled_batch_G_real.csv`
- `data/raw/media_labeled_batch_H_real.csv`

### Latest strict audit snapshot (`reports/dataset_audit.json`)
- files: `8`
- raw rows: `994`
- clean rows: `781`
- class counts:
  - `AVOID_PERSON`: `208`
  - `MOVE_TO_CHAIR`: `202`
  - `CHECK_TABLE`: `170`
  - `EXPLORE`: `201`
- imbalance ratio: `1.2235` (passes `<=1.3`)
- real share: `1.000` (passes `>=0.6`)
- synthetic share: `0.000` (passes `<=0.4`)
- independent real batches/sources: `6` (passes)
- `ready_for_training=true`

---

## Promotion Readiness (Latest)

Latest normal cycle is **not promoted** (`reports/promotion_summary.json`).

Gate summary vs promoted baseline:
- test non-regression: **PASS**
  - test accuracy: `0.7414 -> 0.7647`
- key-class non-regression: **FAIL**
  - `CHECK_TABLE` F1: `0.7368 -> 0.7018` (regression)
  - `MOVE_TO_CHAIR` F1: `0.7333 -> 0.7458` (pass)
- fresh-real hard improvement: **FAIL**
  - accuracy delta: `0.0` (required `+0.10`)
  - macro F1 delta: `0.0` (required `+0.10`)

Latest training metrics (`reports/metrics.json`):
- test accuracy: `0.7647`
- macro F1: `0.7626`
- fresh-real eval: accuracy `1.000`, macro F1 `0.250` on `12` rows

---

## Main Risk Right Now

Primary blocker is now concentrated in two areas:
- `CHECK_TABLE` retention is below promoted-baseline F1.
- Fresh-real holdout is saturated and too narrow (all-positive behavior), so improvement delta remains zero and fails policy.

---

## Next Steps (Priority Order)

1. Run targeted `CHECK_TABLE` correction pass.
- Relabel/drop ambiguous table-vs-chair rows from latest review artifacts.
- Prioritize true table-interaction contexts; drop ambiguous intent frames.

2. Improve fresh-real holdout diversity.
- Add at least two new independent real batches with mixed/occlusion/table-chair transition coverage.
- Ensure fresh holdout includes meaningful class variety, not single-class dominance.

3. Re-run guarded cycle and compare to promoted baseline.
- Keep strict gates unchanged.
- Promote only if all three pass together:
  - test non-regression
  - key-class non-regression
  - fresh-real hard improvement thresholds

---

## Important Constraints To Keep

- Keep reasoning model training as **MLP only**.
- Keep strict audit/preprocess/pipeline/promotion gates enabled.
- Keep default training source as real-first raw data (`data/raw`) with archive excluded unless explicitly opted in for experiments.
