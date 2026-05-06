# Branch Summary: `cwehbe`

## Sprint Outcome (As of 2026-05-03)

The branch is operational with strict gates and repeatable training cycles, but the latest model is still not promotable.

---

## Update (2026-05-05) - Balanced Run (Promotion Ignored)

Summary of what happened:
- Ran a balanced training cycle using `data/raw_balanced/*.csv` (200 rows, 50 per class; 19% rebalance rows).
- Generated a new model and metrics, and refreshed the confusion matrix.
- Promotion gate was intentionally ignored for this cycle.

Latest balanced-run metrics (`reports/metrics.json`):
- test accuracy: `0.750`
- macro F1: `0.711`
- fresh-real accuracy: `0.762`
- fresh-real macro F1: `0.658`

Confusion matrix highlights (test set):
- `CHECK_TABLE` -> `EXPLORE` is the dominant error (4/5).
- `AVOID_PERSON` -> `MOVE_TO_CHAIR` appears occasionally (1/5).
- `MOVE_TO_CHAIR` and `EXPLORE` are otherwise stable in test.

Fresh-real error pattern:
- `CHECK_TABLE` collapses to `EXPLORE` (16/17).
- `AVOID_PERSON` sometimes flips to `MOVE_TO_CHAIR` (4/25).

What is missing / still needed:
- More real `CHECK_TABLE` examples to break the table-vs-explore collapse.
- Better real `EXPLORE` diversity (not just table-focus scenes).
- Reduce reliance on rebalance rows by collecting additional real batches.

What is currently true:
- Strict audit, preprocessing, review, and training gates are active end-to-end.
- Promotion baseline is initialized at `reports/metrics_promoted_baseline.json`.
- MLP-only training remains enforced.
- Latest normal cycle is **not promoted**.

---

## Update (2026-05-07) - MOVE_TO_CHAIR Recovery Pool + Seed Sweep

Summary of what happened:
- Built a reviewed-only `MOVE_TO_CHAIR` recovery pool from reviewed archive + staging + raw sources.
- Expanded candidates and curated to hard/ambiguous rows (needs_review=1 or auto_label != label).
- Ran 3-seed sweep (17/42/123) using oversample; no non-regressing run vs promoted baseline.

Seed sweep highlights (best-of, not promotable):
- Best `MOVE_TO_CHAIR` F1: `0.716` (seed `42`)
- Test accuracy: `0.776`
- Fresh-real macro F1: `0.622`

Notes:
- Clean rows stayed at `913` despite added recovery pool rows (dedup is filtering out duplicates).
- Fresh-real holdout size dropped to `36` rows (eval rows `27`), making deltas noisy.

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
- `data/raw/media_labeled_stage2_check_table_train_20260506.csv`
- `data/raw/media_labeled_stage2_curated_train_20260506.csv`
- `data/raw/media_labeled_stage2_train_refix_20260506.csv`
- `data/raw/move_recovery_pool_20260507.csv`
- `data/raw/session_20260507_002204.csv` (live labeling, in progress)
- `data/raw/zz_fresh_real_holdout_20260506.csv`

### Latest strict audit snapshot (seed sweep 2026-05-07)
- files: `9` (seed sweep 2026-05-07, pre live session)
- raw rows: `1646`
- clean rows: `913`
- class counts:
  - `AVOID_PERSON`: `208`
  - `MOVE_TO_CHAIR`: `242`
  - `CHECK_TABLE`: `256`
  - `EXPLORE`: `207`
- imbalance ratio: `1.24` (passes `<=1.3`)
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
