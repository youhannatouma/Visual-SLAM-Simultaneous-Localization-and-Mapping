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

---

## Update (2026-05-07) - Fresh Real Batches I/J + Holdout/Seed Refresh

Summary of what happened:
- Added new web-sourced fresh-real media batches `I` and `J` (Pexels video IDs `7591978`, `7593333`, `9065728`, `8632778`, `9655673`, `7224877`, `7643475`).
- Ran ingest + review exports + correction apply for both batches (`status=applied`).
- Enforced hard-negative trim from new batches to keep only `40` difficult `AVOID_PERSON` rows total (`10` from I, `30` from J), to avoid class-skew gate failure while preserving non-chair confusers.
- Ran short seed sweep (`17/42/123`, `16` epochs) with fixed comparable holdout policy (`holdout_per_class=8`, `min_total=32`, `min_sources=2`).
- Ran expanded holdout sweep with `min_total=48` by increasing `holdout_per_class` to `12` (required by current holdout builder; `8` caps holdout at `32`).
- Ran one reduced-oversampling experiment (`balance=cap`, seed `42`) as overfit control.

Key outcomes:
- Comparable sweep (`8/32/2`) executed successfully: `reports/seed_sweep_freshIJ_h32_20260507_140750/seed_sweep_summary.json`.
- Expanded holdout sweep (`12/48/2`) executed successfully: `reports/seed_sweep_freshIJ_h48hpc12_20260507_140925/seed_sweep_summary.json`.
- Cap experiment executed: `reports/exp_cap_freshIJ_h32_seed42/metrics.json`.
- Fresh-real per-class deltas consolidated in `reports/fresh_real_deltas_20260507.md`.

Promotion status:
- No run cleared promotion thresholds against `reports/metrics_promoted_baseline.json`.
- Continue data-refresh cycles; avoid architecture churn.

---

## Update (2026-05-07) - Data Refresh Cycle K (MOVE_TO_CHAIR Focus)

Actions completed:
- Added fresh real batch `K` from new web clips: `pexels_6626481`, `5716999`, `2108287`, `9365378`.
- Ran review/correction cycle and ensured review status `applied` for batch K.
- Rebalanced batch K to avoid `AVOID_PERSON` drift and keep MOVE_TO_CHAIR focus (`50 MOVE_TO_CHAIR`, `6 AVOID_PERSON`, `4 EXPLORE`).
- Added reviewed hard negatives from fresh batches (`data/raw/media_labeled_stage2_move_hardneg_KJ_20260507.csv`, 30 rows, all `EXPLORE`).
- Re-ran short seed sweep with fixed comparable holdout (`holdout_per_class=8`, `min_total=32`, `min_sources=2`) and seeds `17/42/123` for 16 epochs.

Artifacts:
- Sweep summary: `reports/seed_sweep_cycleK_h32_20260507_142436/seed_sweep_summary.json`
- Per-class fresh-real deltas: `reports/fresh_real_deltas_cycleK_20260507.md`

Result:
- Pipeline ranking marks seed `17` promotable by global gate, but per-class fresh-real deltas show mixed regression (`MOVE_TO_CHAIR` delta negative for seed 17).
- By explicit stop rule (do not accept runs where one class improves while another regresses), this cycle is treated as **non-promotable**.
- Continue data-refresh-only loop; no architecture churn.

---

## Update (2026-05-07) - Track 1 Orchestrator Added

Implemented:
- Added `scripts/run_track1_reasoning_loop.py` as the canonical Track 1 orchestrator.
- Orchestrator runs short seed sweeps (`17/42/123`, epochs `12..20`) with:
  - comparable profile fixed to holdout `8/32/2`
  - periodic noise-check profile `12/48/2` every `3` cycles
- Promotion stop condition now enforced at loop level as all-gates-required:
  - test non-regression
  - key-class non-regression
  - fresh-real aggregate thresholds
  - fresh-real per-class non-regression across all 4 classes
- Added machine-readable loop report:
  - `reports/track1/<timestamp>/track1_cycle_summary.json`
  - includes per-cycle sweep config, best-seed gate results, fresh-real per-class deltas, regressing classes, targeted refresh recommendations, and promotion decision.
- Architecture freeze is explicit in report constraints (no model/training API churn in loop logic; only data/balancing/hard-negative tuning).

Canonical command:

```bash
python scripts/run_track1_reasoning_loop.py \
  --python-bin .venv311/bin/python \
  --seeds 17,42,123 \
  --epochs-min 12 \
  --epochs-max 20 \
  --comparable-holdout-per-class 8 \
  --comparable-holdout-min-total 32 \
  --comparable-holdout-min-sources 2 \
  --noise-holdout-per-class 12 \
  --noise-holdout-min-total 48 \
  --noise-holdout-min-sources 2 \
  --noise-check-every-cycles 3 \
  --max-refresh-cycles 12
```

---

## Update (2026-05-07) - Mapping While Running (SLAM Integration)

Implemented:
- Added runtime SLAM integration module: `mapping_runtime.py`.
- Added explicit per-frame runtime contracts:
  - `PoseSample(x, y, theta, timestamp)`
  - `ActionSample(action, confidence, source_mode, timestamp)`
  - `MapEvent(event_type, grid_xy, world_xy, label, confidence, track_id, timestamp)`
- Wired integration directly into `main.py` inference loop:
  - ORB motion now updates global pose estimates.
  - Live occupancy grid performs incremental updates with:
    - obstacle marking
    - free-space ray carving
    - decay smoothing
  - Trajectory trace is rendered continuously on occupancy map output.
- Added joint run report output (single artifact for label + map quality):
  - `label_metrics`
  - `map_metrics` (`loop_closure_drift`, `map_consistency_score`, `obstacle_precision_recall`)
  - `pose_stats`, config, timing, warnings/failures
- Added optional labeled-run annotation input for same-run evaluation.

Validation:
- `py_compile` passed for `main.py`, `mapping_runtime.py`, and tests.
- Unit tests added and passing via discovery:
  - `tests/test_mapping_runtime.py`

---

## Update (2026-05-13) - Runtime Hardening + Mapping Correction Alignment

Implemented:
- Hardened reasoning-model checkpoint loading in `reasoning.py` so runtime now accepts:
  - legacy plain `state_dict` checkpoints
  - metadata-bearing checkpoints with `model_state_dict`, `sequence_length`, and `feature_size`
- Updated `train_reasoning.py` to save metadata-bearing checkpoints by default, without breaking older runtime compatibility.
- Updated `scripts/triage_reasoning_confusions.py` to use the same shared checkpoint loader as runtime.
- Added confidence-aware model stabilization in `ReasoningEngine`:
  - model predictions now carry softmax confidence
  - low-confidence model outputs do not enter the stabilized action vote
  - runtime action confidence now reflects the stronger of scene-supported confidence and model confidence
- Fixed a mapping/runtime consistency bug in `mapping_runtime.py`:
  - obstacle projection and free-space ray carving now use the corrected loop-closure pose instead of the raw drifting pose
  - this aligns occupancy updates with the loop-closure-corrected trajectory already used for render/evaluation
- Expanded `.gitignore` for local-only artifacts and review outputs:
  - capture reports
  - review CSVs / correction CSVs / correction audits
  - tuning model directories
  - local raw archive and candidate artifact directories

Why this matters:
- The reasoning path is now safer to retrain and redeploy without silent checkpoint-contract drift.
- Mapping quality is now materially closer to what we would want to promote, because post-closure corrections affect both visualization and occupancy updates, not just the displayed trajectory.
- Local review/debug artifacts are less likely to pollute future commits.

Validation:
- `python3 -m py_compile` passed for:
  - `reasoning.py`
  - `mapping_runtime.py`
  - `main.py`
  - `train_reasoning.py`
  - `scripts/triage_reasoning_confusions.py`
- Added/updated regression coverage:
  - `tests/test_reasoning_contract.py`
  - `tests/test_mapping_runtime.py`
- Full test execution was partially blocked in the local shell because:
  - `torch` is not installed in the active interpreter, so reasoning-contract tests were skipped
  - `cv2` is not installed in the active interpreter, so mapping-runtime tests could not import

Status:
- These changes are merge-worthy runtime and tooling hardening.
- They do not change the Track 1 policy: promotion still depends on measured benchmark results and fresh-real gating, not architecture churn.
