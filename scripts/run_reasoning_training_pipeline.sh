#!/usr/bin/env bash
set -euo pipefail

INPUT_GLOB="data/raw/*.csv"
OUT_DIR="data/processed"
REPORT="reports/dataset_audit.json"
REPORT_DIR="reports"
MANIFEST_PATH="data/manifest/dataset_manifest.json"
CHANGELOG_PATH="data/manifest/CHANGELOG.md"
MIN_PER_CLASS=50
MAX_IMBALANCE=1.5
MIN_REAL_SHARE=0.15
BALANCE="oversample"
SEED=42
TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=32
TRAIN_LR=1e-3
TRAIN_WEIGHT_DECAY=0.0
TRAIN_LABEL_SMOOTHING=0.0
COLLECT_FIRST=0
PYTHON_BIN="venv/Scripts/python.exe"
MODEL_PATH="models/reasoning_model.pt"
ALLOW_ARCHIVE_EXPERIMENT=0
DISK_MIN_FREE_GB=1.0
PRUNE=1
PROMOTION_BASELINE_PATH="reports/metrics_promoted_baseline.json"
PROMOTION_SUMMARY_PATH="reports/promotion_summary.json"
FRESH_REAL_MIN_IMPROVE_ACC=-0.06
FRESH_REAL_MIN_IMPROVE_MACRO_F1=-0.03
FRESH_REAL_ABSOLUTE_MIN_ACC=0.50
FRESH_REAL_ABSOLUTE_MIN_MACRO_F1=0.55
ALLOW_FRESH_REAL_PER_CLASS_REGRESSION=1
ESTABLISH_PROMOTION_BASELINE=0
PROMOTION_WRITE_BASELINE=1
PROMOTION_STRICT=1
HOLDOUT_PER_CLASS=12
HOLDOUT_MIN_TOTAL=48
HOLDOUT_MIN_SOURCES=4
HOLDOUT_MIN_REVIEWED_PER_CLASS=0
HOLDOUT_MIN_CLASS_SOURCES=0
HOLDOUT_REQUIRE_SCENARIO_TAGS=0
TRAINING_PROFILE="comparable"
REAL_REVIEWED_WEIGHT=1.0
HARD_NEGATIVE_WEIGHT=1.0
CLASS_TARGET_WEIGHTS=""

usage() {
  cat <<USAGE
Usage: scripts/run_reasoning_training_pipeline.sh [options]

Options:
  --collect-first                 Launch main.py first for manual data collection.
  --python-bin <bin>             Python executable (default: .venv311/bin/python).
  --input-glob <glob>            Raw input glob (default: data/raw/*.csv).
  --out-dir <dir>                Processed output directory (default: data/processed).
  --report <path>                Audit report path (default: reports/dataset_audit.json).
  --report-dir <dir>             Report output directory (default: reports).
  --manifest-path <path>         Dataset manifest output (default: data/manifest/dataset_manifest.json).
  --changelog-path <path>        Dataset changelog path (default: data/manifest/CHANGELOG.md).
  --min-per-class <n>            Minimum samples per class (default: 50).
  --max-imbalance <ratio>        Max class imbalance ratio (default: 1.5).
  --min-real-share <ratio>       Minimum real data share gate (default: 0.15).
  --balance <none|cap|oversample> Balancing strategy (default: oversample).
  --seed <n>                     Random seed for preprocessing (default: 42).
  --epochs <n>                   Training epochs (default: 30).
  --batch-size <n>               Training batch size (default: 32).
  --lr <float>                   Training learning rate (default: 1e-3).
  --weight-decay <float>         Adam weight decay (default: 0.0).
  --label-smoothing <float>      Cross-entropy label smoothing (default: 0.0).
  --model-path <path>            Model output path (default: models/reasoning_model.pt).
  --allow-archive-experiment     Allow training from globs that include data/raw_archive.
  --disk-min-free-gb <n>         Minimum free disk guard before running (default: 1.0).
  --no-prune                     Disable aggressive pruning preflight.
  --promotion-baseline-path <path> Baseline metrics used for promotion checks (default: reports/metrics_promoted_baseline.json).
  --promotion-summary <path>     Promotion summary JSON output (default: reports/promotion_summary.json).
  --no-promotion-write           Do not update the promoted baseline when a run is promotable.
  --allow-non-promotable         Do not exit with an error when promotion gates fail.
  --fresh-real-min-improve-acc <x> Minimum required fresh-real accuracy delta vs promoted baseline (default: -0.06).
  --fresh-real-min-improve-macro-f1 <x> Minimum required fresh-real macro F1 delta vs promoted baseline (default: -0.03).
  --fresh-real-absolute-min-acc <x> Minimum absolute fresh-real accuracy floor (default: 0.50).
  --fresh-real-absolute-min-macro-f1 <x> Minimum absolute fresh-real macro F1 floor (default: 0.55).
  --enforce-fresh-real-per-class-non-regression Re-enable the old mixed-sign class stop rule.
  --establish-promotion-baseline If baseline is missing, save current metrics as baseline and mark run non-promotable.
  --holdout-per-class <n>        Holdout rows per class (default: 12).
  --holdout-min-total <n>        Minimum total holdout rows (default: 48).
  --holdout-min-sources <n>      Minimum distinct holdout sources (default: 4).
  --holdout-min-reviewed-per-class <n> Minimum reviewed holdout rows per class.
  --holdout-min-class-sources <n> Minimum distinct holdout sources per class.
  --holdout-require-scenario-tags Require scenario tags on fresh-real holdout rows.
  --training-profile <profile>   Training sampling profile: comparable|real_recovery.
  --real-reviewed-weight <x>     Sampling multiplier for reviewed real rows.
  --hard-negative-weight <x>     Sampling multiplier for auto-label disagreement rows.
  --class-target-weights <spec>  LABEL:WEIGHT pairs for reviewed real class emphasis.
  -h, --help                     Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect-first)
      COLLECT_FIRST=1
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --input-glob)
      INPUT_GLOB="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    --manifest-path)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --changelog-path)
      CHANGELOG_PATH="$2"
      shift 2
      ;;
    --min-per-class)
      MIN_PER_CLASS="$2"
      shift 2
      ;;
    --max-imbalance)
      MAX_IMBALANCE="$2"
      shift 2
      ;;
    --balance)
      BALANCE="$2"
      shift 2
      ;;
    --min-real-share)
      MIN_REAL_SHARE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      TRAIN_LR="$2"
      shift 2
      ;;
    --weight-decay)
      TRAIN_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --label-smoothing)
      TRAIN_LABEL_SMOOTHING="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --allow-archive-experiment)
      ALLOW_ARCHIVE_EXPERIMENT=1
      shift
      ;;
    --disk-min-free-gb)
      DISK_MIN_FREE_GB="$2"
      shift 2
      ;;
    --no-prune)
      PRUNE=0
      shift
      ;;
    --promotion-baseline-path)
      PROMOTION_BASELINE_PATH="$2"
      shift 2
      ;;
    --promotion-summary)
      PROMOTION_SUMMARY_PATH="$2"
      shift 2
      ;;
    --no-promotion-write)
      PROMOTION_WRITE_BASELINE=0
      shift
      ;;
    --allow-non-promotable)
      PROMOTION_STRICT=0
      shift
      ;;
    --fresh-real-min-improve-acc)
      FRESH_REAL_MIN_IMPROVE_ACC="$2"
      shift 2
      ;;
    --fresh-real-min-improve-macro-f1)
      FRESH_REAL_MIN_IMPROVE_MACRO_F1="$2"
      shift 2
      ;;
    --fresh-real-absolute-min-acc)
      FRESH_REAL_ABSOLUTE_MIN_ACC="$2"
      shift 2
      ;;
    --fresh-real-absolute-min-macro-f1)
      FRESH_REAL_ABSOLUTE_MIN_MACRO_F1="$2"
      shift 2
      ;;
    --enforce-fresh-real-per-class-non-regression)
      ALLOW_FRESH_REAL_PER_CLASS_REGRESSION=0
      shift
      ;;
    --establish-promotion-baseline)
      ESTABLISH_PROMOTION_BASELINE=1
      shift
      ;;
    --holdout-per-class)
      HOLDOUT_PER_CLASS="$2"
      shift 2
      ;;
    --holdout-min-total)
      HOLDOUT_MIN_TOTAL="$2"
      shift 2
      ;;
    --holdout-min-sources)
      HOLDOUT_MIN_SOURCES="$2"
      shift 2
      ;;
    --holdout-min-reviewed-per-class)
      HOLDOUT_MIN_REVIEWED_PER_CLASS="$2"
      shift 2
      ;;
    --holdout-min-class-sources)
      HOLDOUT_MIN_CLASS_SOURCES="$2"
      shift 2
      ;;
    --holdout-require-scenario-tags)
      HOLDOUT_REQUIRE_SCENARIO_TAGS=1
      shift
      ;;
    --training-profile)
      TRAINING_PROFILE="$2"
      shift 2
      ;;
    --real-reviewed-weight)
      REAL_REVIEWED_WEIGHT="$2"
      shift 2
      ;;
    --hard-negative-weight)
      HARD_NEGATIVE_WEIGHT="$2"
      shift 2
      ;;
    --class-target-weights)
      CLASS_TARGET_WEIGHTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$ALLOW_ARCHIVE_EXPERIMENT" -ne 1 && "$INPUT_GLOB" == *"raw_archive"* ]]; then
  echo "Archive policy gate failed: data/raw_archive is excluded by default. Use --allow-archive-experiment to override." >&2
  exit 1
fi

if [[ "$COLLECT_FIRST" -eq 1 ]]; then
  echo "[1/4] Collect labeled sessions with main.py (manual)."
  echo "      Label keys: a=AVOID_PERSON c=MOVE_TO_CHAIR t=CHECK_TABLE e=EXPLORE"
  echo "      Press ESC to finish collection and continue pipeline."
  "$PYTHON_BIN" main.py
fi

mkdir -p "$(dirname "$REPORT")" "$OUT_DIR" "$REPORT_DIR" "$(dirname "$MODEL_PATH")"

echo "[0/5] Running disk preflight + retention..."
PRUNE_ARGS=()
if [[ "$PRUNE" -eq 1 ]]; then
  PRUNE_ARGS+=(--prune)
fi
"$PYTHON_BIN" scripts/manage_artifacts.py \
  --min-free-gb "$DISK_MIN_FREE_GB" \
  --budget-report reports/artifact_budget.json \
  --protect-prefix "$(basename "$REPORT")" \
  --protect-prefix "$(basename "$PROMOTION_BASELINE_PATH")" \
  "${PRUNE_ARGS[@]}"

echo "[2/4] Running dataset audit..."
"$PYTHON_BIN" scripts/audit_reasoning_data.py \
  --input-glob "$INPUT_GLOB" \
  --min-per-class "$MIN_PER_CLASS" \
  --max-class-imbalance-ratio "$MAX_IMBALANCE" \
  --min-real-share "$MIN_REAL_SHARE" \
  --require-two-real-batches \
  --report "$REPORT"

READY_FOR_TRAINING=$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path

report_path = Path(r'''$REPORT''')
if not report_path.exists():
    raise SystemExit("false")

with report_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("true" if bool(data.get("ready_for_training", False)) else "false")
PY
)

if [[ "$READY_FOR_TRAINING" != "true" ]]; then
  echo "Audit gate failed: ready_for_training is false in $REPORT" >&2
  exit 1
fi

echo "[3/4] Running preprocessing..."
PREP_EXTRA_ARGS=()
if [[ "$HOLDOUT_REQUIRE_SCENARIO_TAGS" -eq 1 ]]; then
  PREP_EXTRA_ARGS+=(--holdout-require-scenario-tags)
fi
"$PYTHON_BIN" scripts/prepare_reasoning_data.py \
  --input-glob "$INPUT_GLOB" \
  --out-dir "$OUT_DIR" \
  --balance "$BALANCE" \
  --min-per-class "$MIN_PER_CLASS" \
  --min-real-share "$MIN_REAL_SHARE" \
  --enforce-review-applied \
  --require-two-real-batches-for-holdout \
  --holdout-latest-real-source \
  --holdout-per-class "$HOLDOUT_PER_CLASS" \
  --holdout-min-total "$HOLDOUT_MIN_TOTAL" \
  --holdout-min-sources "$HOLDOUT_MIN_SOURCES" \
  --holdout-min-reviewed-per-class "$HOLDOUT_MIN_REVIEWED_PER_CLASS" \
  --holdout-min-class-sources "$HOLDOUT_MIN_CLASS_SOURCES" \
  --seed "$SEED" \
  "${PREP_EXTRA_ARGS[@]}"

echo "[3.5/4] Writing dataset manifest snapshot..."
"$PYTHON_BIN" scripts/create_dataset_manifest.py \
  --input-glob "$INPUT_GLOB" \
  --processed-dir "$OUT_DIR" \
  --manifest-path "$MANIFEST_PATH" \
  --changelog-path "$CHANGELOG_PATH"

echo "[4/4] Training reasoning model..."
METRICS_PATH="$REPORT_DIR/metrics.json"
if [[ -f "$METRICS_PATH" ]]; then
  cp "$METRICS_PATH" "$REPORT_DIR/metrics_previous.json"
fi
"$PYTHON_BIN" train_reasoning.py \
  --train "$OUT_DIR/train.csv" \
  --val "$OUT_DIR/val.csv" \
  --test "$OUT_DIR/test.csv" \
  --fresh-real-eval "$OUT_DIR/fresh_real_eval.csv" \
  --report-dir "$REPORT_DIR" \
  --model "$MODEL_PATH" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --lr "$TRAIN_LR" \
  --weight-decay "$TRAIN_WEIGHT_DECAY" \
  --label-smoothing "$TRAIN_LABEL_SMOOTHING" \
  --seed "$SEED" \
  --training-profile "$TRAINING_PROFILE" \
  --real-reviewed-weight "$REAL_REVIEWED_WEIGHT" \
  --hard-negative-weight "$HARD_NEGATIVE_WEIGHT" \
  --class-target-weights "$CLASS_TARGET_WEIGHTS"

PROMOTION_ARGS=(
  --current-metrics-path "$REPORT_DIR/metrics.json"
  --baseline-path "$PROMOTION_BASELINE_PATH"
  --summary-path "$PROMOTION_SUMMARY_PATH"
  --fresh-real-min-improve-acc "$FRESH_REAL_MIN_IMPROVE_ACC"
  --fresh-real-min-improve-macro-f1 "$FRESH_REAL_MIN_IMPROVE_MACRO_F1"
  --fresh-real-absolute-min-acc "$FRESH_REAL_ABSOLUTE_MIN_ACC"
  --fresh-real-absolute-min-macro-f1 "$FRESH_REAL_ABSOLUTE_MIN_MACRO_F1"
  --fresh-real-min-total "$HOLDOUT_MIN_TOTAL"
)
if [[ "$ESTABLISH_PROMOTION_BASELINE" -eq 1 ]]; then
  PROMOTION_ARGS+=(--establish-promotion-baseline)
fi
if [[ "$PROMOTION_WRITE_BASELINE" -eq 0 ]]; then
  PROMOTION_ARGS+=(--no-promotion-write)
fi
if [[ "$PROMOTION_STRICT" -eq 0 ]]; then
  PROMOTION_ARGS+=(--allow-non-promotable)
fi
if [[ "$ALLOW_FRESH_REAL_PER_CLASS_REGRESSION" -eq 1 ]]; then
  PROMOTION_ARGS+=(--allow-fresh-real-per-class-regression)
fi
"$PYTHON_BIN" scripts/run_promotion_summary.py "${PROMOTION_ARGS[@]}"

echo "Pipeline completed successfully."
