#!/usr/bin/env bash
set -euo pipefail

INPUT_GLOB="data/raw/*.csv"
OUT_DIR="data/processed"
REPORT="reports/dataset_audit.json"
MANIFEST_PATH="data/manifest/dataset_manifest.json"
CHANGELOG_PATH="data/manifest/CHANGELOG.md"
MIN_PER_CLASS=50
MAX_IMBALANCE=1.3
MIN_REAL_SHARE=0.6
MAX_SYNTHETIC_SHARE=0.4
BALANCE="cap"
SEED=42
TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=32
TRAIN_LR=1e-3
COLLECT_FIRST=0
PYTHON_BIN=".venv311/bin/python"
ALLOW_ARCHIVE_EXPERIMENT=0
DISK_MIN_FREE_GB=1.0
PRUNE=1

usage() {
  cat <<USAGE
Usage: scripts/run_reasoning_training_pipeline.sh [options]

Options:
  --collect-first                 Launch main.py first for manual data collection.
  --python-bin <bin>             Python executable (default: .venv311/bin/python).
  --input-glob <glob>            Raw input glob (default: data/raw/*.csv).
  --out-dir <dir>                Processed output directory (default: data/processed).
  --report <path>                Audit report path (default: reports/dataset_audit.json).
  --manifest-path <path>         Dataset manifest output (default: data/manifest/dataset_manifest.json).
  --changelog-path <path>        Dataset changelog path (default: data/manifest/CHANGELOG.md).
  --min-per-class <n>            Minimum samples per class (default: 50).
  --max-imbalance <ratio>        Max class imbalance ratio (default: 1.3).
  --min-real-share <ratio>       Minimum real data share gate (default: 0.6).
  --max-synthetic-share <ratio>  Maximum synthetic data share gate (default: 0.4).
  --balance <none|cap|oversample> Balancing strategy (default: cap).
  --seed <n>                     Random seed for preprocessing (default: 42).
  --epochs <n>                   Training epochs (default: 30).
  --batch-size <n>               Training batch size (default: 32).
  --lr <float>                   Training learning rate (default: 1e-3).
  --allow-archive-experiment     Allow training from globs that include data/raw_archive.
  --disk-min-free-gb <n>         Minimum free disk guard before running (default: 1.0).
  --no-prune                     Disable aggressive pruning preflight.
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
    --max-synthetic-share)
      MAX_SYNTHETIC_SHARE="$2"
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

mkdir -p "$(dirname "$REPORT")" "$OUT_DIR"

echo "[0/5] Running disk preflight + retention..."
PRUNE_ARGS=()
if [[ "$PRUNE" -eq 1 ]]; then
  PRUNE_ARGS+=(--prune)
fi
"$PYTHON_BIN" scripts/manage_artifacts.py \
  --min-free-gb "$DISK_MIN_FREE_GB" \
  --budget-report reports/artifact_budget.json \
  --protect-prefix "$(basename "$REPORT")" \
  "${PRUNE_ARGS[@]}"

echo "[2/4] Running dataset audit..."
"$PYTHON_BIN" scripts/audit_reasoning_data.py \
  --input-glob "$INPUT_GLOB" \
  --min-per-class "$MIN_PER_CLASS" \
  --max-class-imbalance-ratio "$MAX_IMBALANCE" \
  --min-real-share "$MIN_REAL_SHARE" \
  --max-synthetic-share "$MAX_SYNTHETIC_SHARE" \
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
"$PYTHON_BIN" scripts/prepare_reasoning_data.py \
  --input-glob "$INPUT_GLOB" \
  --out-dir "$OUT_DIR" \
  --balance "$BALANCE" \
  --min-per-class "$MIN_PER_CLASS" \
  --min-real-share "$MIN_REAL_SHARE" \
  --max-synthetic-share "$MAX_SYNTHETIC_SHARE" \
  --enforce-review-applied \
  --require-two-real-batches-for-holdout \
  --holdout-latest-real-source \
  --seed "$SEED"

echo "[3.5/4] Writing dataset manifest snapshot..."
"$PYTHON_BIN" scripts/create_dataset_manifest.py \
  --input-glob "$INPUT_GLOB" \
  --processed-dir "$OUT_DIR" \
  --manifest-path "$MANIFEST_PATH" \
  --changelog-path "$CHANGELOG_PATH"

echo "[4/4] Training reasoning model..."
if [[ -f reports/metrics.json ]]; then
  cp reports/metrics.json reports/metrics_previous.json
fi
"$PYTHON_BIN" train_reasoning.py \
  --train "$OUT_DIR/train.csv" \
  --val "$OUT_DIR/val.csv" \
  --test "$OUT_DIR/test.csv" \
  --fresh-real-eval "$OUT_DIR/fresh_real_eval.csv" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --lr "$TRAIN_LR"

if [[ -f reports/metrics_previous.json && -f reports/metrics.json ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

old = json.loads(Path("reports/metrics_previous.json").read_text(encoding="utf-8"))
new = json.loads(Path("reports/metrics.json").read_text(encoding="utf-8"))

tol = 0.005
old_test = float(old.get("accuracy", {}).get("test", 0.0))
new_test = float(new.get("accuracy", {}).get("test", 0.0))
if new_test + tol < old_test:
    raise SystemExit(
        f"Promotion gate failed: test accuracy dropped from {old_test:.4f} to {new_test:.4f}"
    )

for label in ("CHECK_TABLE", "MOVE_TO_CHAIR"):
    old_f1 = float(old.get("per_class", {}).get(label, {}).get("f1", 0.0))
    new_f1 = float(new.get("per_class", {}).get(label, {}).get("f1", 0.0))
    if new_f1 + 1e-12 < old_f1:
        raise SystemExit(
            f"Promotion gate failed: {label} F1 dropped from {old_f1:.4f} to {new_f1:.4f}"
        )

old_fresh = old.get("fresh_real_eval", {}).get("accuracy")
new_fresh = new.get("fresh_real_eval", {}).get("accuracy")
if old_fresh is not None and new_fresh is not None:
    old_fresh = float(old_fresh)
    new_fresh = float(new_fresh)
    if new_fresh + tol < old_fresh:
        raise SystemExit(
            f"Promotion gate failed: fresh real eval accuracy dropped from {old_fresh:.4f} to {new_fresh:.4f}"
        )

print("Promotion gates passed.")
PY
fi

echo "Pipeline completed successfully."
