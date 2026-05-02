#!/usr/bin/env bash
set -euo pipefail

INPUT_GLOB="data/raw/*.csv"
OUT_DIR="data/processed"
REPORT="reports/dataset_audit.json"
MIN_PER_CLASS=50
MAX_IMBALANCE=2.0
BALANCE="cap"
SEED=42
TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=32
TRAIN_LR=1e-3
COLLECT_FIRST=0
PYTHON_BIN="python3"

usage() {
  cat <<USAGE
Usage: scripts/run_reasoning_training_pipeline.sh [options]

Options:
  --collect-first                 Launch main.py first for manual data collection.
  --python-bin <bin>             Python executable (default: python3).
  --input-glob <glob>            Raw input glob (default: data/raw/*.csv).
  --out-dir <dir>                Processed output directory (default: data/processed).
  --report <path>                Audit report path (default: reports/dataset_audit.json).
  --min-per-class <n>            Minimum samples per class (default: 50).
  --max-imbalance <ratio>        Max class imbalance ratio (default: 2.0).
  --balance <none|cap|oversample> Balancing strategy (default: cap).
  --seed <n>                     Random seed for preprocessing (default: 42).
  --epochs <n>                   Training epochs (default: 30).
  --batch-size <n>               Training batch size (default: 32).
  --lr <float>                   Training learning rate (default: 1e-3).
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

if [[ "$COLLECT_FIRST" -eq 1 ]]; then
  echo "[1/4] Collect labeled sessions with main.py (manual)."
  echo "      Label keys: a=AVOID_PERSON c=MOVE_TO_CHAIR t=CHECK_TABLE e=EXPLORE"
  echo "      Press ESC to finish collection and continue pipeline."
  "$PYTHON_BIN" main.py
fi

mkdir -p "$(dirname "$REPORT")" "$OUT_DIR"

echo "[2/4] Running dataset audit..."
"$PYTHON_BIN" scripts/audit_reasoning_data.py \
  --input-glob "$INPUT_GLOB" \
  --min-per-class "$MIN_PER_CLASS" \
  --max-class-imbalance-ratio "$MAX_IMBALANCE" \
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
  --seed "$SEED"

echo "[4/4] Training reasoning model..."
"$PYTHON_BIN" train_reasoning.py \
  --train "$OUT_DIR/train.csv" \
  --val "$OUT_DIR/val.csv" \
  --test "$OUT_DIR/test.csv" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --lr "$TRAIN_LR"

echo "Pipeline completed successfully."
