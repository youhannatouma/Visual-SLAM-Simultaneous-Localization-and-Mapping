#!/usr/bin/env bash
set -euo pipefail

INPUT_GLOB="data/raw/*.csv"
OUT_DIR="data/processed"
REPORT="reports/dataset_audit.json"
REPORT_DIR="reports"
MANIFEST_PATH="data/manifest/dataset_manifest.json"
CHANGELOG_PATH="data/manifest/CHANGELOG.md"
MIN_PER_CLASS=50
MAX_IMBALANCE=1.35
MIN_REAL_SHARE=0.6
MAX_SYNTHETIC_SHARE=0.4
BALANCE="oversample"
SEED=42
TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=32
TRAIN_LR=1e-3
COLLECT_FIRST=0
PYTHON_BIN=".venv311/bin/python"
MODEL_PATH="models/reasoning_model.pt"
ALLOW_ARCHIVE_EXPERIMENT=0
DISK_MIN_FREE_GB=1.0
PRUNE=1
PROMOTION_BASELINE_PATH="reports/metrics_promoted_baseline.json"
PROMOTION_SUMMARY_PATH="reports/promotion_summary.json"
FRESH_REAL_MIN_IMPROVE_ACC=0.10
FRESH_REAL_MIN_IMPROVE_MACRO_F1=0.10
ESTABLISH_PROMOTION_BASELINE=0
PROMOTION_WRITE_BASELINE=1
PROMOTION_STRICT=1
HOLDOUT_PER_CLASS=8
HOLDOUT_MIN_TOTAL=32
HOLDOUT_MIN_SOURCES=2

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
  --max-imbalance <ratio>        Max class imbalance ratio (default: 1.35).
  --min-real-share <ratio>       Minimum real data share gate (default: 0.6).
  --max-synthetic-share <ratio>  Maximum synthetic data share gate (default: 0.4).
  --balance <none|cap|oversample> Balancing strategy (default: oversample).
  --seed <n>                     Random seed for preprocessing (default: 42).
  --epochs <n>                   Training epochs (default: 30).
  --batch-size <n>               Training batch size (default: 32).
  --lr <float>                   Training learning rate (default: 1e-3).
  --model-path <path>            Model output path (default: models/reasoning_model.pt).
  --allow-archive-experiment     Allow training from globs that include data/raw_archive.
  --disk-min-free-gb <n>         Minimum free disk guard before running (default: 1.0).
  --no-prune                     Disable aggressive pruning preflight.
  --promotion-baseline-path <path> Baseline metrics used for promotion checks (default: reports/metrics_promoted_baseline.json).
  --promotion-summary <path>     Promotion summary JSON output (default: reports/promotion_summary.json).
  --no-promotion-write           Do not update the promoted baseline when a run is promotable.
  --allow-non-promotable         Do not exit with an error when promotion gates fail.
  --fresh-real-min-improve-acc <x> Minimum required fresh-real accuracy improvement vs promoted baseline (default: 0.10).
  --fresh-real-min-improve-macro-f1 <x> Minimum required fresh-real macro F1 improvement vs promoted baseline (default: 0.10).
  --establish-promotion-baseline If baseline is missing, save current metrics as baseline and mark run non-promotable.
  --holdout-per-class <n>        Holdout rows per class (default: 8).
  --holdout-min-total <n>        Minimum total holdout rows (default: 32).
  --holdout-min-sources <n>      Minimum distinct holdout sources (default: 2).
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
  --holdout-per-class "$HOLDOUT_PER_CLASS" \
  --holdout-min-total "$HOLDOUT_MIN_TOTAL" \
  --holdout-min-sources "$HOLDOUT_MIN_SOURCES" \
  --seed "$SEED"

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
  --seed "$SEED"

"$PYTHON_BIN" - <<PY
import json
import shutil
from pathlib import Path

new_path = Path(r'''$REPORT_DIR''') / "metrics.json"
if not new_path.exists():
  raise SystemExit(f"Promotion gate failed: {new_path} is missing.")

baseline_path = Path(r'''$PROMOTION_BASELINE_PATH''')
summary_path = Path(r'''$PROMOTION_SUMMARY_PATH''')
summary_path.parent.mkdir(parents=True, exist_ok=True)
baseline_path.parent.mkdir(parents=True, exist_ok=True)

fresh_acc_threshold = float(r'''$FRESH_REAL_MIN_IMPROVE_ACC''')
fresh_f1_threshold = float(r'''$FRESH_REAL_MIN_IMPROVE_MACRO_F1''')
establish_mode = bool(int(r'''$ESTABLISH_PROMOTION_BASELINE'''))
write_baseline = bool(int(r'''$PROMOTION_WRITE_BASELINE'''))
strict_mode = bool(int(r'''$PROMOTION_STRICT'''))

new = json.loads(new_path.read_text(encoding="utf-8"))

def get_float(obj, *keys):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    try:
        return float(cur)
    except Exception:
        return None

gate_results = {}
recommended_actions = []
promotable = True
status = "promoted"
failure_reasons = []

if not baseline_path.exists():
    if establish_mode:
        shutil.copy2(new_path, baseline_path)
        promotable = False
        status = "baseline_established_non_promotable"
        failure_reasons.append("No promoted baseline existed; baseline established for next-cycle comparison.")
        recommended_actions.append("Run one more full cycle to compare against the newly established promoted baseline.")
    else:
        raise SystemExit(
            "Promotion gate failed: baseline metrics file is missing. "
            "Use --establish-promotion-baseline once to initialize."
        )
else:
    old = json.loads(baseline_path.read_text(encoding="utf-8"))
    tol = 0.005

    old_test = get_float(old, "accuracy", "test")
    new_test = get_float(new, "accuracy", "test")
    test_ok = (old_test is not None and new_test is not None and new_test + tol >= old_test)
    gate_results["test_accuracy_non_regression"] = {
        "passed": bool(test_ok),
        "old": old_test,
        "new": new_test,
        "tolerance": tol,
    }
    if not test_ok:
        promotable = False
        failure_reasons.append(f"Test accuracy regressed ({old_test} -> {new_test}).")
        recommended_actions.append("Collect more diverse real samples and rebalance weak classes before retraining.")

    key_class_fail = False
    key_class_rows = {}
    for label in ("CHECK_TABLE", "MOVE_TO_CHAIR"):
        old_f1 = get_float(old, "per_class", label, "f1")
        new_f1 = get_float(new, "per_class", label, "f1")
        ok = old_f1 is not None and new_f1 is not None and new_f1 + 1e-12 >= old_f1
        key_class_rows[label] = {"passed": bool(ok), "old_f1": old_f1, "new_f1": new_f1}
        if not ok:
            key_class_fail = True
    gate_results["key_class_f1_non_regression"] = {"passed": not key_class_fail, "classes": key_class_rows}
    if key_class_fail:
        promotable = False
        failure_reasons.append("Key class F1 regression detected.")
        recommended_actions.append("Increase reviewed real data for CHECK_TABLE/MOVE_TO_CHAIR edge cases.")

    old_fresh_acc = get_float(old, "fresh_real_eval", "accuracy")
    new_fresh_acc = get_float(new, "fresh_real_eval", "accuracy")
    old_fresh_f1 = get_float(old, "fresh_real_eval", "macro_f1")
    new_fresh_f1 = get_float(new, "fresh_real_eval", "macro_f1")

    acc_delta = None if old_fresh_acc is None or new_fresh_acc is None else (new_fresh_acc - old_fresh_acc)
    f1_delta = None if old_fresh_f1 is None or new_fresh_f1 is None else (new_fresh_f1 - old_fresh_f1)
    fresh_acc_ok = acc_delta is not None and acc_delta >= fresh_acc_threshold
    fresh_f1_ok = f1_delta is not None and f1_delta >= fresh_f1_threshold
    gate_results["fresh_real_improvement"] = {
        "passed": bool(fresh_acc_ok and fresh_f1_ok),
        "old": {"accuracy": old_fresh_acc, "macro_f1": old_fresh_f1},
        "new": {"accuracy": new_fresh_acc, "macro_f1": new_fresh_f1},
        "delta": {"accuracy": acc_delta, "macro_f1": f1_delta},
        "required_delta": {"accuracy": fresh_acc_threshold, "macro_f1": fresh_f1_threshold},
    }
    if not (fresh_acc_ok and fresh_f1_ok):
        promotable = False
        failure_reasons.append(
            "Fresh-real improvement gate not met "
            f"(delta accuracy={acc_delta}, delta macro_f1={f1_delta})."
        )
        recommended_actions.append("Collect at least two new independent real batches focused on low-light/clutter/occlusion/mixed scenes.")

    # Stop-rule gate: disallow promotion when fresh-real improves in aggregate
    # but one or more classes regress (mixed-sign class deltas).
    class_labels = ("AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE")
    class_gate_rows = {}
    class_regression = False
    for label in class_labels:
        old_cls = get_float(old, "fresh_real_eval", "per_class", label, "f1")
        new_cls = get_float(new, "fresh_real_eval", "per_class", label, "f1")
        ok = old_cls is not None and new_cls is not None and (new_cls + 1e-12 >= old_cls)
        class_gate_rows[label] = {
            "passed": bool(ok),
            "old_f1": old_cls,
            "new_f1": new_cls,
            "delta_f1": None if old_cls is None or new_cls is None else (new_cls - old_cls),
        }
        if not ok:
            class_regression = True
    gate_results["fresh_real_per_class_non_regression"] = {
        "passed": not class_regression,
        "classes": class_gate_rows,
    }
    if class_regression:
        promotable = False
        failure_reasons.append("Fresh-real per-class regression detected (mixed-sign deltas).")
        recommended_actions.append("Continue data-refresh cycles and target regressing fresh-real classes before promotion.")

    if promotable:
      if write_baseline:
        shutil.copy2(new_path, baseline_path)
        status = "promoted"
      else:
        status = "promotable_no_write"
    else:
      status = "not_promoted"

summary = {
    "status": status,
    "promotable": bool(promotable),
    "baseline_path": str(baseline_path),
    "current_metrics_path": str(new_path),
    "gate_results": gate_results,
    "failure_reasons": failure_reasons,
    "recommended_actions": recommended_actions,
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Promotion summary: {summary_path}")

if not promotable and strict_mode:
  raise SystemExit("Promotion gate failed: run is non-promotable. See promotion summary JSON for details.")
PY

echo "Pipeline completed successfully."
