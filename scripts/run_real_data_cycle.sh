#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=".venv311/bin/python"
MEDIA_DIR_A=""
MEDIA_DIR_B=""
BATCH_A_ID="batch_C_real"
BATCH_B_ID="batch_D_real"
SCENARIO_A="mixed"
SCENARIO_B="low_light"
REVIEW_CORRECTIONS_A=""
REVIEW_CORRECTIONS_B=""
VIDEO_STRIDE=10
INPUT_GLOB="data/raw/*.csv"
CYCLE_REPORT="reports/cycle_report.json"
RUN_PIPELINE=1
PIPELINE_EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: scripts/run_real_data_cycle.sh [options] --media-dir-a <path> --media-dir-b <path> --review-corrections-a <csv> --review-corrections-b <csv>

Runs one full cycle:
  ingest A -> ingest B -> correction audit/coverage checks -> guarded training pipeline

Options:
  --python-bin <bin>             Python executable (default: .venv311/bin/python)
  --media-dir-a <path>           Media directory for batch A (required)
  --media-dir-b <path>           Media directory for batch B (required)
  --batch-a-id <id>              Batch A id (default: batch_C_real)
  --batch-b-id <id>              Batch B id (default: batch_D_real)
  --scenario-a <name>            Batch A scenario tag (default: mixed)
  --scenario-b <name>            Batch B scenario tag (default: low_light)
  --review-corrections-a <csv>   Correction CSV for batch A (required)
  --review-corrections-b <csv>   Correction CSV for batch B (required)
  --video-stride <n>             Video frame stride (default: 10)
  --input-glob <glob>            Raw input glob for pipeline (default: data/raw/*.csv)
  --cycle-report <path>          Cycle summary JSON output (default: reports/cycle_report.json)
  --no-run-pipeline              Stop after ingest + cycle checks
  --pipeline-arg <arg>           Extra arg forwarded to run_reasoning_training_pipeline.sh (repeatable)
  -h, --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --media-dir-a)
      MEDIA_DIR_A="$2"
      shift 2
      ;;
    --media-dir-b)
      MEDIA_DIR_B="$2"
      shift 2
      ;;
    --batch-a-id)
      BATCH_A_ID="$2"
      shift 2
      ;;
    --batch-b-id)
      BATCH_B_ID="$2"
      shift 2
      ;;
    --scenario-a)
      SCENARIO_A="$2"
      shift 2
      ;;
    --scenario-b)
      SCENARIO_B="$2"
      shift 2
      ;;
    --review-corrections-a)
      REVIEW_CORRECTIONS_A="$2"
      shift 2
      ;;
    --review-corrections-b)
      REVIEW_CORRECTIONS_B="$2"
      shift 2
      ;;
    --video-stride)
      VIDEO_STRIDE="$2"
      shift 2
      ;;
    --input-glob)
      INPUT_GLOB="$2"
      shift 2
      ;;
    --cycle-report)
      CYCLE_REPORT="$2"
      shift 2
      ;;
    --no-run-pipeline)
      RUN_PIPELINE=0
      shift
      ;;
    --pipeline-arg)
      PIPELINE_EXTRA_ARGS+=("$2")
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

if [[ -z "$MEDIA_DIR_A" || -z "$MEDIA_DIR_B" ]]; then
  echo "Both --media-dir-a and --media-dir-b are required." >&2
  exit 2
fi
if [[ -z "$REVIEW_CORRECTIONS_A" || -z "$REVIEW_CORRECTIONS_B" ]]; then
  echo "Both --review-corrections-a and --review-corrections-b are required." >&2
  exit 2
fi
if [[ ! -f "$REVIEW_CORRECTIONS_A" || ! -f "$REVIEW_CORRECTIONS_B" ]]; then
  echo "Correction CSV file not found." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RAW_A="data/raw/media_labeled_${BATCH_A_ID}_${TS}.csv"
RAW_B="data/raw/media_labeled_${BATCH_B_ID}_${TS}.csv"
REVIEW_A="reports/media_review_${BATCH_A_ID}_${TS}.csv"
REVIEW_B="reports/media_review_${BATCH_B_ID}_${TS}.csv"
AUDIT_A="reports/correction_audit_${BATCH_A_ID}_${TS}.json"
AUDIT_B="reports/correction_audit_${BATCH_B_ID}_${TS}.json"

echo "[1/5] Ingesting batch A (${BATCH_A_ID})..."
"$PYTHON_BIN" scripts/build_reasoning_data_from_media.py \
  --media-dir "$MEDIA_DIR_A" \
  --batch-id "$BATCH_A_ID" \
  --scenario "$SCENARIO_A" \
  --video-stride "$VIDEO_STRIDE" \
  --out-csv "$RAW_A" \
  --review-out "$REVIEW_A" \
  --review-corrections "$REVIEW_CORRECTIONS_A" \
  --correction-audit-out "$AUDIT_A"

echo "[2/5] Ingesting batch B (${BATCH_B_ID})..."
"$PYTHON_BIN" scripts/build_reasoning_data_from_media.py \
  --media-dir "$MEDIA_DIR_B" \
  --batch-id "$BATCH_B_ID" \
  --scenario "$SCENARIO_B" \
  --video-stride "$VIDEO_STRIDE" \
  --out-csv "$RAW_B" \
  --review-out "$REVIEW_B" \
  --review-corrections "$REVIEW_CORRECTIONS_B" \
  --correction-audit-out "$AUDIT_B"

echo "[3/5] Aggregating cycle coverage + correction audits..."
"$PYTHON_BIN" - <<PY
import glob
import json
import os
from pathlib import Path

coverage_files = sorted(glob.glob("reports/batch_coverage_*.json"), key=os.path.getmtime, reverse=True)[:4]
audit_files = [r'''$AUDIT_A''', r'''$AUDIT_B''']
required_scenarios = {"low_light", "clutter", "occlusion", "mixed", "long_range"}

scenario_counts = {}
for path in coverage_files:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        continue
    scenario = str(payload.get("scenario", "mixed"))
    rows = int(payload.get("rows_total", 0))
    scenario_counts[scenario] = scenario_counts.get(scenario, 0) + rows

seen = {k for k, v in scenario_counts.items() if v > 0}
missing = sorted(required_scenarios - seen)

summary = {
    "timestamp": int(__import__("time").time()),
    "batch_ids": [r'''$BATCH_A_ID''', r'''$BATCH_B_ID'''],
    "raw_files": [r'''$RAW_A''', r'''$RAW_B'''],
    "review_files": [r'''$REVIEW_A''', r'''$REVIEW_B'''],
    "correction_audit_files": audit_files,
    "coverage_files_sampled": coverage_files,
    "scenario_rows": scenario_counts,
    "required_scenarios": sorted(required_scenarios),
    "missing_scenarios": missing,
    "gates": {
        "two_independent_real_batches": True,
        "scenario_coverage_complete": len(missing) == 0,
    },
}
Path(r'''$CYCLE_REPORT''').parent.mkdir(parents=True, exist_ok=True)
Path(r'''$CYCLE_REPORT''').write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Cycle report written: {r'''$CYCLE_REPORT'''}")
if missing:
    print("Warning: scenario coverage missing:", ", ".join(missing))
PY

if [[ "$RUN_PIPELINE" -eq 0 ]]; then
  echo "[4/5] Skipping guarded pipeline (--no-run-pipeline)."
  exit 0
fi

echo "[4/5] Running guarded pipeline..."
scripts/run_reasoning_training_pipeline.sh \
  --python-bin "$PYTHON_BIN" \
  --input-glob "$INPUT_GLOB" \
  "${PIPELINE_EXTRA_ARGS[@]}"

echo "[5/5] Cycle completed."
