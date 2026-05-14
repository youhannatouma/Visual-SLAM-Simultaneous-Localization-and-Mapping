import argparse
import json
import subprocess
import time
from pathlib import Path

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]
RECOVERY_POLICY = {
    "fresh_real_accuracy_min": 0.68,
    "fresh_real_macro_f1_min": 0.65,
    "check_table_f1_min": 0.50,
    "other_class_f1_min": 0.55,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track 1 orchestrator: short seed sweeps + targeted refresh guidance for "
            "fresh-real regressions with alternating comparable and real-recovery runs."
        )
    )
    parser.add_argument("--python-bin", default=".venv311/bin/python", help="Python executable to run helper scripts")
    parser.add_argument("--sweep-script", default="scripts/run_reasoning_seed_sweep.py", help="Seed sweep script path")
    parser.add_argument("--baseline", default="reports/metrics_promoted_baseline.json", help="Promoted baseline metrics path")
    parser.add_argument("--seeds", default="17,42,123", help="Comma-separated seeds")
    parser.add_argument("--epochs-min", type=int, default=12, help="Minimum epochs for short sweeps")
    parser.add_argument("--epochs-max", type=int, default=20, help="Maximum epochs for short sweeps")
    parser.add_argument(
        "--real-recovery-every-cycles",
        type=int,
        default=2,
        help="Run the more aggressive real_recovery profile every N cycles.",
    )

    parser.add_argument("--comparable-holdout-per-class", type=int, default=24)
    parser.add_argument("--comparable-holdout-min-total", type=int, default=96)
    parser.add_argument("--comparable-holdout-min-sources", type=int, default=3)
    parser.add_argument("--comparable-holdout-min-reviewed-per-class", type=int, default=24)
    parser.add_argument("--comparable-holdout-min-class-sources", type=int, default=3)

    parser.add_argument("--recovery-holdout-per-class", type=int, default=24)
    parser.add_argument("--recovery-holdout-min-total", type=int, default=96)
    parser.add_argument("--recovery-holdout-min-sources", type=int, default=3)
    parser.add_argument("--recovery-holdout-min-reviewed-per-class", type=int, default=24)
    parser.add_argument("--recovery-holdout-min-class-sources", type=int, default=3)

    parser.add_argument("--max-refresh-cycles", type=int, default=12, help="Safety cap for cycle count")
    parser.add_argument("--require-scenario-tags", action="store_true")

    parser.add_argument("--track1-report-root", default="reports/track1", help="Root directory for Track 1 reports")
    parser.add_argument("--processed-root", default="data/processed_track1", help="Root directory for per-cycle processed data")
    parser.add_argument("--model-root", default="models/track1", help="Root directory for per-cycle models")

    parser.add_argument("--keep-going", action="store_true", help="Continue if one seed fails inside a sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print and record planned commands without executing")
    return parser.parse_args()


def parse_seeds(text):
    seeds = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError("No valid seeds passed")
    return seeds


def cycle_epochs(cycle_idx, epochs_min, epochs_max):
    if epochs_min > epochs_max:
        raise ValueError("--epochs-min cannot be greater than --epochs-max")
    span = (epochs_max - epochs_min) + 1
    return epochs_min + ((cycle_idx - 1) % span)


def cycle_profile(cycle_idx, every_n):
    if every_n <= 0:
        raise ValueError("--real-recovery-every-cycles must be >= 1")
    return "real_recovery" if cycle_idx % every_n == 0 else "comparable"


def build_sweep_cmd(args, profile_name, epochs, cycle_report_dir, cycle_processed_dir, cycle_model_dir):
    if profile_name == "real_recovery":
        holdout_per_class = args.recovery_holdout_per_class
        holdout_min_total = args.recovery_holdout_min_total
        holdout_min_sources = args.recovery_holdout_min_sources
        holdout_min_reviewed_per_class = args.recovery_holdout_min_reviewed_per_class
        holdout_min_class_sources = args.recovery_holdout_min_class_sources
        training_profile = "real_recovery"
        real_reviewed_weight = 2.5
        hard_negative_weight = 1.75
        class_target_weights = "CHECK_TABLE:2.4,MOVE_TO_CHAIR:1.8,EXPLORE:1.2"
    else:
        holdout_per_class = args.comparable_holdout_per_class
        holdout_min_total = args.comparable_holdout_min_total
        holdout_min_sources = args.comparable_holdout_min_sources
        holdout_min_reviewed_per_class = args.comparable_holdout_min_reviewed_per_class
        holdout_min_class_sources = args.comparable_holdout_min_class_sources
        training_profile = "comparable"
        real_reviewed_weight = 1.0
        hard_negative_weight = 1.0
        class_target_weights = ""

    cmd = [
        args.python_bin,
        args.sweep_script,
        "--seeds",
        args.seeds,
        "--python-bin",
        args.python_bin,
        "--epochs",
        str(epochs),
        "--holdout-per-class",
        str(holdout_per_class),
        "--holdout-min-total",
        str(holdout_min_total),
        "--holdout-min-sources",
        str(holdout_min_sources),
        "--holdout-min-reviewed-per-class",
        str(holdout_min_reviewed_per_class),
        "--holdout-min-class-sources",
        str(holdout_min_class_sources),
        "--baseline",
        args.baseline,
        "--out-root",
        str(cycle_report_dir),
        "--processed-root",
        str(cycle_processed_dir),
        "--model-root",
        str(cycle_model_dir),
        "--training-profile",
        training_profile,
        "--real-reviewed-weight",
        str(real_reviewed_weight),
        "--hard-negative-weight",
        str(hard_negative_weight),
    ]
    if class_target_weights:
        cmd.extend(["--class-target-weights", class_target_weights])
    if args.require_scenario_tags:
        cmd.append("--holdout-require-scenario-tags")
    if args.keep_going:
        cmd.append("--keep-going")
    return cmd


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_gate(gates, name):
    node = gates.get(name, {})
    return bool(node.get("passed", False)), node


def build_class_delta_map(gates):
    class_gate = gates.get("fresh_real_per_class_non_regression", {})
    classes = class_gate.get("classes", {}) if isinstance(class_gate, dict) else {}
    out = {}
    for label in ACTION_CLASSES:
        row = classes.get(label, {}) if isinstance(classes, dict) else {}
        out[label] = {
            "old_f1": row.get("old_f1"),
            "new_f1": row.get("new_f1"),
            "delta_f1": row.get("delta_f1"),
            "passed": bool(row.get("passed", False)),
        }
    return out


def regressing_classes_from_deltas(delta_map):
    regressing = []
    for label in ACTION_CLASSES:
        delta = delta_map.get(label, {}).get("delta_f1")
        if delta is None:
            regressing.append(label)
            continue
        if float(delta) < 0.0:
            regressing.append(label)
    return regressing


def build_refresh_recommendations(regressing):
    if not regressing:
        return ["No regressing fresh-real classes. Continue standard reviewed data expansion."]

    recs = []
    if "MOVE_TO_CHAIR" in regressing:
        recs.append(
            "Prioritize MOVE_TO_CHAIR refresh: collect hard chair-approach transitions, occlusion scenes, and table/chair boundary cases."
        )
    if "CHECK_TABLE" in regressing:
        recs.append(
            "Prioritize CHECK_TABLE refresh: collect true table-interaction frames under clutter/low-light and remove ambiguous table-vs-explore rows."
        )
    if "AVOID_PERSON" in regressing:
        recs.append(
            "Refresh AVOID_PERSON negatives: add crowded/person-adjacent scenes and hard confusers that currently drift into motion classes."
        )
    if "EXPLORE" in regressing:
        recs.append(
            "Refresh EXPLORE diversity: add non-table exploratory contexts to reduce collapse with CHECK_TABLE and MOVE_TO_CHAIR."
        )

    recs.append(
        "Constrain refresh to regressing classes only, and tune only data composition/balancing/hard-negative ratio (no architecture changes)."
    )
    return recs


def top_confusion_pairs(confusion_matrix):
    pairs = []
    if not isinstance(confusion_matrix, list):
        return pairs
    for true_idx, row in enumerate(confusion_matrix):
        if not isinstance(row, list):
            continue
        for pred_idx, count in enumerate(row):
            if true_idx == pred_idx:
                continue
            if count:
                pairs.append(
                    {
                        "true_label": ACTION_CLASSES[true_idx],
                        "pred_label": ACTION_CLASSES[pred_idx],
                        "count": int(count),
                    }
                )
    pairs.sort(key=lambda item: (-item["count"], item["true_label"], item["pred_label"]))
    return pairs


def evaluate_recovery_policy(metrics):
    fresh_real = metrics.get("fresh_real_eval", {}) if isinstance(metrics, dict) else {}
    per_class = fresh_real.get("per_class", {}) if isinstance(fresh_real, dict) else {}
    accuracy = fresh_real.get("accuracy")
    macro_f1 = fresh_real.get("macro_f1")

    class_rows = {}
    failed_labels = []
    for label in ACTION_CLASSES:
        f1 = None
        if isinstance(per_class.get(label), dict):
            f1 = per_class[label].get("f1")
        threshold = RECOVERY_POLICY["check_table_f1_min"] if label == "CHECK_TABLE" else RECOVERY_POLICY["other_class_f1_min"]
        passed = f1 is not None and float(f1) >= float(threshold)
        class_rows[label] = {
            "f1": f1,
            "threshold": float(threshold),
            "passed": bool(passed),
        }
        if not passed:
            failed_labels.append(label)

    accuracy_passed = accuracy is not None and float(accuracy) >= float(RECOVERY_POLICY["fresh_real_accuracy_min"])
    macro_f1_passed = macro_f1 is not None and float(macro_f1) >= float(RECOVERY_POLICY["fresh_real_macro_f1_min"])
    worst_label = min(
        ACTION_CLASSES,
        key=lambda label: float(class_rows[label]["f1"]) if class_rows[label]["f1"] is not None else float("-inf"),
    )
    return {
        "passed": bool(accuracy_passed and macro_f1_passed and not failed_labels),
        "fresh_real_accuracy": {
            "value": accuracy,
            "threshold": float(RECOVERY_POLICY["fresh_real_accuracy_min"]),
            "passed": bool(accuracy_passed),
        },
        "fresh_real_macro_f1": {
            "value": macro_f1,
            "threshold": float(RECOVERY_POLICY["fresh_real_macro_f1_min"]),
            "passed": bool(macro_f1_passed),
        },
        "per_class": class_rows,
        "worst_class": worst_label,
        "failed_labels": failed_labels,
    }


def build_recovery_summary(metrics, processed_metadata, evaluation):
    fresh_real = metrics.get("fresh_real_eval", {}) if isinstance(metrics, dict) else {}
    confusion_pairs = top_confusion_pairs(fresh_real.get("confusion_matrix", []))
    holdout_qa = (
        processed_metadata.get("holdout_latest_real_source", {}).get("qa", {})
        if isinstance(processed_metadata, dict)
        else {}
    )
    summary = {
        "fresh_real_accuracy": fresh_real.get("accuracy"),
        "fresh_real_macro_f1": fresh_real.get("macro_f1"),
        "rows": fresh_real.get("rows"),
        "confusion_matrix": fresh_real.get("confusion_matrix"),
        "top_confusion_pairs": confusion_pairs[:6],
        "worst_class": evaluation.get("recovery_policy", {}).get("worst_class"),
        "holdout_quality": {
            "promotion_grade_ready": holdout_qa.get("promotion_grade_ready"),
            "rows": holdout_qa.get("rows"),
            "usable_sequence_rows": holdout_qa.get("usable_sequence_rows"),
            "per_class": holdout_qa.get("per_class"),
            "per_class_reviewed_rows": holdout_qa.get("per_class_reviewed_rows"),
            "per_class_source_counts": holdout_qa.get("per_class_source_counts"),
            "per_class_scenarios": holdout_qa.get("per_class_scenarios"),
            "reviewed_row_percent": holdout_qa.get("reviewed_row_percent"),
        },
        "recovery_policy": evaluation.get("recovery_policy", {}),
        "targeted_refresh_recommendations": evaluation.get("targeted_refresh_recommendations", []),
    }
    if confusion_pairs:
        summary["top_confusion_pair"] = confusion_pairs[0]
    return summary


def evaluate_cycle(best_seed, sweep_summary, cycle_dir, processed_cycle_dir):
    ranked = sweep_summary.get("ranked_runs", [])
    best = sweep_summary.get("best") or {}
    if best_seed is None:
        return {
            "status": "no_completed_runs",
            "promotion": {"all_gates_pass": False},
            "recovery_policy": {"passed": False, "worst_class": "CHECK_TABLE", "failed_labels": ACTION_CLASSES},
            "fresh_real_per_class": {label: {"delta_f1": None, "passed": False} for label in ACTION_CLASSES},
            "regressing_classes": ACTION_CLASSES,
            "targeted_refresh_recommendations": ["Sweep produced no valid seed run; fix pipeline errors first."],
            "ranked_runs": ranked,
            "best": best,
        }

    promotion_path = Path(cycle_dir) / f"seed_{best_seed}" / "promotion_summary.json"
    metrics_path = Path(cycle_dir) / f"seed_{best_seed}" / "metrics.json"
    if not promotion_path.exists():
        return {
            "status": "missing_promotion_summary",
            "promotion": {"all_gates_pass": False},
            "recovery_policy": {"passed": False, "worst_class": "CHECK_TABLE", "failed_labels": ACTION_CLASSES},
            "fresh_real_per_class": {label: {"delta_f1": None, "passed": False} for label in ACTION_CLASSES},
            "regressing_classes": ACTION_CLASSES,
            "targeted_refresh_recommendations": [
                f"Missing promotion summary for best seed {best_seed}: {promotion_path}"
            ],
            "ranked_runs": ranked,
            "best": best,
        }

    promotion = read_json(promotion_path)
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    processed_metadata = {}
    metadata_path = Path(processed_cycle_dir) / f"seed_{best_seed}" / "metadata.json"
    if metadata_path.exists():
        processed_metadata = read_json(metadata_path)
    gates = promotion.get("gate_results", {})
    test_ok, _ = get_gate(gates, "test_accuracy_non_regression")
    key_ok, _ = get_gate(gates, "key_class_f1_non_regression")
    fresh_agg_ok, fresh_agg = get_gate(gates, "fresh_real_improvement")
    fresh_class_ok, _ = get_gate(gates, "fresh_real_per_class_non_regression")

    delta_map = build_class_delta_map(gates)
    regressing = regressing_classes_from_deltas(delta_map)
    recovery_policy = evaluate_recovery_policy(metrics)
    confusion_pairs = top_confusion_pairs(metrics.get("fresh_real_eval", {}).get("confusion_matrix", []))
    top_pair = confusion_pairs[0] if confusion_pairs else None
    recommendations = build_refresh_recommendations(regressing or recovery_policy.get("failed_labels", []))
    if top_pair:
        recommendations.insert(
            0,
            "Top confusion to collect next: "
            f"{top_pair['true_label']} -> {top_pair['pred_label']} "
            f"(count={top_pair['count']}).",
        )
    holdout_quality = processed_metadata.get("holdout_latest_real_source", {}).get("qa", {}) if isinstance(processed_metadata, dict) else {}
    all_gates_pass = bool(
        test_ok
        and key_ok
        and fresh_agg_ok
        and fresh_class_ok
        and not regressing
        and recovery_policy.get("passed", False)
        and bool(holdout_quality.get("promotion_grade_ready", False))
    )

    result = {
        "status": "evaluated",
        "promotion": {
            "promotable_flag": bool(promotion.get("promotable", False)),
            "status": promotion.get("status"),
            "all_gates_pass": all_gates_pass,
            "test_non_regression_passed": test_ok,
            "key_class_non_regression_passed": key_ok,
            "fresh_real_aggregate_threshold_passed": fresh_agg_ok,
            "fresh_real_per_class_non_regression_passed": fresh_class_ok,
            "fresh_real_aggregate": {
                "old": fresh_agg.get("old"),
                "new": fresh_agg.get("new"),
                "delta": fresh_agg.get("delta"),
                "required_delta": fresh_agg.get("required_delta"),
            },
            "failure_reasons": promotion.get("failure_reasons", []),
            "recommended_actions": promotion.get("recommended_actions", []),
        },
        "recovery_policy": recovery_policy,
        "fresh_real_per_class": delta_map,
        "regressing_classes": regressing,
        "top_confusion_pair": top_pair,
        "holdout_quality": holdout_quality,
        "targeted_refresh_recommendations": recommendations,
        "ranked_runs": ranked,
        "best": best,
        "best_seed_promotion_summary": str(promotion_path),
    }
    recovery_summary = build_recovery_summary(metrics, processed_metadata, result)
    recovery_summary_path = Path(cycle_dir) / "fresh_real_recovery_summary.json"
    recovery_summary_path.write_text(json.dumps(recovery_summary, indent=2), encoding="utf-8")
    result["fresh_real_recovery_summary"] = str(recovery_summary_path)
    return result


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_report_dir = Path(args.track1_report_root) / stamp
    base_processed_dir = Path(args.processed_root) / stamp
    base_model_dir = Path(args.model_root) / stamp

    base_report_dir.mkdir(parents=True, exist_ok=True)
    base_processed_dir.mkdir(parents=True, exist_ok=True)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    cycles = []
    final_status = "max_cycles_reached_without_recovery_pass"
    promoted_cycle = None
    consecutive_comparable_recovery_passes = 0

    for cycle_idx in range(1, args.max_refresh_cycles + 1):
        profile = cycle_profile(cycle_idx, args.real_recovery_every_cycles)
        epochs = cycle_epochs(cycle_idx, args.epochs_min, args.epochs_max)

        cycle_id = f"cycle_{cycle_idx:02d}_{profile}"
        cycle_report_dir = base_report_dir / cycle_id
        cycle_processed_dir = base_processed_dir / cycle_id
        cycle_model_dir = base_model_dir / cycle_id
        cycle_report_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_sweep_cmd(args, profile, epochs, cycle_report_dir, cycle_processed_dir, cycle_model_dir)

        cycle_record = {
            "cycle_index": cycle_idx,
            "cycle_id": cycle_id,
            "profile": profile,
            "is_real_recovery": bool(profile == "real_recovery"),
            "epochs": epochs,
            "seeds": seeds,
            "sweep_command": cmd,
            "sweep_summary_path": str(cycle_report_dir / "seed_sweep_summary.json"),
            "evaluation": {},
        }

        if args.dry_run:
            cycle_record["evaluation"] = {
                "status": "dry_run_not_executed",
                "promotion": {"all_gates_pass": False},
                "recovery_policy": {"passed": False, "worst_class": "CHECK_TABLE", "failed_labels": ACTION_CLASSES},
                "fresh_real_per_class": {label: {"delta_f1": None, "passed": False} for label in ACTION_CLASSES},
                "regressing_classes": ACTION_CLASSES,
                "targeted_refresh_recommendations": [
                    "Dry-run mode: execute without --dry-run to evaluate promotion gates and deltas."
                ],
            }
            cycles.append(cycle_record)
            continue

        subprocess.run(cmd, check=True)

        summary_path = Path(cycle_record["sweep_summary_path"])
        if not summary_path.exists():
            cycle_record["evaluation"] = {
                "status": "missing_seed_sweep_summary",
                "promotion": {"all_gates_pass": False},
                "recovery_policy": {"passed": False, "worst_class": "CHECK_TABLE", "failed_labels": ACTION_CLASSES},
                "fresh_real_per_class": {label: {"delta_f1": None, "passed": False} for label in ACTION_CLASSES},
                "regressing_classes": ACTION_CLASSES,
                "targeted_refresh_recommendations": [
                    "Sweep summary missing. Inspect sweep logs and fix failing seed runs before next cycle."
                ],
            }
            cycles.append(cycle_record)
            continue

        sweep_summary = read_json(summary_path)
        best_seed = None
        if sweep_summary.get("best") and sweep_summary["best"].get("seed") is not None:
            best_seed = int(sweep_summary["best"]["seed"])

        evaluation = evaluate_cycle(best_seed, sweep_summary, cycle_report_dir, cycle_processed_dir)
        cycle_record["evaluation"] = evaluation
        cycles.append(cycle_record)

        recovery_passed = bool(evaluation.get("recovery_policy", {}).get("passed", False))
        if profile == "comparable" and recovery_passed:
            consecutive_comparable_recovery_passes += 1
        elif profile == "comparable":
            consecutive_comparable_recovery_passes = 0

        if evaluation.get("promotion", {}).get("all_gates_pass", False):
            final_status = "promotable_under_recovery_policy"
            promoted_cycle = cycle_idx
            break
        if consecutive_comparable_recovery_passes >= 2:
            final_status = "ready_to_tighten_canonical_promotion_gates"
            promoted_cycle = cycle_idx
            break

    report = {
        "track": "track_1_reasoning_strong_labels",
        "timestamp": stamp,
        "status": final_status,
        "promoted_cycle": promoted_cycle,
        "constraints": {
            "architecture_frozen": True,
            "allowed_tuning_knobs": [
                "data_composition",
                "balancing_strategy",
                "hard_negative_ratio",
                "hard_negative_pool_selection",
            ],
            "forbidden_changes": [
                "model_architecture",
                "training_api_contract",
                "feature_schema_churn",
            ],
        },
        "config": {
            "baseline": args.baseline,
            "seeds": seeds,
            "epochs_min": args.epochs_min,
            "epochs_max": args.epochs_max,
            "real_recovery_every_cycles": args.real_recovery_every_cycles,
            "comparable_holdout": {
                "holdout_per_class": args.comparable_holdout_per_class,
                "holdout_min_total": args.comparable_holdout_min_total,
                "holdout_min_sources": args.comparable_holdout_min_sources,
                "holdout_min_reviewed_per_class": args.comparable_holdout_min_reviewed_per_class,
                "holdout_min_class_sources": args.comparable_holdout_min_class_sources,
            },
            "recovery_holdout": {
                "holdout_per_class": args.recovery_holdout_per_class,
                "holdout_min_total": args.recovery_holdout_min_total,
                "holdout_min_sources": args.recovery_holdout_min_sources,
                "holdout_min_reviewed_per_class": args.recovery_holdout_min_reviewed_per_class,
                "holdout_min_class_sources": args.recovery_holdout_min_class_sources,
            },
            "require_scenario_tags": bool(args.require_scenario_tags),
            "max_refresh_cycles": args.max_refresh_cycles,
            "dry_run": bool(args.dry_run),
            "recovery_policy": RECOVERY_POLICY,
        },
        "cycles": cycles,
    }

    out_path = base_report_dir / "track1_cycle_summary.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Track 1 summary: {out_path}")


if __name__ == "__main__":
    main()
