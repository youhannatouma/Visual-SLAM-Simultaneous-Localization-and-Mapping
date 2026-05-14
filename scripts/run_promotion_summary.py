import argparse
import json
import shutil
from pathlib import Path


def get_float(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if cur is None:
        return None
    try:
        return float(cur)
    except Exception:
        return None


def get_int(obj, *keys):
    value = get_float(obj, *keys)
    if value is None:
        return None
    return int(value)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate promotion status for a metrics.json run.")
    p.add_argument("--current-metrics-path", required=True)
    p.add_argument("--baseline-path", required=True)
    p.add_argument("--summary-path", required=True)
    p.add_argument("--fresh-real-min-improve-acc", type=float, default=0.10)
    p.add_argument("--fresh-real-min-improve-macro-f1", type=float, default=0.10)
    p.add_argument("--fresh-real-absolute-min-acc", type=float, default=None)
    p.add_argument("--fresh-real-absolute-min-macro-f1", type=float, default=None)
    p.add_argument("--fresh-real-min-total", type=int, default=48)
    p.add_argument("--establish-promotion-baseline", action="store_true")
    p.add_argument("--no-promotion-write", action="store_true")
    p.add_argument("--allow-non-promotable", action="store_true")
    p.add_argument("--allow-fresh-real-per-class-regression", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    current_path = Path(args.current_metrics_path)
    baseline_path = Path(args.baseline_path)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    if not current_path.exists():
        raise SystemExit(f"Current metrics file not found: {current_path}")

    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline_present = baseline_path.exists()
    if baseline_present:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = {}

    gate_results = {}
    recommended_actions = []
    promotable = True
    status = "promoted"
    failure_reasons = []

    gate_results["baseline_present"] = {"passed": bool(baseline_present), "path": str(baseline_path)}
    if not baseline_present:
        if args.establish_promotion_baseline:
            shutil.copy2(current_path, baseline_path)
            promotable = False
            status = "baseline_established_non_promotable"
            failure_reasons.append("No promoted baseline existed; baseline established for next-cycle comparison.")
            recommended_actions.append("Run one more full cycle to compare against the newly established promoted baseline.")
        else:
            promotable = False
            status = "baseline_missing"
            failure_reasons.append("Promoted baseline metrics missing; cannot compute promotion gates.")
            recommended_actions.append("Run once with --establish-promotion-baseline to seed baseline.")

    if baseline_present:
        tol = 0.005
        old_test = get_float(baseline, "accuracy", "test")
        new_test = get_float(current, "accuracy", "test")
        test_ok = old_test is not None and new_test is not None and (new_test + tol >= old_test)
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
        key_class_missing = []
        for label in ("CHECK_TABLE", "MOVE_TO_CHAIR"):
            old_f1 = get_float(baseline, "per_class", label, "f1")
            new_f1 = get_float(current, "per_class", label, "f1")
            ok = old_f1 is not None and new_f1 is not None and (new_f1 + 1e-12 >= old_f1)
            if old_f1 is None or new_f1 is None:
                key_class_missing.append(label)
            key_class_rows[label] = {"passed": bool(ok), "old_f1": old_f1, "new_f1": new_f1}
            if not ok:
                key_class_fail = True
        gate_results["key_class_f1_non_regression"] = {"passed": not key_class_fail, "classes": key_class_rows}
        if key_class_missing:
            promotable = False
            failure_reasons.append(f"Key-class F1 missing for: {', '.join(sorted(set(key_class_missing)))}.")
        elif key_class_fail:
            promotable = False
            failure_reasons.append("Key class F1 regression detected.")
            recommended_actions.append("Increase reviewed real data for CHECK_TABLE/MOVE_TO_CHAIR edge cases.")

        fresh_rows = get_int(current, "fresh_real_eval", "rows")
        seq_len = get_int(current, "sequence_length") or 1
        min_eval_rows = max(1, int(args.fresh_real_min_total) - int(seq_len) + 1)
        fresh_eval_ok = fresh_rows is not None and fresh_rows >= min_eval_rows
        gate_results["fresh_real_eval_min_rows"] = {
            "passed": bool(fresh_eval_ok),
            "rows": fresh_rows,
            "min_rows": min_eval_rows,
        }
        if not fresh_eval_ok:
            promotable = False
            failure_reasons.append(f"Fresh-real eval too small or missing (rows={fresh_rows}, min_rows={min_eval_rows}).")

        old_fresh_acc = get_float(baseline, "fresh_real_eval", "accuracy")
        new_fresh_acc = get_float(current, "fresh_real_eval", "accuracy")
        old_fresh_f1 = get_float(baseline, "fresh_real_eval", "macro_f1")
        new_fresh_f1 = get_float(current, "fresh_real_eval", "macro_f1")
        acc_delta = None if old_fresh_acc is None or new_fresh_acc is None else (new_fresh_acc - old_fresh_acc)
        f1_delta = None if old_fresh_f1 is None or new_fresh_f1 is None else (new_fresh_f1 - old_fresh_f1)
        fresh_acc_ok = acc_delta is not None and acc_delta >= float(args.fresh_real_min_improve_acc)
        fresh_f1_ok = f1_delta is not None and f1_delta >= float(args.fresh_real_min_improve_macro_f1)
        abs_acc_ok = True if args.fresh_real_absolute_min_acc is None else (
            new_fresh_acc is not None and new_fresh_acc >= float(args.fresh_real_absolute_min_acc)
        )
        abs_f1_ok = True if args.fresh_real_absolute_min_macro_f1 is None else (
            new_fresh_f1 is not None and new_fresh_f1 >= float(args.fresh_real_absolute_min_macro_f1)
        )
        gate_results["fresh_real_improvement"] = {
            "passed": bool(fresh_eval_ok and fresh_acc_ok and fresh_f1_ok and abs_acc_ok and abs_f1_ok),
            "old": {"accuracy": old_fresh_acc, "macro_f1": old_fresh_f1},
            "new": {"accuracy": new_fresh_acc, "macro_f1": new_fresh_f1},
            "delta": {"accuracy": acc_delta, "macro_f1": f1_delta},
            "required_delta": {
                "accuracy": float(args.fresh_real_min_improve_acc),
                "macro_f1": float(args.fresh_real_min_improve_macro_f1),
            },
            "absolute_minimums": {
                "accuracy": args.fresh_real_absolute_min_acc,
                "macro_f1": args.fresh_real_absolute_min_macro_f1,
            },
        }
        if old_fresh_acc is None or new_fresh_acc is None or old_fresh_f1 is None or new_fresh_f1 is None:
            promotable = False
            failure_reasons.append("Fresh-real metrics missing in baseline or current metrics.")
        elif not (fresh_eval_ok and fresh_acc_ok and fresh_f1_ok and abs_acc_ok and abs_f1_ok):
            promotable = False
            failure_reasons.append(
                "Fresh-real gate not met "
                f"(delta accuracy={acc_delta}, delta macro_f1={f1_delta}, "
                f"abs accuracy={new_fresh_acc}, abs macro_f1={new_fresh_f1})."
            )

        class_labels = ("AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE")
        class_gate_rows = {}
        class_regression = False
        class_missing = []
        for label in class_labels:
            old_cls = get_float(baseline, "fresh_real_eval", "per_class", label, "f1")
            new_cls = get_float(current, "fresh_real_eval", "per_class", label, "f1")
            ok = old_cls is not None and new_cls is not None and (new_cls + 1e-12 >= old_cls)
            if old_cls is None or new_cls is None:
                class_missing.append(label)
            class_gate_rows[label] = {
                "passed": bool(ok),
                "old_f1": old_cls,
                "new_f1": new_cls,
                "delta_f1": None if old_cls is None or new_cls is None else (new_cls - old_cls),
            }
            if not ok:
                class_regression = True
        class_gate_passed = args.allow_fresh_real_per_class_regression or not class_regression
        gate_results["fresh_real_per_class_non_regression"] = {
            "passed": bool(class_gate_passed),
            "allow_regression": bool(args.allow_fresh_real_per_class_regression),
            "classes": class_gate_rows,
        }
        if class_missing:
            promotable = False
            failure_reasons.append(f"Fresh-real per-class F1 missing for: {', '.join(sorted(set(class_missing)))}.")
        elif class_regression and not args.allow_fresh_real_per_class_regression:
            promotable = False
            failure_reasons.append("Fresh-real per-class regression detected (mixed-sign deltas).")

    if promotable:
        if not args.no_promotion_write:
            shutil.copy2(current_path, baseline_path)
            status = "promoted"
        else:
            status = "promotable_no_write"
    elif status == "promoted":
        status = "not_promoted"

    summary = {
        "status": status,
        "promotable": bool(promotable),
        "baseline_path": str(baseline_path),
        "current_metrics_path": str(current_path),
        "gate_results": gate_results,
        "failure_reasons": failure_reasons,
        "recommended_actions": recommended_actions,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Promotion summary: {summary_path}")
    if not promotable and not args.allow_non_promotable:
        raise SystemExit("Promotion gate failed: run is non-promotable. See promotion summary JSON for details.")


if __name__ == "__main__":
    main()
