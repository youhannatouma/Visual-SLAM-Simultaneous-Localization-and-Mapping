import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a multi-seed reasoning pipeline sweep and select the best non-regressing run"
    )
    parser.add_argument(
        "--seeds",
        default="17,42,123",
        help="Comma-separated list of seeds to run.",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv311/bin/python",
        help="Python executable to pass to the pipeline.",
    )
    parser.add_argument(
        "--pipeline",
        default="scripts/run_reasoning_training_pipeline.sh",
        help="Pipeline script path.",
    )
    parser.add_argument(
        "--baseline",
        default="reports/metrics_promoted_baseline.json",
        help="Baseline metrics for non-regression checks.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Root output directory for per-seed reports.",
    )
    parser.add_argument(
        "--processed-root",
        default="",
        help="Root directory for per-seed processed outputs.",
    )
    parser.add_argument(
        "--model-root",
        default="",
        help="Root directory for per-seed model outputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue sweep even if a seed run fails.",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy the best run to canonical reports and model outputs.",
    )
    parser.add_argument(
        "--canonical-report-dir",
        default="reports",
        help="Canonical report directory for promotion (default: reports).",
    )
    parser.add_argument(
        "--canonical-model",
        default="models/reasoning_model.pt",
        help="Canonical model path for promotion.",
    )
    parser.add_argument(
        "--disk-min-free-gb",
        type=float,
        default=1.0,
        help="Minimum free disk (GB) passed to the guarded pipeline.",
    )
    parser.add_argument(
        "--holdout-per-class",
        type=int,
        default=12,
        help="Rows per class used to assemble fresh-real holdout.",
    )
    parser.add_argument(
        "--holdout-min-total",
        type=int,
        default=48,
        help="Minimum total rows required in fresh-real holdout.",
    )
    parser.add_argument(
        "--holdout-min-sources",
        type=int,
        default=4,
        help="Minimum distinct real sources required in fresh-real holdout.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs forwarded to guarded pipeline.",
    )
    parser.add_argument(
        "--holdout-min-reviewed-per-class",
        type=int,
        default=0,
        help="Minimum reviewed holdout rows per class passed through to preprocessing.",
    )
    parser.add_argument(
        "--holdout-min-class-sources",
        type=int,
        default=0,
        help="Minimum holdout source diversity per class passed through to preprocessing.",
    )
    parser.add_argument(
        "--holdout-require-scenario-tags",
        action="store_true",
        help="Require scenario tags on holdout rows.",
    )
    parser.add_argument(
        "--training-profile",
        default="comparable",
        choices=["comparable", "real_recovery"],
        help="Training profile passed through to train_reasoning.py.",
    )
    parser.add_argument(
        "--real-reviewed-weight",
        type=float,
        default=1.0,
        help="Sampling multiplier for reviewed real rows.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help="Sampling multiplier for hard negatives.",
    )
    parser.add_argument(
        "--class-target-weights",
        default="",
        help="LABEL:WEIGHT pairs for reviewed real row emphasis.",
    )
    return parser.parse_args()


def parse_seeds(text):
    items = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise ValueError("No valid seeds provided")
    return items


def run_pipeline(
    pipeline,
    python_bin,
    seed,
    report_dir,
    out_dir,
    model_path,
    summary_path,
    baseline_path,
    disk_min_free_gb,
    holdout_per_class,
    holdout_min_total,
    holdout_min_sources,
    epochs,
    holdout_min_reviewed_per_class,
    holdout_min_class_sources,
    holdout_require_scenario_tags,
    training_profile,
    real_reviewed_weight,
    hard_negative_weight,
    class_target_weights,
):
    report_path = os.path.join(report_dir, "dataset_audit.json")
    manifest_path = os.path.join(report_dir, "manifest", "dataset_manifest.json")
    changelog_path = os.path.join(report_dir, "manifest", "CHANGELOG.md")
    cmd = [
        pipeline,
        "--python-bin",
        python_bin,
        "--seed",
        str(seed),
        "--report",
        report_path,
        "--report-dir",
        report_dir,
        "--out-dir",
        out_dir,
        "--model-path",
        model_path,
        "--promotion-summary",
        summary_path,
        "--promotion-baseline-path",
        baseline_path,
        "--manifest-path",
        manifest_path,
        "--changelog-path",
        changelog_path,
        "--disk-min-free-gb",
        str(disk_min_free_gb),
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
        "--epochs",
        str(epochs),
        "--training-profile",
        str(training_profile),
        "--real-reviewed-weight",
        str(real_reviewed_weight),
        "--hard-negative-weight",
        str(hard_negative_weight),
        "--no-promotion-write",
        "--allow-non-promotable",
    ]
    if class_target_weights:
        cmd.extend(["--class-target-weights", class_target_weights])
    if holdout_require_scenario_tags:
        cmd.append("--holdout-require-scenario-tags")
    subprocess.run(cmd, check=True)


def load_metrics(path):
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}, None, {}, None, None, None, None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_acc = data.get("accuracy", {}).get("test")
    per_class = data.get("per_class", {})
    macro_f1 = data.get("macro_f1")
    fresh_real = data.get("fresh_real_eval", {})
    fresh_real_acc = fresh_real.get("accuracy")
    fresh_real_macro_f1 = fresh_real.get("macro_f1")
    fresh_real_per_class = fresh_real.get("per_class", {})
    worst_fresh_real_f1 = min(
        (
            float(row.get("f1", 0.0) or 0.0)
            for row in fresh_real_per_class.values()
        ),
        default=0.0,
    )
    return data, test_acc, per_class, macro_f1, fresh_real_acc, fresh_real_macro_f1, worst_fresh_real_f1


def load_promotion_summary(path):
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_non_regressing(baseline, candidate):
    _, base_acc, base_classes, _, _, _, _ = baseline
    _, cand_acc, cand_classes, _, _, _, _ = candidate

    if base_acc is None or cand_acc is None:
        return False
    if float(cand_acc) + 1e-12 < float(base_acc):
        return False
    for label in ACTION_CLASSES:
        base_f1 = base_classes.get(label, {}).get("f1")
        cand_f1 = cand_classes.get(label, {}).get("f1")
        if base_f1 is None or cand_f1 is None:
            return False
        if float(cand_f1) + 1e-12 < float(base_f1):
            return False
    return True


def select_and_rank_runs(baseline, candidates, promotion_summaries):
    ranked = []
    for seed, metrics_path in candidates.items():
        metrics = load_metrics(metrics_path)
        _, test_acc, per_class, macro_f1, fresh_real_acc, fresh_real_macro_f1, worst_fresh_real_f1 = metrics
        move_f1 = per_class.get("MOVE_TO_CHAIR", {}).get("f1", 0.0)
        promotion = promotion_summaries.get(seed, {})
        promotable = bool(promotion.get("promotable", False))
        non_regressing = is_non_regressing(baseline, metrics)
        failure_reasons = promotion.get("failure_reasons", [])
        recommended_actions = promotion.get("recommended_actions", [])
        ranked.append(
            {
                "seed": int(seed),
                "promotable": promotable,
                "non_regressing": bool(non_regressing),
                "test_accuracy": float(test_acc or 0.0),
                "move_to_chair_f1": float(move_f1 or 0.0),
                "macro_f1": float(macro_f1 or 0.0),
                "fresh_real_accuracy": float(fresh_real_acc or 0.0),
                "fresh_real_macro_f1": float(fresh_real_macro_f1 or 0.0),
                "worst_fresh_real_f1": float(worst_fresh_real_f1 or 0.0),
                "metrics": metrics_path,
                "failure_reasons": failure_reasons,
                "recommended_actions": recommended_actions,
            }
        )

    if not ranked:
        return None, []

    ranked.sort(
        key=lambda r: (
            r["fresh_real_macro_f1"],
            r["worst_fresh_real_f1"],
            1 if r["non_regressing"] else 0,
            r["test_accuracy"],
            r["macro_f1"],
            1 if r["promotable"] else 0,
            r["move_to_chair_f1"],
        ),
        reverse=True,
    )
    return ranked[0], ranked


def promote_best(best_seed, report_dir, model_path, canonical_report_dir, canonical_model):
    os.makedirs(canonical_report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(canonical_model), exist_ok=True)

    src_metrics = Path(report_dir) / "metrics.json"
    src_confusion = Path(report_dir) / "confusion_matrix.png"
    dst_metrics = Path(canonical_report_dir) / "metrics.json"
    dst_confusion = Path(canonical_report_dir) / "confusion_matrix.png"

    if src_metrics.exists():
        dst_metrics.write_text(src_metrics.read_text(encoding="utf-8"), encoding="utf-8")
    if src_confusion.exists():
        dst_confusion.write_bytes(src_confusion.read_bytes())

    if Path(model_path).exists():
        Path(canonical_model).write_bytes(Path(model_path).read_bytes())

    print(f"Promoted seed {best_seed} to {canonical_report_dir} and {canonical_model}")


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or f"reports/seed_sweep_{timestamp}"
    processed_root = args.processed_root or f"data/processed_sweep_{timestamp}"
    model_root = args.model_root or f"models/seed_sweep_{timestamp}"

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    baseline = load_metrics(args.baseline)
    if not baseline[0]:
        print(f"Warning: baseline metrics not found at {args.baseline}; non-regression checks will be skipped.")

    metrics_paths = {}
    promotion_summaries = {}
    for seed in seeds:
        run_dir = os.path.join(out_root, f"seed_{seed}")
        out_dir = os.path.join(processed_root, f"seed_{seed}")
        model_path = os.path.join(model_root, f"seed_{seed}.pt")
        summary_path = os.path.join(run_dir, "promotion_summary.json")
        os.makedirs(run_dir, exist_ok=True)

        try:
            run_pipeline(
                args.pipeline,
                args.python_bin,
                seed,
                run_dir,
                out_dir,
                model_path,
                summary_path,
                args.baseline,
                args.disk_min_free_gb,
                args.holdout_per_class,
                args.holdout_min_total,
                args.holdout_min_sources,
                args.epochs,
                args.holdout_min_reviewed_per_class,
                args.holdout_min_class_sources,
                args.holdout_require_scenario_tags,
                args.training_profile,
                args.real_reviewed_weight,
                args.hard_negative_weight,
                args.class_target_weights,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Seed {seed} failed: {exc}")
            if not args.keep_going:
                raise
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            metrics_paths[seed] = metrics_path
        promotion_summaries[seed] = load_promotion_summary(summary_path)

    best, ranked = select_and_rank_runs(baseline, metrics_paths, promotion_summaries)
    summary = {
        "baseline": args.baseline,
        "runs": metrics_paths,
        "best": None,
        "ranked_runs": ranked,
    }

    if best is not None:
        best_seed = best["seed"]
        move_f1 = best["move_to_chair_f1"]
        macro_f1 = best["macro_f1"]
        metrics_path = best["metrics"]
        summary["best"] = {
            "seed": best_seed,
            "test_accuracy": float(best.get("test_accuracy", 0.0)),
            "move_to_chair_f1": float(move_f1),
            "macro_f1": float(macro_f1),
            "fresh_real_accuracy": float(best.get("fresh_real_accuracy", 0.0)),
            "fresh_real_macro_f1": float(best.get("fresh_real_macro_f1", 0.0)),
            "worst_fresh_real_f1": float(best.get("worst_fresh_real_f1", 0.0)),
            "metrics": metrics_path,
            "promotable": bool(best.get("promotable", False)),
            "failure_reasons": best.get("failure_reasons", []),
            "recommended_actions": best.get("recommended_actions", []),
        }

        if args.promote_best and bool(best.get("promotable", False)):
            promote_best(
                best_seed,
                os.path.dirname(metrics_path),
                os.path.join(model_root, f"seed_{best_seed}.pt"),
                args.canonical_report_dir,
                args.canonical_model,
            )
        elif args.promote_best:
            print("Best run is not promotable; skipping promotion write.")
    else:
        print("No completed runs found.")

    summary_path = Path(out_root) / "seed_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Seed sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
