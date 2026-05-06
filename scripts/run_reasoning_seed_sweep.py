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


def run_pipeline(pipeline, python_bin, seed, report_dir, out_dir, model_path, summary_path, baseline_path):
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
        "--no-promotion-write",
        "--allow-non-promotable",
    ]
    subprocess.run(cmd, check=True)


def load_metrics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    test_acc = data.get("accuracy", {}).get("test")
    per_class = data.get("per_class", {})
    macro_f1 = data.get("macro_f1")
    return data, test_acc, per_class, macro_f1


def is_non_regressing(baseline, candidate):
    _, base_acc, base_classes, _ = baseline
    _, cand_acc, cand_classes, _ = candidate

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


def select_best(baseline, candidates):
    eligible = []
    for seed, metrics_path in candidates.items():
        metrics = load_metrics(metrics_path)
        if is_non_regressing(baseline, metrics):
            _, _, per_class, macro_f1 = metrics
            move_f1 = per_class.get("MOVE_TO_CHAIR", {}).get("f1", 0.0)
            eligible.append((seed, move_f1, macro_f1 or 0.0, metrics_path))

    if not eligible:
        return None

    eligible.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return eligible[0]


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

    metrics_paths = {}
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
            )
        except subprocess.CalledProcessError as exc:
            print(f"Seed {seed} failed: {exc}")
            if not args.keep_going:
                raise
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            metrics_paths[seed] = metrics_path

    best = select_best(baseline, metrics_paths)
    summary = {
        "baseline": args.baseline,
        "runs": metrics_paths,
        "best": None,
    }

    if best is not None:
        best_seed, move_f1, macro_f1, metrics_path = best
        summary["best"] = {
            "seed": best_seed,
            "move_to_chair_f1": float(move_f1),
            "macro_f1": float(macro_f1),
            "metrics": metrics_path,
        }

        if args.promote_best:
            promote_best(
                best_seed,
                os.path.dirname(metrics_path),
                os.path.join(model_root, f"seed_{best_seed}.pt"),
                args.canonical_report_dir,
                args.canonical_model,
            )
    else:
        print("No non-regressing runs found.")

    summary_path = Path(out_root) / "seed_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Seed sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
