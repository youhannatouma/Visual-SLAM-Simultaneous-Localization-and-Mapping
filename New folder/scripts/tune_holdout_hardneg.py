import argparse
import json
import subprocess
import time
from pathlib import Path


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Joint tuning for holdout assembly and hard-negative ratio")
    parser.add_argument("--python-bin", default=".venv311/bin/python")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--holdout-per-class-grid", default="8,10,12")
    parser.add_argument("--holdout-min-total-grid", default="32,40")
    parser.add_argument("--hardneg-ratio-grid", default="0.15,0.25,0.35")
    parser.add_argument("--holdout-min-sources", type=int, default=2)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--source-glob", default="data/raw_archive/*.csv")
    parser.add_argument("--baseline", default="reports/metrics_promoted_baseline.json")
    parser.add_argument("--output", default="reports/tuning_holdout_hardneg_summary.json")
    args = parser.parse_args()

    cwd = Path(args.workdir).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    holdout_pc_grid = parse_int_list(args.holdout_per_class_grid)
    holdout_min_total_grid = parse_int_list(args.holdout_min_total_grid)
    hardneg_grid = parse_float_list(args.hardneg_ratio_grid)

    results = []
    rank = 0

    for holdout_pc in holdout_pc_grid:
        for holdout_min_total in holdout_min_total_grid:
            for hardneg in hardneg_grid:
                rank += 1
                exp_id = f"exp_{rank:02d}_hpc{holdout_pc}_hmin{holdout_min_total}_hn{str(hardneg).replace('.', 'p')}"
                out_root = f"reports/tuning_{stamp}/{exp_id}"
                processed_root = f"data/processed_tuning_{stamp}/{exp_id}"
                model_root = f"models/tuning_{stamp}/{exp_id}"

                run([
                    args.python_bin,
                    "scripts/build_move_recovery_pool.py",
                    "--source-glob", args.source_glob,
                    "--allow-unreviewed",
                    "--target-label", "MOVE_TO_CHAIR",
                    "--contrast-labels", "CHECK_TABLE,EXPLORE,AVOID_PERSON",
                    "--contrast-ratio", str(hardneg),
                    "--needs-review-only",
                    "--exclude-existing-glob", "data/raw/*.csv",
                    "--limit", str(args.limit),
                    "--out-csv", "data/raw/move_recovery_pool_20260507.csv",
                    "--write-review-status",
                ], cwd)

                run([
                    args.python_bin,
                    "scripts/run_reasoning_seed_sweep.py",
                    "--seeds", args.seeds,
                    "--python-bin", args.python_bin,
                    "--epochs", str(args.epochs),
                    "--holdout-per-class", str(holdout_pc),
                    "--holdout-min-total", str(holdout_min_total),
                    "--holdout-min-sources", str(args.holdout_min_sources),
                    "--baseline", args.baseline,
                    "--out-root", out_root,
                    "--processed-root", processed_root,
                    "--model-root", model_root,
                    "--keep-going",
                ], cwd)

                summary = read_json(cwd / out_root / "seed_sweep_summary.json")
                ranked = summary.get("ranked_runs", [])
                best = summary.get("best") or {}
                any_promotable = any(bool(x.get("promotable", False)) for x in ranked)
                best_move = float(best.get("move_to_chair_f1", 0.0) or 0.0)
                best_macro = float(best.get("macro_f1", 0.0) or 0.0)
                results.append({
                    "exp_id": exp_id,
                    "holdout_per_class": holdout_pc,
                    "holdout_min_total": holdout_min_total,
                    "holdout_min_sources": args.holdout_min_sources,
                    "hardneg_ratio": hardneg,
                    "seeds": args.seeds,
                    "epochs": args.epochs,
                    "any_promotable": any_promotable,
                    "best_move_to_chair_f1": best_move,
                    "best_macro_f1": best_macro,
                    "summary_path": f"{out_root}/seed_sweep_summary.json",
                    "best": best,
                })

    results.sort(
        key=lambda r: (
            1 if r["any_promotable"] else 0,
            r["best_move_to_chair_f1"],
            r["best_macro_f1"],
        ),
        reverse=True,
    )

    output = {
        "timestamp": stamp,
        "search_space": {
            "holdout_per_class_grid": holdout_pc_grid,
            "holdout_min_total_grid": holdout_min_total_grid,
            "hardneg_ratio_grid": hardneg_grid,
            "holdout_min_sources": args.holdout_min_sources,
            "seeds": args.seeds,
            "epochs": args.epochs,
        },
        "best": results[0] if results else None,
        "results": results,
    }

    out_path = cwd / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote tuning summary: {out_path}")


if __name__ == "__main__":
    main()
