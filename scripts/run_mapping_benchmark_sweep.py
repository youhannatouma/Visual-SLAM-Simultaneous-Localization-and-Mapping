import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_csv_list(raw: str, cast):
    values = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_report(report: dict) -> dict:
    map_metrics = report.get("map_metrics", {})
    config = report.get("config", {})
    primary_metric = map_metrics.get("benchmark_obstacle_metric_primary", {})
    alternate_metric = map_metrics.get("benchmark_obstacle_metric_alternate", {})
    summary = {
        "report_path": report.get("_report_path", ""),
        "backend": config.get("mapping_backend"),
        "det_confidence": config.get("det_confidence"),
        "det_imgsz": config.get("det_imgsz"),
        "footprint_radius": config.get("map_obstacle_footprint_radius_cells"),
        "footprint_shape": config.get("map_obstacle_footprint_shape"),
        "match_radius": config.get("map_obstacle_match_radius_cells"),
        "persistence_frames": config.get("map_obstacle_persistence_frames"),
        "confidence_strength": config.get("map_confidence_strength"),
        "selected_metric": map_metrics.get("benchmark_obstacle_metric"),
        "selected_metric_f1": float(primary_metric.get("f1", 0.0)) if primary_metric.get("available") else 0.0,
        "cell_f1": float(map_metrics.get("obstacle_precision_recall", {}).get("f1", 0.0)) if map_metrics.get("obstacle_precision_recall", {}).get("available") else 0.0,
        "object_f1": float(map_metrics.get("obstacle_object_precision_recall", {}).get("f1", 0.0)) if map_metrics.get("obstacle_object_precision_recall", {}).get("available") else 0.0,
        "alternate_metric_name": map_metrics.get("benchmark_obstacle_metric_alternate_name"),
        "alternate_metric_f1": float(alternate_metric.get("f1", 0.0)) if alternate_metric.get("available") else 0.0,
        "map_consistency": float(map_metrics.get("map_consistency_score", {}).get("score_mean", 0.0)),
        "pose_jitter": float(map_metrics.get("pose_jitter", {}).get("jitter_score", 0.0)),
        "obstacle_persistence": float(map_metrics.get("obstacle_persistence_stability", {}).get("iou_mean", 0.0)),
        "occupancy_concentration": float(map_metrics.get("occupancy_confidence_concentration", {}).get("concentration_score", 0.0)),
        "status": map_metrics.get("mapping_quality_summary", {}).get("status"),
        "promotable": bool(map_metrics.get("mapping_quality_summary", {}).get("promotable", False)),
    }
    return summary


def passes_guardrails(summary: dict, thresholds: dict) -> bool:
    return (
        float(summary.get("map_consistency", 0.0)) >= float(thresholds["map_consistency_min"])
        and float(summary.get("pose_jitter", 0.0)) >= float(thresholds["pose_jitter_min"])
        and float(summary.get("obstacle_persistence", 0.0)) >= float(thresholds["obstacle_persistence_min"])
    )


def candidate_sort_key(summary: dict):
    return (
        1 if passes_guardrails(summary, DEFAULT_THRESHOLDS) else 0,
        float(summary.get("object_f1", 0.0)),
        float(summary.get("cell_f1", 0.0)),
        float(summary.get("occupancy_concentration", 0.0)),
    )


def build_candidates(args) -> list:
    combos = []
    for backend, radius, shape, match_radius, persistence, conf_strength in itertools.product(
        parse_csv_list(args.backends, str),
        parse_csv_list(args.footprint_radii, int),
        parse_csv_list(args.footprint_shapes, str),
        parse_csv_list(args.match_radii, int),
        parse_csv_list(args.persistence_frames, int),
        parse_csv_list(args.confidence_strengths, float),
    ):
        combos.append(
            {
                "backend": backend,
                "footprint_radius": radius,
                "footprint_shape": shape,
                "match_radius": match_radius,
                "persistence_frames": persistence,
                "confidence_strength": conf_strength,
            }
        )
    return combos


def run_candidate(repo_root: Path, args, candidate: dict) -> dict:
    main_path = repo_root / "main.py"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        report_path = tmp.name
    cmd = [
        args.python_bin,
        str(main_path),
        "--benchmark-video",
        args.video,
        "--headless",
        "--run-annotations",
        args.annotations,
        "--run-report-out",
        report_path,
        "--mapping-backend",
        candidate["backend"],
        "--det-confidence",
        str(args.det_confidence),
        "--det-imgsz",
        str(args.det_imgsz),
        "--map-obstacle-footprint-radius-cells",
        str(candidate["footprint_radius"]),
        "--map-obstacle-footprint-shape",
        candidate["footprint_shape"],
        "--map-obstacle-match-radius-cells",
        str(candidate["match_radius"]),
        "--map-obstacle-persistence-frames",
        str(candidate["persistence_frames"]),
        "--map-confidence-strength",
        str(candidate["confidence_strength"]),
    ]
    if args.camera_calibration:
        cmd.extend(["--camera-calibration", args.camera_calibration])
    if candidate["backend"] == "heuristic":
        cmd.append("--no-depth")
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    report = load_report(report_path)
    report["_report_path"] = report_path
    summary = summarize_report(report)
    summary["guardrails_passed"] = passes_guardrails(summary, DEFAULT_THRESHOLDS)
    return summary


def write_summary_artifacts(output_json: Path, output_md: Path, baseline: dict, ranked: list):
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": baseline,
        "ranked_candidates": ranked,
        "best_candidate": ranked[0] if ranked else None,
        "guardrail_thresholds": DEFAULT_THRESHOLDS,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Mapping Benchmark Sweep",
        "",
        f"- Baseline object F1: `{baseline.get('object_f1', 0.0):.4f}`",
        f"- Baseline cell F1: `{baseline.get('cell_f1', 0.0):.4f}`",
    ]
    if ranked:
        best = ranked[0]
        lines.extend(
            [
        f"- Best backend: `{best.get('backend')}`",
        f"- Best detector: conf `{best.get('det_confidence')}`, imgsz `{best.get('det_imgsz')}`",
        f"- Best object F1: `{best.get('object_f1', 0.0):.4f}`",
                f"- Best cell F1: `{best.get('cell_f1', 0.0):.4f}`",
                f"- Best footprint: radius `{best.get('footprint_radius')}`, shape `{best.get('footprint_shape')}`",
            ]
        )
    lines.extend(["", "## Ranked Candidates", ""])
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"{idx}. backend=`{row.get('backend')}` object_f1=`{row.get('object_f1', 0.0):.4f}` "
            f"cell_f1=`{row.get('cell_f1', 0.0):.4f}` guardrails=`{row.get('guardrails_passed')}` "
            f"det_conf=`{row.get('det_confidence')}` imgsz=`{row.get('det_imgsz')}` "
            f"radius=`{row.get('footprint_radius')}` shape=`{row.get('footprint_shape')}` "
            f"match_radius=`{row.get('match_radius')}` persistence=`{row.get('persistence_frames')}` "
            f"conf_strength=`{row.get('confidence_strength')}`"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


DEFAULT_THRESHOLDS = {
    "map_consistency_min": 0.70,
    "pose_jitter_min": 0.40,
    "obstacle_persistence_min": 0.20,
}


def parse_args():
    p = argparse.ArgumentParser(description="Sweep mapping benchmark settings and rank configurations.")
    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter used to run main.py.")
    p.add_argument("--video", required=True, help="Benchmark video path.")
    p.add_argument("--annotations", required=True, help="Benchmark annotation JSON path.")
    p.add_argument("--camera-calibration", default="", help="Optional camera calibration JSON path.")
    p.add_argument("--baselines-report", default="", help="Optional existing run report to use as baseline.")
    p.add_argument("--det-confidence", type=float, default=0.25)
    p.add_argument("--det-imgsz", type=int, default=320)
    p.add_argument("--backends", default="heuristic,depth")
    p.add_argument("--footprint-radii", default="0,1,2")
    p.add_argument("--footprint-shapes", default="square,class_aware,cross,horizontal")
    p.add_argument("--match-radii", default="0,1")
    p.add_argument("--persistence-frames", default="1,2,3")
    p.add_argument("--confidence-strengths", default="0.5,1.0")
    p.add_argument("--output-json", default="reports/runtime/mapping_benchmark_sweep_summary.json")
    p.add_argument("--output-md", default="reports/runtime/mapping_benchmark_sweep_summary.md")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    baseline = {}
    if args.baselines_report:
        baseline_report = load_report(args.baselines_report)
        baseline_report["_report_path"] = args.baselines_report
        baseline = summarize_report(baseline_report)

    ranked = []
    for candidate in build_candidates(args):
        ranked.append(run_candidate(repo_root, args, candidate))
    ranked.sort(key=candidate_sort_key, reverse=True)
    if not baseline and ranked:
        baseline = ranked[0]

    write_summary_artifacts(
        output_json=repo_root / args.output_json,
        output_md=repo_root / args.output_md,
        baseline=baseline,
        ranked=ranked,
    )
    print(f"[Sweep] Wrote {args.output_json}")
    print(f"[Sweep] Wrote {args.output_md}")


if __name__ == "__main__":
    main()
