import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_contract(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_report_sections(report: dict, required_sections):
    missing = [section for section in required_sections if section not in report]
    if missing:
        raise ValueError(f"Runtime report missing required sections: {', '.join(missing)}")


def parse_args():
    p = argparse.ArgumentParser(description="Validate single-machine release readiness.")
    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter used to run runtime smoke validation.")
    p.add_argument("--contract", default="deployment/release_contract.json", help="Release contract JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    contract_path = repo_root / args.contract
    require_path(contract_path, "Release contract")
    contract = load_contract(contract_path)

    artifacts = contract.get("artifacts", {})
    runtime_profile = contract.get("runtime_profile", {})
    required_report_sections = contract.get("required_report_sections", [])

    model_path = repo_root / artifacts["reasoning_model"]
    yolo_path = repo_root / artifacts["yolo_weights"]
    metrics_path = repo_root / artifacts["metrics"]
    promotion_summary_path = repo_root / artifacts["promotion_summary"]
    runtime_report_path = repo_root / artifacts["runtime_validation_report"]

    for label, path in [
        ("Reasoning model checkpoint", model_path),
        ("YOLO weights", yolo_path),
        ("Metrics artifact", metrics_path),
        ("Promotion summary artifact", promotion_summary_path),
    ]:
        require_path(path, label)

    import cv2  # noqa: F401
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    from reasoning import FEATURE_SIZE, ReasoningMLP, SEQUENCE_LENGTH, load_reasoning_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict, metadata = load_reasoning_checkpoint(str(model_path), device)
    feature_size = int(metadata.get("feature_size", FEATURE_SIZE)) if metadata else FEATURE_SIZE
    sequence_length = int(metadata.get("sequence_length", SEQUENCE_LENGTH)) if metadata else SEQUENCE_LENGTH
    if feature_size != FEATURE_SIZE:
        raise ValueError(f"Checkpoint feature size mismatch: {feature_size} != {FEATURE_SIZE}")
    if sequence_length != SEQUENCE_LENGTH:
        raise ValueError(f"Checkpoint sequence length mismatch: {sequence_length} != {SEQUENCE_LENGTH}")
    model = ReasoningMLP(FEATURE_SIZE, sequence_length=SEQUENCE_LENGTH)
    model.load_state_dict(state_dict)
    model.eval()

    promotion_summary = json.loads(promotion_summary_path.read_text(encoding="utf-8"))
    if not bool(promotion_summary.get("promotable", False)):
        raise ValueError(f"Promotion summary is not promotable: {promotion_summary_path}")

    benchmark_video = repo_root / runtime_profile["benchmark_video"]
    annotations = repo_root / runtime_profile["run_annotations"]
    require_path(benchmark_video, "Benchmark video")
    require_path(annotations, "Benchmark annotations")
    runtime_report_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_bin,
        "main.py",
        "--benchmark-video",
        str(benchmark_video),
        "--run-annotations",
        str(annotations),
        "--run-report-out",
        str(runtime_report_path),
        "--max-frames",
        str(runtime_profile.get("max_frames", 90)),
        "--det-confidence",
        str(runtime_profile.get("det_confidence", 0.1)),
        "--det-imgsz",
        str(runtime_profile.get("det_imgsz", 640)),
        "--mapping-backend",
        str(runtime_profile.get("mapping_backend", "heuristic")),
        "--map-benchmark-obstacle-metric",
        str(runtime_profile.get("map_benchmark_obstacle_metric", "object")),
        "--map-obstacle-footprint-radius-cells",
        str(runtime_profile.get("map_obstacle_footprint_radius_cells", 1)),
        "--map-obstacle-footprint-shape",
        str(runtime_profile.get("map_obstacle_footprint_shape", "class_aware")),
    ]
    if runtime_profile.get("headless", True):
        command.append("--headless")
    if runtime_profile.get("no_depth", True):
        command.append("--no-depth")
    calibration = str(runtime_profile.get("camera_calibration", "")).strip()
    if calibration:
        command.extend(["--camera-calibration", str(repo_root / calibration)])

    subprocess.run(command, cwd=str(repo_root), check=True)
    require_path(runtime_report_path, "Runtime validation report")
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    validate_report_sections(runtime_report, required_report_sections)
    print(f"Release validation succeeded: {runtime_report_path}")


if __name__ == "__main__":
    main()
