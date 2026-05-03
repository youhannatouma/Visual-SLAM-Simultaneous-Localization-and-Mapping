import argparse
import json
import os
import time
from pathlib import Path


def bytes_to_gb(value):
    return float(value) / (1024 ** 3)


def dir_size_bytes(path: Path):
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def list_files_sorted(path: Path):
    if not path.exists():
        return []
    items = [p for p in path.glob("*") if p.is_file()]
    return sorted(items, key=lambda p: p.stat().st_mtime, reverse=True)


def enforce_disk_guard(min_free_gb):
    stats = os.statvfs(".")
    free_bytes = stats.f_bavail * stats.f_frsize
    free_gb = bytes_to_gb(free_bytes)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk guard failed: free space {free_gb:.2f} GB < required {min_free_gb:.2f} GB. "
            "Run cleanup or increase storage before ingestion/training."
        )
    return free_gb


def prune_files(path: Path, keep_latest, protect_prefixes, dry_run=False):
    deleted = []
    if not path.exists():
        return deleted

    files = list_files_sorted(path)
    protected = []
    deletable = []
    for p in files:
        if any(p.name.startswith(prefix) for prefix in protect_prefixes):
            protected.append(p)
        else:
            deletable.append(p)

    keep_set = set(deletable[:max(0, keep_latest)])
    for p in deletable:
        if p in keep_set:
            continue
        deleted.append(p)
        if not dry_run:
            try:
                p.unlink()
            except OSError:
                pass
    return deleted


def write_budget_report(report_path: Path):
    report = {
        "timestamp": int(time.time()),
        "sizes_gb": {
            "images": round(bytes_to_gb(dir_size_bytes(Path("/Users/guenayfer/SLAM/images"))), 4),
            "data_raw": round(bytes_to_gb(dir_size_bytes(Path("data/raw"))), 4),
            "data_raw_archive": round(bytes_to_gb(dir_size_bytes(Path("data/raw_archive"))), 4),
            "data_processed": round(bytes_to_gb(dir_size_bytes(Path("data/processed"))), 4),
            "reports": round(bytes_to_gb(dir_size_bytes(Path("reports"))), 4),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Disk guard, pruning, and artifact budget reporting")
    parser.add_argument("--min-free-gb", type=float, default=1.0, help="Fail if free disk space is below this value")
    parser.add_argument("--prune", action="store_true", help="Enable pruning")
    parser.add_argument("--keep-images", type=int, default=20, help="Keep latest N image/video files")
    parser.add_argument("--keep-reports", type=int, default=20, help="Keep latest N report files")
    parser.add_argument("--keep-raw-archive", type=int, default=30, help="Keep latest N raw archive CSV files")
    parser.add_argument(
        "--protect-prefix",
        action="append",
        default=[],
        help="Filename prefix to protect from pruning (can be repeated)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--budget-report", default="reports/artifact_budget.json", help="Artifact budget report output")
    return parser.parse_args()


def main():
    args = parse_args()

    free_gb = enforce_disk_guard(args.min_free_gb)
    print(f"Disk guard OK: free space {free_gb:.2f} GB")

    if args.prune:
        deleted_images = prune_files(
            Path("/Users/guenayfer/SLAM/images"),
            keep_latest=args.keep_images,
            protect_prefixes=args.protect_prefix,
            dry_run=args.dry_run,
        )
        deleted_reports = prune_files(
            Path("reports"),
            keep_latest=args.keep_reports,
            protect_prefixes=args.protect_prefix,
            dry_run=args.dry_run,
        )
        deleted_archive = prune_files(
            Path("data/raw_archive"),
            keep_latest=args.keep_raw_archive,
            protect_prefixes=args.protect_prefix,
            dry_run=args.dry_run,
        )
        print(
            "Prune summary: "
            f"images={len(deleted_images)} reports={len(deleted_reports)} raw_archive={len(deleted_archive)}"
        )

    report = write_budget_report(Path(args.budget_report))
    print(f"Artifact budget report: {args.budget_report}")
    for k, v in report["sizes_gb"].items():
        print(f"  {k:<16} {v:.4f} GB")


if __name__ == "__main__":
    main()
