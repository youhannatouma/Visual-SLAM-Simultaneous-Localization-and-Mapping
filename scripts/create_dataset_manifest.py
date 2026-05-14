import argparse
import fnmatch
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]
EXCLUDED_RAW_FILE_PATTERNS = (
    "zz_fresh_real_holdout_*.csv",
    "media_labeled_stage2_train_refix_*.csv",
    "media_labeled_stage2_move_hardneg_*.csv",
    "move_recovery_pool_*.csv",
    "vid*.csv",
)


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Create reproducible dataset manifest and changelog entry")
    parser.add_argument("--input-glob", default="data/raw/*.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest-path", default="data/manifest/dataset_manifest.json")
    parser.add_argument("--changelog-path", default="data/manifest/CHANGELOG.md")
    parser.add_argument("--dataset-version", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = int(time.time())
    version = args.dataset_version or time.strftime("v%Y%m%d_%H%M%S", time.localtime(timestamp))

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")
    files = [
        f for f in files
        if not any(fnmatch.fnmatch(Path(f).name, pat) for pat in EXCLUDED_RAW_FILE_PATTERNS)
    ]
    if not files:
        raise FileNotFoundError(
            f"All files matched by {args.input_glob} were excluded by raw-file policy"
        )

    frames = []
    file_entries = []
    for f in files:
        p = Path(f)
        df = pd.read_csv(p)
        df["__source_file"] = p.name
        frames.append(df)
        file_entries.append(
            {
                "file": str(p),
                "rows": int(len(df)),
                "sha256": file_sha256(p),
            }
        )

    merged = pd.concat(frames, ignore_index=True)
    source_type_dist = {}
    if "source_type" in merged.columns:
        source_type_dist = {str(k): int(v) for k, v in merged["source_type"].value_counts().to_dict().items()}
    class_dist = {}
    if "label" in merged.columns:
        vc = merged["label"].value_counts().to_dict()
        class_dist = {label: int(vc.get(label, 0)) for label in ACTION_CLASSES}

    processed_dir = Path(args.processed_dir)
    processed_files = []
    for name in ["train.csv", "val.csv", "test.csv", "fresh_real_eval.csv", "metadata.json"]:
        p = processed_dir / name
        if p.exists():
            processed_files.append({"file": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size})

    combined_hash = hashlib.sha256(
        "\n".join([entry["sha256"] for entry in file_entries] + [entry["sha256"] for entry in processed_files]).encode("utf-8")
    ).hexdigest()

    manifest = {
        "dataset_version": version,
        "timestamp": timestamp,
        "input_glob": args.input_glob,
        "raw_files": file_entries,
        "raw_total_rows": int(sum(entry["rows"] for entry in file_entries)),
        "source_type_distribution": source_type_dist,
        "class_distribution": class_dist,
        "processed_files": processed_files,
        "fingerprint_sha256": combined_hash,
        "policy": {
            "default_training_source": "data/raw",
            "raw_archive_excluded_by_default": True,
        },
    }

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    changelog_path = Path(args.changelog_path)
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    if not changelog_path.exists():
        changelog_path.write_text("# Dataset Changelog\n\n", encoding="utf-8")
    with changelog_path.open("a", encoding="utf-8") as f:
        f.write(
            f"## {version}\n"
            f"- timestamp: {timestamp}\n"
            f"- raw files: {len(file_entries)}\n"
            f"- raw rows: {manifest['raw_total_rows']}\n"
            f"- fingerprint: {combined_hash}\n"
            f"- class distribution: {class_dist}\n"
            f"- source distribution: {source_type_dist}\n\n"
        )

    print(f"Manifest written: {manifest_path}")
    print(f"Changelog updated: {changelog_path}")
    print(f"Dataset version: {version}")
    print(f"Fingerprint: {combined_hash}")


if __name__ == "__main__":
    main()
