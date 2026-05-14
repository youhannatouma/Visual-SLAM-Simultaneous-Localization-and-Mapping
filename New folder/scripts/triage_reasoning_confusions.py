#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning import ACTION_CLASSES, FEATURE_SIZE, ReasoningMLP, SEQUENCE_LENGTH, load_reasoning_checkpoint


def load_clean_frame_with_df(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    if "label" not in df.columns:
        raise ValueError(f"Dataset missing 'label' column: {csv_path}")

    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not feature_cols:
        raise ValueError(f"No feature columns f0..fN found in {csv_path}")
    feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]))

    features_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    labels_raw = df["label"].astype(str)

    valid_mask = (
        features_df.notna().all(axis=1)
        & np.isfinite(features_df.to_numpy(dtype=np.float32)).all(axis=1)
        & labels_raw.isin(ACTION_CLASSES)
    )

    features_df = features_df.loc[valid_mask]
    labels_raw = labels_raw.loc[valid_mask]
    cleaned_df = df.loc[valid_mask].copy()

    if features_df.empty:
        raise ValueError(f"No valid rows after cleaning in {csv_path}")
    if len(feature_cols) != FEATURE_SIZE:
        raise ValueError(
            f"Expected {FEATURE_SIZE} canonical features in {csv_path}, "
            f"found {len(feature_cols)}"
        )

    return (
        features_df.to_numpy(dtype=np.float32),
        labels_raw.map(ACTION_CLASSES.index).to_numpy(dtype=np.int64),
        feature_cols,
        cleaned_df,
    )


def build_sequences(features, labels, cleaned_df, sequence_length):
    if len(features) < sequence_length:
        raise ValueError(
            f"Dataset has {len(features)} rows, needs at least {sequence_length} "
            "for sequence evaluation"
        )

    sequences = []
    label_indices = []
    meta_rows = []

    for idx in range(len(features) - sequence_length + 1):
        sequences.append(features[idx:idx + sequence_length])
        label_indices.append(labels[idx + sequence_length - 1])
        meta_rows.append(cleaned_df.iloc[idx + sequence_length - 1])

    return np.stack(sequences, axis=0), np.array(label_indices, dtype=np.int64), meta_rows


def predict_batches(model, sequences, batch_size, device):
    preds = []
    pred_probs = []
    true_probs = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            batch_preds = probs.argmax(dim=1).cpu().numpy()
            batch_pred_probs = probs.max(dim=1).values.cpu().numpy()

            preds.extend(batch_preds.tolist())
            pred_probs.extend(batch_pred_probs.tolist())
            true_probs.extend(probs.cpu().numpy().tolist())

    return (
        np.array(preds, dtype=np.int64),
        np.array(pred_probs, dtype=np.float32),
        np.array(true_probs, dtype=np.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export misclassified CHECK_TABLE vs MOVE_TO_CHAIR rows for review"
    )
    parser.add_argument("--data", required=True, help="Input CSV (processed split or holdout)")
    parser.add_argument("--model", default="models/reasoning_model.pt", help="Model path")
    parser.add_argument("--output", required=True, help="Output CSV for triage rows")
    parser.add_argument("--true-label", default="CHECK_TABLE", help="True label to match")
    parser.add_argument("--pred-label", default="MOVE_TO_CHAIR", help="Predicted label to match")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Sequence length for evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Include both true->pred and pred->true confusions",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on output rows (0 = no cap)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    features, labels, feature_cols, cleaned_df = load_clean_frame_with_df(args.data)
    sequences, true_labels, meta_rows = build_sequences(
        features, labels, cleaned_df, args.sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReasoningMLP(features.shape[1], sequence_length=args.sequence_length).to(device)
    state_dict, _ = load_reasoning_checkpoint(args.model, device)
    model.load_state_dict(state_dict)

    preds, pred_probs, all_probs = predict_batches(
        model, sequences, args.batch_size, device
    )

    true_label = args.true_label
    pred_label = args.pred_label

    rows = []
    for idx, (true_idx, pred_idx) in enumerate(zip(true_labels, preds)):
        true_name = ACTION_CLASSES[true_idx]
        pred_name = ACTION_CLASSES[pred_idx]

        match = (true_name == true_label and pred_name == pred_label)
        if args.bidirectional:
            match = match or (true_name == pred_label and pred_name == true_label)

        if not match:
            continue

        meta = meta_rows[idx]
        row = meta.to_dict()
        row.update(
            {
                "true_label": true_name,
                "pred_label": pred_name,
                "pred_prob": float(pred_probs[idx]),
                "true_prob": float(all_probs[idx][true_idx]),
                "sequence_index": int(idx),
                "row_index": int(meta.name),
            }
        )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No matching confusions found.")
    else:
        if args.max_rows and len(out_df) > args.max_rows:
            out_df = out_df.sample(n=args.max_rows, random_state=42)

        ordered_cols = [
            "true_label",
            "pred_label",
            "pred_prob",
            "true_prob",
            "sequence_index",
            "row_index",
        ] + [c for c in cleaned_df.columns if c not in feature_cols] + feature_cols

        ordered_cols = [c for c in ordered_cols if c in out_df.columns]
        out_df = out_df[ordered_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
