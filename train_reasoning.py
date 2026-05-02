import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from reasoning import ACTION_CLASSES, ReasoningModel

HUGE_DATASET_THRESHOLD = 50000


class ReasoningDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Dataset is empty: {csv_path}")
        if "label" not in df.columns:
            raise ValueError(f"Dataset missing 'label' column: {csv_path}")

        feature_cols = [c for c in df.columns if c.startswith("f")]
        if not feature_cols:
            raise ValueError(f"No feature columns f0..fN found in {csv_path}")
        feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]) if c[1:].isdigit() else c)

        features_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        labels_raw = df["label"].astype(str)

        valid_mask = features_df.notna().all(axis=1) & labels_raw.isin(ACTION_CLASSES)
        features_df = features_df.loc[valid_mask]
        labels_raw = labels_raw.loc[valid_mask]

        if features_df.empty:
            raise ValueError(f"No valid rows after cleaning in {csv_path}")

        self.features = features_df.to_numpy(dtype=np.float32)
        self.labels = labels_raw.map(ACTION_CLASSES.index).to_numpy(dtype=np.int64)
        self.feature_cols = feature_cols
        self.feature_size = self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate_model(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_true = []
    all_pred = []
    total_correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    accuracy = (total_correct / total) if total else 0.0
    return accuracy, np.array(all_true, dtype=np.int64), np.array(all_pred, dtype=np.int64)


def compute_classification_metrics(y_true, y_pred, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(score))

    macro_f1 = float(np.mean(f1)) if f1 else 0.0
    return confusion, precision, recall, f1, macro_f1


def save_confusion_matrix(confusion, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(ACTION_CLASSES)))
    ax.set_yticks(np.arange(len(ACTION_CLASSES)))
    ax.set_xticklabels(ACTION_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(ACTION_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(train_path, val_path, test_path, model_path, report_dir, epochs, batch_size, lr):
    train_dataset = ReasoningDataset(train_path)
    val_dataset = ReasoningDataset(val_path)
    test_dataset = ReasoningDataset(test_path)

    total_rows = len(train_dataset) + len(val_dataset) + len(test_dataset)
    if total_rows >= HUGE_DATASET_THRESHOLD:
        raise RuntimeError(
            "Dataset has "
            f"{total_rows} rows, which meets/exceeds local threshold ({HUGE_DATASET_THRESHOLD}). "
            "Please train remotely (e.g., Colab/server) and copy the resulting model back to models/."
        )

    if train_dataset.feature_size != val_dataset.feature_size or train_dataset.feature_size != test_dataset.feature_size:
        raise ValueError("Feature-size mismatch across train/val/test datasets")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    model = ReasoningModel(train_dataset.feature_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0
        val_acc, _, _ = evaluate_model(model, val_dataset, batch_size, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch}/{epochs} "
            f"loss={epoch_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    print(f"Saved best model to {model_path}")

    test_acc, y_true, y_pred = evaluate_model(model, test_dataset, batch_size, device)
    confusion, precision, recall, f1, macro_f1 = compute_classification_metrics(
        y_true, y_pred, len(ACTION_CLASSES)
    )

    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    for i, label in enumerate(ACTION_CLASSES):
        print(
            f"  {label:<15} precision={precision[i]:.3f} "
            f"recall={recall[i]:.3f} f1={f1[i]:.3f}"
        )

    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    save_confusion_matrix(confusion, cm_path)

    metrics = {
        "threshold_local_rows": HUGE_DATASET_THRESHOLD,
        "dataset_rows": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
            "total": total_rows,
        },
        "accuracy": {
            "best_val": float(best_val_acc),
            "test": float(test_acc),
        },
        "macro_f1": float(macro_f1),
        "per_class": {
            ACTION_CLASSES[i]: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
            }
            for i in range(len(ACTION_CLASSES))
        },
        "confusion_matrix": confusion.tolist(),
    }

    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {cm_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train reasoning model with train/val/test splits")
    parser.add_argument("--train", default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--val", default="data/processed/val.csv", help="Path to validation CSV")
    parser.add_argument("--test", default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--model", default="models/reasoning_model.pt", help="Where to save trained model")
    parser.add_argument("--report-dir", default="reports", help="Where to save metrics and confusion matrix")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_path=args.model,
        report_dir=args.report_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
