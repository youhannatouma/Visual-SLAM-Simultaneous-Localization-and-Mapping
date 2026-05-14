import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from reasoning import ACTION_CLASSES, FEATURE_SIZE, ReasoningMLP as ReasoningModel, SEQUENCE_LENGTH


HUGE_DATASET_THRESHOLD = 50000
REAL_SOURCE_TYPES = {"real_media", "manual_live"}


def normalize_optional_string(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def parse_class_weight_targets(text):
    targets = {}
    if not text:
        return targets
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Invalid --class-target-weights entry. "
                "Expected comma-separated LABEL:WEIGHT pairs."
            )
        label, raw_weight = chunk.split(":", 1)
        label = label.strip().upper()
        if label not in ACTION_CLASSES:
            raise ValueError(
                f"Invalid class in --class-target-weights: {label}. "
                f"Expected one of {ACTION_CLASSES}"
            )
        try:
            weight = float(raw_weight)
        except Exception as exc:
            raise ValueError(
                f"Invalid weight for --class-target-weights entry '{chunk}'"
            ) from exc
        if weight <= 0.0:
            raise ValueError(
                f"Class target weights must be > 0. Got {weight} for {label}"
            )
        targets[label] = weight
    return targets


def load_clean_frame(csv_path, return_clean_df=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training file not found: {csv_path}")

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

    if features_df.empty:
        raise ValueError(f"No valid rows after cleaning in {csv_path}")
    if len(feature_cols) != FEATURE_SIZE:
        raise ValueError(
            f"Expected {FEATURE_SIZE} canonical features in {csv_path}, "
            f"found {len(feature_cols)}"
        )

    result = (
        features_df.to_numpy(dtype=np.float32),
        labels_raw.map(ACTION_CLASSES.index).to_numpy(dtype=np.int64),
        feature_cols,
    )
    if return_clean_df:
        return result + (df.loc[valid_mask].copy(),)
    return result


class SequenceDataset(Dataset):
    def __init__(self, csv_path, sequence_length=SEQUENCE_LENGTH):
        self.features, self.labels, self.feature_cols, self.cleaned_df = load_clean_frame(
            csv_path, return_clean_df=True
        )
        self.sequence_length = sequence_length
        self.feature_size = self.features.shape[1]

        if len(self.features) < self.sequence_length:
            raise ValueError(
                f"Dataset {csv_path} has {len(self.features)} valid rows, "
                f"needs at least {self.sequence_length} for sequence training"
            )

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    def sample_weights(
        self,
        training_profile,
        real_reviewed_weight,
        hard_negative_weight,
        class_target_weights,
    ):
        weights = np.ones(len(self), dtype=np.float32)
        if len(weights) == 0:
            return weights
        if (
            training_profile == "comparable"
            and abs(real_reviewed_weight - 1.0) < 1e-12
            and abs(hard_negative_weight - 1.0) < 1e-12
            and not class_target_weights
        ):
            return weights

        for idx in range(len(weights)):
            meta = self.cleaned_df.iloc[idx + self.sequence_length - 1]
            label = ACTION_CLASSES[int(self.labels[idx + self.sequence_length - 1])]
            source_type = normalize_optional_string(meta.get("source_type")).lower()
            needs_review = normalize_optional_string(meta.get("needs_review")).lower()
            auto_label = normalize_optional_string(meta.get("auto_label")).upper()

            is_real = source_type in REAL_SOURCE_TYPES
            is_reviewed = needs_review not in {"1", "true", "yes", "pending"}
            is_hard_negative = bool(auto_label) and auto_label in ACTION_CLASSES and auto_label != label

            if is_real and is_reviewed:
                weights[idx] *= float(real_reviewed_weight)
                if label in class_target_weights:
                    weights[idx] *= float(class_target_weights[label])

            if is_hard_negative:
                weights[idx] *= float(hard_negative_weight)

        return weights
        
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


def evaluate_optional_dataset(model, csv_path, batch_size, device):
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        return None
    dataset = SequenceDataset(csv_path, sequence_length=SEQUENCE_LENGTH)
    acc, y_true, y_pred = evaluate_model(model, dataset, batch_size, device)
    confusion, precision, recall, f1, macro_f1 = compute_classification_metrics(
        y_true, y_pred, len(ACTION_CLASSES)
    )
    return {
        "path": csv_path,
        "rows": len(dataset),
        "accuracy": float(acc),
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loss(label_smoothing):
    if label_smoothing <= 0.0:
        return nn.CrossEntropyLoss()
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        return nn.CrossEntropyLoss()


def train(
    train_path,
    val_path,
    test_path,
    model_path,
    report_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    label_smoothing,
    fresh_real_eval_path,
    algorithm,
    seed,
    training_profile,
    real_reviewed_weight,
    hard_negative_weight,
    class_target_weights,
):
    if algorithm.lower() != "mlp":
        raise ValueError("Only MLP is supported for reasoning training in this project")

    set_seed(seed)

    train_dataset = SequenceDataset(train_path, sequence_length=SEQUENCE_LENGTH)
    val_dataset = SequenceDataset(val_path, sequence_length=SEQUENCE_LENGTH)
    test_dataset = SequenceDataset(test_path, sequence_length=SEQUENCE_LENGTH)

    total_rows = len(train_dataset) + len(val_dataset) + len(test_dataset)
    if total_rows >= HUGE_DATASET_THRESHOLD:
        raise RuntimeError(
            "Dataset has "
            f"{total_rows} rows, which meets/exceeds local threshold ({HUGE_DATASET_THRESHOLD}). "
            "Please train remotely (e.g., Colab/server) and copy the resulting model back to models/."
        )

    if train_dataset.feature_size != val_dataset.feature_size or train_dataset.feature_size != test_dataset.feature_size:
        raise ValueError("Feature-size mismatch across train/val/test datasets")

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_sample_weights = train_dataset.sample_weights(
        training_profile=training_profile,
        real_reviewed_weight=real_reviewed_weight,
        hard_negative_weight=hard_negative_weight,
        class_target_weights=class_target_weights,
    )
    use_weighted_sampler = not np.allclose(train_sample_weights, 1.0)
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    model = ReasoningModel(train_dataset.feature_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = build_loss(label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "sequence_length": int(SEQUENCE_LENGTH),
        "feature_size": int(train_dataset.feature_size),
        "action_classes": list(ACTION_CLASSES),
        "model_algorithm": "mlp",
        "seed": int(seed),
    }
    torch.save(checkpoint, model_path)
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
        "model_algorithm": "mlp",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": train_dataset.feature_size,
        "feature_columns": train_dataset.feature_cols,
        "seed": int(seed),
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "label_smoothing": float(label_smoothing),
            "profile": training_profile,
            "real_reviewed_weight": float(real_reviewed_weight),
            "hard_negative_weight": float(hard_negative_weight),
            "class_target_weights": {k: float(v) for k, v in class_target_weights.items()},
            "weighted_sampler_enabled": bool(use_weighted_sampler),
        },
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
    fresh_real_metrics = evaluate_optional_dataset(model, fresh_real_eval_path, batch_size, device)
    if fresh_real_metrics is not None:
        metrics["fresh_real_eval"] = fresh_real_metrics
        print(
            "Fresh real eval: "
            f"accuracy={fresh_real_metrics['accuracy']:.3f} "
            f"macro_f1={fresh_real_metrics['macro_f1']:.3f} "
            f"rows={fresh_real_metrics['rows']}"
        )

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
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing")
    parser.add_argument("--fresh-real-eval", default="", help="Optional held-out fresh real eval CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic training")
    parser.add_argument("--algorithm", default="mlp", choices=["mlp"], help="Training algorithm (locked to MLP)")
    parser.add_argument(
        "--training-profile",
        default="comparable",
        choices=["comparable", "real_recovery"],
        help="Sampling profile for training (default: comparable)",
    )
    parser.add_argument(
        "--real-reviewed-weight",
        type=float,
        default=1.0,
        help="Multiplier for reviewed real rows during training sampling.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help="Multiplier for auto-label disagreement hard negatives during training sampling.",
    )
    parser.add_argument(
        "--class-target-weights",
        default="",
        help="Optional comma-separated LABEL:WEIGHT pairs applied to reviewed real rows.",
    )
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
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        fresh_real_eval_path=args.fresh_real_eval,
        algorithm=args.algorithm,
        seed=args.seed,
        training_profile=args.training_profile,
        real_reviewed_weight=args.real_reviewed_weight,
        hard_negative_weight=args.hard_negative_weight,
        class_target_weights=parse_class_weight_targets(args.class_target_weights),
    )
