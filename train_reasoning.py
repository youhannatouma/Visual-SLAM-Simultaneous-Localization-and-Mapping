import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from reasoning import ACTION_CLASSES, ReasoningModel


class ReasoningDataset(Dataset):
    def __init__(self, csv_path):
        self.features = []
        self.labels = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training file not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("Training file is empty")

            for row in reader:
                if len(row) < 2:
                    continue
                *feature_values, label = row
                self.features.append([float(value) for value in feature_values])
                self.labels.append(ACTION_CLASSES.index(label))

        if len(self.features) == 0:
            raise ValueError("No training samples found in the dataset")

        self.feature_size = len(self.features[0])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def train(data_path, model_path, epochs, batch_size, lr):
    dataset = ReasoningDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = ReasoningModel(dataset.feature_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
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

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  train_acc={epoch_acc:.3f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

    # optional final evaluation on the full dataset
    model.eval()
    with torch.no_grad():
        features = torch.tensor(dataset.features, dtype=torch.float32, device=device)
        labels = torch.tensor(dataset.labels, dtype=torch.long, device=device)
        logits = model(features)
        correct = (logits.argmax(dim=1) == labels).sum().item()
        print(f"Final accuracy on full dataset: {correct / len(dataset):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the reasoning model from logged training data")
    parser.add_argument("--data", default="data/reasoning_data.csv", help="Path to the labeled training CSV")
    parser.add_argument("--model", default="models/reasoning_model.pt", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.batch_size, args.lr)
