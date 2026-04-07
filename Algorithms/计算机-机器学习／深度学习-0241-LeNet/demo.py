"""LeNet MVP on sklearn digits (upsampled to 32x32)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LeNetConfig:
    seed: int = 42
    test_size: float = 0.2
    batch_size: int = 64
    epochs: int = 12
    lr: float = 1e-3
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LeNet(nn.Module):
    """Classic LeNet-5 style network for 32x32 grayscale input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.Tanh()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def load_digits_32x32() -> tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    # Original digits images are 8x8 in [0, 16]. Scale to [0, 1] then upsample.
    images = digits.images.astype(np.float32) / 16.0
    images_32 = ndimage.zoom(images, zoom=(1, 4, 4), order=1).astype(np.float32)
    labels = digits.target.astype(np.int64)
    return images_32, labels


def make_loaders(config: LeNetConfig) -> tuple[DataLoader, DataLoader]:
    images, labels = load_digits_32x32()
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_train).unsqueeze(1),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).unsqueeze(1),
        torch.from_numpy(y_test),
    )
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

        all_true.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return total_loss / total, correct / total, y_true, y_pred


def main() -> None:
    config = LeNetConfig()
    set_seed(config.seed)

    device = torch.device(config.device)
    train_loader, test_loader = make_loaders(config)

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[dict[str, float]] = []
    final_true = np.array([])
    final_pred = np.array([])

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        final_true, final_pred = y_true, y_pred
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    history_df = pd.DataFrame(history)
    print("\n[History tail]")
    print(history_df.tail(5).to_string(index=False))

    report = classification_report(final_true, final_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    key_rows = [str(i) for i in range(10)] + ["accuracy", "macro avg", "weighted avg"]
    print("\n[Classification report]")
    print(report_df.loc[key_rows, ["precision", "recall", "f1-score", "support"]].to_string())

    cm = confusion_matrix(final_true, final_pred)
    print("\n[Confusion matrix]")
    print(cm)

    final_acc = float(history_df["test_acc"].iloc[-1])
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    if final_acc < 0.90:
        raise RuntimeError("LeNet MVP accuracy below expected floor (0.90).")


if __name__ == "__main__":
    main()
