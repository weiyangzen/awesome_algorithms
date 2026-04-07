"""Minimal runnable MVP for 卷积神经网络 (CNN).

This script trains a tiny CNN on the sklearn digits dataset.
It is intentionally small and transparent: preprocessing, model,
training loop, and evaluation are all explicit in source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class CNNConfig:
    test_size: float = 0.25
    batch_size: int = 64
    epochs: int = 28
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gaussian_sigma: float = 0.25
    min_accuracy: float = 0.93
    seed: int = 42


class TinyCNN(nn.Module):
    """A compact CNN suitable for 8x8 grayscale digit images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessed_digits(config: CNNConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    # Apply a light Gaussian smoothing filter to demonstrate explicit scipy usage.
    smoothed = np.stack(
        [gaussian_filter(img, sigma=config.gaussian_sigma) for img in images],
        axis=0,
    ).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    x_train, x_test, y_train, y_test = train_test_split(
        smoothed,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x).unsqueeze(1)  # [N, 1, 8, 8]
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = yb.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    preds: List[np.ndarray] = []
    truths: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = yb.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            truths.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(truths)
    acc = float((y_pred == y_true).mean())
    avg_loss = running_loss / max(sample_count, 1)
    return avg_loss, acc, y_true, y_pred


def params_are_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def main() -> None:
    config = CNNConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test = load_preprocessed_digits(config)
    train_loader = make_dataloader(x_train, y_train, config.batch_size, shuffle=True)
    train_eval_loader = make_dataloader(x_train, y_train, config.batch_size, shuffle=False)
    test_loader = make_dataloader(x_test, y_test, config.batch_size, shuffle=False)

    model = TinyCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, float]] = []

    init_train_loss, init_train_acc, _, _ = evaluate(model, train_eval_loader, criterion, device)
    init_test_loss, init_test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    history.append(
        {
            "epoch": 0,
            "train_loss": init_train_loss,
            "train_acc": init_train_acc,
            "test_loss": init_test_loss,
            "test_acc": init_test_acc,
        }
    )

    print(
        f"[epoch 00] train_loss={init_train_loss:.4f} train_acc={init_train_acc:.4f} "
        f"test_loss={init_test_loss:.4f} test_acc={init_test_acc:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss, train_acc, _, _ = evaluate(model, train_eval_loader, criterion, device)
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

        if epoch == 1 or epoch % 3 == 0 or epoch == config.epochs:
            print(
                f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    log_df = pd.DataFrame(history)
    final = log_df.iloc[-1]

    # Validation checks for the MVP.
    assert float(final["train_loss"]) < float(log_df.iloc[0]["train_loss"]), "Train loss did not decrease."
    assert float(final["test_acc"]) >= config.min_accuracy, (
        f"Test accuracy {float(final['test_acc']):.4f} < required {config.min_accuracy:.4f}"
    )
    assert params_are_finite(model), "Model parameters contain NaN/Inf."

    print("\nTraining log tail:")
    print(log_df.tail(6).to_string(index=False))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    print(
        "Final metrics: "
        f"train_loss={float(final['train_loss']):.4f}, "
        f"train_acc={float(final['train_acc']):.4f}, "
        f"test_loss={float(final['test_loss']):.4f}, "
        f"test_acc={float(final['test_acc']):.4f}"
    )
    print("All checks passed.")


if __name__ == "__main__":
    main()
