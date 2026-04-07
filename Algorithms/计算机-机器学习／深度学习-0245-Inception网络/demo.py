"""Minimal runnable MVP for Inception network (CS-0115).

This demo builds a tiny, source-transparent Inception-style CNN for
classification on sklearn digits (8x8 grayscale images).

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class InceptionConfig:
    seed: int = 42
    test_size: float = 0.25
    batch_size: int = 128
    epochs: int = 24
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gaussian_sigma: float = 0.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessed_digits(config: InceptionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sklearn digits, smooth lightly, normalize, then split."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0

    if config.gaussian_sigma > 0:
        images = np.stack(
            [gaussian_filter(img, sigma=config.gaussian_sigma) for img in images],
            axis=0,
        ).astype(np.float32)

    images = np.clip(images, 0.0, 1.0)
    labels = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x).unsqueeze(1).to(torch.float32)
    y_tensor = torch.from_numpy(y).to(torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class InceptionBlock(nn.Module):
    """Classic Inception-style 4-branch block with channel concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()

        def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int, padding: int = 0) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.branch1 = conv_bn_relu(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_bn_relu(in_channels, red_3x3, kernel_size=1),
            conv_bn_relu(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_bn_relu(in_channels, red_5x5, kernel_size=1),
            conv_bn_relu(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class TinyInceptionNet(nn.Module):
    """A small Inception-like network for 8x8 digit classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.inception1 = InceptionBlock(
            in_channels=32,
            out_1x1=16,
            red_3x3=16,
            out_3x3=24,
            red_5x5=8,
            out_5x5=8,
            pool_proj=16,
        )

        self.inception2 = InceptionBlock(
            in_channels=64,
            out_1x1=24,
            red_3x3=24,
            out_3x3=32,
            red_5x5=8,
            out_5x5=8,
            pool_proj=24,
        )

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception3 = InceptionBlock(
            in_channels=88,
            out_1x1=24,
            red_3x3=24,
            out_3x3=32,
            red_5x5=12,
            out_5x5=16,
            pool_proj=24,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.downsample(x)
        x = self.inception3(x)
        return self.classifier(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = xb.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == yb).sum().item()

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    preds: List[int] = []
    trues: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        batch_size = xb.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(yb.cpu().numpy().tolist())

    return total_loss / total_samples, total_correct / total_samples, preds, trues


def main() -> None:
    config = InceptionConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x_train, x_test, y_train, y_test = load_preprocessed_digits(config)
    train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    test_loader = make_dataloader(x_test, y_test, batch_size=config.batch_size, shuffle=False)

    model = TinyInceptionNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    init_train_loss, init_train_acc, _, _ = evaluate(model, train_loader, device)
    init_test_loss, init_test_acc, _, _ = evaluate(model, test_loader, device)
    print(
        "Epoch 00 | "
        f"train_loss={init_train_loss:.4f} train_acc={init_train_acc:.4f} | "
        f"test_loss={init_test_loss:.4f} test_acc={init_test_acc:.4f}"
    )

    history = [
        {
            "epoch": 0,
            "train_loss": init_train_loss,
            "train_acc": init_train_acc,
            "test_loss": init_test_loss,
            "test_acc": init_test_acc,
        }
    ]

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    history_df = pd.DataFrame(history)
    final_train_loss = float(history_df.iloc[-1]["train_loss"])
    best_test_acc = float(history_df["test_acc"].max())

    final_test_loss, final_test_acc, final_preds, final_trues = evaluate(model, test_loader, device)
    cm = confusion_matrix(final_trues, final_preds)
    report = classification_report(final_trues, final_preds, digits=4)

    print("\nTraining log tail:")
    print(history_df.tail(5).to_string(index=False))

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report)

    # Lightweight sanity checks for CI-style validation.
    assert final_train_loss < init_train_loss, "Training loss did not decrease."
    assert best_test_acc >= 0.92, f"Best test accuracy too low: {best_test_acc:.4f}"
    assert np.isfinite(final_test_loss), "Final test loss is non-finite."

    for p in model.parameters():
        assert torch.isfinite(p).all(), "Model parameters contain NaN/Inf."

    print(
        "All checks passed. "
        f"Final test_acc={final_test_acc:.4f}, best_test_acc={best_test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
