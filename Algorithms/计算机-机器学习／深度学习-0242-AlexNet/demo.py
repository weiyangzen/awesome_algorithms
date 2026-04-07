"""AlexNet minimal runnable MVP.

This script trains an AlexNet-style CNN on the offline sklearn digits dataset.
It preserves AlexNet's core architectural ideas:
- 5 convolution layers + 3 fully connected layers
- ReLU nonlinearity
- max pooling
- local response normalization (LRN)
- dropout in classifier

The goal is a compact, honest demo that runs quickly with `uv run python demo.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, zoom
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 64
    test_ratio: float = 0.25
    batch_size: int = 64
    epochs: int = 14
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    min_test_accuracy: float = 0.94


class DigitsDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class AlexNet(nn.Module):
    """AlexNet-style model scaled for 64x64 inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_digit_to_rgb(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize 8x8 grayscale digit to target_size and expand to RGB."""
    scale = float(target_size) / float(image.shape[0])
    resized = zoom(image, zoom=scale, order=1)
    smoothed = gaussian_filter(resized, sigma=0.35)
    smoothed = smoothed.astype(np.float32)

    low = float(smoothed.min())
    high = float(smoothed.max())
    normalized = (smoothed - low) / (high - low + 1e-6)
    rgb = np.stack([normalized, normalized, normalized], axis=0)
    return rgb


def prepare_dataset(config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    images = (digits.images / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)

    rgb_images = np.stack(
        [resize_digit_to_rgb(img, target_size=config.image_size) for img in images],
        axis=0,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        rgb_images,
        labels,
        test_size=config.test_ratio,
        random_state=config.seed,
        stratify=labels,
    )

    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test, y_train, y_test


def run_epoch(
    model: AlexNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.item()) * batch_size
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_true.append(batch_y.detach().cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    avg_loss = total_loss / float(len(loader.dataset))
    acc = float(accuracy_score(true, pred))
    return avg_loss, acc, pred, true


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
    cfg = TrainConfig()
    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_train, x_test, y_train, y_test = prepare_dataset(cfg)

    distribution_df = pd.DataFrame({"label": y_train})
    print("train label distribution:")
    print(distribution_df["label"].value_counts().sort_index().to_string())

    train_loader = DataLoader(
        DigitsDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        DigitsDataset(x_test, y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = AlexNet(num_classes=10).to(device)
    print(f"trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_epoch = -1
    best_test_acc = -1.0
    best_test_pred = np.array([], dtype=np.int64)
    best_test_true = np.array([], dtype=np.int64)

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc, test_pred, test_true = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        if test_acc > best_test_acc:
            best_epoch = epoch
            best_test_acc = test_acc
            best_test_pred = test_pred
            best_test_true = test_true

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    print(f"\nbest epoch: {best_epoch}, best_test_acc={best_test_acc:.4f}")
    print("\nclassification report (best epoch):")
    print(classification_report(best_test_true, best_test_pred, digits=4))

    cm = confusion_matrix(best_test_true, best_test_pred)
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(10)], columns=[f"pred_{i}" for i in range(10)])
    print("confusion matrix:")
    print(cm_df.to_string())

    rng = np.random.default_rng(cfg.seed)
    sample_ids = rng.choice(len(x_test), size=8, replace=False)
    sample_x = torch.tensor(x_test[sample_ids], dtype=torch.float32, device=device)
    with torch.no_grad():
        sample_pred = model(sample_x).argmax(dim=1).cpu().numpy()
    sample_df = pd.DataFrame(
        {
            "index": sample_ids,
            "true": y_test[sample_ids],
            "pred": sample_pred,
        }
    )
    print("sample predictions:")
    print(sample_df.to_string(index=False))

    if best_test_acc < cfg.min_test_accuracy:
        raise RuntimeError(
            f"best test accuracy {best_test_acc:.4f} below threshold {cfg.min_test_accuracy:.4f}; "
            "adjust epochs or learning rate."
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
