"""VGGNet minimal runnable MVP on sklearn digits.

This script implements a tiny VGG-style CNN with explicit Conv stacks:
small 3x3 kernels, repeated blocks, and max-pooling between blocks.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VGGBlock(nn.Module):
    """VGG-style block: (Conv-BN-ReLU) * N -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, conv_count: int) -> None:
        super().__init__()
        if conv_count <= 0:
            raise ValueError(f"conv_count must be positive, got {conv_count}")

        layers = []
        current_in = in_channels
        for _ in range(conv_count):
            layers.extend(
                [
                    nn.Conv2d(
                        current_in,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_in = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyVGGNet(nn.Module):
    """A compact VGG-like classifier for 8x8 grayscale digit images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(1, 16, conv_count=2),   # 8x8 -> 4x4
            VGGBlock(16, 32, conv_count=2),  # 4x4 -> 2x2
            VGGBlock(32, 64, conv_count=2),  # 2x2 -> 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    test_ratio: float = 0.25,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def build_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == yb).sum().item())
            total_count += yb.numel()
            total_loss += float(loss.item() * yb.numel())

    mean_loss = total_loss / max(total_count, 1)
    mean_acc = total_correct / max(total_count, 1)
    return mean_loss, mean_acc


def majority_class_baseline(y_train: torch.Tensor, y_test: torch.Tensor) -> Tuple[int, float]:
    majority_class = int(torch.mode(y_train).values.item())
    baseline_acc = float((y_test == majority_class).float().mean().item())
    return majority_class, baseline_acc


def main() -> None:
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = load_dataset(test_ratio=0.25, random_state=42)
    train_loader, test_loader = build_dataloaders(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=64,
    )

    model = TinyVGGNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 14
    print(f"device: {device}")
    print(f"train shape: X={tuple(x_train.shape)}, y={tuple(y_train.shape)}")
    print(f"test  shape: X={tuple(x_test.shape)}, y={tuple(y_test.shape)}")
    print(f"model: TinyVGGNet (params={sum(p.numel() for p in model.parameters())})")
    print(f"optimizer: Adam(lr=1e-3, weight_decay=1e-4), epochs={epochs}")

    final_train_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        test_loss, test_acc = run_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        final_train_acc = train_acc
        final_test_acc = test_acc

        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

    majority_class, baseline_acc = majority_class_baseline(y_train, y_test)
    print(f"majority baseline: class={majority_class}, acc={baseline_acc:.4f}")

    model.eval()
    with torch.no_grad():
        sample_logits = model(x_test[:8].to(device))
        sample_prob = torch.softmax(sample_logits, dim=1).cpu().numpy()
        sample_pred = np.argmax(sample_prob, axis=1)

    print("sample predictions (first 8 test samples):")
    for i in range(8):
        top_prob = float(np.max(sample_prob[i]))
        print(
            f"  idx={i:02d} true={int(y_test[i].item())} "
            f"pred={int(sample_pred[i])} top_prob={top_prob:.3f}"
        )

    print(f"final train_acc={final_train_acc:.4f}, final test_acc={final_test_acc:.4f}")

    if not np.isfinite(final_test_acc):
        raise RuntimeError("test accuracy is not finite")
    if final_test_acc < 0.93:
        raise RuntimeError(f"test accuracy too low: {final_test_acc:.4f} < 0.93")
    if final_test_acc < baseline_acc + 0.50:
        raise RuntimeError(
            "model did not beat majority baseline by >= 0.50: "
            f"test={final_test_acc:.4f}, baseline={baseline_acc:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
