"""Minimal runnable MVP for EfficientNet on sklearn digits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EfficientNetConfig:
    """Configuration for a tiny EfficientNet-like model."""

    width_mult: float = 1.0
    depth_mult: float = 1.0
    dropout: float = 0.2
    num_classes: int = 10


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def round_filters(filters: int, width_mult: float, divisor: int = 8) -> int:
    if width_mult == 1.0:
        return filters
    scaled = filters * width_mult
    min_depth = divisor
    new_filters = max(min_depth, int(scaled + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * scaled:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_mult: float) -> int:
    return max(1, int(np.ceil(repeats * depth_mult)))


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.reduce(scale)
        scale = self.act(scale)
        scale = self.expand(scale)
        scale = self.gate(scale)
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"Unsupported stride={stride}; expected 1 or 2.")

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, stride=1))
        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
            )
        )
        layers.append(SqueezeExcitation(hidden_dim, se_ratio=0.25))
        layers.append(
            ConvBNAct(
                hidden_dim,
                out_channels,
                kernel_size=1,
                stride=1,
                activation=False,
            )
        )

        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out


class TinyEfficientNet(nn.Module):
    """A compact EfficientNet-style network for 8x8 grayscale images."""

    def __init__(self, cfg: EfficientNetConfig) -> None:
        super().__init__()

        # (expand_ratio, channels, repeats, stride, kernel_size)
        base_blocks = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 1, 5),
            (6, 64, 3, 2, 3),
        ]

        stem_channels = round_filters(16, cfg.width_mult)
        self.stem = ConvBNAct(1, stem_channels, kernel_size=3, stride=1)

        layers: list[nn.Module] = []
        in_channels = stem_channels
        for expand_ratio, out_channels_base, repeats_base, stride, kernel_size in base_blocks:
            out_channels = round_filters(out_channels_base, cfg.width_mult)
            repeats = round_repeats(repeats_base, cfg.depth_mult)
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                layers.append(
                    MBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        drop_rate=0.05,
                    )
                )
                in_channels = out_channels

        self.blocks = nn.Sequential(*layers)
        head_channels = round_filters(128, cfg.width_mult)
        self.head = ConvBNAct(in_channels, head_channels, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_channels, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_dataset(test_size: float = 0.25, random_state: int = 42):
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0,1), got {test_size}")

    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    x = np.expand_dims(x, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return x_train, x_test, y_train, y_test


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_samples += int(yb.size(0))

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> tuple[int, float]:
    majority = int(np.bincount(y_train).argmax())
    acc = float((y_test == majority).mean())
    return majority, acc


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_samples(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    max_samples: int = 8,
) -> list[tuple[int, int, float]]:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_test[:max_samples], dtype=torch.float32, device=device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    result: list[tuple[int, int, float]] = []
    for i in range(xb.size(0)):
        result.append((int(y_test[i]), int(pred[i].item()), float(conf[i].item())))
    return result


def main() -> None:
    set_global_seed(42)
    torch.set_num_threads(1)

    x_train, x_test, y_train, y_test = load_dataset()
    train_loader = make_loader(x_train, y_train, batch_size=64, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size=256, shuffle=False)

    cfg = EfficientNetConfig(width_mult=1.0, depth_mult=1.0, dropout=0.2, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyEfficientNet(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    print(f"device: {device}")
    print(f"train shape: x={x_train.shape}, y={y_train.shape}")
    print(f"test  shape: x={x_test.shape}, y={y_test.shape}")
    print(f"model: TinyEfficientNet, params={count_parameters(model):,}")

    epochs = 18
    final_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
        final_test_acc = test_acc
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    majority_label, baseline_acc = majority_class_baseline(y_train, y_test)
    print(f"majority baseline: class={majority_label}, acc={baseline_acc:.4f}")

    samples = predict_samples(model, x_test, y_test, device, max_samples=8)
    print("sample predictions (true, pred, confidence):")
    for i, (true_y, pred_y, conf) in enumerate(samples):
        print(f"  [{i}] true={true_y}, pred={pred_y}, conf={conf:.4f}")

    if not np.isfinite(final_test_acc):
        raise RuntimeError("test accuracy is not finite")
    if final_test_acc < 0.90:
        raise RuntimeError(f"test accuracy too low: {final_test_acc:.4f} < 0.90")
    if final_test_acc < baseline_acc + 0.60:
        raise RuntimeError(
            f"model did not beat baseline enough: {final_test_acc:.4f} vs {baseline_acc:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
