"""DenseNet MVP on sklearn digits using a tiny PyTorch implementation.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 3e-3
    epochs: int = 10
    test_size: float = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DenseLayer(nn.Module):
    """One DenseNet layer: BN -> ReLU -> 3x3 Conv, then concat with input."""

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.transform(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    """Stack multiple DenseLayer modules."""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Compress channels and downsample feature maps between dense blocks."""

    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        out_channels = max(4, int(in_channels * compression))
        self.out_channels = out_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class TinyDenseNet(nn.Module):
    """A compact DenseNet classifier for 8x8 grayscale images."""

    def __init__(
        self,
        num_classes: int = 10,
        growth_rate: int = 12,
        block_config: tuple[int, int, int] = (3, 3, 3),
        init_channels: int = 16,
        compression: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, init_channels, kernel_size=3, padding=1, bias=False)

        channels = init_channels
        feature_layers: list[nn.Module] = []

        for block_index, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=channels, growth_rate=growth_rate)
            feature_layers.append(block)
            channels = block.out_channels

            if block_index != len(block_config) - 1:
                transition = TransitionLayer(in_channels=channels, compression=compression)
                feature_layers.append(transition)
                channels = transition.out_channels

        self.features = nn.Sequential(*feature_layers)
        self.head = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    digits = load_digits()
    x = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_count += batch_x.size(0)

    mean_loss = total_loss / total_count
    mean_acc = total_correct / total_count
    return mean_loss, mean_acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return float(accuracy_score(y_true, y_pred))


@torch.no_grad()
def collect_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_examples: int = 10,
) -> list[tuple[int, int]]:
    model.eval()
    examples: list[tuple[int, int]] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = batch_y.numpy()

        for pred, target in zip(preds, targets):
            examples.append((int(pred), int(target)))
            if len(examples) >= num_examples:
                return examples
    return examples


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloaders(config)

    model = TinyDenseNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Device: {device}")
    print("Start training TinyDenseNet on sklearn digits...")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if epoch == 1 or epoch % 2 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:02d}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
            )

    test_acc = evaluate(model=model, loader=test_loader, device=device)
    examples = collect_examples(model=model, loader=test_loader, device=device)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Sample predictions (pred, true): {examples}")


if __name__ == "__main__":
    main()
