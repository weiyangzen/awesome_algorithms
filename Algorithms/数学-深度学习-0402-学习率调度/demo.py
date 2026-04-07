"""Minimal runnable MVP for learning-rate scheduling (MATH-0402).

This script compares three explicit LR schedules on the same tiny MLP task:
- constant learning rate,
- step decay,
- warmup + cosine decay.

The schedule formulas are implemented directly (no torch scheduler black box).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn


@dataclass
class ConstantLRScheduler:
    """Keep learning rate fixed across epochs."""

    base_lr: float

    def lr_at(self, epoch: int) -> float:
        del epoch
        return float(self.base_lr)


@dataclass
class StepLRScheduler:
    """Drop LR by gamma at each milestone epoch."""

    base_lr: float
    milestones: Sequence[int]
    gamma: float = 0.2

    def lr_at(self, epoch: int) -> float:
        drops = sum(1 for m in self.milestones if epoch >= m)
        return float(self.base_lr * (self.gamma**drops))


@dataclass
class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    base_lr: float
    total_epochs: int
    warmup_epochs: int = 10
    min_lr_ratio: float = 0.08

    def lr_at(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return float(self.base_lr * (epoch + 1) / max(1, self.warmup_epochs))

        min_lr = self.base_lr * self.min_lr_ratio
        decay_span = max(1, self.total_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / decay_span
        progress = float(min(max(progress, 0.0), 1.0))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr + (self.base_lr - min_lr) * cosine_factor)


@dataclass
class TrainResult:
    """Container for one scheduler run."""

    name: str
    lr_history: List[float]
    loss_history: List[float]
    train_acc: float
    test_acc: float


class TinyMLP(nn.Module):
    """Small MLP for moons classification."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set NumPy + Torch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_moons_dataset(
    n_samples: int = 1200,
    noise: float = 0.22,
    test_ratio: float = 0.25,
    seed: int = 402,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate and standardize a moons dataset."""
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)

    n_train = int(n_samples * (1.0 - test_ratio))
    tr, te = idx[:n_train], idx[n_train:]

    x_train = x[tr].astype(np.float32)
    x_test = x[te].astype(np.float32)
    y_train = y[tr].astype(np.float32)
    y_test = y[te].astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute binary classification accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
        return float((pred == y).to(torch.float32).mean().item())


def train_with_scheduler(
    name: str,
    scheduler: object,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    total_epochs: int,
    seed: int = 2026,
) -> TrainResult:
    """Train TinyMLP with an explicit scheduler that provides lr_at(epoch)."""
    set_seed(seed)

    model = TinyMLP(hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    lr_history: List[float] = []
    loss_history: List[float] = []

    model.train()
    for epoch in range(total_epochs):
        lr = float(scheduler.lr_at(epoch))
        for group in optimizer.param_groups:
            group["lr"] = lr

        lr_history.append(lr)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_history.append(float(loss.item()))

    train_acc = evaluate_accuracy(model, x_train, y_train)
    test_acc = evaluate_accuracy(model, x_test, y_test)

    return TrainResult(
        name=name,
        lr_history=lr_history,
        loss_history=loss_history,
        train_acc=train_acc,
        test_acc=test_acc,
    )


def summarize_result(result: TrainResult) -> str:
    """Build one compact result line."""
    return (
        f"{result.name:14s} "
        f"lr[{result.lr_history[0]:.4f}->{result.lr_history[-1]:.4f}] "
        f"loss[{result.loss_history[0]:.4f}->{result.loss_history[-1]:.4f}] "
        f"train_acc={result.train_acc:.3f} test_acc={result.test_acc:.3f}"
    )


def main() -> None:
    total_epochs = 120
    base_lr = 0.08

    x_train, y_train, x_test, y_test = make_moons_dataset()

    schedules = [
        (
            "constant",
            ConstantLRScheduler(base_lr=base_lr),
        ),
        (
            "step",
            StepLRScheduler(base_lr=base_lr, milestones=(45, 85), gamma=0.2),
        ),
        (
            "warmup_cosine",
            WarmupCosineScheduler(
                base_lr=base_lr,
                total_epochs=total_epochs,
                warmup_epochs=10,
                min_lr_ratio=0.08,
            ),
        ),
    ]

    results: List[TrainResult] = []
    for name, scheduler in schedules:
        result = train_with_scheduler(
            name=name,
            scheduler=scheduler,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            total_epochs=total_epochs,
            seed=2026,
        )
        results.append(result)

    print("Learning-rate scheduling comparison on moons classification")
    print("-" * 92)
    for result in results:
        print(summarize_result(result))

    step_result = next(r for r in results if r.name == "step")
    warmup_result = next(r for r in results if r.name == "warmup_cosine")

    print("-" * 92)
    print(
        "step key LR points:",
        f"e44={step_result.lr_history[44]:.4f},",
        f"e45={step_result.lr_history[45]:.4f},",
        f"e84={step_result.lr_history[84]:.4f},",
        f"e85={step_result.lr_history[85]:.4f}",
    )
    print(
        "warmup+cosine key LR points:",
        f"e0={warmup_result.lr_history[0]:.4f},",
        f"e9={warmup_result.lr_history[9]:.4f},",
        f"e60={warmup_result.lr_history[60]:.4f},",
        f"e119={warmup_result.lr_history[119]:.4f}",
    )

    # Basic quality checks: all schedules should reduce training loss.
    for result in results:
        assert result.loss_history[-1] < 0.8 * result.loss_history[0], (
            f"{result.name}: loss did not decrease enough "
            f"({result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f})"
        )

    # Schedule-shape checks.
    assert step_result.lr_history[45] < step_result.lr_history[44]
    assert step_result.lr_history[85] < step_result.lr_history[84]
    assert warmup_result.lr_history[0] < warmup_result.lr_history[9]
    assert warmup_result.lr_history[-1] < warmup_result.lr_history[10]

    # Task-level sanity checks.
    assert warmup_result.test_acc >= 0.85
    assert step_result.test_acc >= 0.84


if __name__ == "__main__":
    main()
