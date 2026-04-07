"""Minimal runnable MVP for 神经网络 - 感知机.

This demo implements a binary linear perceptron from scratch using NumPy only.
It is deterministic and non-interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class PerceptronConfig:
    n_samples: int = 1000
    n_features: int = 10
    train_ratio: float = 0.8
    learning_rate: float = 1.0
    max_epochs: int = 40
    report_every: int = 5
    seed: int = 42

    def validate(self) -> None:
        if self.n_samples < 200:
            raise ValueError("n_samples must be >= 200")
        if self.n_features < 2:
            raise ValueError("n_features must be >= 2")
        if not (0.5 <= self.train_ratio < 1.0):
            raise ValueError("train_ratio must be in [0.5, 1.0)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")


def dot_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute row-wise dot products x_i · w."""
    return np.einsum("ij,j->i", x, w)


def make_linearly_separable_dataset(
    cfg: PerceptronConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reproducible linear-separable binary classification dataset.

    Labels are encoded as {-1, +1}.
    """
    rng = np.random.default_rng(cfg.seed)

    x = rng.normal(size=(cfg.n_samples, cfg.n_features))
    true_w = rng.normal(size=cfg.n_features)
    true_b = 0.25

    margin_score = dot_rows(x, true_w) + true_b
    y = np.where(margin_score >= 0.0, 1, -1).astype(np.int64)

    # Standardize features for stable training dynamics.
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    x = (x - mu) / sigma

    split = int(cfg.n_samples * cfg.train_ratio)
    return x[:split], y[:split], x[split:], y[split:]


def predict_signed(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    scores = dot_rows(x, w) + b
    return np.where(scores >= 0.0, 1, -1).astype(np.int64)


def accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    pred = predict_signed(x, w, b)
    return float(np.mean(pred == y))


def train_perceptron(
    cfg: PerceptronConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float, int, int, List[Tuple[int, int, float, float]]]:
    """Train a classic online perceptron.

    Update rule on misclassified sample (x_i, y_i):
      w <- w + eta * y_i * x_i
      b <- b + eta * y_i
    where y_i in {-1, +1}.
    """
    rng = np.random.default_rng(cfg.seed + 17)

    n_train, n_features = x_train.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    total_updates = 0

    init_train_acc = accuracy(x_train, y_train, w, b)
    init_test_acc = accuracy(x_test, y_test, w, b)

    logs: List[Tuple[int, int, float, float]] = []
    converged_epoch = -1

    for epoch in range(1, cfg.max_epochs + 1):
        perm = rng.permutation(n_train)
        epoch_errors = 0

        for idx in perm:
            xi = x_train[idx]
            yi = int(y_train[idx])
            margin = yi * (float(np.dot(w, xi)) + b)

            if margin <= 0.0:
                w += cfg.learning_rate * yi * xi
                b += cfg.learning_rate * yi
                epoch_errors += 1
                total_updates += 1

        train_acc = accuracy(x_train, y_train, w, b)
        test_acc = accuracy(x_test, y_test, w, b)

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.max_epochs or epoch_errors == 0:
            logs.append((epoch, epoch_errors, train_acc, test_acc))

        if epoch_errors == 0:
            converged_epoch = epoch
            break

        if not np.all(np.isfinite(w)) or not np.isfinite(b):
            raise FloatingPointError("Perceptron diverged: non-finite parameter encountered")

    final_train_acc = accuracy(x_train, y_train, w, b)
    final_test_acc = accuracy(x_test, y_test, w, b)

    # Deterministic sanity checks for this synthetic setup.
    if total_updates == 0:
        raise AssertionError("Unexpected: no updates happened; dataset/config likely degenerate")
    if final_train_acc < 0.98:
        raise AssertionError(f"Expected final train accuracy >= 0.98, got {final_train_acc:.4f}")
    if final_test_acc < 0.95:
        raise AssertionError(f"Expected final test accuracy >= 0.95, got {final_test_acc:.4f}")

    print(f"Initial train acc: {init_train_acc:.4f}")
    print(f"Initial test acc : {init_test_acc:.4f}")
    return w, b, total_updates, converged_epoch, logs


def main() -> None:
    cfg = PerceptronConfig()
    cfg.validate()

    x_train, y_train, x_test, y_test = make_linearly_separable_dataset(cfg)
    w, b, total_updates, converged_epoch, logs = train_perceptron(cfg, x_train, y_train, x_test, y_test)

    print("\nEpoch logs (epoch, errors, train_acc, test_acc):")
    for epoch, errors, tr_acc, te_acc in logs:
        print(f"  {epoch:>2d} | {errors:>4d} | {tr_acc:.4f} | {te_acc:.4f}")

    final_train_acc = accuracy(x_train, y_train, w, b)
    final_test_acc = accuracy(x_test, y_test, w, b)

    print("\nFinal summary:")
    print(f"  Total updates  : {total_updates}")
    if converged_epoch > 0:
        print(f"  Converged at   : epoch {converged_epoch}")
    else:
        print(f"  Converged at   : not reached within {cfg.max_epochs} epochs")
    print(f"  Final train acc: {final_train_acc:.4f}")
    print(f"  Final test acc : {final_test_acc:.4f}")
    print(f"  ||w||_2        : {np.linalg.norm(w):.6f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
