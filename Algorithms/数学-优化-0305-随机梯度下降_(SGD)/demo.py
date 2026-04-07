"""Minimal runnable MVP for 随机梯度下降 (SGD).

This demo trains an L2-regularized logistic regression model with mini-batch SGD
using only NumPy. It is deterministic and non-interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SGDConfig:
    n_samples: int = 1200
    n_features: int = 12
    train_ratio: float = 0.8
    noise_std: float = 0.8
    reg_lambda: float = 1e-2
    batch_size: int = 32
    epochs: int = 35
    lr0: float = 0.25
    lr_decay: float = 5e-4
    report_every: int = 5
    seed: int = 42

    def validate(self) -> None:
        if self.n_samples < 100:
            raise ValueError("n_samples must be >= 100")
        if self.n_features < 2:
            raise ValueError("n_features must be >= 2")
        if not (0.5 <= self.train_ratio < 1.0):
            raise ValueError("train_ratio must be in [0.5, 1.0)")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr0 <= 0.0:
            raise ValueError("lr0 must be positive")
        if self.lr_decay < 0.0:
            raise ValueError("lr_decay must be non-negative")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")


def dot_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise dot product x_i · w, implemented with einsum for stable portability."""
    return np.einsum("ij,j->i", x, w)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def make_dataset(cfg: SGDConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.n_samples, cfg.n_features))
    true_w = rng.normal(loc=0.0, scale=1.2, size=cfg.n_features)
    logits = dot_rows(x, true_w) + cfg.noise_std * rng.normal(size=cfg.n_samples)
    y = (logits > 0.0).astype(np.float64)

    # Standardize features to improve SGD conditioning.
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    x = (x - mu) / sigma

    split = int(cfg.n_samples * cfg.train_ratio)
    return x[:split], y[:split], x[split:], y[split:]


def logistic_loss_and_grads(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    reg_lambda: float,
) -> Tuple[float, np.ndarray, float]:
    logits = dot_rows(x, w) + b
    p = sigmoid(logits)
    eps = 1e-12
    ce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    loss = float(ce + 0.5 * reg_lambda * np.dot(w, w))

    err = p - y
    grad_w = np.einsum("ij,i->j", x, err) / x.shape[0] + reg_lambda * w
    grad_b = float(np.mean(err))
    return loss, grad_w, grad_b


def accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    pred = (sigmoid(dot_rows(x, w) + b) >= 0.5).astype(np.float64)
    return float(np.mean(pred == y))


def train_sgd(
    cfg: SGDConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, List[Tuple[int, float, float, float]]]:
    rng = np.random.default_rng(cfg.seed + 7)
    n_train, n_features = x_train.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    global_step = 0

    initial_train_loss, _, _ = logistic_loss_and_grads(x_train, y_train, w, b, cfg.reg_lambda)
    initial_test_acc = accuracy(x_test, y_test, w, b)

    logs: List[Tuple[int, float, float, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        perm = rng.permutation(n_train)

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            _, grad_w, grad_b = logistic_loss_and_grads(xb, yb, w, b, cfg.reg_lambda)
            lr_t = cfg.lr0 / (1.0 + cfg.lr_decay * global_step)
            w -= lr_t * grad_w
            b -= lr_t * grad_b
            global_step += 1

            if not np.all(np.isfinite(w)) or not np.isfinite(b):
                raise FloatingPointError("SGD diverged: non-finite parameter encountered")

        train_loss, _, _ = logistic_loss_and_grads(x_train, y_train, w, b, cfg.reg_lambda)
        test_loss, _, _ = logistic_loss_and_grads(x_test, y_test, w, b, cfg.reg_lambda)
        test_acc = accuracy(x_test, y_test, w, b)

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            logs.append((epoch, train_loss, test_loss, test_acc))

    final_train_loss, _, _ = logistic_loss_and_grads(x_train, y_train, w, b, cfg.reg_lambda)
    final_test_acc = accuracy(x_test, y_test, w, b)

    # Deterministic sanity checks for this synthetic task.
    if final_train_loss >= initial_train_loss:
        raise AssertionError(
            f"Expected training loss to decrease, got initial={initial_train_loss:.6f}, final={final_train_loss:.6f}"
        )
    if final_test_acc < 0.84:
        raise AssertionError(f"Expected final test accuracy >= 0.84, got {final_test_acc:.4f}")

    print(f"Initial train loss: {initial_train_loss:.6f}")
    print(f"Initial test acc : {initial_test_acc:.4f}")
    return w, b, initial_train_loss, final_train_loss, logs


def main() -> None:
    cfg = SGDConfig()
    cfg.validate()

    x_train, y_train, x_test, y_test = make_dataset(cfg)
    w, b, init_loss, final_loss, logs = train_sgd(cfg, x_train, y_train, x_test, y_test)

    print("\nEpoch logs (epoch, train_loss, test_loss, test_acc):")
    for epoch, tr_loss, te_loss, te_acc in logs:
        print(f"  {epoch:>2d} | {tr_loss:.6f} | {te_loss:.6f} | {te_acc:.4f}")

    final_test_acc = accuracy(x_test, y_test, w, b)
    print("\nFinal summary:")
    print(f"  Loss drop      : {init_loss:.6f} -> {final_loss:.6f}")
    print(f"  Final test acc : {final_test_acc:.4f}")
    print(f"  ||w||_2        : {np.linalg.norm(w):.6f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
