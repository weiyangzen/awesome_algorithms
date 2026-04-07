"""Early stopping MVP on noisy binary classification.

This demo keeps the algorithm auditable:
- data generation, splitting and feature expansion are implemented manually;
- optimization loop is implemented in NumPy (no high-level trainer black box).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    weights: np.ndarray
    bias: float
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    stopped_early: bool


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    positive = logits >= 0
    negative = ~positive
    out = np.empty_like(logits, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_z = np.exp(logits[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    return out


def bce_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Binary cross-entropy from logits using stable logaddexp."""
    losses = np.logaddexp(0.0, logits) - y_true * logits
    return float(np.mean(losses))


def binary_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(np.float64)
    return float(np.mean(preds == y_true))


def linear_logits(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = x @ w + b
    if not np.isfinite(logits).all():
        raise FloatingPointError("non-finite logits encountered")
    return logits


def evaluate_dataset(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    logits = linear_logits(x, w, b)
    probs = sigmoid(logits)
    return {
        "loss": bce_from_logits(y, logits),
        "accuracy": binary_accuracy(y, probs),
    }


def polynomial_expand_degree2(x: np.ndarray) -> np.ndarray:
    """Build [x_i, x_i*x_j (i<=j)] features. For d=20, output dim = 230."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")

    n_samples, d = x.shape
    cross_terms = []
    for i in range(d):
        xi = x[:, i]
        for j in range(i, d):
            cross_terms.append((xi * x[:, j])[:, None])

    if cross_terms:
        cross = np.hstack(cross_terms)
        return np.hstack([x, cross])
    return x


def standardize_with_train_stats(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std
    return x_train_n, x_val_n, x_test_n


def stratified_split_three_way(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")

    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    for cls in [0.0, 1.0]:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n = cls_idx.size
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val

        # Keep all partitions non-empty for stable metrics.
        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError("class-wise split produced an empty partition")

        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
    )


def make_dataset(seed: int = 403) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)

    n_samples = 1200
    d_raw = 20

    x_raw = rng.normal(0.0, 1.0, size=(n_samples, d_raw))

    # Synthetic nonlinear score to make overfitting possible after feature expansion.
    score = (
        1.3 * x_raw[:, 0]
        - 1.0 * x_raw[:, 1]
        + 0.8 * x_raw[:, 2] * x_raw[:, 3]
        - 0.6 * (x_raw[:, 4] ** 2)
        + 0.7 * np.sin(x_raw[:, 5])
        + 0.25 * rng.normal(size=n_samples)
    )
    prob = sigmoid(score)
    y = rng.binomial(1, prob).astype(np.float64)

    # Label noise to make early-stopping effect more visible.
    flip_mask = rng.uniform(size=n_samples) < 0.18
    y[flip_mask] = 1.0 - y[flip_mask]

    x = polynomial_expand_degree2(x_raw)

    x_train, y_train, x_val, y_val, x_test, y_test = stratified_split_three_way(
        x,
        y,
        train_frac=0.50,
        val_frac=0.25,
        test_frac=0.25,
        seed=seed,
    )

    x_train, x_val, x_test = standardize_with_train_stats(x_train, x_val, x_test)

    return (
        x_train.astype(np.float64),
        y_train.astype(np.float64),
        x_val.astype(np.float64),
        y_val.astype(np.float64),
        x_test.astype(np.float64),
        y_test.astype(np.float64),
    )


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    learning_rate: float = 0.005,
    max_epochs: int = 300,
    batch_size: int = 64,
    l2: float = 1e-4,
    patience: int = 18,
    min_delta: float = 1e-4,
    early_stopping: bool = True,
    restore_best: bool = True,
    seed: int = 403,
) -> FitResult:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")
    if x_val.ndim != 2:
        raise ValueError("x_val must be 2D")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("x_train/x_val feature dimensions must match")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("y_train/y_val must be 1D")
    if x_train.shape[0] != y_train.shape[0] or x_val.shape[0] != y_val.shape[0]:
        raise ValueError("sample counts of X and y must match")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_epochs <= 0 or batch_size <= 0:
        raise ValueError("max_epochs and batch_size must be positive")
    if patience < 0:
        raise ValueError("patience must be non-negative")

    rng = np.random.default_rng(seed)
    n_train, n_features = x_train.shape

    w = rng.normal(loc=0.0, scale=0.01, size=n_features)
    b = 0.0

    best_w = w.copy()
    best_b = b
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    rows = []
    stopped_epoch = max_epochs
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n_train)

        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            logits = linear_logits(xb, w, b)
            probs = sigmoid(logits)
            error = probs - yb

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                grad_w = (xb.T @ error) / idx.size + l2 * w
            grad_b = float(np.mean(error))
            if not np.isfinite(grad_w).all() or not np.isfinite(grad_b):
                raise FloatingPointError("non-finite gradient encountered")

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
            # Keep long-run baseline training numerically stable.
            np.clip(w, -20.0, 20.0, out=w)
            b = float(np.clip(b, -20.0, 20.0))

        train_logits = linear_logits(x_train, w, b)
        val_logits = linear_logits(x_val, w, b)
        train_probs = sigmoid(train_logits)
        val_probs = sigmoid(val_logits)

        train_loss = bce_from_logits(y_train, train_logits) + 0.5 * l2 * float(np.dot(w, w))
        val_loss = bce_from_logits(y_val, val_logits)

        train_acc = binary_accuracy(y_train, train_probs)
        val_acc = binary_accuracy(y_val, val_probs)

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_w = w.copy()
            best_b = b
            wait = 0
        else:
            wait += 1

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "wait": wait,
            }
        )

        if early_stopping and wait >= patience:
            stopped_epoch = epoch
            stopped_early = True
            break

    history = pd.DataFrame(rows)

    if restore_best:
        w = best_w
        b = best_b

    return FitResult(
        weights=w,
        bias=b,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_epoch=stopped_epoch,
        stopped_early=stopped_early,
    )


def print_history_snapshot(name: str, history: pd.DataFrame, rows: int = 5) -> None:
    show = pd.concat([history.head(rows), history.tail(rows)], ignore_index=True)
    print(f"\n[{name}] history snapshot (head+tail)")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


def main() -> None:
    x_train, y_train, x_val, y_val, x_test, y_test = make_dataset(seed=403)

    config = {
        "learning_rate": 0.005,
        "max_epochs": 300,
        "batch_size": 64,
        "l2": 1e-4,
        "patience": 18,
        "min_delta": 1e-4,
        "seed": 403,
    }

    with_early_stop = fit_logistic_regression(
        x_train,
        y_train,
        x_val,
        y_val,
        early_stopping=True,
        restore_best=True,
        **config,
    )

    without_early_stop = fit_logistic_regression(
        x_train,
        y_train,
        x_val,
        y_val,
        early_stopping=False,
        restore_best=False,
        **config,
    )

    es_val = evaluate_dataset(x_val, y_val, with_early_stop.weights, with_early_stop.bias)
    es_test = evaluate_dataset(x_test, y_test, with_early_stop.weights, with_early_stop.bias)

    noes_val = evaluate_dataset(x_val, y_val, without_early_stop.weights, without_early_stop.bias)
    noes_test = evaluate_dataset(x_test, y_test, without_early_stop.weights, without_early_stop.bias)

    print("=== Early Stopping Demo (MATH-0403) ===")
    print(f"train/val/test sizes: {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}")
    print(f"feature dimension after polynomial expansion: {x_train.shape[1]}")

    print("\n--- With Early Stopping ---")
    print(f"stopped_early: {with_early_stop.stopped_early}")
    print(f"stopped_epoch: {with_early_stop.stopped_epoch}")
    print(f"best_epoch: {with_early_stop.best_epoch}")
    print(f"best_val_loss: {with_early_stop.best_val_loss:.6f}")
    print(f"val_loss/val_acc: {es_val['loss']:.6f} / {es_val['accuracy']:.4f}")
    print(f"test_loss/test_acc: {es_test['loss']:.6f} / {es_test['accuracy']:.4f}")

    print("\n--- Without Early Stopping (fixed max_epochs) ---")
    print(f"trained_epochs: {without_early_stop.history.shape[0]}")
    print(f"val_loss/val_acc: {noes_val['loss']:.6f} / {noes_val['accuracy']:.4f}")
    print(f"test_loss/test_acc: {noes_test['loss']:.6f} / {noes_test['accuracy']:.4f}")

    gap = noes_test["loss"] - es_test["loss"]
    print(f"\n(test_loss_without_es - test_loss_with_es): {gap:.6f}")

    print_history_snapshot("with_early_stopping", with_early_stop.history)
    print_history_snapshot("without_early_stopping", without_early_stop.history)


if __name__ == "__main__":
    main()
