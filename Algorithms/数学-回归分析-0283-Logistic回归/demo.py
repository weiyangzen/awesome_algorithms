"""Logistic regression MVP (binary classification, numpy-only optimizer).

This script is deterministic and runs without interactive input.
It implements logistic regression with L2 regularization using
batch gradient descent + backtracking line search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float]


@dataclass
class FitResult:
    weights: np.ndarray
    bias: float
    converged: bool
    iterations: int
    history: List[HistoryItem]


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    positive = z >= 0
    negative = ~positive

    out = np.empty_like(z, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    return out


def validate_dataset(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {x.shape[0]} rows while y has {y.shape[0]} rows.")
    if x.shape[0] < 10:
        raise ValueError("Need at least 10 samples.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"y must be binary in {{0,1}}, got values={unique}.")


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    n_samples = x.shape[0]
    n_test = int(round(n_samples * test_ratio))
    n_test = min(max(n_test, 1), n_samples - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    if np.any(std < 1e-12):
        bad_col = int(np.argmin(std))
        raise ValueError(f"Feature column {bad_col} has near-zero std and cannot be standardized.")

    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std
    return x_train_std, x_test_std, mean, std


def logistic_loss_and_grad(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    l2: float,
) -> Tuple[float, np.ndarray, float]:
    n_samples = x.shape[0]
    logits = x @ w + b
    prob = stable_sigmoid(logits)

    eps = 1e-12
    data_loss = -np.mean(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps))
    reg_loss = 0.5 * l2 * float(np.dot(w, w))
    loss = float(data_loss + reg_loss)

    diff = prob - y
    grad_w = (x.T @ diff) / n_samples + l2 * w
    grad_b = float(np.mean(diff))
    return loss, grad_w, grad_b


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-2,
    lr_init: float = 1.0,
    max_iter: int = 400,
    tol: float = 1e-7,
    armijo_c: float = 1e-4,
    min_lr: float = 1e-8,
) -> FitResult:
    if l2 < 0.0:
        raise ValueError("l2 must be >= 0.")
    if lr_init <= 0.0:
        raise ValueError("lr_init must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    _, n_features = x.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    history: List[HistoryItem] = []
    converged = False

    for step in range(1, max_iter + 1):
        loss, grad_w, grad_b = logistic_loss_and_grad(x=x, y=y, w=w, b=b, l2=l2)
        grad_norm = float(np.sqrt(np.dot(grad_w, grad_w) + grad_b * grad_b))

        # Backtracking line search for stable descent.
        lr = lr_init
        grad_norm_sq = float(np.dot(grad_w, grad_w) + grad_b * grad_b)

        accepted = False
        while lr >= min_lr:
            w_new = w - lr * grad_w
            b_new = b - lr * grad_b
            new_loss, _, _ = logistic_loss_and_grad(x=x, y=y, w=w_new, b=b_new, l2=l2)

            if new_loss <= loss - armijo_c * lr * grad_norm_sq:
                accepted = True
                w, b = w_new, b_new
                loss = new_loss
                break
            lr *= 0.5

        if not accepted:
            # Fallback tiny step to keep progress deterministic.
            w = w - min_lr * grad_w
            b = b - min_lr * grad_b
            loss, _, _ = logistic_loss_and_grad(x=x, y=y, w=w, b=b, l2=l2)
            lr = min_lr

        history.append((step, loss, grad_norm, lr))

        if grad_norm < tol:
            converged = True
            break

        if step >= 2:
            prev_loss = history[-2][1]
            if abs(prev_loss - loss) < tol:
                converged = True
                break

    return FitResult(
        weights=w,
        bias=b,
        converged=converged,
        iterations=len(history),
        history=history,
    )


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return stable_sigmoid(x @ w + b)


def predict_label(x: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    prob = predict_proba(x=x, w=w, b=b)
    return (prob >= threshold).astype(float)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(float)

    tp = float(np.sum((y_pred == 1.0) & (y_true == 1.0)))
    tn = float(np.sum((y_pred == 0.0) & (y_true == 0.0)))
    fp = float(np.sum((y_pred == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((y_pred == 0.0) & (y_true == 1.0)))

    eps = 1e-12
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    y_prob_clip = np.clip(y_prob, eps, 1.0 - eps)
    log_loss = -float(np.mean(y_true * np.log(y_prob_clip) + (1.0 - y_true) * np.log(1.0 - y_prob_clip)))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": float(log_loss),
    }


def is_monotone_nonincreasing(values: List[float], atol: float = 1e-12) -> Tuple[bool, int]:
    violations = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1] + atol:
            violations += 1
    return violations == 0, violations


def make_correlated_binary_dataset(
    seed: int = 2026,
    n_samples: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a moderately collinear binary classification dataset."""
    rng = np.random.default_rng(seed)

    z1 = rng.normal(size=n_samples)
    z2 = rng.normal(size=n_samples)
    z3 = rng.normal(size=n_samples)

    x0 = z1 + 0.10 * rng.normal(size=n_samples)
    x1 = 0.92 * z1 + 0.18 * rng.normal(size=n_samples)
    x2 = z2 + 0.12 * rng.normal(size=n_samples)
    x3 = 0.90 * z2 + 0.20 * rng.normal(size=n_samples)
    x4 = z3 + 0.15 * rng.normal(size=n_samples)
    x5 = 0.85 * z3 + 0.22 * rng.normal(size=n_samples)
    x6 = 0.45 * z1 - 0.40 * z2 + 0.30 * rng.normal(size=n_samples)
    x7 = -0.35 * z1 + 0.50 * z3 + 0.25 * rng.normal(size=n_samples)

    x = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7]).astype(float)

    true_w = np.array([1.25, -0.95, 0.85, 0.55, -0.80, 0.70, -0.65, 0.40], dtype=float)
    true_b = -0.15

    logits = x @ true_w + true_b + 0.30 * rng.normal(size=n_samples)
    prob = stable_sigmoid(logits)
    y = (rng.random(n_samples) < prob).astype(float)

    if np.unique(y).size < 2:
        raise RuntimeError("Generated dataset has only one class; try another random seed.")

    return x, y, true_w


def main() -> None:
    x, y, true_w = make_correlated_binary_dataset(seed=2026, n_samples=1200)
    validate_dataset(x=x, y=y)

    x_train, x_test, y_train, y_test = train_test_split(x=x, y=y, test_ratio=0.25, seed=2026)
    x_train_std, x_test_std, _, std = standardize_from_train(x_train=x_train, x_test=x_test)
    true_w_std = true_w * std

    result = fit_logistic_regression(
        x=x_train_std,
        y=y_train,
        l2=2e-2,
        lr_init=1.0,
        max_iter=400,
        tol=1e-7,
    )

    train_prob = predict_proba(x_train_std, result.weights, result.bias)
    test_prob = predict_proba(x_test_std, result.weights, result.bias)

    train_metrics = binary_metrics(y_true=y_train, y_prob=train_prob)
    test_metrics = binary_metrics(y_true=y_test, y_prob=test_prob)

    losses = [item[1] for item in result.history]
    monotone_ok, monotone_violations = is_monotone_nonincreasing(losses)

    base_rate = float(np.mean(y_train))
    majority_acc = max(base_rate, 1.0 - base_rate)

    print("=== Logistic Regression MVP (MATH-0283) ===")
    print(f"train_samples: {x_train.shape[0]}, test_samples: {x_test.shape[0]}, features: {x_train.shape[1]}")
    print(f"positive_rate(train): {base_rate:.4f}, positive_rate(test): {np.mean(y_test):.4f}")

    print("\n=== Optimization ===")
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"initial_loss: {losses[0]:.8f}")
    print(f"final_loss:   {losses[-1]:.8f}")
    print(f"loss_monotone_nonincreasing: {monotone_ok} (violations={monotone_violations})")
    print(f"final_grad_norm: {result.history[-1][2]:.8e}")

    print("\n=== Train Metrics ===")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    coef_error = float(np.linalg.norm(result.weights - true_w_std))
    print("\n=== Coefficients ===")
    print("learned_weights:", np.array2string(result.weights, precision=4, separator=", "))
    print("true_weights_in_standardized_space:", np.array2string(true_w_std, precision=4, separator=", "))
    print(f"l2_error_vs_true_standardized_weights: {coef_error:.6f}")

    checks = {
        "loss_decreased": losses[-1] < losses[0],
        "better_than_majority_on_test": test_metrics["accuracy"] > majority_acc,
        "f1_reasonable": test_metrics["f1"] > 0.65,
        "monotone_loss": monotone_ok,
    }
    all_pass = all(checks.values())

    print("\n=== Global Checks ===")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print(f"global_checks_pass: {all_pass}")


if __name__ == "__main__":
    main()
