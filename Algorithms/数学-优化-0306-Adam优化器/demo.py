"""Minimal runnable MVP for Adam optimizer (MATH-0306).

This demo implements Adam from scratch with NumPy and uses it to train
binary logistic regression on a synthetic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class AdamState:
    """Optimizer state for Adam."""

    m: np.ndarray
    v: np.ndarray
    t: int


def make_synthetic_binary_dataset(
    n_per_class: int = 200,
    seed: int = 306,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a linearly-separable-ish binary classification dataset.

    Returns:
        X: shape (2*n_per_class, 3), where last column is bias feature 1.
        y: shape (2*n_per_class,), values in {0, 1}.
    """
    rng = np.random.default_rng(seed)

    mean0 = np.array([-1.2, -1.0])
    mean1 = np.array([1.1, 1.3])
    cov = np.array([[0.7, 0.2], [0.2, 0.6]])

    x0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean1, cov, size=n_per_class)

    x = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.float64), np.ones(n_per_class, dtype=np.float64)]
    )

    # Standardize each real feature for better conditioning.
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-12
    x = (x - x_mean) / x_std

    # Append bias feature so we optimize one parameter vector only.
    bias = np.ones((x.shape[0], 1), dtype=np.float64)
    x_with_bias = np.hstack([x, bias])
    return x_with_bias, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z_clip = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def logistic_loss_and_grad(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Binary cross-entropy loss and gradient.

    Args:
        w: parameter vector, shape (d,)
        x: design matrix, shape (n, d)
        y: labels in {0,1}, shape (n,)
        l2: L2 regularization coefficient on non-bias weights

    Returns:
        loss: scalar loss
        grad: gradient vector, shape (d,)
    """
    n = x.shape[0]
    # Use einsum to keep behavior deterministic across BLAS backends.
    logits = np.einsum("nd,d->n", x, w)
    p = sigmoid(logits)

    eps = 1e-12
    bce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))

    # Exclude bias term from L2 (bias is the last entry).
    w_no_bias = w.copy()
    w_no_bias[-1] = 0.0
    reg = 0.5 * l2 * float(np.dot(w_no_bias, w_no_bias))

    loss = bce + reg

    grad = np.einsum("nd,n->d", x, (p - y)) / n
    grad += l2 * w_no_bias
    return float(loss), grad


def adam_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: AdamState,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, AdamState]:
    """Apply one Adam update step."""
    t = state.t + 1
    m = beta1 * state.m + (1.0 - beta1) * grad
    v = beta2 * state.v + (1.0 - beta2) * (grad * grad)

    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)

    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w_new, AdamState(m=m, v=v, t=t)


def train_logistic_with_adam(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, List[float]]:
    """Train logistic regression parameters with Adam."""
    d = x.shape[1]
    w = np.zeros(d, dtype=np.float64)
    state = AdamState(m=np.zeros_like(w), v=np.zeros_like(w), t=0)

    history: List[float] = []
    for _ in range(epochs):
        loss, grad = logistic_loss_and_grad(w, x, y, l2=l2)
        w, state = adam_step(
            w,
            grad,
            state,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        history.append(loss)

    return w, history


def accuracy(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    p = sigmoid(np.einsum("nd,d->n", x, w))
    y_pred = (p >= 0.5).astype(np.float64)
    return float(np.mean(y_pred == y))


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 306,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic random split."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    tr = idx[:n_train]
    te = idx[n_train:]
    return x[tr], y[tr], x[te], y[te]


def finite_diff_gradient_check(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    delta: float = 1e-5,
    check_dims: Sequence[int] = (0, 1, 2),
) -> float:
    """Lightweight gradient check; returns max abs error on selected dims."""
    _, analytic = logistic_loss_and_grad(w, x, y, l2=l2)

    max_err = 0.0
    for i in check_dims:
        w_pos = w.copy()
        w_neg = w.copy()
        w_pos[i] += delta
        w_neg[i] -= delta
        l_pos, _ = logistic_loss_and_grad(w_pos, x, y, l2=l2)
        l_neg, _ = logistic_loss_and_grad(w_neg, x, y, l2=l2)
        numeric = (l_pos - l_neg) / (2.0 * delta)
        err = abs(numeric - analytic[i])
        max_err = max(max_err, err)
    return max_err


def main() -> None:
    print("Adam Optimizer MVP (MATH-0306)")
    print("=" * 64)

    x, y = make_synthetic_binary_dataset(n_per_class=220, seed=306)
    x_train, y_train, x_test, y_test = split_train_test(x, y, train_ratio=0.8, seed=42)

    w0 = np.zeros(x_train.shape[1], dtype=np.float64)
    grad_err = finite_diff_gradient_check(w0, x_train, y_train, l2=1e-3)
    print(f"gradient-check max abs error: {grad_err:.3e}")

    w, history = train_logistic_with_adam(
        x_train,
        y_train,
        epochs=320,
        lr=0.05,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        l2=1e-3,
    )

    train_acc = accuracy(w, x_train, y_train)
    test_acc = accuracy(w, x_test, y_test)

    print(f"initial loss: {history[0]:.6f}")
    print(f"mid loss (epoch 160): {history[159]:.6f}")
    print(f"final loss: {history[-1]:.6f}")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"learned weights: {np.array2string(w, precision=4)}")

    if not (history[-1] < history[0]):
        raise RuntimeError("training did not decrease loss")
    if not (test_acc > 0.85):
        raise RuntimeError("test accuracy is unexpectedly low")

    print("=" * 64)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
