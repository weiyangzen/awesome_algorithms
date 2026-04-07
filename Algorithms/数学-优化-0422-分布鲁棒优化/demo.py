"""Minimal runnable MVP for Distributionally Robust Optimization (MATH-0422).

This demo implements a small Wasserstein-DRO logistic regression solver from
scratch (NumPy only) and compares it with ERM on in-distribution / shifted data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class TrainResult:
    """Container for one optimization run."""

    w: np.ndarray
    loss_history: List[float]
    epsilon: float


def sample_binary_gaussian(
    rng: np.random.Generator,
    n: int,
    mean_pos: np.ndarray,
    mean_neg: np.ndarray,
    cov_pos: np.ndarray,
    cov_neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a balanced binary dataset with labels in {-1, +1}."""
    n_pos = n // 2
    n_neg = n - n_pos

    x_pos = rng.multivariate_normal(mean_pos, cov_pos, size=n_pos)
    x_neg = rng.multivariate_normal(mean_neg, cov_neg, size=n_neg)
    y_pos = np.ones(n_pos, dtype=np.float64)
    y_neg = -np.ones(n_neg, dtype=np.float64)

    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])

    idx = np.arange(n)
    rng.shuffle(idx)
    return x[idx], y[idx]


def build_datasets(seed: int = 422) -> Dict[str, np.ndarray]:
    """Create train / ID-test / shifted-test splits."""
    rng = np.random.default_rng(seed)

    train_cov_pos = np.array([[1.0, 0.15], [0.15, 0.9]])
    train_cov_neg = np.array([[0.95, -0.10], [-0.10, 1.05]])
    test_cov_id = np.array([[1.0, 0.10], [0.10, 1.0]])
    test_cov_shift = np.array([[1.6, 0.35], [0.35, 1.5]])

    x_train, y_train = sample_binary_gaussian(
        rng=rng,
        n=800,
        mean_pos=np.array([1.5, 1.2]),
        mean_neg=np.array([-1.5, -1.1]),
        cov_pos=train_cov_pos,
        cov_neg=train_cov_neg,
    )
    x_test_id, y_test_id = sample_binary_gaussian(
        rng=rng,
        n=500,
        mean_pos=np.array([1.45, 1.1]),
        mean_neg=np.array([-1.45, -1.0]),
        cov_pos=test_cov_id,
        cov_neg=test_cov_id,
    )
    x_test_shift, y_test_shift = sample_binary_gaussian(
        rng=rng,
        n=500,
        mean_pos=np.array([0.75, 0.55]),
        mean_neg=np.array([-0.75, -0.50]),
        cov_pos=test_cov_shift,
        cov_neg=test_cov_shift,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test_id": x_test_id,
        "y_test_id": y_test_id,
        "x_test_shift": x_test_shift,
        "y_test_shift": y_test_shift,
    }


def standardize_from_train(
    x_train: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Standardize all sets using train statistics only."""
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-12

    out: List[np.ndarray] = [(x_train - mu) / sigma]
    for x in others:
        out.append((x - mu) / sigma)
    return tuple(out)


def append_bias(x: np.ndarray) -> np.ndarray:
    """Append constant bias feature 1.0."""
    bias = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.hstack([x, bias])


def dro_logistic_loss_grad(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    l2: float,
) -> Tuple[float, np.ndarray]:
    """Compute DRO logistic objective and gradient.

    Objective:
        mean_i log(1 + exp(-(y_i * <w, x_i> - epsilon * ||w_no_bias||_2)))
        + 0.5 * l2 * ||w_no_bias||_2^2
    """
    margins = y * np.einsum("nd,d->n", x, w)

    norm_linear = float(np.linalg.norm(w[:-1]))
    robust_margins = margins - epsilon * norm_linear

    losses = np.logaddexp(0.0, -robust_margins)
    reg = 0.5 * l2 * float(np.dot(w[:-1], w[:-1]))
    loss = float(np.mean(losses) + reg)

    # sigma(-m) = 1 / (1 + exp(m))
    weights = 1.0 / (1.0 + np.exp(np.clip(robust_margins, -50.0, 50.0)))

    norm_grad = np.zeros_like(w)
    if norm_linear > 1e-12:
        norm_grad[:-1] = w[:-1] / norm_linear

    # d(robust_margin_i)/dw = y_i * x_i - epsilon * d||w_no_bias||/dw
    drobust = y[:, None] * x - epsilon * norm_grad[None, :]
    grad = -np.mean(weights[:, None] * drobust, axis=0)

    reg_grad = np.zeros_like(w)
    reg_grad[:-1] = w[:-1]
    grad += l2 * reg_grad
    return loss, grad


def fit_dro_logistic(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    l2: float = 1e-3,
    lr: float = 0.15,
    epochs: int = 450,
) -> TrainResult:
    """Full-batch gradient descent for DRO logistic regression."""
    w = np.zeros(x.shape[1], dtype=np.float64)
    history: List[float] = []

    for _ in range(epochs):
        loss, grad = dro_logistic_loss_grad(w=w, x=x, y=y, epsilon=epsilon, l2=l2)
        w = w - lr * grad
        history.append(loss)

    return TrainResult(w=w, loss_history=history, epsilon=epsilon)


def accuracy(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy for labels in {-1, +1}."""
    scores = np.einsum("nd,d->n", x, w)
    pred = np.where(scores >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == y))


def finite_diff_gradient_check(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    l2: float = 1e-3,
    delta: float = 1e-5,
    check_dims: Sequence[int] = (0, 1, 2),
) -> float:
    """Numerically verify gradient on selected dimensions."""
    w = np.array([0.17, -0.11, 0.05], dtype=np.float64)
    _, analytic = dro_logistic_loss_grad(w=w, x=x, y=y, epsilon=epsilon, l2=l2)

    max_abs_err = 0.0
    for dim in check_dims:
        w_pos = w.copy()
        w_neg = w.copy()
        w_pos[dim] += delta
        w_neg[dim] -= delta

        f_pos, _ = dro_logistic_loss_grad(w=w_pos, x=x, y=y, epsilon=epsilon, l2=l2)
        f_neg, _ = dro_logistic_loss_grad(w=w_neg, x=x, y=y, epsilon=epsilon, l2=l2)
        numeric = (f_pos - f_neg) / (2.0 * delta)
        max_abs_err = max(max_abs_err, abs(numeric - analytic[dim]))
    return max_abs_err


def main() -> None:
    print("Distributionally Robust Optimization MVP (MATH-0422)")
    print("=" * 72)

    raw = build_datasets(seed=422)
    x_train, x_test_id, x_test_shift = standardize_from_train(
        raw["x_train"], raw["x_test_id"], raw["x_test_shift"]
    )
    x_train = append_bias(x_train)
    x_test_id = append_bias(x_test_id)
    x_test_shift = append_bias(x_test_shift)
    y_train = raw["y_train"]
    y_test_id = raw["y_test_id"]
    y_test_shift = raw["y_test_shift"]

    eps_dro = 0.60

    grad_err_erm = finite_diff_gradient_check(x_train, y_train, epsilon=0.0, l2=2e-3)
    grad_err_dro = finite_diff_gradient_check(x_train, y_train, epsilon=eps_dro, l2=2e-3)
    print(f"gradient-check ERM (eps=0.0) max abs err: {grad_err_erm:.3e}")
    print(f"gradient-check DRO (eps={eps_dro:.2f}) max abs err: {grad_err_dro:.3e}")

    erm = fit_dro_logistic(
        x=x_train,
        y=y_train,
        epsilon=0.0,
        l2=2e-3,
        lr=0.18,
        epochs=420,
    )
    dro = fit_dro_logistic(
        x=x_train,
        y=y_train,
        epsilon=eps_dro,
        l2=2e-3,
        lr=0.15,
        epochs=420,
    )

    metrics = {
        "ERM": {
            "train_acc": accuracy(erm.w, x_train, y_train),
            "id_acc": accuracy(erm.w, x_test_id, y_test_id),
            "shift_acc": accuracy(erm.w, x_test_shift, y_test_shift),
            "loss0": erm.loss_history[0],
            "lossT": erm.loss_history[-1],
        },
        "DRO": {
            "train_acc": accuracy(dro.w, x_train, y_train),
            "id_acc": accuracy(dro.w, x_test_id, y_test_id),
            "shift_acc": accuracy(dro.w, x_test_shift, y_test_shift),
            "loss0": dro.loss_history[0],
            "lossT": dro.loss_history[-1],
        },
    }

    print("-" * 72)
    for name in ("ERM", "DRO"):
        m = metrics[name]
        print(
            f"{name:>3} | loss: {m['loss0']:.4f} -> {m['lossT']:.4f} "
            f"| train: {m['train_acc']:.4f} | id: {m['id_acc']:.4f} | shift: {m['shift_acc']:.4f}"
        )

    shift_gain = metrics["DRO"]["shift_acc"] - metrics["ERM"]["shift_acc"]
    print("-" * 72)
    print(f"DRO shifted-distribution accuracy gain over ERM: {shift_gain:+.4f}")
    print("=" * 72)

    if max(grad_err_erm, grad_err_dro) > 5e-4:
        raise RuntimeError("gradient check failed")
    if not (metrics["ERM"]["lossT"] < metrics["ERM"]["loss0"]):
        raise RuntimeError("ERM training did not decrease loss")
    if not (metrics["DRO"]["lossT"] < metrics["DRO"]["loss0"]):
        raise RuntimeError("DRO training did not decrease loss")
    if metrics["DRO"]["shift_acc"] < 0.70:
        raise RuntimeError("DRO shifted accuracy is unexpectedly low")

    print("Run completed successfully.")


if __name__ == "__main__":
    main()
