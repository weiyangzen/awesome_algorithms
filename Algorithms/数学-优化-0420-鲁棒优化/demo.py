"""Minimal runnable MVP for Robust Optimization (MATH-0420).

This demo implements a robust linear regression objective with feature-matrix
uncertainty:

    min_w max_{||Delta||_F <= rho} (1/(2n)) ||(X + Delta) w - y||_2^2
          + (l2/2) ||w||_2^2

Using the induced-norm identity, the inner max has a closed form:

    max_{||Delta||_F <= rho} ||(X + Delta)w - y||_2
    = ||Xw - y||_2 + rho ||w||_2

So we optimize:

    f_rob(w) = (1/(2n)) (||Xw - y||_2 + rho ||w||_2)^2 + (l2/2) ||w||_2^2

The script compares:
1) nominal least squares (rho=0 form),
2) robust least squares (rho>0),
on ID and shifted test measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class TrainResult:
    """Container for one gradient-descent run."""

    w: np.ndarray
    loss_history: List[float]
    name: str


def sample_correlated_gaussian(
    rng: np.random.Generator, n: int, chol: np.ndarray
) -> np.ndarray:
    """Sample N(0, cov) where cov = chol @ chol.T using einsum."""
    z = rng.normal(0.0, 1.0, size=(n, chol.shape[0]))
    return np.einsum("nd,fd->nf", z, chol)


def build_dataset(
    seed: int = 420,
    n_train: int = 320,
    n_test: int = 320,
    d: int = 10,
) -> Dict[str, np.ndarray]:
    """Build train / ID-test / shifted-test sets.

    We generate latent clean features, then add measurement noise. Shifted test
    has stronger feature perturbation, which stresses robustness to X-noise.
    """
    rng = np.random.default_rng(seed)
    cov = 0.35 * np.ones((d, d), dtype=np.float64) + 0.65 * np.eye(d, dtype=np.float64)
    chol = np.linalg.cholesky(cov)

    w_true = rng.normal(0.0, 1.0, size=d)
    w_true /= max(np.linalg.norm(w_true), 1e-12)

    x_train_latent = sample_correlated_gaussian(rng, n_train, chol)
    x_test_latent = sample_correlated_gaussian(rng, n_test, chol)

    y_train = np.einsum("nd,d->n", x_train_latent, w_true) + rng.normal(
        0.0, 0.25, size=n_train
    )
    y_test = np.einsum("nd,d->n", x_test_latent, w_true) + rng.normal(
        0.0, 0.25, size=n_test
    )

    x_train_obs = x_train_latent + rng.normal(0.0, 0.15, size=(n_train, d))
    x_test_id = x_test_latent + rng.normal(0.0, 0.15, size=(n_test, d))
    x_test_shift = x_test_latent + rng.normal(0.0, 0.60, size=(n_test, d))

    return {
        "x_train": x_train_obs,
        "y_train": y_train,
        "x_test_id": x_test_id,
        "y_test_id": y_test,
        "x_test_shift": x_test_shift,
        "y_test_shift": y_test,
    }


def standardize_from_train(
    x_train: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Standardize all matrices using train statistics only."""
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-12

    out: List[np.ndarray] = [(x_train - mu) / sigma]
    for x in others:
        out.append((x - mu) / sigma)
    return tuple(out)


def nominal_loss_grad(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float
) -> Tuple[float, np.ndarray]:
    """Nominal least-squares objective and gradient."""
    n = float(y.shape[0])
    residual = np.einsum("nd,d->n", x, w) - y
    loss = 0.5 * float(np.dot(residual, residual)) / n + 0.5 * l2 * float(np.dot(w, w))
    grad = np.einsum("nd,n->d", x, residual) / n + l2 * w
    return loss, grad


def robust_loss_grad(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    l2: float,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """Robust objective and gradient for Frobenius-bounded X uncertainty."""
    n = float(y.shape[0])
    residual = np.einsum("nd,d->n", x, w) - y

    a = float(np.linalg.norm(residual))
    b = float(np.linalg.norm(w))
    a_safe = max(a, eps)
    b_safe = max(b, eps)

    scalar = a + rho * b
    loss = 0.5 * (scalar * scalar) / n + 0.5 * l2 * b * b

    grad_a = np.einsum("nd,n->d", x, residual) / a_safe
    grad_b = w / b_safe
    grad = (scalar / n) * (grad_a + rho * grad_b) + l2 * w
    return loss, grad


def fit_gradient_descent(
    obj_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    d: int,
    name: str,
    epochs: int,
    lr: float,
) -> TrainResult:
    """Full-batch gradient descent with mild inverse-time decay."""
    w = np.zeros(d, dtype=np.float64)
    history: List[float] = []

    for t in range(epochs):
        loss, grad = obj_grad_fn(w)
        step = lr / (1.0 + 0.0015 * t)
        w = w - step * grad
        history.append(loss)

    return TrainResult(w=w, loss_history=history, name=name)


def finite_diff_gradient_check(
    loss_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    w: np.ndarray,
    delta: float = 1e-6,
) -> float:
    """Numerically verify gradient via centered finite differences."""
    _, analytic = loss_grad_fn(w)
    max_abs_err = 0.0
    for i in range(w.shape[0]):
        w_pos = w.copy()
        w_neg = w.copy()
        w_pos[i] += delta
        w_neg[i] -= delta

        f_pos, _ = loss_grad_fn(w_pos)
        f_neg, _ = loss_grad_fn(w_neg)
        numeric = (f_pos - f_neg) / (2.0 * delta)
        max_abs_err = max(max_abs_err, abs(numeric - analytic[i]))
    return max_abs_err


def mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean squared prediction error."""
    residual = np.einsum("nd,d->n", x, w) - y
    return float(np.mean(residual * residual))


def robust_objective_value(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, rho: float, l2: float
) -> float:
    """Convenience wrapper for robust objective value only."""
    loss, _ = robust_loss_grad(w=w, x=x, y=y, rho=rho, l2=l2)
    return loss


def worst_case_residual_norm(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, rho: float, eps: float = 1e-12
) -> Tuple[float, float, float]:
    """Compute residual norm and one explicit worst-case perturbation result.

    Returns:
    - nominal residual norm ||Xw - y||,
    - adversarial residual norm ||(X + Delta*)w - y||,
    - ||Delta*||_F.
    """
    residual = np.einsum("nd,d->n", x, w) - y
    a = float(np.linalg.norm(residual))
    b = float(np.linalg.norm(w))
    if a < eps or b < eps:
        return a, a, 0.0

    u = residual / a
    v = w / b
    delta_star = rho * np.outer(u, v)  # rank-1, Fro norm = rho
    adv_residual = residual + np.einsum("nd,d->n", delta_star, w)
    adv_norm = float(np.linalg.norm(adv_residual))
    delta_norm = float(np.linalg.norm(delta_star, ord="fro"))
    return a, adv_norm, delta_norm


def main() -> None:
    print("Robust Optimization MVP (MATH-0420)")
    print("=" * 72)

    data = build_dataset(seed=420)
    x_train, x_test_id, x_test_shift = standardize_from_train(
        data["x_train"], data["x_test_id"], data["x_test_shift"]
    )
    y_train = data["y_train"]
    y_test_id = data["y_test_id"]
    y_test_shift = data["y_test_shift"]

    rho = 1.60
    l2 = 5e-3
    epochs = 900

    d = x_train.shape[1]
    w_probe = np.linspace(-0.35, 0.40, num=d, dtype=np.float64)

    nominal_fn = lambda w: nominal_loss_grad(w=w, x=x_train, y=y_train, l2=l2)
    robust_fn = lambda w: robust_loss_grad(w=w, x=x_train, y=y_train, rho=rho, l2=l2)

    grad_err_nominal = finite_diff_gradient_check(nominal_fn, w_probe)
    grad_err_robust = finite_diff_gradient_check(robust_fn, w_probe)
    print(f"gradient-check nominal max abs err: {grad_err_nominal:.3e}")
    print(f"gradient-check robust  max abs err: {grad_err_robust:.3e}")

    nominal = fit_gradient_descent(
        obj_grad_fn=nominal_fn,
        d=d,
        name="Nominal",
        epochs=epochs,
        lr=0.08,
    )
    robust = fit_gradient_descent(
        obj_grad_fn=robust_fn,
        d=d,
        name="Robust",
        epochs=epochs,
        lr=0.07,
    )

    nominal_train_mse = mse(x_train, y_train, nominal.w)
    robust_train_mse = mse(x_train, y_train, robust.w)
    nominal_id_mse = mse(x_test_id, y_test_id, nominal.w)
    robust_id_mse = mse(x_test_id, y_test_id, robust.w)
    nominal_shift_mse = mse(x_test_shift, y_test_shift, nominal.w)
    robust_shift_mse = mse(x_test_shift, y_test_shift, robust.w)

    nom_base, nom_adv, nom_delta = worst_case_residual_norm(
        x_train, y_train, nominal.w, rho
    )
    rob_base, rob_adv, rob_delta = worst_case_residual_norm(
        x_train, y_train, robust.w, rho
    )

    robust_obj_on_nominal = robust_objective_value(
        nominal.w, x_train, y_train, rho=rho, l2=l2
    )
    robust_obj_on_robust = robust_objective_value(
        robust.w, x_train, y_train, rho=rho, l2=l2
    )

    print("-" * 72)
    print(f"{'Model':<10} {'Train MSE':>12} {'ID MSE':>12} {'Shift MSE':>12}")
    print(
        f"{'Nominal':<10} {nominal_train_mse:>12.6f} {nominal_id_mse:>12.6f} {nominal_shift_mse:>12.6f}"
    )
    print(
        f"{'Robust':<10} {robust_train_mse:>12.6f} {robust_id_mse:>12.6f} {robust_shift_mse:>12.6f}"
    )
    print("-" * 72)
    print(
        f"nominal loss: {nominal.loss_history[0]:.6f} -> {nominal.loss_history[-1]:.6f}"
    )
    print(
        f"robust  loss: {robust.loss_history[0]:.6f} -> {robust.loss_history[-1]:.6f}"
    )
    print(
        f"robust objective on nominal solution: {robust_obj_on_nominal:.6f}\n"
        f"robust objective on robust  solution: {robust_obj_on_robust:.6f}"
    )
    print(
        f"worst-case train residual norm (nominal solution): {nom_base:.6f} -> {nom_adv:.6f} (||Delta||_F={nom_delta:.3f})"
    )
    print(
        f"worst-case train residual norm (robust  solution): {rob_base:.6f} -> {rob_adv:.6f} (||Delta||_F={rob_delta:.3f})"
    )
    print(
        f"shift MSE delta (robust - nominal): {robust_shift_mse - nominal_shift_mse:+.6f}"
    )

    # Basic health checks.
    assert grad_err_nominal < 1e-4, "Nominal gradient check failed."
    assert grad_err_robust < 1e-4, "Robust gradient check failed."
    assert nominal.loss_history[-1] < nominal.loss_history[0], "Nominal loss did not decrease."
    assert robust.loss_history[-1] < robust.loss_history[0], "Robust loss did not decrease."
    assert robust_obj_on_robust <= robust_obj_on_nominal + 1e-8, "Robust optimizer result is inconsistent."
    assert rob_adv <= nom_adv + 1e-6, "Robust solution should reduce worst-case residual norm in this setup."


if __name__ == "__main__":
    main()
