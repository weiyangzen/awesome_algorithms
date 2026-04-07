"""Gauss-Newton MVP for nonlinear least squares.

Model used in this demo:
    y = a * exp(b * x) + c

The script generates synthetic data, checks the analytic Jacobian against a
finite-difference approximation, runs a damped Gauss-Newton solver, and prints
deterministic diagnostics. No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float, float, float, bool]


@dataclass
class GNResult:
    theta: np.ndarray
    history: List[HistoryItem]
    converged: bool
    reason: str


def exponential_model(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
    return a * np.exp(b * x) + c


def residuals(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return exponential_model(x, theta) - y


def jacobian(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a, b = float(theta[0]), float(theta[1])
    exp_bx = np.exp(b * x)

    j = np.empty((x.shape[0], 3), dtype=float)
    j[:, 0] = exp_bx
    j[:, 1] = a * x * exp_bx
    j[:, 2] = 1.0
    return j


def objective(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    r = residuals(x, y, theta)
    return 0.5 * float(np.mean(r * r))


def finite_difference_jacobian_error(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-6,
) -> float:
    analytic = jacobian(x, theta)
    approx = np.empty_like(analytic)

    for j in range(theta.size):
        direction = np.zeros_like(theta)
        direction[j] = eps
        plus = residuals(x, y=y, theta=theta + direction)
        minus = residuals(x, y=y, theta=theta - direction)
        approx[:, j] = (plus - minus) / (2.0 * eps)

    denom = np.linalg.norm(approx)
    if denom < 1e-15:
        denom = 1.0
    return float(np.linalg.norm(analytic - approx) / denom)


def solve_damped_normal_equation(j: np.ndarray, r: np.ndarray, damping: float) -> np.ndarray:
    n_params = j.shape[1]
    normal_mat = j.T @ j + damping * np.eye(n_params)
    normal_rhs = -(j.T @ r)

    try:
        return np.linalg.solve(normal_mat, normal_rhs)
    except np.linalg.LinAlgError:
        # Fallback keeps the MVP robust when normal_mat is near singular.
        return np.linalg.lstsq(normal_mat, normal_rhs, rcond=None)[0]


def gauss_newton(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    max_iters: int = 80,
    grad_tol: float = 1e-10,
    step_tol: float = 1e-10,
    loss_tol: float = 1e-12,
    damping_init: float = 1e-8,
    max_damping_trials: int = 12,
) -> GNResult:
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}.")
    if theta0.shape != (3,):
        raise ValueError(f"theta0 must have shape (3,), got {theta0.shape}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(theta0)):
        raise ValueError("x, y, theta0 must contain only finite numbers.")

    theta = theta0.astype(float).copy()
    damping = float(damping_init)
    history: List[HistoryItem] = []

    for it in range(1, max_iters + 1):
        r = residuals(x, y, theta)
        j = jacobian(x, theta)

        current_obj = 0.5 * float(np.mean(r * r))
        grad = (j.T @ r) / x.size
        grad_inf = float(np.max(np.abs(grad)))

        if grad_inf < grad_tol:
            history.append((it, current_obj, current_obj, 0.0, grad_inf, damping, True))
            return GNResult(theta=theta, history=history, converged=True, reason="grad_tol")

        accepted = False
        trial_obj = current_obj
        step_norm = 0.0
        damping_used = damping

        for _ in range(max_damping_trials):
            delta = solve_damped_normal_equation(j=j, r=r, damping=damping_used)
            step_norm = float(np.linalg.norm(delta))
            theta_trial = theta + delta
            trial_obj = objective(x, y, theta_trial)

            if np.isfinite(trial_obj) and trial_obj < current_obj:
                theta = theta_trial
                accepted = True
                break

            damping_used *= 10.0

        history.append((it, current_obj, trial_obj, step_norm, grad_inf, damping_used, accepted))

        if not accepted:
            return GNResult(theta=theta, history=history, converged=False, reason="no_descent_step")

        damping = max(damping_used * 0.3, 1e-14)

        if step_norm < step_tol * (np.linalg.norm(theta) + step_tol):
            return GNResult(theta=theta, history=history, converged=True, reason="step_tol")

        if abs(current_obj - trial_obj) < loss_tol * (1.0 + current_obj):
            return GNResult(theta=theta, history=history, converged=True, reason="loss_tol")

    return GNResult(theta=theta, history=history, converged=False, reason="max_iters")


def make_synthetic_data(
    seed: int = 2026,
    n_samples: int = 80,
    noise_std: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0, n_samples)

    theta_true = np.array([2.4, -1.35, 0.55], dtype=float)
    y_clean = exponential_model(x, theta_true)
    y_noisy = y_clean + noise_std * rng.normal(size=n_samples)

    return x, y_noisy, y_clean, theta_true


def print_history(history: List[HistoryItem], max_lines: int = 8) -> None:
    print("iter | obj_before         | obj_after          | step_norm         | grad_inf          | damping        | accepted")
    print("---------------------------------------------------------------------------------------------------------------")

    shown = min(len(history), max_lines)
    for i in range(shown):
        it, obj_before, obj_after, step_norm, grad_inf, damping, accepted = history[i]
        print(
            f"{it:4d} | {obj_before:18.10e} | {obj_after:18.10e} | {step_norm:16.8e} | "
            f"{grad_inf:16.8e} | {damping:14.6e} | {accepted}"
        )

    if len(history) > max_lines:
        omitted = len(history) - max_lines
        it, obj_before, obj_after, step_norm, grad_inf, damping, accepted = history[-1]
        print(f"... ({omitted} more iterations omitted)")
        print(
            f"{it:4d} | {obj_before:18.10e} | {obj_after:18.10e} | {step_norm:16.8e} | "
            f"{grad_inf:16.8e} | {damping:14.6e} | {accepted}  (last)"
        )


def main() -> None:
    x, y_noisy, y_clean, theta_true = make_synthetic_data(seed=2026)
    theta0 = np.array([1.0, -0.20, 0.10], dtype=float)

    jac_err = finite_difference_jacobian_error(x=x, y=y_noisy, theta=theta0, eps=1e-6)

    result = gauss_newton(
        x=x,
        y=y_noisy,
        theta0=theta0,
        max_iters=80,
        grad_tol=1e-10,
        step_tol=1e-10,
        loss_tol=1e-12,
        damping_init=1e-8,
        max_damping_trials=12,
    )

    y_fit = exponential_model(x, result.theta)
    rmse_noisy = float(np.sqrt(np.mean((y_fit - y_noisy) ** 2)))
    rmse_clean = float(np.sqrt(np.mean((y_fit - y_clean) ** 2)))
    theta_l2_error = float(np.linalg.norm(result.theta - theta_true))

    print("Gauss-Newton demo: nonlinear least squares on exponential model")
    print(f"dataset: n={x.size}, x_range=[{x.min():.2f}, {x.max():.2f}]")
    print(f"theta_true: {np.array2string(theta_true, precision=6, suppress_small=True)}")
    print(f"theta_init: {np.array2string(theta0, precision=6, suppress_small=True)}")
    print(f"finite-difference Jacobian relative error: {jac_err:.6e}")

    print_history(result.history, max_lines=10)

    print("\n=== Final Metrics ===")
    print(f"converged: {result.converged} (reason={result.reason})")
    print(f"iterations: {len(result.history)}")
    print(f"theta_hat: {np.array2string(result.theta, precision=6, suppress_small=True)}")
    print(f"theta_l2_error: {theta_l2_error:.6e}")
    print(f"rmse_noisy: {rmse_noisy:.6e}")
    print(f"rmse_clean: {rmse_clean:.6e}")

    pass_flag = (
        jac_err < 1e-6
        and result.converged
        and rmse_noisy < 0.06
        and rmse_clean < 0.04
        and theta_l2_error < 0.30
    )
    print(f"global checks pass: {pass_flag}")


if __name__ == "__main__":
    main()
