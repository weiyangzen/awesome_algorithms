"""Principle of Least Action MVP using a fixed-endpoint harmonic oscillator path.

This script solves a boundary-value problem by minimizing a discretized action
functional directly over interior path nodes, then validates the result against
an analytic trajectory and perturbation tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ActionParams:
    """Physical and numerical parameters for the least-action demo."""

    mass: float = 1.0
    spring_k: float = 1.0
    x_start: float = 0.2
    x_end: float = -0.1
    t_end: float = 1.2
    num_points: int = 180
    maxiter: int = 2000
    random_trials: int = 256
    perturb_scale: float = 0.015
    random_seed: int = 7


def time_grid(params: ActionParams) -> np.ndarray:
    """Uniform time grid."""

    return np.linspace(0.0, params.t_end, params.num_points)


def build_path(interior: np.ndarray, params: ActionParams) -> np.ndarray:
    """Construct full path x(t) with fixed endpoints and free interior nodes."""

    x = np.empty(params.num_points, dtype=float)
    x[0] = params.x_start
    x[-1] = params.x_end
    x[1:-1] = interior
    return x


def action_and_gradient(interior: np.ndarray, params: ActionParams, dt: float) -> tuple[float, np.ndarray]:
    """Return discretized action and gradient w.r.t. interior path nodes.

    Discretization on intervals [i, i+1]:
      S = sum_i 0.5*m*(dx_i^2/dt) - 0.25*k*dt*(x_i^2 + x_{i+1}^2)
    where dx_i = x_{i+1} - x_i.
    """

    x = build_path(interior, params)
    dx = x[1:] - x[:-1]

    kinetic_part = 0.5 * params.mass * (dx * dx) / dt
    potential_part = 0.25 * params.spring_k * dt * (x[:-1] * x[:-1] + x[1:] * x[1:])
    action = float(np.sum(kinetic_part - potential_part))

    grad_full = np.zeros_like(x)
    grad_left = (params.mass / dt) * (x[:-1] - x[1:]) - 0.5 * params.spring_k * dt * x[:-1]
    grad_right = (params.mass / dt) * (x[1:] - x[:-1]) - 0.5 * params.spring_k * dt * x[1:]

    grad_full[:-1] += grad_left
    grad_full[1:] += grad_right

    return action, grad_full[1:-1]


def analytic_harmonic_path(params: ActionParams, t: np.ndarray) -> np.ndarray:
    """Closed-form path for x'' + omega^2 x = 0 with fixed endpoints."""

    omega = np.sqrt(params.spring_k / params.mass)
    sin_term = np.sin(omega * params.t_end)
    if abs(sin_term) < 1e-8:
        raise ValueError("sin(omega * T) is too small; choose different t_end or parameters.")

    coeff_b = (params.x_end - params.x_start * np.cos(omega * params.t_end)) / sin_term
    return params.x_start * np.cos(omega * t) + coeff_b * np.sin(omega * t)


def optimize_path(params: ActionParams) -> dict[str, np.ndarray | float | int | bool | str]:
    """Minimize action over interior nodes and compute diagnostics."""

    t = time_grid(params)
    dt = float(t[1] - t[0])

    interior_init = np.linspace(params.x_start, params.x_end, params.num_points)[1:-1]

    result = minimize(
        fun=lambda z: action_and_gradient(z, params, dt)[0],
        x0=interior_init,
        jac=lambda z: action_and_gradient(z, params, dt)[1],
        method="L-BFGS-B",
        options={"maxiter": params.maxiter, "ftol": 1e-15, "gtol": 1e-12, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x_opt = build_path(result.x, params)
    action_opt, grad_opt = action_and_gradient(result.x, params, dt)

    x_exact = analytic_harmonic_path(params, t)
    err = x_opt - x_exact
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs_err = float(np.max(np.abs(err)))

    rng = np.random.default_rng(params.random_seed)
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    deltas = []
    for _ in range(params.random_trials):
        raw = rng.standard_normal(result.x.shape)
        smooth = np.convolve(raw, kernel, mode="same")
        perturbed = result.x + params.perturb_scale * smooth
        perturbed_action, _ = action_and_gradient(perturbed, params, dt)
        deltas.append(perturbed_action - action_opt)

    delta_array = np.asarray(deltas, dtype=float)

    return {
        "t": t,
        "x_opt": x_opt,
        "x_exact": x_exact,
        "action_opt": action_opt,
        "residual_inf": float(np.max(np.abs(grad_opt))),
        "residual_rms": float(np.sqrt(np.mean(grad_opt * grad_opt))),
        "rmse_vs_exact": rmse,
        "max_abs_err_vs_exact": max_abs_err,
        "delta_action_min": float(np.min(delta_array)),
        "delta_action_mean": float(np.mean(delta_array)),
        "delta_action_max": float(np.max(delta_array)),
        "positive_delta_ratio": float(np.mean(delta_array > 0.0)),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "status": str(result.message),
        "success": bool(result.success),
    }


def print_report(metrics: dict[str, np.ndarray | float | int | bool | str], params: ActionParams) -> None:
    """Print concise diagnostics for non-interactive execution."""

    summary = pd.DataFrame(
        [
            {"metric": "optimizer_success", "value": str(metrics["success"])},
            {"metric": "optimizer_status", "value": str(metrics["status"])},
            {"metric": "nit", "value": f"{metrics['nit']}"},
            {"metric": "nfev", "value": f"{metrics['nfev']}"},
            {"metric": "action_opt", "value": f"{metrics['action_opt']:.8f}"},
            {"metric": "residual_inf", "value": f"{metrics['residual_inf']:.3e}"},
            {"metric": "residual_rms", "value": f"{metrics['residual_rms']:.3e}"},
            {"metric": "rmse_vs_exact", "value": f"{metrics['rmse_vs_exact']:.3e}"},
            {"metric": "max_abs_err_vs_exact", "value": f"{metrics['max_abs_err_vs_exact']:.3e}"},
            {"metric": "delta_action_min", "value": f"{metrics['delta_action_min']:.6f}"},
            {"metric": "delta_action_mean", "value": f"{metrics['delta_action_mean']:.6f}"},
            {"metric": "delta_action_max", "value": f"{metrics['delta_action_max']:.6f}"},
            {"metric": "positive_delta_ratio", "value": f"{metrics['positive_delta_ratio']:.2%}"},
        ]
    )

    print("=== Principle of Least Action MVP (Harmonic Oscillator BVP) ===")
    print(
        "params:",
        {
            "m": params.mass,
            "k": params.spring_k,
            "x_start": params.x_start,
            "x_end": params.x_end,
            "t_end": params.t_end,
            "num_points": params.num_points,
            "random_trials": params.random_trials,
            "perturb_scale": params.perturb_scale,
        },
    )
    print(summary.to_string(index=False))


def main() -> None:
    params = ActionParams()
    metrics = optimize_path(params)
    print_report(metrics, params)

    if metrics["residual_inf"] > 1e-4:
        raise AssertionError("Discrete Euler-Lagrange residual is too large.")
    if metrics["rmse_vs_exact"] > 5e-4:
        raise AssertionError("Optimized path deviates too much from analytic solution.")
    if metrics["positive_delta_ratio"] < 0.99:
        raise AssertionError("Perturbation check failed: action did not consistently increase.")
    if metrics["delta_action_min"] <= 0.0:
        raise AssertionError("Found perturbation with non-positive action increment.")


if __name__ == "__main__":
    main()
