"""Variational Principle MVP.

We solve a fixed-endpoint least-action problem for a 1D harmonic oscillator:
    L(x, x_dot) = 0.5 * m * x_dot^2 - 0.5 * k * x^2

The continuous stationary-action condition yields Euler-Lagrange:
    m * x_ddot + k * x = 0

This demo discretizes the path and minimizes the discrete action with scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# Physical parameters (choose T < pi / omega to keep the action minimum branch simple)
MASS = 1.0
SPRING_K = 1.0
X0 = 1.0
XT = 0.2
T_FINAL = 1.2
N_GRID = 80
RNG_SEED = 42


def analytic_path(t: np.ndarray, m: float, k: float, x0: float, xt: float, t_final: float) -> np.ndarray:
    """Closed-form path satisfying m x'' + k x = 0 and the two boundary conditions."""
    omega = np.sqrt(k / m)
    sin_term = np.sin(omega * t_final)
    if np.isclose(sin_term, 0.0):
        raise ValueError("Choose T_FINAL so that sin(omega * T_FINAL) is not ~0.")

    a = x0
    b = (xt - x0 * np.cos(omega * t_final)) / sin_term
    return a * np.cos(omega * t) + b * np.sin(omega * t)


def stitch_path(internal: np.ndarray, x0: float, xt: float) -> np.ndarray:
    """Rebuild the full path [x0, internal..., xT]."""
    x = np.empty(internal.size + 2, dtype=float)
    x[0] = x0
    x[-1] = xt
    x[1:-1] = internal
    return x


def discrete_action(internal: np.ndarray, m: float, k: float, x0: float, xt: float, t_final: float) -> float:
    """Compute left-Riemann discrete action for a candidate path."""
    x = stitch_path(internal, x0, xt)
    dt = t_final / (x.size - 1)

    v = (x[1:] - x[:-1]) / dt
    kinetic = 0.5 * m * v**2
    potential = 0.5 * k * x[:-1] ** 2
    lagrangian = kinetic - potential

    return float(np.sum(lagrangian) * dt)


def discrete_el_residual(x: np.ndarray, m: float, k: float, t_final: float) -> np.ndarray:
    """Residual of the discrete Euler-Lagrange equation at interior nodes."""
    dt = t_final / (x.size - 1)
    # m * (x_{i+1} - 2x_i + x_{i-1}) / dt^2 + k * x_i = 0
    return m * (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt**2) + k * x[1:-1]


def optimize_path(
    m: float,
    k: float,
    x0: float,
    xt: float,
    t_final: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Optimize interior points to minimize action."""
    t = np.linspace(0.0, t_final, n_grid)
    linear_guess = np.linspace(x0, xt, n_grid)
    initial_internal = linear_guess[1:-1]

    result = minimize(
        fun=discrete_action,
        x0=initial_internal,
        args=(m, k, x0, xt, t_final),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    optimized_path = stitch_path(result.x, x0, xt)
    linear_action = discrete_action(initial_internal, m, k, x0, xt, t_final)
    optimized_action = float(result.fun)

    return t, linear_guess, optimized_path, linear_action, optimized_action


def perturbation_check(
    optimized_internal: np.ndarray,
    m: float,
    k: float,
    x0: float,
    xt: float,
    t_final: float,
    n_trials: int = 200,
    scale: float = 2e-2,
) -> float:
    """Return the fraction of random perturbations with larger action than optimized path."""
    rng = np.random.default_rng(RNG_SEED)
    s_star = discrete_action(optimized_internal, m, k, x0, xt, t_final)

    count_higher = 0
    for _ in range(n_trials):
        delta = rng.normal(0.0, scale, size=optimized_internal.shape)
        s_try = discrete_action(optimized_internal + delta, m, k, x0, xt, t_final)
        if s_try > s_star:
            count_higher += 1

    return count_higher / n_trials


def main() -> None:
    t, x_linear, x_opt, s_linear, s_opt = optimize_path(
        m=MASS,
        k=SPRING_K,
        x0=X0,
        xt=XT,
        t_final=T_FINAL,
        n_grid=N_GRID,
    )

    x_ref = analytic_path(t, MASS, SPRING_K, X0, XT, T_FINAL)
    max_abs_error = float(np.max(np.abs(x_opt - x_ref)))
    rms_error = float(np.sqrt(np.mean((x_opt - x_ref) ** 2)))

    residual = discrete_el_residual(x_opt, MASS, SPRING_K, T_FINAL)
    max_el_residual = float(np.max(np.abs(residual)))

    perturb_ratio = perturbation_check(
        optimized_internal=x_opt[1:-1],
        m=MASS,
        k=SPRING_K,
        x0=X0,
        xt=XT,
        t_final=T_FINAL,
    )

    sampled = np.linspace(0, N_GRID - 1, 8, dtype=int)
    table = pd.DataFrame(
        {
            "t": t[sampled],
            "x_linear": x_linear[sampled],
            "x_optimized": x_opt[sampled],
            "x_analytic": x_ref[sampled],
            "abs_err": np.abs(x_opt[sampled] - x_ref[sampled]),
        }
    )

    print("=== Variational Principle MVP (Harmonic Oscillator) ===")
    print(f"grid points             : {N_GRID}")
    print(f"boundary                : x(0)={X0:.3f}, x(T)={XT:.3f}, T={T_FINAL:.3f}")
    print(f"action(linear guess)    : {s_linear:.8f}")
    print(f"action(optimized path)  : {s_opt:.8f}")
    print(f"action improvement      : {s_linear - s_opt:.8f}")
    print(f"max |x_opt - x_analytic|: {max_abs_error:.6e}")
    print(f"rms |x_opt - x_analytic|: {rms_error:.6e}")
    print(f"max EL residual         : {max_el_residual:.6e}")
    print(f"perturbation higher-ratio (n=200): {perturb_ratio:.3f}")
    print("\nSampled trajectory comparison:")
    print(table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))


if __name__ == "__main__":
    main()
