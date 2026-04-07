"""Minimal runnable MVP for London Equations.

This script demonstrates:
1) Second London equation + Maxwell -> Meissner screening profile B(x)
2) Estimation of London penetration depth lambda_L from synthetic noisy data
3) First London equation dynamics dJ/dt = (n_s e^2 / m) E for a pulsed electric field
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class LondonConfig:
    # True penetration depth used to generate synthetic "experiment".
    lambda_l: float = 45e-9  # m
    b0_surface: float = 0.08  # T
    domain_factor: float = 10.0  # L = domain_factor * lambda_l
    n_grid: int = 240

    # Synthetic measurement noise for B(x)
    noise_std: float = 0.0015  # T
    random_seed: int = 42

    # First London equation dynamics
    e_pulse: float = 3e-5  # V/m
    dt: float = 2e-12  # s
    n_steps: int = 2400
    pulse_steps: int = 480


def london_superfluid_density(lambda_l: float) -> float:
    """n_s from lambda_L^2 = m/(mu0 n_s e^2)."""
    return constants.m_e / (constants.mu_0 * constants.e**2 * lambda_l**2)


def meissner_profile_exp(x: np.ndarray, b0: float, lambda_l: float) -> np.ndarray:
    """Semi-infinite analytic Meissner profile B(x) = B0 exp(-x/lambda)."""
    return b0 * np.exp(-x / lambda_l)


def solve_second_london_fd(
    *,
    lambda_l: float,
    b0: float,
    length: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve B'' = B/lambda^2 by finite differences with two Dirichlet boundaries.

    Boundaries:
    B(0) = b0
    B(L) = b0 * exp(-L/lambda)

    This makes the semi-infinite analytic profile compatible with the finite domain.
    """
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3")

    x = np.linspace(0.0, length, n_grid)
    dx = x[1] - x[0]

    b_left = b0
    b_right = float(meissner_profile_exp(np.array([length]), b0, lambda_l)[0])

    n_int = n_grid - 2
    main = np.full(n_int, -2.0 - (dx**2) / (lambda_l**2), dtype=float)
    off = np.full(n_int - 1, 1.0, dtype=float)

    a = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
    rhs = np.zeros(n_int, dtype=float)
    rhs[0] -= b_left
    rhs[-1] -= b_right

    b_int = np.linalg.solve(a, rhs)

    b = np.empty(n_grid, dtype=float)
    b[0] = b_left
    b[-1] = b_right
    b[1:-1] = b_int
    return x, b


def estimate_lambda_from_data(
    x: np.ndarray,
    b_obs: np.ndarray,
    *,
    b0_guess: float,
    lambda_guess: float,
) -> tuple[float, float]:
    """Fit B(x) = B0 exp(-x/lambda) with scipy.optimize.curve_fit."""
    popt, _ = curve_fit(
        meissner_profile_exp,
        x,
        b_obs,
        p0=(b0_guess, lambda_guess),
        bounds=([0.0, 1e-10], [1.0, 1e-5]),
        maxfev=20000,
    )
    b0_hat, lambda_hat = float(popt[0]), float(popt[1])
    return b0_hat, lambda_hat


def second_london_residual(
    x: np.ndarray,
    b: np.ndarray,
    lambda_l: float,
) -> float:
    """Relative residual of dJ/dx = -B/(mu0 lambda^2) in 1D.

    With B = B_z(x), J = J_y(x), Maxwell gives J = -(1/mu0) dB/dx.
    """
    d_b_dx = np.gradient(b, x)
    j = -(1.0 / constants.mu_0) * d_b_dx
    d_j_dx = np.gradient(j, x)

    rhs = -b / (constants.mu_0 * lambda_l**2)

    interior = slice(2, -2)  # reduce edge finite-difference artifacts
    num = np.linalg.norm(d_j_dx[interior] - rhs[interior])
    den = np.linalg.norm(rhs[interior]) + 1e-18
    return float(num / den)


def simulate_first_london_current(
    *,
    lambda_l: float,
    e_pulse: float,
    dt: float,
    n_steps: int,
    pulse_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Euler integration of dJ/dt = alpha * E(t), alpha = n_s e^2 / m."""
    if pulse_steps >= n_steps:
        raise ValueError("pulse_steps must be < n_steps")

    ns = london_superfluid_density(lambda_l)
    alpha = ns * constants.e**2 / constants.m_e

    t = np.arange(n_steps, dtype=float) * dt
    e_t = np.zeros(n_steps, dtype=float)
    e_t[:pulse_steps] = e_pulse

    j = np.zeros(n_steps, dtype=float)
    for k in range(1, n_steps):
        j[k] = j[k - 1] + alpha * e_t[k - 1] * dt

    j_post = j[pulse_steps:]
    drift_span = float(np.max(j_post) - np.min(j_post))
    j_end = float(j[-1])
    return t, e_t, j, drift_span, j_end


def main() -> None:
    cfg = LondonConfig()
    rng = np.random.default_rng(cfg.random_seed)

    length = cfg.domain_factor * cfg.lambda_l

    # 1) Second London equation numerical profile
    x, b_fd = solve_second_london_fd(
        lambda_l=cfg.lambda_l,
        b0=cfg.b0_surface,
        length=length,
        n_grid=cfg.n_grid,
    )
    b_ref = meissner_profile_exp(x, cfg.b0_surface, cfg.lambda_l)

    profile_rel_err = float(np.linalg.norm(b_fd - b_ref) / (np.linalg.norm(b_ref) + 1e-18))
    second_eq_rel_res = second_london_residual(x, b_fd, cfg.lambda_l)

    # 2) Fit lambda from noisy synthetic measurements
    b_obs = b_ref + rng.normal(0.0, cfg.noise_std, size=x.size)
    b0_hat, lambda_hat = estimate_lambda_from_data(
        x,
        b_obs,
        b0_guess=cfg.b0_surface * 0.9,
        lambda_guess=cfg.lambda_l * 1.2,
    )
    lambda_rel_err = abs(lambda_hat - cfg.lambda_l) / cfg.lambda_l

    # 3) First London equation current dynamics
    _, _, j_t, drift_span, j_end = simulate_first_london_current(
        lambda_l=cfg.lambda_l,
        e_pulse=cfg.e_pulse,
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        pulse_steps=cfg.pulse_steps,
    )

    # Sanity checks: robust but meaningful
    assert profile_rel_err < 2.5e-3, f"FD profile error too large: {profile_rel_err:.3e}"
    assert second_eq_rel_res < 4.0e-2, f"Second London residual too large: {second_eq_rel_res:.3e}"
    assert lambda_rel_err < 0.08, f"Lambda fit relative error too large: {lambda_rel_err:.3e}"
    assert drift_span < 1e-9 * (abs(j_end) + 1.0), f"Post-pulse drift too large: {drift_span:.3e}"

    ns = london_superfluid_density(cfg.lambda_l)

    summary = pd.DataFrame(
        [
            {
                "metric": "FD_vs_analytic_profile_rel_error",
                "value": profile_rel_err,
                "unit": "1",
            },
            {
                "metric": "Second_London_relative_residual",
                "value": second_eq_rel_res,
                "unit": "1",
            },
            {
                "metric": "lambda_true_m",
                "value": cfg.lambda_l,
                "unit": "m",
            },
            {
                "metric": "lambda_fitted_m",
                "value": lambda_hat,
                "unit": "m",
            },
            {
                "metric": "lambda_fit_relative_error",
                "value": lambda_rel_err,
                "unit": "1",
            },
            {
                "metric": "B0_fitted_T",
                "value": b0_hat,
                "unit": "T",
            },
            {
                "metric": "n_s_from_lambda",
                "value": ns,
                "unit": "m^-3",
            },
            {
                "metric": "J_end_after_pulse",
                "value": j_end,
                "unit": "A/m^2",
            },
            {
                "metric": "J_post_pulse_drift_span",
                "value": drift_span,
                "unit": "A/m^2",
            },
        ]
    )

    sample_idx = np.linspace(0, len(x) - 1, 8, dtype=int)
    profile_table = pd.DataFrame(
        {
            "x_nm": x[sample_idx] * 1e9,
            "B_fd_T": b_fd[sample_idx],
            "B_analytic_T": b_ref[sample_idx],
            "abs_diff_T": np.abs(b_fd[sample_idx] - b_ref[sample_idx]),
        }
    )

    print("=== London Equations MVP ===")
    print(f"lambda_true = {cfg.lambda_l:.3e} m, domain_length = {length:.3e} m")
    print("\n[Summary Metrics]")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("\n[Profile Sample: finite-difference vs analytic]")
    print(profile_table.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
