"""Pendulum period formula MVP.

This script compares four period estimates for a simple pendulum:
1) Small-angle formula
2) Finite-amplitude series correction (up to theta0^4)
3) Elliptic-integral exact formula
4) Numerical ODE estimate from zero crossings
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import ellipk


@dataclass(frozen=True)
class PendulumConfig:
    """Physical and numerical configuration for the MVP."""

    length_m: float = 1.20
    gravity_mps2: float = 9.81
    theta0_rad: float = 0.90
    omega0_rad_s: float = 0.0
    t_start_s: float = 0.0
    t_end_s: float = 35.0
    num_points: int = 4200
    rtol: float = 1e-9
    atol: float = 1e-11


def small_angle_period(length_m: float, gravity_mps2: float) -> float:
    """Return T0 = 2*pi*sqrt(L/g)."""

    return float(2.0 * math.pi * math.sqrt(length_m / gravity_mps2))


def finite_amplitude_period_exact(theta_amp_rad: float, length_m: float, gravity_mps2: float) -> float:
    """Return exact oscillation period using complete elliptic integral.

    T = 4 * sqrt(L/g) * K(k^2), k = sin(theta_amp / 2)
    """

    k = math.sin(0.5 * abs(theta_amp_rad))
    m_param = float(k * k)
    return float(4.0 * math.sqrt(length_m / gravity_mps2) * ellipk(m_param))


def finite_amplitude_period_series(theta_amp_rad: float, length_m: float, gravity_mps2: float) -> float:
    """Return series-corrected period up to O(theta^4).

    T \approx T0 * (1 + theta^2/16 + 11*theta^4/3072)
    """

    t0 = small_angle_period(length_m, gravity_mps2)
    th = abs(theta_amp_rad)
    correction = 1.0 + (th**2) / 16.0 + 11.0 * (th**4) / 3072.0
    return float(t0 * correction)


def pendulum_rhs(_t: float, y: np.ndarray, cfg: PendulumConfig) -> np.ndarray:
    """RHS for y = [theta, omega], with theta'' + (g/L) sin(theta) = 0."""

    theta, omega = y
    theta_ddot = -(cfg.gravity_mps2 / cfg.length_m) * math.sin(theta)
    return np.array([omega, theta_ddot], dtype=float)


def specific_mechanical_energy(theta: np.ndarray, omega: np.ndarray, cfg: PendulumConfig) -> np.ndarray:
    """Return energy per unit mass for diagnostics."""

    kinetic = 0.5 * (cfg.length_m**2) * np.square(omega)
    potential = cfg.gravity_mps2 * cfg.length_m * (1.0 - np.cos(theta))
    return kinetic + potential


def estimate_period_upward_crossings(t: np.ndarray, theta: np.ndarray) -> tuple[float, int]:
    """Estimate period from upward zero crossings with linear interpolation."""

    crossings: list[float] = []
    for i in range(theta.size - 1):
        y0 = float(theta[i])
        y1 = float(theta[i + 1])
        if y0 < 0.0 <= y1 and y1 != y0:
            alpha = -y0 / (y1 - y0)
            t_cross = float(t[i] + alpha * (t[i + 1] - t[i]))
            crossings.append(t_cross)

    if len(crossings) < 2:
        return float("nan"), len(crossings)

    periods = np.diff(np.array(crossings, dtype=float))
    return float(np.mean(periods)), len(crossings)


def make_formula_comparison_table(length_m: float, gravity_mps2: float) -> pd.DataFrame:
    """Return a small amplitude sweep table for formula comparison."""

    amplitudes = np.array([0.05, 0.20, 0.50, 0.90, 1.20], dtype=float)
    t0 = small_angle_period(length_m, gravity_mps2)

    rows: list[dict[str, float]] = []
    for theta_amp in amplitudes:
        t_series = finite_amplitude_period_series(theta_amp, length_m, gravity_mps2)
        t_exact = finite_amplitude_period_exact(theta_amp, length_m, gravity_mps2)
        rows.append(
            {
                "theta_amp_deg": float(np.rad2deg(theta_amp)),
                "small_angle_period_s": t0,
                "series_period_s": t_series,
                "exact_period_s": t_exact,
                "small_vs_exact_rel_err": float(abs(t0 - t_exact) / t_exact),
                "series_vs_exact_rel_err": float(abs(t_series - t_exact) / t_exact),
            }
        )

    return pd.DataFrame(rows)


def simulate(cfg: PendulumConfig) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Integrate the nonlinear pendulum and return trajectory + metrics."""

    if cfg.length_m <= 0.0 or cfg.gravity_mps2 <= 0.0:
        raise ValueError("length_m and gravity_mps2 must be positive.")
    if cfg.num_points < 200:
        raise ValueError("num_points must be >= 200.")

    t_eval = np.linspace(cfg.t_start_s, cfg.t_end_s, cfg.num_points)
    y0 = np.array([cfg.theta0_rad, cfg.omega0_rad_s], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: pendulum_rhs(t, y, cfg),
        t_span=(cfg.t_start_s, cfg.t_end_s),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    theta = sol.y[0]
    omega = sol.y[1]
    energy = specific_mechanical_energy(theta, omega, cfg)

    traj = pd.DataFrame(
        {
            "t_s": sol.t,
            "theta_rad": theta,
            "theta_deg": np.rad2deg(theta),
            "omega_rad_s": omega,
            "energy_per_mass": energy,
        }
    )

    energy0 = float(energy[0])
    max_rel_energy_drift = float(np.max(np.abs((energy - energy0) / max(1e-12, abs(energy0)))))

    theta_amp_obs = float(np.max(np.abs(theta)))
    t_small = small_angle_period(cfg.length_m, cfg.gravity_mps2)
    t_series = finite_amplitude_period_series(theta_amp_obs, cfg.length_m, cfg.gravity_mps2)
    t_exact = finite_amplitude_period_exact(theta_amp_obs, cfg.length_m, cfg.gravity_mps2)
    t_est, crossing_count = estimate_period_upward_crossings(sol.t, theta)

    summary = {
        "length_m": float(cfg.length_m),
        "gravity_mps2": float(cfg.gravity_mps2),
        "theta0_deg": float(np.rad2deg(cfg.theta0_rad)),
        "observed_amp_deg": float(np.rad2deg(theta_amp_obs)),
        "small_angle_period_s": t_small,
        "series_period_s": t_series,
        "exact_period_s": t_exact,
        "estimated_period_s": float(t_est),
        "num_vs_exact_rel_err": float(abs(t_est - t_exact) / t_exact),
        "small_vs_exact_rel_err": float(abs(t_small - t_exact) / t_exact),
        "series_vs_exact_rel_err": float(abs(t_series - t_exact) / t_exact),
        "max_rel_energy_drift": max_rel_energy_drift,
        "upward_zero_crossings": float(crossing_count),
    }

    formula_table = make_formula_comparison_table(cfg.length_m, cfg.gravity_mps2)
    return traj, summary, formula_table


def main() -> None:
    cfg = PendulumConfig()
    traj, summary, formula_table = simulate(cfg)

    print("Pendulum Period Formula MVP")
    print(
        f"L={cfg.length_m:.4f} m, g={cfg.gravity_mps2:.4f} m/s^2, "
        f"theta0={cfg.theta0_rad:.6f} rad ({np.rad2deg(cfg.theta0_rad):.3f} deg), "
        f"omega0={cfg.omega0_rad_s:.6f} rad/s"
    )
    print(
        f"time=[{cfg.t_start_s:.1f}, {cfg.t_end_s:.1f}] s, num_points={cfg.num_points}, "
        f"rtol={cfg.rtol:.1e}, atol={cfg.atol:.1e}"
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "small_angle_period_s", "value": f"{summary['small_angle_period_s']:.8f}"},
            {"metric": "series_period_s", "value": f"{summary['series_period_s']:.8f}"},
            {"metric": "exact_period_s", "value": f"{summary['exact_period_s']:.8f}"},
            {"metric": "estimated_period_s", "value": f"{summary['estimated_period_s']:.8f}"},
            {"metric": "num_vs_exact_rel_err", "value": f"{summary['num_vs_exact_rel_err']:.3e}"},
            {"metric": "small_vs_exact_rel_err", "value": f"{summary['small_vs_exact_rel_err']:.3e}"},
            {"metric": "series_vs_exact_rel_err", "value": f"{summary['series_vs_exact_rel_err']:.3e}"},
            {"metric": "max_rel_energy_drift", "value": f"{summary['max_rel_energy_drift']:.3e}"},
            {"metric": "upward_zero_crossings", "value": f"{int(summary['upward_zero_crossings'])}"},
        ]
    )

    print("\nsummary:")
    print(summary_df.to_string(index=False))

    print("\nformula_sweep:")
    print(formula_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    print("\ntrajectory_head:")
    print(traj.head(6).to_string(index=False))
    print("\ntrajectory_tail:")
    print(traj.tail(6).to_string(index=False))

    if not np.isfinite(summary["estimated_period_s"]):
        raise AssertionError("Failed to estimate period from upward zero crossings.")
    if summary["num_vs_exact_rel_err"] > 2e-3:
        raise AssertionError("Numerical period estimate is too far from elliptic-integral exact period.")
    if summary["series_vs_exact_rel_err"] > summary["small_vs_exact_rel_err"]:
        raise AssertionError("Series correction should improve over small-angle formula at this amplitude.")
    if summary["max_rel_energy_drift"] > 5e-7:
        raise AssertionError("Energy drift too large; check integration tolerance.")


if __name__ == "__main__":
    main()
