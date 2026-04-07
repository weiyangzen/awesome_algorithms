"""Physical pendulum MVP.

Model:
    I_p * theta_ddot + c * theta_dot + m * g * d * sin(theta) = 0

where I_p is the inertia about the pivot:
    I_p = I_cm + m * d^2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import ellipk


@dataclass(frozen=True)
class PhysicalPendulumConfig:
    """Physical and numerical parameters for a rigid-body pendulum."""

    mass: float = 1.2
    gravity: float = 9.81
    com_distance: float = 0.22
    inertia_cm: float = 0.075
    damping: float = 0.0
    theta0: float = 0.60
    omega0: float = 0.0
    t_start: float = 0.0
    t_end: float = 30.0
    num_points: int = 2400
    rtol: float = 1e-9
    atol: float = 1e-11


def pivot_inertia(cfg: PhysicalPendulumConfig) -> float:
    """Return inertia about the pivot via the parallel-axis theorem."""

    return cfg.inertia_cm + cfg.mass * cfg.com_distance**2


def pendulum_rhs(_t: float, y: np.ndarray, cfg: PhysicalPendulumConfig) -> np.ndarray:
    """RHS for y = [theta, omega]."""

    theta, omega = y
    i_p = pivot_inertia(cfg)
    restoring = (cfg.mass * cfg.gravity * cfg.com_distance / i_p) * np.sin(theta)
    damping_term = (cfg.damping / i_p) * omega
    theta_ddot = -restoring - damping_term
    return np.array([omega, theta_ddot], dtype=float)


def mechanical_energy(theta: np.ndarray, omega: np.ndarray, cfg: PhysicalPendulumConfig) -> np.ndarray:
    """Mechanical energy with zero potential at the bottom equilibrium."""

    i_p = pivot_inertia(cfg)
    kinetic = 0.5 * i_p * omega**2
    potential = cfg.mass * cfg.gravity * cfg.com_distance * (1.0 - np.cos(theta))
    return kinetic + potential


def estimate_upward_crossing_period(t: np.ndarray, theta: np.ndarray) -> tuple[float, int]:
    """Estimate oscillation period using upward zero crossings.

    Returns (mean_period, count_of_crossings_used).
    """

    crossings: list[float] = []
    for i in range(len(theta) - 1):
        y0 = theta[i]
        y1 = theta[i + 1]
        if y0 < 0.0 <= y1 and y1 != y0:
            alpha = -y0 / (y1 - y0)
            t_cross = t[i] + alpha * (t[i + 1] - t[i])
            crossings.append(float(t_cross))

    if len(crossings) < 2:
        return float("nan"), len(crossings)

    periods = np.diff(np.array(crossings, dtype=float))
    return float(np.mean(periods)), len(crossings)


def exact_period_finite_amplitude(cfg: PhysicalPendulumConfig, theta_amp: float) -> float:
    """Exact period for oscillatory motion using complete elliptic integral."""

    i_p = pivot_inertia(cfg)
    k = np.sin(0.5 * abs(theta_amp))
    m_param = float(k * k)
    return float(4.0 * np.sqrt(i_p / (cfg.mass * cfg.gravity * cfg.com_distance)) * ellipk(m_param))


def simulate(cfg: PhysicalPendulumConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Integrate the ODE and return trajectory + diagnostics."""

    if cfg.mass <= 0.0 or cfg.gravity <= 0.0 or cfg.com_distance <= 0.0:
        raise ValueError("mass/gravity/com_distance must be positive.")
    if pivot_inertia(cfg) <= 0.0:
        raise ValueError("pivot inertia must be positive.")
    if cfg.num_points < 10:
        raise ValueError("num_points must be >= 10.")

    t_eval = np.linspace(cfg.t_start, cfg.t_end, cfg.num_points)
    y0 = np.array([cfg.theta0, cfg.omega0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, y: pendulum_rhs(t, y, cfg),
        t_span=(cfg.t_start, cfg.t_end),
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
    energy = mechanical_energy(theta, omega, cfg)
    df = pd.DataFrame(
        {
            "t": sol.t,
            "theta_rad": theta,
            "theta_deg": np.rad2deg(theta),
            "omega_rad_s": omega,
            "energy_j": energy,
        }
    )

    e0 = float(energy[0])
    max_rel_energy_drift = float(np.max(np.abs((energy - e0) / max(1e-12, abs(e0)))))

    i_p = pivot_inertia(cfg)
    small_angle_period = float(2.0 * np.pi * np.sqrt(i_p / (cfg.mass * cfg.gravity * cfg.com_distance)))
    theta_amp_obs = float(np.max(np.abs(theta)))
    finite_amp_period = exact_period_finite_amplitude(cfg, theta_amp_obs)
    estimated_period, crossing_count = estimate_upward_crossing_period(sol.t, theta)

    rel_err_vs_exact = float(abs(estimated_period - finite_amp_period) / finite_amp_period)
    rel_err_vs_small = float(abs(estimated_period - small_angle_period) / small_angle_period)

    summary = {
        "pivot_inertia_kg_m2": i_p,
        "observed_amplitude_deg": float(np.rad2deg(theta_amp_obs)),
        "small_angle_period_s": small_angle_period,
        "finite_amplitude_period_s": finite_amp_period,
        "estimated_period_s": estimated_period,
        "period_rel_err_vs_exact": rel_err_vs_exact,
        "period_rel_err_vs_small_angle": rel_err_vs_small,
        "max_rel_energy_drift": max_rel_energy_drift,
        "upward_zero_crossings": float(crossing_count),
    }
    return df, summary


def main() -> None:
    cfg = PhysicalPendulumConfig()
    traj, summary = simulate(cfg)

    print("Physical Pendulum MVP")
    print(
        f"m={cfg.mass}, g={cfg.gravity}, d={cfg.com_distance}, "
        f"I_cm={cfg.inertia_cm}, I_p={summary['pivot_inertia_kg_m2']:.6f}"
    )
    print(
        f"theta0={cfg.theta0:.6f} rad ({np.rad2deg(cfg.theta0):.3f} deg), "
        f"omega0={cfg.omega0:.6f} rad/s, damping={cfg.damping}"
    )
    print(
        f"time_span=[{cfg.t_start}, {cfg.t_end}], num_points={cfg.num_points}, "
        f"rtol={cfg.rtol}, atol={cfg.atol}"
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "observed_amplitude_deg", "value": f"{summary['observed_amplitude_deg']:.6f}"},
            {"metric": "small_angle_period_s", "value": f"{summary['small_angle_period_s']:.8f}"},
            {"metric": "finite_amplitude_period_s", "value": f"{summary['finite_amplitude_period_s']:.8f}"},
            {"metric": "estimated_period_s", "value": f"{summary['estimated_period_s']:.8f}"},
            {"metric": "period_rel_err_vs_exact", "value": f"{summary['period_rel_err_vs_exact']:.3e}"},
            {
                "metric": "period_rel_err_vs_small_angle",
                "value": f"{summary['period_rel_err_vs_small_angle']:.3e}",
            },
            {"metric": "max_rel_energy_drift", "value": f"{summary['max_rel_energy_drift']:.3e}"},
            {"metric": "upward_zero_crossings", "value": f"{int(summary['upward_zero_crossings'])}"},
        ]
    )
    print("\nsummary:")
    print(summary_df.to_string(index=False))

    print("\ntrajectory_head:")
    print(traj.head(5).to_string(index=False))
    print("\ntrajectory_tail:")
    print(traj.tail(5).to_string(index=False))

    if summary["max_rel_energy_drift"] > 5e-7:
        raise AssertionError("Energy drift too large; verify integration settings.")
    if summary["period_rel_err_vs_exact"] > 2e-3:
        raise AssertionError("Estimated period deviates too much from finite-amplitude theory.")


if __name__ == "__main__":
    main()
