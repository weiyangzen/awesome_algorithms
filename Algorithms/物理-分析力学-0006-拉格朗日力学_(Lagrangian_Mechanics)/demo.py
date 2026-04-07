"""Lagrangian mechanics MVP: single pendulum from Euler-Lagrange equation.

This script demonstrates a minimal, source-traceable implementation of
Lagrangian mechanics using one generalized coordinate (theta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PendulumParams:
    """Physical and simulation parameters for a simple pendulum."""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.0
    theta0: float = 0.45
    omega0: float = 0.0
    t_end: float = 12.0
    num_steps: int = 2000


def lagrangian(theta: np.ndarray, omega: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Return L(theta, omega) = T - V."""

    inertia = params.mass * params.length * params.length
    kinetic = 0.5 * inertia * omega * omega
    potential = params.mass * params.gravity * params.length * (1.0 - np.cos(theta))
    return kinetic - potential


def mechanical_energy(theta: np.ndarray, omega: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Return total mechanical energy E(theta, omega) = T + V."""

    inertia = params.mass * params.length * params.length
    kinetic = 0.5 * inertia * omega * omega
    potential = params.mass * params.gravity * params.length * (1.0 - np.cos(theta))
    return kinetic + potential


def rhs(_t: float, y: np.ndarray, params: PendulumParams) -> np.ndarray:
    """State derivative for y = [theta, omega]."""

    theta, omega = y
    inertia = params.mass * params.length * params.length
    alpha = -(params.gravity / params.length) * np.sin(theta) - (params.damping / inertia) * omega
    return np.array([omega, alpha], dtype=float)


def estimate_period_from_zero_crossings(t: np.ndarray, theta: np.ndarray) -> float:
    """Estimate period from upward zero crossings by linear interpolation."""

    crossings: list[float] = []
    for i in range(len(theta) - 1):
        a = theta[i]
        b = theta[i + 1]
        if a <= 0.0 < b:
            ta = t[i]
            tb = t[i + 1]
            frac = -a / (b - a)
            crossings.append(ta + frac * (tb - ta))

    if len(crossings) < 2:
        return float("nan")

    diffs = np.diff(np.asarray(crossings, dtype=float))
    return float(np.mean(diffs))


def simulate(params: PendulumParams) -> dict[str, np.ndarray | float]:
    """Integrate pendulum dynamics and compute diagnostics."""

    t_eval = np.linspace(0.0, params.t_end, params.num_steps)
    y0 = np.array([params.theta0, params.omega0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(0.0, params.t_end),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t = sol.t
    theta = sol.y[0]
    omega = sol.y[1]

    inertia = params.mass * params.length * params.length
    l_val = lagrangian(theta, omega, params)
    energy = mechanical_energy(theta, omega, params)

    d_l_dtheta = -params.mass * params.gravity * params.length * np.sin(theta)
    d_l_domega = inertia * omega
    q_nonconservative = -params.damping * omega

    ddt_d_l_domega = np.gradient(d_l_domega, t)
    el_residual = ddt_d_l_domega - d_l_dtheta - q_nonconservative

    baseline = max(1.0, abs(float(energy[0])))
    energy_rel_drift = float(np.max(np.abs(energy - energy[0])) / baseline)
    max_theta_deg = float(np.rad2deg(np.max(np.abs(theta))))

    period_sim = estimate_period_from_zero_crossings(t, theta)
    period_small_angle = float(2.0 * np.pi * np.sqrt(params.length / params.gravity))
    period_rel_error = float(abs(period_sim - period_small_angle) / period_small_angle)

    return {
        "t": t,
        "theta": theta,
        "omega": omega,
        "lagrangian": l_val,
        "energy": energy,
        "energy_rel_drift": energy_rel_drift,
        "max_theta_deg": max_theta_deg,
        "period_sim": period_sim,
        "period_small_angle": period_small_angle,
        "period_rel_error": period_rel_error,
        "el_residual_rms": float(np.sqrt(np.mean(el_residual * el_residual))),
        "el_residual_max_abs": float(np.max(np.abs(el_residual))),
    }


def print_report(result: dict[str, np.ndarray | float], params: PendulumParams) -> None:
    """Print concise, non-interactive diagnostics."""

    summary = pd.DataFrame(
        [
            {"metric": "theta0_deg", "value": f"{np.rad2deg(params.theta0):.3f}"},
            {"metric": "max_abs_theta_deg", "value": f"{result['max_theta_deg']:.3f}"},
            {"metric": "period_sim_s", "value": f"{result['period_sim']:.6f}"},
            {"metric": "period_small_angle_s", "value": f"{result['period_small_angle']:.6f}"},
            {"metric": "period_rel_error", "value": f"{result['period_rel_error']:.4%}"},
            {"metric": "energy_rel_drift", "value": f"{result['energy_rel_drift']:.2e}"},
            {"metric": "el_residual_rms", "value": f"{result['el_residual_rms']:.2e}"},
            {"metric": "el_residual_max_abs", "value": f"{result['el_residual_max_abs']:.2e}"},
            {
                "metric": "L_range",
                "value": (
                    f"[{np.min(result['lagrangian']):.4f}, {np.max(result['lagrangian']):.4f}]"
                ),
            },
            {
                "metric": "E_range",
                "value": f"[{np.min(result['energy']):.4f}, {np.max(result['energy']):.4f}]",
            },
        ]
    )

    print("=== Lagrangian Mechanics MVP (Simple Pendulum) ===")
    print(
        "params:",
        {
            "m": params.mass,
            "l": params.length,
            "g": params.gravity,
            "c": params.damping,
            "theta0_deg": float(np.round(np.rad2deg(params.theta0), 3)),
            "omega0": params.omega0,
            "t_end": params.t_end,
            "num_steps": params.num_steps,
        },
    )
    print(summary.to_string(index=False))


def main() -> None:
    params = PendulumParams()
    result = simulate(params)
    print_report(result, params)

    if result["energy_rel_drift"] > 5e-4:
        raise AssertionError("Energy drift too large; check integration setup.")
    if result["period_rel_error"] > 0.03:
        raise AssertionError("Period deviates too much from small-angle estimate.")
    if result["el_residual_rms"] > 3e-2:
        raise AssertionError("Euler-Lagrange residual too large.")


if __name__ == "__main__":
    main()
