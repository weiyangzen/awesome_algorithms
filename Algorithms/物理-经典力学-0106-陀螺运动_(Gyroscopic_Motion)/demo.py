"""Gyroscopic motion MVP: symmetric heavy top with one fixed point.

This script integrates a classical gyroscope model using Euler angles
(theta, phi, psi) and checks core invariants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class GyroscopeParams:
    """Physical and simulation parameters for a symmetric heavy top."""

    i_perp: float = 0.02
    i_spin: float = 0.04
    mass: float = 0.5
    gravity: float = 9.81
    com_distance: float = 0.1
    theta0: float = np.deg2rad(20.0)
    theta_dot0: float = 0.25
    phi_dot0: float = 0.068
    psi_dot0: float = 180.0
    t_end: float = 8.0
    num_steps: int = 2000


EPS = 1e-9


def conserved_momenta(params: GyroscopeParams) -> tuple[float, float]:
    """Return conserved generalized momenta (p_phi, p_psi)."""

    theta0 = params.theta0
    b = params.i_spin * (params.psi_dot0 + params.phi_dot0 * np.cos(theta0))
    a = params.i_perp * (np.sin(theta0) ** 2) * params.phi_dot0 + b * np.cos(theta0)
    return a, b


def euler_rates(
    theta: np.ndarray | float,
    a_const: float,
    b_const: float,
    params: GyroscopeParams,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Compute (phi_dot, psi_dot) from current theta and conserved momenta."""

    sin_theta = np.sin(theta)
    sin2 = np.maximum(sin_theta * sin_theta, EPS)
    phi_dot = (a_const - b_const * np.cos(theta)) / (params.i_perp * sin2)
    psi_dot = b_const / params.i_spin - phi_dot * np.cos(theta)
    return phi_dot, psi_dot


def rhs(
    _t: float,
    y: np.ndarray,
    params: GyroscopeParams,
    a_const: float,
    b_const: float,
) -> np.ndarray:
    """State derivative for y = [theta, theta_dot, phi, psi]."""

    theta, theta_dot, _phi, _psi = y
    phi_dot, psi_dot = euler_rates(theta, a_const, b_const, params)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    theta_ddot = (
        params.i_perp * (phi_dot**2) * sin_theta * cos_theta
        - b_const * phi_dot * sin_theta
        + params.mass * params.gravity * params.com_distance * sin_theta
    ) / params.i_perp

    return np.array([theta_dot, theta_ddot, phi_dot, psi_dot], dtype=float)


def total_energy(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    phi_dot: np.ndarray,
    psi_dot: np.ndarray,
    params: GyroscopeParams,
) -> np.ndarray:
    """Mechanical energy T + V for the heavy symmetric top."""

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    kinetic = 0.5 * params.i_perp * (theta_dot**2 + (phi_dot**2) * (sin_theta**2))
    kinetic += 0.5 * params.i_spin * (psi_dot + phi_dot * cos_theta) ** 2
    potential = params.mass * params.gravity * params.com_distance * cos_theta
    return kinetic + potential


def simulate(params: GyroscopeParams) -> dict[str, np.ndarray | float]:
    """Run ODE integration and return trajectories + diagnostics."""

    a_const, b_const = conserved_momenta(params)
    t_eval = np.linspace(0.0, params.t_end, params.num_steps)
    y0 = np.array([params.theta0, params.theta_dot0, 0.0, 0.0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, a_const, b_const),
        t_span=(0.0, params.t_end),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    theta = sol.y[0]
    theta_dot = sol.y[1]
    phi = sol.y[2]
    psi = sol.y[3]

    phi_dot, psi_dot = euler_rates(theta, a_const, b_const, params)
    energy = total_energy(theta, theta_dot, phi_dot, psi_dot, params)

    spin_rate = psi_dot + phi_dot * np.cos(theta)
    p_psi = params.i_spin * spin_rate
    p_phi = params.i_perp * (np.sin(theta) ** 2) * phi_dot + p_psi * np.cos(theta)

    baseline = max(1.0, abs(energy[0]))
    energy_rel_drift = float(np.max(np.abs(energy - energy[0])) / baseline)
    ppsi_rel_drift = float(np.max(np.abs(p_psi - p_psi[0])) / max(1.0, abs(p_psi[0])))
    pphi_rel_drift = float(np.max(np.abs(p_phi - p_phi[0])) / max(1.0, abs(p_phi[0])))

    start_idx = int(0.3 * len(sol.t))
    mean_precession = float(np.mean(phi_dot[start_idx:]))
    steady_precession = float(
        params.mass * params.gravity * params.com_distance / (params.i_spin * np.mean(spin_rate[start_idx:]))
    )
    nutation_amplitude_deg = float(np.rad2deg(theta.max() - theta.min()))

    return {
        "t": sol.t,
        "theta": theta,
        "phi": phi,
        "psi": psi,
        "phi_dot": phi_dot,
        "psi_dot": psi_dot,
        "energy": energy,
        "spin_rate": spin_rate,
        "energy_rel_drift": energy_rel_drift,
        "ppsi_rel_drift": ppsi_rel_drift,
        "pphi_rel_drift": pphi_rel_drift,
        "mean_precession": mean_precession,
        "steady_precession": steady_precession,
        "nutation_amplitude_deg": nutation_amplitude_deg,
    }


def print_report(result: dict[str, np.ndarray | float], params: GyroscopeParams) -> None:
    """Print concise, non-interactive diagnostics."""

    theta = result["theta"]
    phi = result["phi"]
    spin_rate = result["spin_rate"]

    summary = pd.DataFrame(
        [
            {
                "metric": "theta_range_deg",
                "value": f"[{np.rad2deg(theta.min()):.3f}, {np.rad2deg(theta.max()):.3f}]",
            },
            {
                "metric": "nutation_amplitude_deg",
                "value": f"{result['nutation_amplitude_deg']:.3f}",
            },
            {
                "metric": "mean_precession_rad_s",
                "value": f"{result['mean_precession']:.6f}",
            },
            {
                "metric": "fast_spin_formula_rad_s",
                "value": f"{result['steady_precession']:.6f}",
            },
            {
                "metric": "final_phi_deg",
                "value": f"{np.rad2deg(phi[-1]):.3f}",
            },
            {
                "metric": "mean_spin_rad_s",
                "value": f"{np.mean(spin_rate):.3f}",
            },
            {
                "metric": "energy_rel_drift",
                "value": f"{result['energy_rel_drift']:.2e}",
            },
            {
                "metric": "p_psi_rel_drift",
                "value": f"{result['ppsi_rel_drift']:.2e}",
            },
            {
                "metric": "p_phi_rel_drift",
                "value": f"{result['pphi_rel_drift']:.2e}",
            },
        ]
    )

    print("=== Gyroscopic Motion MVP (Symmetric Heavy Top) ===")
    print(
        "params:",
        {
            "I_perp": params.i_perp,
            "I_spin": params.i_spin,
            "m": params.mass,
            "g": params.gravity,
            "l": params.com_distance,
            "theta0_deg": float(np.round(np.rad2deg(params.theta0), 3)),
            "theta_dot0": params.theta_dot0,
            "phi_dot0": params.phi_dot0,
            "psi_dot0": params.psi_dot0,
            "t_end": params.t_end,
        },
    )
    print(summary.to_string(index=False))


def main() -> None:
    params = GyroscopeParams()
    result = simulate(params)
    print_report(result, params)

    # Minimal sanity checks for this MVP (should hold for stable integration).
    if result["energy_rel_drift"] > 5e-4:
        raise AssertionError("Energy drift too large; check integration setup.")
    if result["mean_precession"] <= 0.0:
        raise AssertionError("Expected forward precession but got non-positive mean rate.")
    if result["nutation_amplitude_deg"] < 0.05:
        raise AssertionError("Nutation amplitude unexpectedly tiny; scenario may be degenerate.")


if __name__ == "__main__":
    main()
