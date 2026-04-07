"""Minimal runnable MVP for the Two-Body Problem.

This script solves the planar two-body problem in relative coordinates:
    r'' = -mu * r / |r|^3
and validates key physical invariants for an elliptical orbit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, r2_score


Array = np.ndarray


@dataclass(frozen=True)
class OrbitConfig:
    """Physical and numerical setup for one elliptic orbit simulation."""

    m1: float = 1.0
    m2: float = 0.2
    gravitational_constant: float = 1.0
    semi_major_axis: float = 2.0
    eccentricity: float = 0.35
    samples: int = 4001

    @property
    def total_mass(self) -> float:
        return self.m1 + self.m2

    @property
    def mu(self) -> float:
        return self.gravitational_constant * self.total_mass

    @property
    def orbital_period(self) -> float:
        # Kepler's third law for two-body relative motion.
        return 2.0 * np.pi * np.sqrt(self.semi_major_axis**3 / self.mu)


@dataclass
class OrbitResult:
    time: Array
    state: Array
    invariants: Dict[str, Array]
    modeled_radius: Array
    summary_metrics: Dict[str, float]


def make_periapsis_initial_state(config: OrbitConfig) -> Array:
    """Return state [x, y, vx, vy] at periapsis for a planar elliptic orbit."""
    if not (0.0 <= config.eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1 for an ellipse")

    e = config.eccentricity
    a = config.semi_major_axis
    mu = config.mu

    r_periapsis = a * (1.0 - e)
    v_periapsis = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    return np.array([r_periapsis, 0.0, 0.0, v_periapsis], dtype=np.float64)


def two_body_relative_rhs(_: float, state: Array, mu: float) -> Array:
    """Relative dynamics: r' = v, v' = -mu * r / |r|^3."""
    x, y, vx, vy = state
    r2 = x * x + y * y
    r = np.sqrt(r2)
    if r <= 1e-12:
        raise ValueError("radius is too close to zero; singular gravitational acceleration")

    inv_r3 = 1.0 / (r2 * r)
    ax = -mu * x * inv_r3
    ay = -mu * y * inv_r3
    return np.array([vx, vy, ax, ay], dtype=np.float64)


def integrate_relative_orbit(config: OrbitConfig) -> Tuple[Array, Array]:
    """Integrate one period of relative motion with high-accuracy ODE solver."""
    state0 = make_periapsis_initial_state(config)
    time = np.linspace(0.0, config.orbital_period, config.samples, dtype=np.float64)

    solution = solve_ivp(
        fun=lambda t, y: two_body_relative_rhs(t, y, config.mu),
        t_span=(float(time[0]), float(time[-1])),
        y0=state0,
        t_eval=time,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"integration failed: {solution.message}")

    state = np.asarray(solution.y.T, dtype=np.float64)
    return time, state


def compute_invariants(state: Array, mu: float) -> Dict[str, Array]:
    """Compute orbital invariants and geometric quantities along trajectory."""
    r_vec = state[:, 0:2]
    v_vec = state[:, 2:4]

    radius = np.linalg.norm(r_vec, axis=1)
    speed_sq = np.sum(v_vec * v_vec, axis=1)

    specific_energy = 0.5 * speed_sq - mu / radius
    specific_angular_momentum = r_vec[:, 0] * v_vec[:, 1] - r_vec[:, 1] * v_vec[:, 0]

    rv_dot = np.sum(r_vec * v_vec, axis=1)
    ecc_vec = ((speed_sq - mu / radius)[:, None] * r_vec - rv_dot[:, None] * v_vec) / mu
    eccentricity = np.linalg.norm(ecc_vec, axis=1)

    true_anomaly = np.arctan2(r_vec[:, 1], r_vec[:, 0])

    return {
        "radius": radius,
        "speed_sq": speed_sq,
        "specific_energy": specific_energy,
        "specific_angular_momentum": specific_angular_momentum,
        "ecc_vec_x": ecc_vec[:, 0],
        "ecc_vec_y": ecc_vec[:, 1],
        "eccentricity": eccentricity,
        "true_anomaly": true_anomaly,
    }


def project_to_barycentric(state: Array, m1: float, m2: float) -> Dict[str, Array]:
    """Map relative state to barycentric coordinates for COM/momentum checks."""
    total_mass = m1 + m2
    rel_pos = state[:, 0:2]
    rel_vel = state[:, 2:4]

    r1 = -(m2 / total_mass) * rel_pos
    r2 = (m1 / total_mass) * rel_pos
    v1 = -(m2 / total_mass) * rel_vel
    v2 = (m1 / total_mass) * rel_vel

    com = (m1 * r1 + m2 * r2) / total_mass
    total_momentum = m1 * v1 + m2 * v2

    return {
        "com": com,
        "total_momentum": total_momentum,
    }


def conic_radius_model(theta: Array, p: float, e: float, omega: float) -> Array:
    """Polar conic equation r = p / (1 + e cos(theta - omega))."""
    denom = 1.0 + e * np.cos(theta - omega)
    return p / denom


def summarize_metrics(
    config: OrbitConfig,
    state: Array,
    invariants: Dict[str, Array],
    modeled_radius: Array,
) -> Dict[str, float]:
    """Create numerical diagnostics used for assertions and final report."""
    radius = invariants["radius"]
    energy = invariants["specific_energy"]
    h = invariants["specific_angular_momentum"]
    ecc = invariants["eccentricity"]

    energy0 = float(energy[0])
    h0 = float(h[0])
    ecc0 = float(ecc[0])

    rel_energy_drift = float(np.max(np.abs((energy - energy0) / energy0)))
    rel_h_drift = float(np.max(np.abs((h - h0) / h0)))
    max_abs_ecc_drift = float(np.max(np.abs(ecc - ecc0)))

    expected_energy = -config.mu / (2.0 * config.semi_major_axis)
    expected_h = np.sqrt(config.mu * config.semi_major_axis * (1.0 - config.eccentricity**2))

    energy_model_error = float(abs(energy0 - expected_energy))
    h_model_error = float(abs(h0 - expected_h))

    radius_mae = float(mean_absolute_error(radius, modeled_radius))
    radius_r2 = float(r2_score(radius, modeled_radius))

    closure_pos_error = float(np.linalg.norm(state[-1, 0:2] - state[0, 0:2]))
    closure_vel_error = float(np.linalg.norm(state[-1, 2:4] - state[0, 2:4]))

    bary = project_to_barycentric(state, m1=config.m1, m2=config.m2)
    com_max_norm = float(np.max(np.linalg.norm(bary["com"], axis=1)))
    momentum_max_norm = float(np.max(np.linalg.norm(bary["total_momentum"], axis=1)))

    energy_t = torch.tensor(energy, dtype=torch.float64)
    torch_energy_span = float((torch.max(energy_t) - torch.min(energy_t)).item())

    return {
        "period": float(config.orbital_period),
        "initial_energy": energy0,
        "expected_energy": float(expected_energy),
        "initial_h": h0,
        "expected_h": float(expected_h),
        "rel_energy_drift": rel_energy_drift,
        "rel_h_drift": rel_h_drift,
        "max_abs_ecc_drift": max_abs_ecc_drift,
        "energy_model_error": energy_model_error,
        "h_model_error": h_model_error,
        "radius_mae": radius_mae,
        "radius_r2": radius_r2,
        "closure_pos_error": closure_pos_error,
        "closure_vel_error": closure_vel_error,
        "com_max_norm": com_max_norm,
        "momentum_max_norm": momentum_max_norm,
        "torch_energy_span": torch_energy_span,
    }


def build_sample_table(
    time: Array,
    state: Array,
    invariants: Dict[str, Array],
    modeled_radius: Array,
    rows: int = 9,
) -> pd.DataFrame:
    """Build a compact tabular snapshot for manual inspection."""
    if rows < 2:
        raise ValueError("rows must be at least 2")
    idx = np.linspace(0, time.shape[0] - 1, rows, dtype=int)

    return pd.DataFrame(
        {
            "t": time[idx],
            "x": state[idx, 0],
            "y": state[idx, 1],
            "vx": state[idx, 2],
            "vy": state[idx, 3],
            "r_numeric": invariants["radius"][idx],
            "r_model": modeled_radius[idx],
            "energy": invariants["specific_energy"][idx],
            "h": invariants["specific_angular_momentum"][idx],
            "ecc": invariants["eccentricity"][idx],
        }
    )


def run_two_body_demo() -> OrbitResult:
    """Execute one complete simulation + diagnostics pass."""
    config = OrbitConfig()
    time, state = integrate_relative_orbit(config)
    invariants = compute_invariants(state, mu=config.mu)

    h0 = float(invariants["specific_angular_momentum"][0])
    e0 = float(invariants["eccentricity"][0])
    e_vec0 = np.array([invariants["ecc_vec_x"][0], invariants["ecc_vec_y"][0]], dtype=np.float64)
    omega0 = float(np.arctan2(e_vec0[1], e_vec0[0]))
    p0 = (h0 * h0) / config.mu

    modeled_radius = conic_radius_model(
        theta=invariants["true_anomaly"],
        p=p0,
        e=e0,
        omega=omega0,
    )

    metrics = summarize_metrics(config, state, invariants, modeled_radius)

    # Deterministic pass/fail checks for MVP quality.
    assert metrics["rel_energy_drift"] < 1e-7, (
        f"energy drift too large: {metrics['rel_energy_drift']:.3e}"
    )
    assert metrics["rel_h_drift"] < 1e-7, f"angular momentum drift too large: {metrics['rel_h_drift']:.3e}"
    assert metrics["max_abs_ecc_drift"] < 3e-7, (
        f"eccentricity drift too large: {metrics['max_abs_ecc_drift']:.3e}"
    )
    assert metrics["radius_r2"] > 0.9999, f"conic fit r^2 too low: {metrics['radius_r2']:.6f}"
    assert metrics["radius_mae"] < 2e-4, f"conic fit MAE too large: {metrics['radius_mae']:.3e}"
    assert metrics["closure_pos_error"] < 2e-3, (
        f"orbit closure position error too large: {metrics['closure_pos_error']:.3e}"
    )
    assert metrics["closure_vel_error"] < 2e-3, (
        f"orbit closure velocity error too large: {metrics['closure_vel_error']:.3e}"
    )
    assert metrics["com_max_norm"] < 1e-12, f"COM drift too large: {metrics['com_max_norm']:.3e}"
    assert metrics["momentum_max_norm"] < 1e-12, (
        f"total momentum drift too large: {metrics['momentum_max_norm']:.3e}"
    )

    return OrbitResult(
        time=time,
        state=state,
        invariants=invariants,
        modeled_radius=modeled_radius,
        summary_metrics=metrics,
    )


def main() -> None:
    result = run_two_body_demo()

    print("=== Two-Body Problem MVP ===")
    for key, value in result.summary_metrics.items():
        print(f"{key:>22s}: {value:.12e}")

    table = build_sample_table(
        time=result.time,
        state=result.state,
        invariants=result.invariants,
        modeled_radius=result.modeled_radius,
        rows=9,
    )
    print("\n=== Orbit Snapshot (9 points) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
