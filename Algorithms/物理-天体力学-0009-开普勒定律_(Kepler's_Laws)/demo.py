"""Minimal runnable MVP for Kepler's Laws.

The script demonstrates all three Kepler laws in a planar two-body setting:
1) First law: orbit is an ellipse with the central body at one focus.
2) Second law: equal areas are swept in equal times.
3) Third law: T^2 / a^3 is approximately constant for the same gravitational parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, r2_score


Array = np.ndarray


@dataclass(frozen=True)
class OrbitConfig:
    """Physical and numerical configuration for one elliptic orbit."""

    m1: float = 1.0
    m2: float = 0.2
    gravitational_constant: float = 1.0
    semi_major_axis: float = 2.0
    eccentricity: float = 0.35
    samples: int = 4001

    @property
    def mu(self) -> float:
        return self.gravitational_constant * (self.m1 + self.m2)

    @property
    def orbital_period_theory(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.semi_major_axis**3 / self.mu)


@dataclass
class FirstLawResult:
    invariants: Dict[str, Array]
    modeled_radius: Array
    metrics: Dict[str, float]


@dataclass
class SecondLawResult:
    segment_table: pd.DataFrame
    metrics: Dict[str, float]


@dataclass
class ThirdLawResult:
    period_table: pd.DataFrame
    metrics: Dict[str, float]


def make_periapsis_initial_state(config: OrbitConfig) -> Array:
    """Return [x, y, vx, vy] at periapsis for an elliptic orbit."""
    if not (0.0 <= config.eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1 for an ellipse")

    a = config.semi_major_axis
    e = config.eccentricity
    mu = config.mu

    r_periapsis = a * (1.0 - e)
    v_periapsis = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    return np.array([r_periapsis, 0.0, 0.0, v_periapsis], dtype=np.float64)


def two_body_rhs(_: float, state: Array, mu: float) -> Array:
    """Relative two-body ODE in planar form."""
    x, y, vx, vy = state
    r2 = x * x + y * y
    r = np.sqrt(r2)
    if r <= 1e-12:
        raise ValueError("radius too small; singular gravity acceleration")

    inv_r3 = 1.0 / (r2 * r)
    ax = -mu * x * inv_r3
    ay = -mu * y * inv_r3
    return np.array([vx, vy, ax, ay], dtype=np.float64)


def integrate_sampled_orbit(config: OrbitConfig, periods: float = 1.0) -> Tuple[Array, Array]:
    """Integrate orbit and return sampled trajectory for diagnostics."""
    if periods <= 0.0:
        raise ValueError("periods must be positive")

    state0 = make_periapsis_initial_state(config)
    t_end = periods * config.orbital_period_theory

    n_points = max(2, int(round(config.samples * periods)))
    time = np.linspace(0.0, t_end, n_points, dtype=np.float64)

    solution = solve_ivp(
        fun=lambda t, y: two_body_rhs(t, y, config.mu),
        t_span=(float(time[0]), float(time[-1])),
        y0=state0,
        t_eval=time,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"orbit integration failed: {solution.message}")

    state = np.asarray(solution.y.T, dtype=np.float64)
    return time, state


def compute_invariants(state: Array, mu: float) -> Dict[str, Array]:
    """Compute geometric and dynamical invariants along trajectory."""
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
        "specific_energy": specific_energy,
        "specific_angular_momentum": specific_angular_momentum,
        "ecc_vec_x": ecc_vec[:, 0],
        "ecc_vec_y": ecc_vec[:, 1],
        "eccentricity": eccentricity,
        "true_anomaly": true_anomaly,
    }


def conic_radius_model(theta: Array, p: float, e: float, omega: float) -> Array:
    """Focus-centered conic equation: r = p / (1 + e cos(theta - omega))."""
    denom = 1.0 + e * np.cos(theta - omega)
    return p / denom


def evaluate_first_law(config: OrbitConfig, state: Array) -> FirstLawResult:
    """Validate Kepler's first law using conic consistency and invariants."""
    inv = compute_invariants(state, mu=config.mu)

    h0 = float(inv["specific_angular_momentum"][0])
    e_vec0 = np.array([inv["ecc_vec_x"][0], inv["ecc_vec_y"][0]], dtype=np.float64)
    e0 = float(np.linalg.norm(e_vec0))
    omega0 = float(np.arctan2(e_vec0[1], e_vec0[0]))
    p0 = (h0 * h0) / config.mu

    modeled_radius = conic_radius_model(inv["true_anomaly"], p=p0, e=e0, omega=omega0)

    radius = inv["radius"]
    energy = inv["specific_energy"]
    eccentricity = inv["eccentricity"]

    energy0 = float(energy[0])
    expected_energy = -config.mu / (2.0 * config.semi_major_axis)

    rel_energy_drift = float(np.max(np.abs((energy - energy0) / energy0)))
    max_abs_ecc_drift = float(np.max(np.abs(eccentricity - eccentricity[0])))

    metrics = {
        "initial_eccentricity": e0,
        "initial_semi_latus_rectum": float(p0),
        "initial_argument_of_periapsis": omega0,
        "conic_radius_mae": float(mean_absolute_error(radius, modeled_radius)),
        "conic_radius_r2": float(r2_score(radius, modeled_radius)),
        "rel_energy_drift": rel_energy_drift,
        "energy_model_error": float(abs(energy0 - expected_energy)),
        "max_abs_ecc_drift": max_abs_ecc_drift,
    }

    return FirstLawResult(invariants=inv, modeled_radius=modeled_radius, metrics=metrics)


def evaluate_second_law(time: Array, state: Array, h_reference: float) -> SecondLawResult:
    """Validate Kepler's second law via equal-time area sweep rates."""
    dt = np.diff(time)
    x = state[:, 0]
    y = state[:, 1]

    cross = x[:-1] * y[1:] - y[:-1] * x[1:]
    swept_area = 0.5 * np.abs(cross)
    area_rate = swept_area / dt

    expected_rate = 0.5 * abs(h_reference)
    rel_dev = np.abs((area_rate - expected_rate) / expected_rate)

    area_rate_t = torch.tensor(area_rate, dtype=torch.float64)
    torch_span = float((torch.max(area_rate_t) - torch.min(area_rate_t)).item())

    metrics = {
        "expected_area_rate": float(expected_rate),
        "mean_area_rate": float(np.mean(area_rate)),
        "max_rel_area_rate_dev": float(np.max(rel_dev)),
        "area_rate_cv": float(np.std(area_rate) / np.mean(area_rate)),
        "torch_area_rate_span": torch_span,
    }

    n_samples = min(8, area_rate.shape[0])
    idx = np.linspace(0, area_rate.shape[0] - 1, n_samples, dtype=int)
    segment_table = pd.DataFrame(
        {
            "t_start": time[idx],
            "t_end": time[idx + 1],
            "delta_t": dt[idx],
            "swept_area": swept_area[idx],
            "area_rate": area_rate[idx],
            "rel_dev": rel_dev[idx],
        }
    )

    return SecondLawResult(segment_table=segment_table, metrics=metrics)


def estimate_period_by_event(config: OrbitConfig) -> float:
    """Estimate one orbital period by detecting positive y-crossing events."""
    state0 = make_periapsis_initial_state(config)

    def y_crossing(_: float, state: Array) -> float:
        return float(state[1])

    y_crossing.direction = 1.0  # type: ignore[attr-defined]
    y_crossing.terminal = False  # type: ignore[attr-defined]

    t_end = 1.6 * config.orbital_period_theory
    solution = solve_ivp(
        fun=lambda t, y: two_body_rhs(t, y, config.mu),
        t_span=(0.0, t_end),
        y0=state0,
        events=y_crossing,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=config.orbital_period_theory / 200.0,
    )
    if not solution.success:
        raise RuntimeError(f"period estimation integration failed: {solution.message}")

    event_times = np.asarray(solution.t_events[0], dtype=np.float64)
    candidates = event_times[event_times > 1e-8]
    if candidates.size == 0:
        raise RuntimeError("failed to capture a full-period event crossing")

    return float(candidates[0])


def evaluate_third_law(base_config: OrbitConfig) -> ThirdLawResult:
    """Validate Kepler's third law across multiple semi-major axes."""
    a_values = np.array([0.9, 1.3, 1.8, 2.5, 3.3], dtype=np.float64)

    period_estimates = []
    period_theory = []
    constants = []

    for a in a_values:
        cfg = replace(base_config, semi_major_axis=float(a), eccentricity=0.2)
        t_est = estimate_period_by_event(cfg)
        period_estimates.append(t_est)
        period_theory.append(cfg.orbital_period_theory)
        constants.append((t_est * t_est) / (a * a * a))

    period_estimates_arr = np.asarray(period_estimates, dtype=np.float64)
    period_theory_arr = np.asarray(period_theory, dtype=np.float64)
    constants_arr = np.asarray(constants, dtype=np.float64)

    expected_constant = 4.0 * np.pi * np.pi / base_config.mu
    expected_t2 = expected_constant * (a_values**3)
    estimated_t2 = period_estimates_arr**2

    rel_err = np.abs((constants_arr - expected_constant) / expected_constant)

    constants_t = torch.tensor(constants_arr, dtype=torch.float64)
    metrics = {
        "third_law_expected_constant": float(expected_constant),
        "third_law_mean_constant": float(np.mean(constants_arr)),
        "third_law_max_rel_error": float(np.max(rel_err)),
        "third_law_rel_spread": float((np.max(constants_arr) - np.min(constants_arr)) / expected_constant),
        "third_law_mae_t2": float(mean_absolute_error(expected_t2, estimated_t2)),
        "third_law_r2": float(r2_score(expected_t2, estimated_t2)),
        "third_law_torch_constant_span": float((torch.max(constants_t) - torch.min(constants_t)).item()),
    }

    period_table = pd.DataFrame(
        {
            "a": a_values,
            "T_est": period_estimates_arr,
            "T_theory": period_theory_arr,
            "T2_over_a3": constants_arr,
            "rel_error": rel_err,
        }
    )

    return ThirdLawResult(period_table=period_table, metrics=metrics)


def build_orbit_snapshot(
    time: Array,
    state: Array,
    invariants: Dict[str, Array],
    modeled_radius: Array,
    rows: int = 9,
) -> pd.DataFrame:
    """Build compact trajectory table for quick manual inspection."""
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
            "ecc": invariants["eccentricity"][idx],
        }
    )


def main() -> None:
    config = OrbitConfig()

    time, state = integrate_sampled_orbit(config)
    first = evaluate_first_law(config, state)

    h_reference = float(first.invariants["specific_angular_momentum"][0])
    second = evaluate_second_law(time, state, h_reference=h_reference)

    third = evaluate_third_law(config)

    # Deterministic acceptance thresholds for CI-style validation.
    assert first.metrics["conic_radius_r2"] > 0.9999, (
        f"first law R^2 too low: {first.metrics['conic_radius_r2']:.6f}"
    )
    assert first.metrics["conic_radius_mae"] < 2e-4, (
        f"first law MAE too large: {first.metrics['conic_radius_mae']:.3e}"
    )
    assert first.metrics["rel_energy_drift"] < 1e-7, (
        f"energy drift too large: {first.metrics['rel_energy_drift']:.3e}"
    )

    assert second.metrics["area_rate_cv"] < 3e-3, (
        f"second law CV too high: {second.metrics['area_rate_cv']:.3e}"
    )
    assert second.metrics["max_rel_area_rate_dev"] < 6e-3, (
        f"second law max relative deviation too high: {second.metrics['max_rel_area_rate_dev']:.3e}"
    )

    assert third.metrics["third_law_r2"] > 0.999999, (
        f"third law R^2 too low: {third.metrics['third_law_r2']:.8f}"
    )
    assert third.metrics["third_law_max_rel_error"] < 3e-3, (
        f"third law max relative error too high: {third.metrics['third_law_max_rel_error']:.3e}"
    )

    print("=== Kepler's Laws MVP ===")
    print("\n[First Law Metrics]")
    for k, v in first.metrics.items():
        print(f"{k:>28s}: {v:.12e}")

    print("\n[Second Law Metrics]")
    for k, v in second.metrics.items():
        print(f"{k:>28s}: {v:.12e}")

    print("\n[Third Law Metrics]")
    for k, v in third.metrics.items():
        print(f"{k:>28s}: {v:.12e}")

    orbit_table = build_orbit_snapshot(
        time=time,
        state=state,
        invariants=first.invariants,
        modeled_radius=first.modeled_radius,
        rows=9,
    )

    print("\n=== Orbit Snapshot (9 points) ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(orbit_table.to_string(index=False, float_format=lambda x: f"{x: .6e}"))

    print("\n=== Equal-Time Sweep Segment Snapshot ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(second.segment_table.to_string(index=False, float_format=lambda x: f"{x: .6e}"))

    print("\n=== Third Law Period Table ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(third.period_table.to_string(index=False, float_format=lambda x: f"{x: .6e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
