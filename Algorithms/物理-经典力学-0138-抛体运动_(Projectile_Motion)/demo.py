"""Minimal runnable MVP for projectile motion.

This script demonstrates a transparent pipeline for ideal projectile motion:
1) Forward simulation with analytic equations.
2) Flight-time root solving with scipy.brentq.
3) Inverse parameter estimation from noisy observations (scikit-learn).
4) Cross-check of numpy and torch trajectory computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ProjectileParams:
    """Physical parameters for ideal projectile motion."""

    x0: float
    y0: float
    speed: float
    angle_deg: float
    gravity: float = 9.81

    def validate(self) -> None:
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")


@dataclass(frozen=True)
class ForwardSummary:
    """Key scalar diagnostics from forward simulation."""

    t_flight: float
    range_x: float
    t_peak: float
    max_height: float
    energy_rel_drift_max: float


def initial_velocity_components(params: ProjectileParams) -> Tuple[float, float]:
    """Return (vx0, vy0) from launch speed and angle (degrees)."""
    theta = np.deg2rad(params.angle_deg)
    vx0 = params.speed * np.cos(theta)
    vy0 = params.speed * np.sin(theta)
    return float(vx0), float(vy0)


def y_position(t: np.ndarray | float, params: ProjectileParams, vy0: float) -> np.ndarray | float:
    """Vertical position y(t) under constant gravity."""
    return params.y0 + vy0 * t - 0.5 * params.gravity * t * t


def flight_time_to_ground(params: ProjectileParams) -> float:
    """Solve y(t)=0 for the positive root using Brent's method."""
    vx0, vy0 = initial_velocity_components(params)
    _ = vx0  # keep naming symmetric and explicit

    # Analytic positive root provides a safe upper bracket for brentq.
    discriminant = vy0 * vy0 + 2.0 * params.gravity * params.y0
    if discriminant < 0.0:
        raise ValueError("No real flight time root: discriminant < 0")

    t_upper_guess = (vy0 + np.sqrt(discriminant)) / params.gravity
    t_upper = max(float(t_upper_guess) * 1.2, 1e-3)

    f = lambda t: y_position(t, params, vy0)
    f0 = float(f(0.0))
    fu = float(f(t_upper))

    if f0 == 0.0:
        # Degenerate edge case: starts exactly at ground.
        return 0.0

    if fu > 0.0:
        # Expand bracket if needed; usually unnecessary for physically valid settings.
        growth = 1
        while fu > 0.0 and growth <= 20:
            t_upper *= 1.8
            fu = float(f(t_upper))
            growth += 1
        if fu > 0.0:
            raise RuntimeError("Failed to bracket positive flight-time root for y(t)=0")

    root = brentq(lambda t: float(f(t)), 0.0, t_upper, xtol=1e-12, rtol=1e-12, maxiter=100)
    return float(root)


def simulate_trajectory_numpy(params: ProjectileParams, num_samples: int = 240) -> pd.DataFrame:
    """Generate trajectory table using analytic formulas."""
    if num_samples < 20:
        raise ValueError("num_samples must be >= 20")

    params.validate()
    vx0, vy0 = initial_velocity_components(params)
    t_flight = flight_time_to_ground(params)

    t = np.linspace(0.0, t_flight, num_samples, dtype=np.float64)
    x = params.x0 + vx0 * t
    y = y_position(t, params, vy0)
    vx = np.full_like(t, vx0)
    vy = vy0 - params.gravity * t

    df = pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
        }
    )
    return df


def compute_summary_metrics(df: pd.DataFrame, params: ProjectileParams) -> ForwardSummary:
    """Compute physically meaningful scalar metrics for sanity checks."""
    vx0, vy0 = initial_velocity_components(params)
    t_flight = float(df["t"].iloc[-1])
    range_x = float(df["x"].iloc[-1] - params.x0)

    t_peak = max(vy0 / params.gravity, 0.0)
    max_height = float(params.y0 + vy0 * t_peak - 0.5 * params.gravity * t_peak * t_peak)

    # Mechanical energy E = T + U, with U = g*y for unit mass.
    energy = 0.5 * (df["vx"].to_numpy() ** 2 + df["vy"].to_numpy() ** 2) + params.gravity * df["y"].to_numpy()
    e0 = float(energy[0])
    energy_rel_drift_max = float(np.max(np.abs((energy - e0) / (abs(e0) + 1e-14))))

    # Guard against hidden inconsistencies.
    assert abs(vx0 - float(df["vx"].iloc[0])) < 1e-12

    return ForwardSummary(
        t_flight=t_flight,
        range_x=range_x,
        t_peak=t_peak,
        max_height=max_height,
        energy_rel_drift_max=energy_rel_drift_max,
    )


def build_noisy_observations(df: pd.DataFrame, noise_std: float = 0.03, stride: int = 10) -> pd.DataFrame:
    """Subsample trajectory and add reproducible Gaussian noise to x/y observations."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")

    obs = df.iloc[::stride, :][["t", "x", "y"]].copy().reset_index(drop=True)
    rng = np.random.default_rng(20260407)
    obs["x"] += rng.normal(loc=0.0, scale=noise_std, size=len(obs))
    obs["y"] += rng.normal(loc=0.0, scale=noise_std, size=len(obs))
    return obs


def estimate_parameters_from_observations(observations: pd.DataFrame) -> Dict[str, float]:
    """Estimate launch and gravity parameters from noisy (t, x, y) observations."""
    t = observations["t"].to_numpy(dtype=np.float64)
    x = observations["x"].to_numpy(dtype=np.float64)
    y = observations["y"].to_numpy(dtype=np.float64)

    # x(t) = b0 + b1*t
    reg_x = LinearRegression(fit_intercept=True)
    reg_x.fit(t.reshape(-1, 1), x)
    x0_est = float(reg_x.intercept_)
    vx0_est = float(reg_x.coef_[0])

    # y(t) = c0 + c1*t + c2*t^2, where c2 = -g/2
    features_y = np.column_stack([t, t * t])
    reg_y = LinearRegression(fit_intercept=True)
    reg_y.fit(features_y, y)
    y0_est = float(reg_y.intercept_)
    vy0_est = float(reg_y.coef_[0])
    c2_est = float(reg_y.coef_[1])

    g_est = float(-2.0 * c2_est)
    speed_est = float(np.hypot(vx0_est, vy0_est))
    angle_est_deg = float(np.rad2deg(np.arctan2(vy0_est, vx0_est)))

    return {
        "x0_est": x0_est,
        "y0_est": y0_est,
        "vx0_est": vx0_est,
        "vy0_est": vy0_est,
        "g_est": g_est,
        "speed_est": speed_est,
        "angle_est_deg": angle_est_deg,
        "x_r2": float(reg_x.score(t.reshape(-1, 1), x)),
        "y_r2": float(reg_y.score(features_y, y)),
    }


def torch_trajectory(params: ProjectileParams, t_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x(t), y(t) with torch for cross-implementation consistency checks."""
    vx0, vy0 = initial_velocity_components(params)

    # Ensure writable contiguous storage before bridging NumPy -> Torch.
    t_numpy = np.array(t_values, dtype=np.float64, copy=True)
    t_torch = torch.from_numpy(t_numpy)
    x = params.x0 + vx0 * t_torch
    y = params.y0 + vy0 * t_torch - 0.5 * params.gravity * t_torch * t_torch

    return x.detach().cpu().numpy(), y.detach().cpu().numpy()


def main() -> None:
    params = ProjectileParams(
        x0=0.0,
        y0=1.2,
        speed=22.0,
        angle_deg=37.0,
        gravity=9.81,
    )

    print("=== Demo A: Forward simulation (analytic projectile model) ===")
    trajectory = simulate_trajectory_numpy(params, num_samples=240)
    summary = compute_summary_metrics(trajectory, params)

    print(f"flight_time={summary.t_flight:.6f} s")
    print(f"range={summary.range_x:.6f} m")
    print(f"peak_time={summary.t_peak:.6f} s")
    print(f"max_height={summary.max_height:.6f} m")
    print(f"max_relative_energy_drift={summary.energy_rel_drift_max:.3e}")

    print("\n=== Demo B: Inverse estimation from noisy observations ===")
    observations = build_noisy_observations(trajectory, noise_std=0.03, stride=10)
    estimate = estimate_parameters_from_observations(observations)

    vx0_true, vy0_true = initial_velocity_components(params)
    print(
        "true  : x0={:.4f}, y0={:.4f}, vx0={:.4f}, vy0={:.4f}, g={:.4f}, speed={:.4f}, angle={:.4f}".format(
            params.x0,
            params.y0,
            vx0_true,
            vy0_true,
            params.gravity,
            params.speed,
            params.angle_deg,
        )
    )
    print(
        "est   : x0={x0_est:.4f}, y0={y0_est:.4f}, vx0={vx0_est:.4f}, vy0={vy0_est:.4f}, g={g_est:.4f}, speed={speed_est:.4f}, angle={angle_est_deg:.4f}".format(
            **estimate
        )
    )
    print("fit R^2: x={:.6f}, y={:.6f}".format(estimate["x_r2"], estimate["y_r2"]))

    print("\n=== Demo C: Torch consistency check ===")
    t_values = trajectory["t"].to_numpy(dtype=np.float64)
    x_torch, y_torch = torch_trajectory(params, t_values)

    x_numpy = trajectory["x"].to_numpy(dtype=np.float64)
    y_numpy = trajectory["y"].to_numpy(dtype=np.float64)

    max_abs_diff_x = float(np.max(np.abs(x_numpy - x_torch)))
    max_abs_diff_y = float(np.max(np.abs(y_numpy - y_torch)))
    print(f"max_abs_diff_x={max_abs_diff_x:.3e}")
    print(f"max_abs_diff_y={max_abs_diff_y:.3e}")

    # Deterministic quality gates for this MVP.
    assert summary.t_flight > 0.0
    assert summary.range_x > 0.0
    assert summary.max_height > params.y0
    assert summary.energy_rel_drift_max < 1e-11

    assert abs(estimate["vx0_est"] - vx0_true) < 0.08
    assert abs(estimate["vy0_est"] - vy0_true) < 0.08
    assert abs(estimate["g_est"] - params.gravity) < 0.20
    assert abs(estimate["speed_est"] - params.speed) < 0.08
    assert abs(estimate["angle_est_deg"] - params.angle_deg) < 0.35
    assert estimate["x_r2"] > 0.99995
    assert estimate["y_r2"] > 0.99980

    assert max_abs_diff_x < 1e-12
    assert max_abs_diff_y < 1e-12

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
