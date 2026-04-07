"""Minimal runnable MVP for the law of conservation of energy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.constants import pi
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class OscillatorConfig:
    """Configuration for a 1D undamped mass-spring oscillator."""

    mass_kg: float = 1.5
    spring_k_npm: float = 9.0
    x0_m: float = 0.2
    v0_mps: float = 0.0
    dt_s: float = 0.01
    num_steps: int = 6000


def validate_config(cfg: OscillatorConfig) -> None:
    """Validate simulation configuration."""
    if cfg.mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")
    if cfg.spring_k_npm <= 0.0:
        raise ValueError("spring_k_npm must be positive.")
    if cfg.dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    if cfg.num_steps <= 0:
        raise ValueError("num_steps must be positive.")


def acceleration(x: float | np.ndarray, mass_kg: float, spring_k_npm: float) -> float | np.ndarray:
    """Harmonic restoring acceleration a = -(k/m) * x."""
    return -(spring_k_npm / mass_kg) * x


def mechanical_energy(
    x: float | np.ndarray,
    v: float | np.ndarray,
    mass_kg: float,
    spring_k_npm: float,
) -> float | np.ndarray:
    """Total mechanical energy E = 1/2 m v^2 + 1/2 k x^2."""
    return 0.5 * mass_kg * np.square(v) + 0.5 * spring_k_npm * np.square(x)


def simulate_explicit_euler(cfg: OscillatorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate oscillator using explicit Euler integration."""
    t = np.arange(cfg.num_steps + 1, dtype=np.float64) * cfg.dt_s
    x = np.zeros(cfg.num_steps + 1, dtype=np.float64)
    v = np.zeros(cfg.num_steps + 1, dtype=np.float64)
    e = np.zeros(cfg.num_steps + 1, dtype=np.float64)

    x[0] = cfg.x0_m
    v[0] = cfg.v0_mps
    e[0] = float(mechanical_energy(x[0], v[0], cfg.mass_kg, cfg.spring_k_npm))

    for i in range(cfg.num_steps):
        a = float(acceleration(x[i], cfg.mass_kg, cfg.spring_k_npm))
        x[i + 1] = x[i] + cfg.dt_s * v[i]
        v[i + 1] = v[i] + cfg.dt_s * a
        e[i + 1] = float(mechanical_energy(x[i + 1], v[i + 1], cfg.mass_kg, cfg.spring_k_npm))

    return t, x, v, e


def simulate_velocity_verlet(cfg: OscillatorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate oscillator using Velocity-Verlet integration."""
    t = np.arange(cfg.num_steps + 1, dtype=np.float64) * cfg.dt_s
    x = np.zeros(cfg.num_steps + 1, dtype=np.float64)
    v = np.zeros(cfg.num_steps + 1, dtype=np.float64)
    e = np.zeros(cfg.num_steps + 1, dtype=np.float64)

    x[0] = cfg.x0_m
    v[0] = cfg.v0_mps
    e[0] = float(mechanical_energy(x[0], v[0], cfg.mass_kg, cfg.spring_k_npm))

    a = float(acceleration(x[0], cfg.mass_kg, cfg.spring_k_npm))
    for i in range(cfg.num_steps):
        x_next = x[i] + cfg.dt_s * v[i] + 0.5 * (cfg.dt_s**2) * a
        a_next = float(acceleration(x_next, cfg.mass_kg, cfg.spring_k_npm))
        v_next = v[i] + 0.5 * cfg.dt_s * (a + a_next)

        x[i + 1] = x_next
        v[i + 1] = v_next
        e[i + 1] = float(mechanical_energy(x_next, v_next, cfg.mass_kg, cfg.spring_k_npm))
        a = a_next

    return t, x, v, e


def estimate_period_seconds(x: np.ndarray, t: np.ndarray) -> float:
    """Estimate period from distance between displacement peaks."""
    prominence = max(1e-12, 0.05 * float(np.std(x)))
    peaks, _ = find_peaks(x, prominence=prominence)
    if peaks.size < 2:
        raise RuntimeError("Not enough peaks to estimate period.")
    return float(np.mean(np.diff(t[peaks])))


def relative_energy_drift(e: np.ndarray) -> float:
    """Max relative drift from initial energy."""
    e0 = float(e[0])
    return float(np.max(np.abs(e - e0)) / abs(e0))


def energy_trend_slope(t: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Linear trend slope of energy vs time (J/s) and R^2."""
    model = LinearRegression()
    model.fit(t.reshape(-1, 1), e)
    slope = float(model.coef_[0])
    r2 = float(model.score(t.reshape(-1, 1), e))
    return slope, r2


def dt_scan(
    base_cfg: OscillatorConfig,
    dt_values: np.ndarray,
    total_time_s: float,
) -> pd.DataFrame:
    """Run fixed-horizon simulations across multiple dt values."""
    rows: list[dict[str, float]] = []
    for dt in dt_values:
        steps = int(round(total_time_s / float(dt)))
        cfg = OscillatorConfig(
            mass_kg=base_cfg.mass_kg,
            spring_k_npm=base_cfg.spring_k_npm,
            x0_m=base_cfg.x0_m,
            v0_mps=base_cfg.v0_mps,
            dt_s=float(dt),
            num_steps=steps,
        )
        _, _, _, e_euler = simulate_explicit_euler(cfg)
        _, _, _, e_verlet = simulate_velocity_verlet(cfg)
        rows.append(
            {
                "dt_s": float(dt),
                "steps": float(steps),
                "drift_euler": relative_energy_drift(e_euler),
                "drift_verlet": relative_energy_drift(e_verlet),
            }
        )
    return pd.DataFrame(rows).sort_values("dt_s", ascending=False, ignore_index=True)


def loglog_convergence_order(df: pd.DataFrame, drift_col: str) -> tuple[float, float]:
    """Fit log(drift)=p*log(dt)+c and return (p, R^2)."""
    x = np.log(df["dt_s"].to_numpy(dtype=np.float64)).reshape(-1, 1)
    y = np.log(df[drift_col].to_numpy(dtype=np.float64))

    model = LinearRegression()
    model.fit(x, y)
    order = float(model.coef_[0])
    r2 = float(model.score(x, y))
    return order, r2


def torch_energy_consistency(
    x: np.ndarray,
    v: np.ndarray,
    mass_kg: float,
    spring_k_npm: float,
) -> float:
    """Cross-check NumPy energy with PyTorch implementation."""
    x_t = torch.tensor(x, dtype=torch.float64)
    v_t = torch.tensor(v, dtype=torch.float64)
    e_t = 0.5 * mass_kg * torch.square(v_t) + 0.5 * spring_k_npm * torch.square(x_t)
    e_np = mechanical_energy(x, v, mass_kg, spring_k_npm)
    return float(np.max(np.abs(e_t.numpy() - e_np)))


def main() -> None:
    cfg = OscillatorConfig()
    validate_config(cfg)

    t_euler, x_euler, v_euler, e_euler = simulate_explicit_euler(cfg)
    t_verlet, x_verlet, v_verlet, e_verlet = simulate_velocity_verlet(cfg)

    period_theory = 2.0 * pi * math.sqrt(cfg.mass_kg / cfg.spring_k_npm)
    period_euler = estimate_period_seconds(x_euler, t_euler)
    period_verlet = estimate_period_seconds(x_verlet, t_verlet)

    period_err_euler = abs(period_euler - period_theory) / period_theory
    period_err_verlet = abs(period_verlet - period_theory) / period_theory

    drift_euler = relative_energy_drift(e_euler)
    drift_verlet = relative_energy_drift(e_verlet)

    slope_euler, slope_r2_euler = energy_trend_slope(t_euler, e_euler)
    slope_verlet, slope_r2_verlet = energy_trend_slope(t_verlet, e_verlet)

    dt_values = np.array([0.04, 0.02, 0.01, 0.005], dtype=np.float64)
    drift_scan = dt_scan(cfg, dt_values=dt_values, total_time_s=40.0)

    order_euler, order_r2_euler = loglog_convergence_order(drift_scan, "drift_euler")
    order_verlet, order_r2_verlet = loglog_convergence_order(drift_scan, "drift_verlet")

    torch_diff = torch_energy_consistency(x_verlet, v_verlet, cfg.mass_kg, cfg.spring_k_npm)

    summary = pd.DataFrame(
        [
            ("theory_period_s", period_theory, "s"),
            ("euler_period_s", period_euler, "s"),
            ("verlet_period_s", period_verlet, "s"),
            ("euler_period_rel_error", period_err_euler, "ratio"),
            ("verlet_period_rel_error", period_err_verlet, "ratio"),
            ("euler_energy_rel_drift", drift_euler, "ratio"),
            ("verlet_energy_rel_drift", drift_verlet, "ratio"),
            ("euler_energy_trend_slope", slope_euler, "J/s"),
            ("verlet_energy_trend_slope", slope_verlet, "J/s"),
            ("euler_energy_trend_r2", slope_r2_euler, "[0,1]"),
            ("verlet_energy_trend_r2", slope_r2_verlet, "[0,1]"),
            ("euler_drift_order", order_euler, "log-log slope"),
            ("verlet_drift_order", order_verlet, "log-log slope"),
            ("euler_order_fit_r2", order_r2_euler, "[0,1]"),
            ("verlet_order_fit_r2", order_r2_verlet, "[0,1]"),
            ("torch_numpy_energy_max_abs_diff", torch_diff, "J"),
        ],
        columns=["metric", "value", "unit"],
    )

    sample_idx = np.linspace(0, cfg.num_steps, 8, dtype=int)
    snapshots = pd.DataFrame(
        {
            "t_s": t_verlet[sample_idx],
            "x_verlet_m": x_verlet[sample_idx],
            "v_verlet_mps": v_verlet[sample_idx],
            "E_verlet_J": e_verlet[sample_idx],
            "x_euler_m": x_euler[sample_idx],
            "v_euler_mps": v_euler[sample_idx],
            "E_euler_J": e_euler[sample_idx],
        }
    )

    print("=== Conservation of Energy MVP: 1D Harmonic Oscillator ===")
    print(summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.8e}"))
    print("\n=== Energy Drift vs Step Size ===")
    print(drift_scan.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print("\n=== Trajectory Snapshot (Verlet vs Euler) ===")
    print(snapshots.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    if period_err_verlet > 1e-2:
        raise RuntimeError("Verlet period error is too large.")
    if drift_verlet > 5e-3:
        raise RuntimeError("Verlet energy drift is too large for this MVP setup.")
    if drift_euler <= drift_verlet:
        raise RuntimeError("Euler should drift more than Verlet in this setup.")
    if torch_diff > 1e-12:
        raise RuntimeError("Torch/NumPy energy cross-check failed.")


if __name__ == "__main__":
    main()
