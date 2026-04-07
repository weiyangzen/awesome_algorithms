"""Resonance theory MVP for a damped driven harmonic oscillator.

Model:
    m*x'' + c*x' + k*x = F0*cos(omega*t)

What this script demonstrates:
1) analytic frequency response (steady-state amplitude and phase),
2) displacement resonance prediction,
3) ODE simulation with solve_ivp,
4) steady-state harmonic fitting and cross-check against theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ResonanceConfig:
    """Physical and numerical configuration for resonance analysis."""

    m: float = 1.0
    c: float = 0.45
    k: float = 25.0
    f0: float = 1.0
    omega_min: float = 1.0
    omega_max: float = 9.0
    omega_points: int = 161
    t_transient: float = 35.0
    t_sample: float = 18.0
    points_per_period: int = 160
    rtol: float = 1e-8
    atol: float = 1e-10


def validate_config(cfg: ResonanceConfig) -> None:
    scalar_values = {
        "m": cfg.m,
        "c": cfg.c,
        "k": cfg.k,
        "f0": cfg.f0,
        "omega_min": cfg.omega_min,
        "omega_max": cfg.omega_max,
        "t_transient": cfg.t_transient,
        "t_sample": cfg.t_sample,
        "rtol": cfg.rtol,
        "atol": cfg.atol,
    }
    for key, value in scalar_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value!r}")

    if cfg.m <= 0.0:
        raise ValueError("m must be positive")
    if cfg.c < 0.0:
        raise ValueError("c must be non-negative")
    if cfg.k <= 0.0:
        raise ValueError("k must be positive")
    if cfg.f0 <= 0.0:
        raise ValueError("f0 must be positive")
    if cfg.omega_min <= 0.0 or cfg.omega_max <= cfg.omega_min:
        raise ValueError("require 0 < omega_min < omega_max")
    if cfg.omega_points < 11:
        raise ValueError("omega_points must be >= 11")
    if cfg.t_transient <= 0.0 or cfg.t_sample <= 0.0:
        raise ValueError("t_transient and t_sample must be positive")
    if cfg.points_per_period < 24:
        raise ValueError("points_per_period must be >= 24")
    if cfg.rtol <= 0.0 or cfg.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")

    # Displacement resonance peak exists only when this quantity is positive.
    if cfg.k / cfg.m - (cfg.c**2) / (2.0 * cfg.m**2) <= 0.0:
        raise ValueError("parameters too heavily damped: no displacement resonance peak")


def natural_frequency(cfg: ResonanceConfig) -> float:
    return math.sqrt(cfg.k / cfg.m)


def damping_ratio(cfg: ResonanceConfig) -> float:
    return cfg.c / (2.0 * math.sqrt(cfg.k * cfg.m))


def theoretical_resonance_frequency(cfg: ResonanceConfig) -> float:
    """Displacement resonance frequency for underdamped driven oscillator."""

    value = cfg.k / cfg.m - (cfg.c**2) / (2.0 * cfg.m**2)
    return math.sqrt(value)


def steady_state_amplitude(cfg: ResonanceConfig, omega: np.ndarray) -> np.ndarray:
    denom = np.sqrt((cfg.k - cfg.m * omega**2) ** 2 + (cfg.c * omega) ** 2)
    return cfg.f0 / denom


def steady_state_phase_lag(cfg: ResonanceConfig, omega: np.ndarray) -> np.ndarray:
    # x_ss(t) = A(omega) * cos(omega*t - phi)
    return np.arctan2(cfg.c * omega, cfg.k - cfg.m * omega**2)


def frequency_response_table(cfg: ResonanceConfig) -> pd.DataFrame:
    omega_grid = np.linspace(cfg.omega_min, cfg.omega_max, cfg.omega_points, dtype=float)
    amp = steady_state_amplitude(cfg, omega_grid)
    phase = steady_state_phase_lag(cfg, omega_grid)

    return pd.DataFrame(
        {
            "omega_rad_s": omega_grid,
            "freq_hz": omega_grid / (2.0 * math.pi),
            "amplitude_m": amp,
            "phase_lag_rad": phase,
        }
    )


def rhs_forced_oscillator(
    t: float,
    y: np.ndarray,
    m: float,
    c: float,
    k: float,
    f0: float,
    omega_drive: float,
) -> np.ndarray:
    x, v = y
    a = (f0 * math.cos(omega_drive * t) - c * v - k * x) / m
    return np.array([v, a], dtype=float)


def simulate_trajectory(cfg: ResonanceConfig, omega_drive: float) -> tuple[np.ndarray, np.ndarray]:
    t_total = cfg.t_transient + cfg.t_sample
    period = 2.0 * math.pi / omega_drive
    dt = period / float(cfg.points_per_period)
    n_points = int(math.ceil(t_total / dt)) + 1
    times = np.linspace(0.0, t_total, n_points, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs_forced_oscillator(t, y, cfg.m, cfg.c, cfg.k, cfg.f0, omega_drive),
        t_span=(0.0, t_total),
        y0=np.array([0.0, 0.0], dtype=float),
        t_eval=times,
        method="DOP853",
        max_step=dt,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for omega={omega_drive:.6f}: {sol.message}")

    x = sol.y[0]
    return times, x


def fit_steady_harmonic(times: np.ndarray, x: np.ndarray, omega_drive: float) -> dict[str, float]:
    """Fit x(t)=a*cos(wt)+b*sin(wt)+d on the steady segment via least squares."""

    wt = omega_drive * times
    design = np.column_stack([np.cos(wt), np.sin(wt), np.ones_like(times)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)

    a = float(coef[0])
    b = float(coef[1])
    d = float(coef[2])
    fitted = design @ coef

    amplitude = float(math.hypot(a, b))
    phase_lag = float(math.atan2(b, a))
    rmse = float(np.sqrt(np.mean((x - fitted) ** 2)))

    return {
        "amplitude_est_m": amplitude,
        "phase_lag_est_rad": phase_lag,
        "offset_est_m": d,
        "fit_rmse_m": rmse,
    }


def wrap_phase_error(estimated: float, theoretical: float) -> float:
    return float(math.atan2(math.sin(estimated - theoretical), math.cos(estimated - theoretical)))


def analyze_single_frequency(cfg: ResonanceConfig, omega_drive: float) -> dict[str, float]:
    times, x = simulate_trajectory(cfg, omega_drive)
    mask = times >= cfg.t_transient
    fit = fit_steady_harmonic(times[mask], x[mask], omega_drive)

    amp_theory = float(steady_state_amplitude(cfg, np.array([omega_drive], dtype=float))[0])
    phase_theory = float(steady_state_phase_lag(cfg, np.array([omega_drive], dtype=float))[0])

    amp_rel_err = abs(fit["amplitude_est_m"] - amp_theory) / max(amp_theory, 1e-12)
    phase_err = wrap_phase_error(fit["phase_lag_est_rad"], phase_theory)

    return {
        "omega_rad_s": omega_drive,
        "freq_hz": omega_drive / (2.0 * math.pi),
        "amplitude_theory_m": amp_theory,
        "amplitude_est_m": fit["amplitude_est_m"],
        "amplitude_rel_err": amp_rel_err,
        "phase_lag_theory_rad": phase_theory,
        "phase_lag_est_rad": fit["phase_lag_est_rad"],
        "phase_err_rad": phase_err,
        "fit_rmse_m": fit["fit_rmse_m"],
        "offset_est_m": fit["offset_est_m"],
    }


def main() -> None:
    cfg = ResonanceConfig()
    validate_config(cfg)

    omega0 = natural_frequency(cfg)
    zeta = damping_ratio(cfg)
    omega_res_theory = theoretical_resonance_frequency(cfg)

    response_df = frequency_response_table(cfg)
    idx_peak = int(response_df["amplitude_m"].idxmax())
    omega_peak_grid = float(response_df.loc[idx_peak, "omega_rad_s"])
    amp_peak_grid = float(response_df.loc[idx_peak, "amplitude_m"])
    grid_step = float(response_df.loc[1, "omega_rad_s"] - response_df.loc[0, "omega_rad_s"])

    analysis_omegas = np.array(
        [0.90 * omega_res_theory, omega_res_theory, 1.10 * omega_res_theory],
        dtype=float,
    )
    analysis_rows = [analyze_single_frequency(cfg, float(w)) for w in analysis_omegas]
    analysis_df = pd.DataFrame(analysis_rows)

    center_amp = float(analysis_df.iloc[1]["amplitude_est_m"])
    side_amps = np.array(
        [
            float(analysis_df.iloc[0]["amplitude_est_m"]),
            float(analysis_df.iloc[2]["amplitude_est_m"]),
        ]
    )

    max_amp_rel_err = float(analysis_df["amplitude_rel_err"].max())
    max_phase_err = float(np.max(np.abs(analysis_df["phase_err_rad"].to_numpy())))
    max_fit_rmse = float(analysis_df["fit_rmse_m"].max())

    summary_df = pd.DataFrame(
        {
            "metric": [
                "natural_frequency_rad_s",
                "damping_ratio",
                "theoretical_resonance_rad_s",
                "grid_peak_rad_s",
                "abs_peak_diff_rad_s",
                "grid_peak_amplitude_m",
                "max_amplitude_relative_error",
                "max_abs_phase_error_rad",
                "max_fit_rmse_m",
            ],
            "value": [
                omega0,
                zeta,
                omega_res_theory,
                omega_peak_grid,
                abs(omega_peak_grid - omega_res_theory),
                amp_peak_grid,
                max_amp_rel_err,
                max_phase_err,
                max_fit_rmse,
            ],
        }
    )

    top_resonant = response_df.nlargest(5, columns="amplitude_m").sort_values("omega_rad_s")

    print("Resonance Theory MVP (damped driven oscillator)")
    print("Parameters:")
    print(cfg)

    print("\nSummary metrics:")
    print(summary_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.8e}"))

    print("\nSteady-state fit vs theory near resonance:")
    print(analysis_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    print("\nTop-5 amplitudes from analytic frequency sweep:")
    print(top_resonant.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    # Non-interactive acceptance checks.
    assert abs(omega_peak_grid - omega_res_theory) <= 1.5 * grid_step, (
        "grid peak frequency deviates too much from theoretical resonance"
    )
    assert max_amp_rel_err < 0.02, f"amplitude relative error too large: {max_amp_rel_err}"
    assert max_phase_err < 0.08, f"phase error too large: {max_phase_err}"
    assert max_fit_rmse < 0.003, f"steady fit rmse too large: {max_fit_rmse}"
    assert center_amp > float(np.max(side_amps)), "resonance check failed: center amplitude is not the largest"


if __name__ == "__main__":
    main()
