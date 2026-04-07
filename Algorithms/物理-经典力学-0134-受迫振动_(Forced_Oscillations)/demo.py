"""Forced oscillations MVP (single DOF, damped and sinusoidally driven).

Model:
    m x'' + c x' + k x = F0 cos(omega_d t)

This script demonstrates:
1) direct ODE integration in time domain,
2) analytic steady-state amplitude/phase formula,
3) harmonic fitting on trajectory tail to estimate numeric steady-state response,
4) frequency sweep and resonance diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


EPS = 1e-12


@dataclass(frozen=True)
class ForcedOscillationConfig:
    """Physical and numerical settings for the forced oscillator."""

    m: float = 1.0
    c: float = 0.35
    k: float = 20.0
    f0: float = 1.5

    x0: float = 0.0
    v0: float = 0.0

    drive_omega: float = 4.2
    t_start: float = 0.0
    t_end: float = 80.0
    num_points: int = 5000

    freq_min_ratio: float = 0.4
    freq_max_ratio: float = 1.8
    num_freqs: int = 19
    sweep_t_end: float = 70.0
    sweep_num_points: int = 3000

    tail_fraction: float = 0.35
    rtol: float = 1e-9
    atol: float = 1e-11


def validate_config(cfg: ForcedOscillationConfig) -> None:
    """Validate physical and simulation parameters."""

    if cfg.m <= 0.0:
        raise ValueError("m must be > 0")
    if cfg.k <= 0.0:
        raise ValueError("k must be > 0")
    if cfg.c < 0.0:
        raise ValueError("c must be >= 0")
    if cfg.f0 < 0.0:
        raise ValueError("f0 must be >= 0")

    if cfg.t_end <= cfg.t_start:
        raise ValueError("Require t_end > t_start")
    if cfg.num_points < 100:
        raise ValueError("num_points must be >= 100")
    if cfg.sweep_t_end <= 0.0:
        raise ValueError("sweep_t_end must be > 0")
    if cfg.sweep_num_points < 200:
        raise ValueError("sweep_num_points must be >= 200")
    if cfg.num_freqs < 5:
        raise ValueError("num_freqs must be >= 5")
    if not (0.0 < cfg.tail_fraction < 0.9):
        raise ValueError("tail_fraction must be in (0, 0.9)")
    if not (0.0 < cfg.freq_min_ratio < cfg.freq_max_ratio):
        raise ValueError("Require 0 < freq_min_ratio < freq_max_ratio")


def natural_frequency(cfg: ForcedOscillationConfig) -> float:
    """Natural angular frequency omega0."""

    return float(np.sqrt(cfg.k / cfg.m))


def damping_ratio(cfg: ForcedOscillationConfig) -> float:
    """Dimensionless damping ratio zeta."""

    return float(cfg.c / (2.0 * np.sqrt(cfg.k * cfg.m)))


def steady_state_amplitude_phase(
    m: float,
    c: float,
    k: float,
    f0: float,
    omega: float,
) -> tuple[float, float]:
    """Closed-form steady-state amplitude and phase for sinusoidal forcing."""

    denom = np.sqrt((k - m * omega * omega) ** 2 + (c * omega) ** 2)
    amplitude = f0 / max(denom, EPS)
    phase = float(np.arctan2(c * omega, k - m * omega * omega))
    return float(amplitude), phase


def wrapped_phase_diff(a: float, b: float) -> float:
    """Return absolute phase difference in [0, pi]."""

    raw = a - b
    wrapped = (raw + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(wrapped))


def rhs(
    t: float,
    y: np.ndarray,
    m: float,
    c: float,
    k: float,
    f0: float,
    omega: float,
) -> np.ndarray:
    """First-order state equation y=[x,v]."""

    x = y[0]
    v = y[1]
    a = (f0 * np.cos(omega * t) - c * v - k * x) / m
    return np.array([v, a], dtype=float)


def integrate_forced_oscillator(
    cfg: ForcedOscillationConfig,
    omega: float,
    t_end: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate forced oscillator and return (t, x, v)."""

    t = np.linspace(cfg.t_start, t_end, num_points)
    y0 = np.array([cfg.x0, cfg.v0], dtype=float)

    sol = solve_ivp(
        fun=lambda tt, yy: rhs(tt, yy, cfg.m, cfg.c, cfg.k, cfg.f0, omega),
        t_span=(cfg.t_start, t_end),
        y0=y0,
        t_eval=t,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    x = sol.y[0]
    v = sol.y[1]
    return t, x, v


def fit_harmonic_response(
    t: np.ndarray,
    x: np.ndarray,
    omega: float,
    tail_fraction: float,
) -> tuple[float, float, float, float]:
    """Fit tail segment to x ~= a*cos(omega t)+b*sin(omega t)+bias.

    Returns:
    - amplitude
    - phase (x ~= A*cos(omega t - phase))
    - bias
    - rmse
    """

    n = t.size
    start = int((1.0 - tail_fraction) * n)
    t_tail = t[start:]
    x_tail = x[start:]

    design = np.column_stack(
        [
            np.cos(omega * t_tail),
            np.sin(omega * t_tail),
            np.ones_like(t_tail),
        ]
    )
    coeff, *_ = np.linalg.lstsq(design, x_tail, rcond=None)
    a, b, bias = coeff

    amplitude = float(np.hypot(a, b))
    phase = float(np.arctan2(b, a))

    x_pred = design @ coeff
    rmse = float(np.sqrt(np.mean((x_tail - x_pred) ** 2)))
    return amplitude, phase, float(bias), rmse


def mean_tail(values: np.ndarray, tail_fraction: float) -> float:
    """Tail mean helper."""

    n = values.size
    start = int((1.0 - tail_fraction) * n)
    return float(np.mean(values[start:]))


def run_single_frequency_validation(cfg: ForcedOscillationConfig) -> dict[str, float]:
    """Validate one frequency point against analytic steady-state formula."""

    omega = cfg.drive_omega
    t, x, v = integrate_forced_oscillator(cfg, omega=omega, t_end=cfg.t_end, num_points=cfg.num_points)

    amp_theory, phase_theory = steady_state_amplitude_phase(cfg.m, cfg.c, cfg.k, cfg.f0, omega)
    amp_num, phase_num, bias, fit_rmse = fit_harmonic_response(t, x, omega, cfg.tail_fraction)

    amp_rel_err = abs(amp_num - amp_theory) / max(amp_theory, EPS)
    phase_abs_err = wrapped_phase_diff(phase_num, phase_theory)

    force = cfg.f0 * np.cos(omega * t)
    pin = force * v
    pdiss = cfg.c * v * v
    mean_pin = mean_tail(pin, cfg.tail_fraction)
    mean_pdiss = mean_tail(pdiss, cfg.tail_fraction)
    power_balance_rel_err = abs(mean_pin - mean_pdiss) / max(abs(mean_pdiss), EPS)

    return {
        "omega_main": float(omega),
        "amp_theory": float(amp_theory),
        "amp_numeric": float(amp_num),
        "amp_rel_err": float(amp_rel_err),
        "phase_theory_rad": float(phase_theory),
        "phase_numeric_rad": float(phase_num),
        "phase_abs_err_rad": float(phase_abs_err),
        "fit_bias": float(bias),
        "fit_rmse": float(fit_rmse),
        "mean_input_power": float(mean_pin),
        "mean_dissipation_power": float(mean_pdiss),
        "power_balance_rel_err": float(power_balance_rel_err),
    }


def run_frequency_sweep(cfg: ForcedOscillationConfig) -> pd.DataFrame:
    """Sweep drive frequency and compare numeric vs analytic steady-state amplitude."""

    omega0 = natural_frequency(cfg)
    omegas = np.linspace(cfg.freq_min_ratio * omega0, cfg.freq_max_ratio * omega0, cfg.num_freqs)

    rows: list[dict[str, float]] = []
    for omega in omegas:
        t, x, _ = integrate_forced_oscillator(
            cfg,
            omega=float(omega),
            t_end=cfg.sweep_t_end,
            num_points=cfg.sweep_num_points,
        )
        amp_num, phase_num, _, fit_rmse = fit_harmonic_response(t, x, float(omega), cfg.tail_fraction)
        amp_theory, phase_theory = steady_state_amplitude_phase(cfg.m, cfg.c, cfg.k, cfg.f0, float(omega))
        amp_rel_err = abs(amp_num - amp_theory) / max(amp_theory, EPS)
        phase_abs_err = wrapped_phase_diff(phase_num, phase_theory)

        rows.append(
            {
                "omega_drive_rad_s": float(omega),
                "freq_drive_hz": float(omega / (2.0 * np.pi)),
                "amp_theory": float(amp_theory),
                "amp_numeric": float(amp_num),
                "amp_rel_err": float(amp_rel_err),
                "phase_theory_rad": float(phase_theory),
                "phase_numeric_rad": float(phase_num),
                "phase_abs_err_rad": float(phase_abs_err),
                "fit_rmse": float(fit_rmse),
            }
        )

    return pd.DataFrame(rows)


def theoretical_peak_frequency(cfg: ForcedOscillationConfig) -> float:
    """Displacement-amplitude resonance peak for underdamped case (if exists)."""

    value = cfg.k / cfg.m - (cfg.c * cfg.c) / (2.0 * cfg.m * cfg.m)
    if value <= 0.0:
        return float("nan")
    return float(np.sqrt(value))


def build_summary(
    cfg: ForcedOscillationConfig,
    single: dict[str, float],
    sweep: pd.DataFrame,
) -> dict[str, float | str]:
    """Aggregate key diagnostics and decision flags."""

    idx_num = int(sweep["amp_numeric"].to_numpy().argmax())
    idx_theory = int(sweep["amp_theory"].to_numpy().argmax())

    omega_peak_numeric = float(sweep.iloc[idx_num]["omega_drive_rad_s"])
    omega_peak_theory_grid = float(sweep.iloc[idx_theory]["omega_drive_rad_s"])
    omega_peak_formula = theoretical_peak_frequency(cfg)

    median_amp_rel_err = float(sweep["amp_rel_err"].median())
    max_amp_rel_err = float(sweep["amp_rel_err"].max())
    median_phase_abs_err = float(sweep["phase_abs_err_rad"].median())

    checks = {
        "main_amp_rel_err_lt_5pct": single["amp_rel_err"] < 0.05,
        "main_phase_abs_err_lt_0p10": single["phase_abs_err_rad"] < 0.10,
        "median_sweep_amp_rel_err_lt_8pct": median_amp_rel_err < 0.08,
        "power_balance_rel_err_lt_8pct": single["power_balance_rel_err"] < 0.08,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "omega0_rad_s": natural_frequency(cfg),
        "zeta": damping_ratio(cfg),
        "omega_main_rad_s": single["omega_main"],
        "amp_main_theory": single["amp_theory"],
        "amp_main_numeric": single["amp_numeric"],
        "amp_main_rel_err": single["amp_rel_err"],
        "phase_main_abs_err_rad": single["phase_abs_err_rad"],
        "power_balance_rel_err": single["power_balance_rel_err"],
        "omega_peak_formula_rad_s": omega_peak_formula,
        "omega_peak_theory_grid_rad_s": omega_peak_theory_grid,
        "omega_peak_numeric_rad_s": omega_peak_numeric,
        "median_sweep_amp_rel_err": median_amp_rel_err,
        "max_sweep_amp_rel_err": max_amp_rel_err,
        "median_sweep_phase_abs_err_rad": median_phase_abs_err,
        "check_main_amp_rel_err_lt_5pct": str(checks["main_amp_rel_err_lt_5pct"]),
        "check_main_phase_abs_err_lt_0p10": str(checks["main_phase_abs_err_lt_0p10"]),
        "check_median_sweep_amp_rel_err_lt_8pct": str(checks["median_sweep_amp_rel_err_lt_8pct"]),
        "check_power_balance_rel_err_lt_8pct": str(checks["power_balance_rel_err_lt_8pct"]),
        "validation_status": status,
    }


def main() -> None:
    cfg = ForcedOscillationConfig()
    validate_config(cfg)

    single = run_single_frequency_validation(cfg)
    sweep = run_frequency_sweep(cfg)
    summary = build_summary(cfg, single, sweep)

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Forced Oscillations MVP (PHYS-0134) ===")
    print(
        "params: "
        f"m={cfg.m}, c={cfg.c}, k={cfg.k}, f0={cfg.f0}, "
        f"x0={cfg.x0}, v0={cfg.v0}"
    )
    print(
        f"main_omega={cfg.drive_omega:.6f} rad/s, "
        f"omega0={summary['omega0_rad_s']:.6f} rad/s, "
        f"zeta={summary['zeta']:.6f}"
    )
    print(
        f"time_main=[{cfg.t_start}, {cfg.t_end}], num_points_main={cfg.num_points}, "
        f"sweep_t_end={cfg.sweep_t_end}, num_freqs={cfg.num_freqs}"
    )

    print("\nMain-frequency validation:")
    main_df = pd.DataFrame([single])
    print(main_df.to_string(index=False))

    print("\nSweep (first 8 rows):")
    print(sweep.head(8).to_string(index=False))

    print("\nSweep around numeric peak (top 5 by amp_numeric):")
    print(sweep.sort_values("amp_numeric", ascending=False).head(5).to_string(index=False))

    print("\nSummary:")
    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    print(summary_df.to_string(index=False))

    if summary["validation_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
