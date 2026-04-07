"""Minimal runnable MVP for laser principle (PHYS-0097).

This script implements a semiclassical class-B laser rate-equation model:
- population inversion n(t)
- cavity photon number s(t)

It validates key laser-principle behavior:
1) threshold pump p_th = cavity_loss / gain
2) strong above-threshold increase in steady-state photon number
3) near-linear L-I relation above threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class LaserConfig:
    gain: float = 1.0  # stimulated-emission gain coefficient g
    cavity_loss: float = 0.8  # photon loss coefficient k
    beta_sp: float = 1e-4  # spontaneous-emission coupling beta
    pump_min: float = 0.2
    pump_max: float = 1.4
    pump_points: int = 25
    t_end: float = 120.0
    steps: int = 3000
    n0: float = 0.0
    s0: float = 1e-8


@dataclass
class PumpSample:
    pump: float
    n_ss: float
    s_ss: float


@dataclass
class ValidationReport:
    threshold_theory: float
    threshold_estimated: float
    threshold_abs_error: float
    low_pump: float
    low_photon_ss: float
    high_pump: float
    high_photon_ss: float
    high_low_ratio: float
    slope_relative_error: float


def laser_rhs(_: float, y: np.ndarray, pump: float, cfg: LaserConfig) -> np.ndarray:
    """Rate equations in normalized units.

    dn/dt = pump - n - g*n*s
    ds/dt = -k*s + g*n*s + beta*n
    """
    n, s = y
    dn_dt = pump - n - cfg.gain * n * s
    ds_dt = -cfg.cavity_loss * s + cfg.gain * n * s + cfg.beta_sp * n
    return np.array([dn_dt, ds_dt], dtype=float)


def simulate_single_pump(cfg: LaserConfig, pump: float) -> PumpSample:
    t_eval = np.linspace(0.0, cfg.t_end, cfg.steps)
    y0 = np.array([cfg.n0, cfg.s0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: laser_rhs(t, y, pump, cfg),
        t_span=(0.0, cfg.t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed for pump={pump:.4f}: {sol.message}")

    n_ss = float(sol.y[0, -1])
    s_ss = float(sol.y[1, -1])
    if s_ss < -1e-9:
        raise RuntimeError("Steady photon number became non-physical (negative).")

    return PumpSample(pump=float(pump), n_ss=n_ss, s_ss=max(0.0, s_ss))


def sweep_pump(cfg: LaserConfig) -> list[PumpSample]:
    pumps = np.linspace(cfg.pump_min, cfg.pump_max, cfg.pump_points)
    return [simulate_single_pump(cfg, float(p)) for p in pumps]


def estimate_threshold_from_curve(pumps: np.ndarray, photons: np.ndarray) -> float:
    """Estimate threshold by maximal curvature in log(1+s) curve."""
    transformed = np.log1p(photons)
    second_derivative = np.gradient(np.gradient(transformed, pumps), pumps)
    idx = int(np.argmax(second_derivative))
    return float(pumps[idx])


def estimate_slope_above_threshold(pumps: np.ndarray, photons: np.ndarray, p_th: float) -> float:
    mask = pumps >= p_th + 0.15
    if np.count_nonzero(mask) < 4:
        raise RuntimeError("Not enough above-threshold samples for slope fit.")

    coef = np.polyfit(pumps[mask], photons[mask], deg=1)
    slope = float(coef[0])
    return slope


def build_table(samples: list[PumpSample], cfg: LaserConfig) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "pump": [x.pump for x in samples],
            "n_ss": [x.n_ss for x in samples],
            "s_ss": [x.s_ss for x in samples],
        }
    )
    df["n_threshold"] = cfg.cavity_loss / cfg.gain
    df["above_threshold"] = df["pump"] > df["n_threshold"]
    return df


def make_validation_report(df: pd.DataFrame, cfg: LaserConfig) -> ValidationReport:
    pumps = df["pump"].to_numpy(dtype=float)
    photons = df["s_ss"].to_numpy(dtype=float)

    p_th_theory = cfg.cavity_loss / cfg.gain
    p_th_est = estimate_threshold_from_curve(pumps, photons)

    low_idx = int(np.argmin(np.abs(pumps - (p_th_theory - 0.2))))
    high_idx = int(np.argmin(np.abs(pumps - (p_th_theory + 0.25))))

    low_pump = float(pumps[low_idx])
    high_pump = float(pumps[high_idx])
    low_s = float(photons[low_idx])
    high_s = float(photons[high_idx])

    high_low_ratio = high_s / max(low_s, 1e-12)

    fitted_slope = estimate_slope_above_threshold(pumps, photons, p_th_theory)
    slope_theory = 1.0 / cfg.cavity_loss  # beta->0 approximation from steady-state branch
    slope_relative_error = abs(fitted_slope - slope_theory) / slope_theory

    return ValidationReport(
        threshold_theory=p_th_theory,
        threshold_estimated=p_th_est,
        threshold_abs_error=abs(p_th_est - p_th_theory),
        low_pump=low_pump,
        low_photon_ss=low_s,
        high_pump=high_pump,
        high_photon_ss=high_s,
        high_low_ratio=high_low_ratio,
        slope_relative_error=slope_relative_error,
    )


def print_preview(df: pd.DataFrame) -> None:
    center = len(df) // 2
    preview = pd.concat([df.head(4), df.iloc[center - 2 : center + 2], df.tail(4)], axis=0)
    print(preview.to_string(index=False))


def main() -> None:
    cfg = LaserConfig(
        gain=1.0,
        cavity_loss=0.8,
        beta_sp=1e-4,
        pump_min=0.2,
        pump_max=1.4,
        pump_points=25,
        t_end=120.0,
        steps=3000,
        n0=0.0,
        s0=1e-8,
    )

    samples = sweep_pump(cfg)
    table = build_table(samples, cfg)
    report = make_validation_report(table, cfg)

    checks = {
        "threshold abs error < 0.08": report.threshold_abs_error < 0.08,
        "above/below photon ratio > 30": report.high_low_ratio > 30.0,
        "above-threshold slope rel error < 0.15": report.slope_relative_error < 0.15,
    }

    print("=== Laser Principle MVP (PHYS-0097) ===")
    print("Model: semiclassical rate equations for inversion n and photon number s")

    print("\n[Threshold validation]")
    print(
        "theory p_th = {pth:.4f}, estimated p_th = {pest:.4f}, abs_error = {err:.4f}".format(
            pth=report.threshold_theory,
            pest=report.threshold_estimated,
            err=report.threshold_abs_error,
        )
    )

    print("\n[Below vs above threshold]")
    print(
        "low pump {lp:.3f} -> s_ss = {ls:.6f}; high pump {hp:.3f} -> s_ss = {hs:.6f}; ratio = {rt:.2f}".format(
            lp=report.low_pump,
            ls=report.low_photon_ss,
            hp=report.high_pump,
            hs=report.high_photon_ss,
            rt=report.high_low_ratio,
        )
    )

    print("\n[L-I linearity check]")
    print(
        "fitted slope error (relative to 1/k) = {err:.4f}".format(
            err=report.slope_relative_error,
        )
    )

    print("\n[Pump sweep sample]")
    print_preview(table)

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
