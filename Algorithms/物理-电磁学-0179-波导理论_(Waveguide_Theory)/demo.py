"""Minimal runnable MVP for Waveguide Theory.

This demo models a rectangular metallic waveguide in TE mode (default TE10,
WR-90 dimensions), computes cutoff and dispersion quantities, and exports
frequency-sweep tables for quick inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants


EPS = 1e-12


@dataclass(frozen=True)
class WaveguideConfig:
    """Configuration for rectangular-waveguide TE-mode analysis."""

    a_m: float = 22.86e-3
    b_m: float = 10.16e-3
    m: int = 1
    n: int = 0
    eps_r: float = 1.0
    mu_r: float = 1.0

    sweep_start_ghz: float = 5.0
    sweep_stop_ghz: float = 18.0
    sweep_points: int = 260

    operating_freq_ghz: float = 10.0
    length_m: float = 0.5


def medium_speed(eps_r: float, mu_r: float) -> float:
    """Return wave speed in a homogeneous medium."""

    return constants.c / np.sqrt(eps_r * mu_r)


def medium_impedance(eps_r: float, mu_r: float) -> float:
    """Return intrinsic impedance in a homogeneous medium."""

    return np.sqrt(constants.mu_0 * mu_r / (constants.epsilon_0 * eps_r))


def cutoff_frequency_rectangular(
    a_m: float,
    b_m: float,
    m: int,
    n: int,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> float:
    """Cutoff frequency for rectangular-waveguide TEmn/TMmn index pair."""

    if a_m <= 0.0 or b_m <= 0.0:
        raise ValueError("waveguide dimensions must be positive")
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError("invalid mode indices: require non-negative and not both zero")
    if eps_r <= 0.0 or mu_r <= 0.0:
        raise ValueError("material parameters must be positive")

    c_med = medium_speed(eps_r, mu_r)
    return 0.5 * c_med * np.sqrt((m / a_m) ** 2 + (n / b_m) ** 2)


def te_mode_dispersion(
    f_hz: np.ndarray,
    fc_hz: float,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute TE-mode dispersion quantities for an array of frequencies."""

    f_hz = np.asarray(f_hz, dtype=float)
    if np.any(f_hz <= 0.0):
        raise ValueError("all frequencies must be positive")

    c_med = medium_speed(eps_r, mu_r)
    eta_med = medium_impedance(eps_r, mu_r)

    k = 2.0 * np.pi * f_hz / c_med
    ratio = (fc_hz / f_hz) ** 2
    propagating = f_hz > fc_hz * (1.0 + EPS)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(propagating, k * np.sqrt(1.0 - ratio), np.nan)
        alpha_ev = np.where(~propagating, k * np.sqrt(np.maximum(ratio - 1.0, 0.0)), np.nan)
        lambda_g = np.where(propagating, 2.0 * np.pi / beta, np.nan)
        v_phase = np.where(propagating, 2.0 * np.pi * f_hz / beta, np.nan)
        v_group = np.where(propagating, c_med * np.sqrt(1.0 - ratio), np.nan)
        z_te = np.where(propagating, eta_med / np.sqrt(1.0 - ratio), np.nan)

    return {
        "propagating": propagating,
        "beta": beta,
        "alpha_ev": alpha_ev,
        "lambda_g": lambda_g,
        "v_phase": v_phase,
        "v_group": v_group,
        "z_te": z_te,
    }


def dominant_te10_profile(a_m: float, samples: int = 121) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized TE10 Ey profile along x at a fixed z,t (shape only)."""

    x = np.linspace(0.0, a_m, samples)
    ey = np.sin(np.pi * x / a_m)
    return x, ey


def build_dispersion_table(cfg: WaveguideConfig) -> tuple[pd.DataFrame, float]:
    """Build a sweep table with propagating and below-cutoff regions."""

    f_ghz = np.linspace(cfg.sweep_start_ghz, cfg.sweep_stop_ghz, cfg.sweep_points)
    f_hz = f_ghz * 1e9

    fc_hz = cutoff_frequency_rectangular(
        a_m=cfg.a_m,
        b_m=cfg.b_m,
        m=cfg.m,
        n=cfg.n,
        eps_r=cfg.eps_r,
        mu_r=cfg.mu_r,
    )
    disp = te_mode_dispersion(f_hz=f_hz, fc_hz=fc_hz, eps_r=cfg.eps_r, mu_r=cfg.mu_r)

    table = pd.DataFrame(
        {
            "frequency_ghz": f_ghz,
            "propagating": disp["propagating"],
            "beta_rad_per_m": disp["beta"],
            "alpha_ev_np_per_m": disp["alpha_ev"],
            "lambda_g_mm": disp["lambda_g"] * 1e3,
            "v_phase_over_c": disp["v_phase"] / constants.c,
            "v_group_over_c": disp["v_group"] / constants.c,
            "z_te_ohm": disp["z_te"],
        }
    )
    return table, fc_hz


def evaluate_operating_point(cfg: WaveguideConfig, fc_hz: float) -> dict[str, float]:
    """Evaluate key quantities at one operating frequency."""

    f0_hz = cfg.operating_freq_ghz * 1e9
    disp0 = te_mode_dispersion(
        f_hz=np.array([f0_hz], dtype=float),
        fc_hz=fc_hz,
        eps_r=cfg.eps_r,
        mu_r=cfg.mu_r,
    )

    if not bool(disp0["propagating"][0]):
        raise RuntimeError("operating frequency is below cutoff; choose f0 > fc")

    beta = float(disp0["beta"][0])
    lambda_g_m = float(disp0["lambda_g"][0])
    vp = float(disp0["v_phase"][0])
    vg = float(disp0["v_group"][0])
    z_te = float(disp0["z_te"][0])

    gamma = 1j * beta  # lossless-waveguide approximation
    transfer = np.exp(-gamma * cfg.length_m)

    return {
        "f0_hz": f0_hz,
        "fc_hz": fc_hz,
        "beta": beta,
        "lambda0_m": constants.c / f0_hz,
        "lambda_g_m": lambda_g_m,
        "v_phase_over_c": vp / constants.c,
        "v_group_over_c": vg / constants.c,
        "z_te_ohm": z_te,
        "transfer_mag": float(np.abs(transfer)),
        "transfer_phase_rad": float(np.angle(transfer)),
    }


def main() -> None:
    cfg = WaveguideConfig()

    table, fc_hz = build_dispersion_table(cfg)
    op = evaluate_operating_point(cfg, fc_hz)

    x_m, ey = dominant_te10_profile(cfg.a_m)
    profile = pd.DataFrame(
        {
            "x_mm": x_m * 1e3,
            "Ey_norm": ey,
        }
    )

    out_table = Path(__file__).with_name("waveguide_dispersion.csv")
    out_profile = Path(__file__).with_name("te10_profile.csv")
    table.to_csv(out_table, index=False)
    profile.to_csv(out_profile, index=False)

    print("Waveguide Theory MVP: Rectangular guide TE10 dispersion")
    print(f"dimensions (a x b) : {cfg.a_m*1e3:.3f} mm x {cfg.b_m*1e3:.3f} mm")
    print(f"mode               : TE{cfg.m}{cfg.n}")
    print(f"cutoff frequency   : {fc_hz/1e9:.6f} GHz")
    print(f"operating frequency: {cfg.operating_freq_ghz:.6f} GHz")
    print(f"beta               : {op['beta']:.6f} rad/m")
    print(f"lambda0            : {op['lambda0_m']*1e3:.6f} mm")
    print(f"lambda_g           : {op['lambda_g_m']*1e3:.6f} mm")
    print(f"v_phase/c          : {op['v_phase_over_c']:.6f}")
    print(f"v_group/c          : {op['v_group_over_c']:.6f}")
    print(f"TE impedance       : {op['z_te_ohm']:.6f} Ohm")
    print(f"transfer |H|       : {op['transfer_mag']:.6f} (lossless approx)")
    print(f"transfer phase     : {op['transfer_phase_rad']:.6f} rad")

    sample_rows = table.iloc[np.linspace(0, len(table) - 1, 8, dtype=int)]
    print("\nSweep samples:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(sample_rows.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Sanity checks around WR-90 TE10.
    if not (6.4 < fc_hz / 1e9 < 6.7):
        raise RuntimeError("cutoff frequency outside expected WR-90 TE10 range")
    if not (op["v_phase_over_c"] > 1.0 and 0.0 < op["v_group_over_c"] < 1.0):
        raise RuntimeError("phase/group velocity relation is invalid")
    if not abs(op["v_phase_over_c"] * op["v_group_over_c"] - 1.0) < 1e-6:
        raise RuntimeError("expected v_phase * v_group = c^2 relation violated")
    if not (35.0 < op["lambda_g_m"] * 1e3 < 45.0):
        raise RuntimeError("guided wavelength at 10 GHz is outside expected range")

    print(f"\nSaved sweep table to: {out_table}")
    print(f"Saved TE10 profile to: {out_profile}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
