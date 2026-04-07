"""Minimal runnable MVP for Antenna Theory.

This demo models a thin center-fed dipole (default: half-wave, L=lambda/2),
computes its normalized radiation pattern U(theta), estimates directivity by
numerical integration, extracts HPBW, and evaluates a basic Friis link budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants
from scipy.integrate import simpson
from scipy.optimize import brentq


EPS = 1e-9


@dataclass(frozen=True)
class AntennaConfig:
    """Configuration for the dipole radiation/link calculation."""

    frequency_hz: float = 2.4e9
    length_over_lambda: float = 0.5
    theta_samples: int = 20001
    tx_power_w: float = 1.0
    link_distance_m: float = 100.0


def dipole_field_factor(theta_rad: np.ndarray, kL: float) -> np.ndarray:
    """Return far-field angular factor F(theta) for a thin dipole.

    F(theta) = [cos((kL/2) cos(theta)) - cos(kL/2)] / sin(theta)

    For theta close to 0 or pi, the exact limit goes to 0 for finite dipole
    lengths. We clip singular values to 0 for numerical stability.
    """

    sin_t = np.sin(theta_rad)
    numerator = np.cos(0.5 * kL * np.cos(theta_rad)) - np.cos(0.5 * kL)
    raw = np.divide(numerator, sin_t, out=np.zeros_like(theta_rad), where=np.abs(sin_t) > EPS)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def radiation_intensity(theta_rad: np.ndarray, length_over_lambda: float) -> np.ndarray:
    """Compute unnormalized radiation intensity U(theta) proportional to |F|^2."""

    kL = 2.0 * np.pi * length_over_lambda
    f_theta = dipole_field_factor(theta_rad, kL)
    return f_theta**2


def find_half_power_beamwidth(theta: np.ndarray, normalized_u: np.ndarray) -> tuple[float, float, float]:
    """Find left/right -3 dB angles and HPBW in radians.

    Uses dense-sampled interpolation + root finding.
    """

    peak_idx = int(np.argmax(normalized_u))

    def interp_minus_half(x: float) -> float:
        return float(np.interp(x, theta, normalized_u) - 0.5)

    left = brentq(interp_minus_half, theta[0], theta[peak_idx])
    right = brentq(interp_minus_half, theta[peak_idx], theta[-1])
    return left, right, right - left


def compute_dipole_metrics(cfg: AntennaConfig) -> dict[str, float | np.ndarray]:
    """Compute radiation, directivity, HPBW, effective aperture, and Friis link."""

    theta = np.linspace(EPS, np.pi - EPS, cfg.theta_samples)
    u = radiation_intensity(theta, cfg.length_over_lambda)

    u_max = float(np.max(u))
    if not np.isfinite(u_max) or u_max <= 0.0:
        raise ValueError("invalid radiation pattern: maximum intensity must be positive")

    u_norm = u / u_max

    # P_rad = integral over sphere of U(theta, phi) dOmega.
    # For an ideal thin linear dipole U is independent of phi.
    p_rad = 2.0 * np.pi * simpson(u * np.sin(theta), x=theta)
    directivity = 4.0 * np.pi * u_max / p_rad

    theta_l, theta_r, hpbw = find_half_power_beamwidth(theta, u_norm)

    wavelength_m = constants.c / cfg.frequency_hz
    gain_linear = directivity  # Assume 100% radiation efficiency for MVP.

    # Friis transmission equation in free space.
    p_rx_w = (
        cfg.tx_power_w
        * gain_linear
        * gain_linear
        * (wavelength_m / (4.0 * np.pi * cfg.link_distance_m)) ** 2
    )
    p_rx_dbm = 10.0 * np.log10(p_rx_w / 1e-3)
    effective_aperture_m2 = gain_linear * wavelength_m**2 / (4.0 * np.pi)

    return {
        "theta_rad": theta,
        "u_norm": u_norm,
        "directivity": float(directivity),
        "hpbw_deg": float(np.degrees(hpbw)),
        "theta_left_deg": float(np.degrees(theta_l)),
        "theta_right_deg": float(np.degrees(theta_r)),
        "wavelength_m": float(wavelength_m),
        "gain_linear": float(gain_linear),
        "effective_aperture_m2": float(effective_aperture_m2),
        "p_rx_w": float(p_rx_w),
        "p_rx_dbm": float(p_rx_dbm),
    }


def build_pattern_table(theta_rad: np.ndarray, u_norm: np.ndarray) -> pd.DataFrame:
    """Create a compact table of sampled normalized pattern values."""

    angles_deg = np.arange(0, 181, 15, dtype=float)
    sampled = np.interp(np.deg2rad(np.clip(angles_deg, 0.0 + EPS, 180.0 - EPS)), theta_rad, u_norm)

    # Force the two poles to zero to reflect the analytical limit.
    sampled[0] = 0.0
    sampled[-1] = 0.0

    table = pd.DataFrame(
        {
            "theta_deg": angles_deg.astype(int),
            "normalized_power": sampled,
            "power_dB": 10.0 * np.log10(np.maximum(sampled, 1e-12)),
        }
    )
    return table


def main() -> None:
    cfg = AntennaConfig()
    metrics = compute_dipole_metrics(cfg)

    pattern_table = build_pattern_table(
        theta_rad=metrics["theta_rad"],
        u_norm=metrics["u_norm"],
    )

    out_csv = Path(__file__).with_name("pattern_samples.csv")
    pattern_table.to_csv(out_csv, index=False)

    print("Antenna Theory MVP: Thin half-wave dipole + Friis link")
    print(f"frequency        : {cfg.frequency_hz/1e9:.3f} GHz")
    print(f"length/lambda    : {cfg.length_over_lambda:.3f}")
    print(f"wavelength       : {metrics['wavelength_m']:.6f} m")
    print(f"directivity      : {metrics['directivity']:.6f} ({10*np.log10(metrics['directivity']):.3f} dBi)")
    print(f"HPBW             : {metrics['hpbw_deg']:.3f} deg")
    print(f"-3 dB angles     : {metrics['theta_left_deg']:.3f} deg, {metrics['theta_right_deg']:.3f} deg")
    print(f"effective area   : {metrics['effective_aperture_m2']:.6e} m^2")
    print(f"tx power         : {cfg.tx_power_w:.3f} W")
    print(f"distance         : {cfg.link_distance_m:.3f} m")
    print(f"received power   : {metrics['p_rx_w']:.6e} W ({metrics['p_rx_dbm']:.3f} dBm)")

    print("\nPattern samples (15 deg step):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(pattern_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Basic sanity checks for half-wave dipole numerical result.
    if not (1.5 < metrics["directivity"] < 1.8):
        raise RuntimeError("directivity is outside expected half-wave dipole range")
    if not (60.0 < metrics["hpbw_deg"] < 110.0):
        raise RuntimeError("HPBW is outside expected half-wave dipole range")

    print(f"\nSaved table to: {out_csv}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
