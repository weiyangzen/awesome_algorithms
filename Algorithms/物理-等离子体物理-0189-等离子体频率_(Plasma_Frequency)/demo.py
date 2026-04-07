"""Plasma frequency MVP demo.

This script demonstrates:
1) analytical plasma-frequency computation,
2) numerical oscillation-based frequency recovery,
3) cold-plasma EM cutoff analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physical constants (SI).
EPS0 = 8.854_187_812_8e-12
E_CHARGE = 1.602_176_634e-19
M_ELECTRON = 9.109_383_701_5e-31
C_LIGHT = 299_792_458.0
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PlasmaConfig:
    """Configuration for the minimal plasma-frequency experiment."""

    n_e: float = 1.0e16
    x0: float = 1.0e-5
    v0: float = 0.0
    dt_factor: float = 0.02
    steps: int = 40_000
    probe_frequency_hz: float = 1.0e9


def plasma_angular_frequency(n_e: float) -> float:
    """Electron plasma angular frequency omega_pe in rad/s."""
    if n_e <= 0.0:
        raise ValueError("n_e must be positive.")
    return float(np.sqrt(n_e * E_CHARGE**2 / (EPS0 * M_ELECTRON)))


def plasma_frequency_hz(n_e: float) -> float:
    """Electron plasma frequency f_pe in Hz."""
    return plasma_angular_frequency(n_e) / TWO_PI


def critical_density_for_frequency(f_hz: float) -> float:
    """Critical density where omega_pe == 2*pi*f."""
    if f_hz <= 0.0:
        raise ValueError("f_hz must be positive.")
    omega = TWO_PI * f_hz
    return float(EPS0 * M_ELECTRON * omega**2 / E_CHARGE**2)


def simulate_cold_plasma_oscillation(cfg: PlasmaConfig) -> dict[str, np.ndarray | float]:
    """Simulate x'' + omega_pe^2 x = 0 with symplectic Euler."""
    omega = plasma_angular_frequency(cfg.n_e)
    dt = cfg.dt_factor / omega
    steps = cfg.steps

    t = np.arange(steps, dtype=float) * dt
    x = np.empty(steps, dtype=float)
    v = np.empty(steps, dtype=float)
    energy = np.empty(steps, dtype=float)

    x[0] = cfg.x0
    v[0] = cfg.v0
    energy[0] = 0.5 * (v[0] ** 2 + (omega * x[0]) ** 2)

    omega_sq = omega * omega
    for i in range(steps - 1):
        v_next = v[i] - omega_sq * x[i] * dt
        x_next = x[i] + v_next * dt

        v[i + 1] = v_next
        x[i + 1] = x_next
        energy[i + 1] = 0.5 * (v_next**2 + (omega * x_next) ** 2)

    return {"t": t, "x": x, "v": v, "energy": energy, "dt": dt, "omega_ref": omega}


def estimate_angular_frequency_fft(signal: np.ndarray, dt: float) -> float:
    """Estimate dominant angular frequency from FFT peak with quadratic refinement."""
    centered = np.asarray(signal, dtype=float) - float(np.mean(signal))
    n = centered.size
    if n < 16:
        raise ValueError("Signal is too short for robust FFT estimation.")

    window = np.hanning(n)
    spectrum = np.fft.rfft(centered * window)
    magnitudes = np.abs(spectrum)
    freqs_hz = np.fft.rfftfreq(n, d=dt)

    peak = int(np.argmax(magnitudes[1:]) + 1)
    f_peak = freqs_hz[peak]

    # Local parabolic interpolation for sub-bin correction.
    if 1 <= peak < magnitudes.size - 1:
        y0 = np.log(max(magnitudes[peak - 1], 1e-300))
        y1 = np.log(max(magnitudes[peak], 1e-300))
        y2 = np.log(max(magnitudes[peak + 1], 1e-300))
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-18:
            delta = 0.5 * (y0 - y2) / denom
            bin_width = freqs_hz[1] - freqs_hz[0]
            f_peak = f_peak + delta * bin_width

    return float(TWO_PI * f_peak)


def build_density_scan_table(n_values: np.ndarray) -> pd.DataFrame:
    """Build omega/f/period table for a density sweep."""
    omega = np.array([plasma_angular_frequency(float(n)) for n in n_values], dtype=float)
    freq = omega / TWO_PI
    period = TWO_PI / omega
    return pd.DataFrame(
        {
            "n_e_m^-3": n_values,
            "omega_pe_rad_s^-1": omega,
            "f_pe_Hz": freq,
            "period_s": period,
        }
    )


def cutoff_analysis_table(n_values: np.ndarray, probe_frequency_hz: float) -> pd.DataFrame:
    """Cold-plasma cutoff analysis for a fixed probe frequency."""
    omega_probe = TWO_PI * probe_frequency_hz
    omega_pe = np.array([plasma_angular_frequency(float(n)) for n in n_values], dtype=float)

    propagate = omega_probe >= omega_pe
    skin_depth = np.full_like(omega_pe, np.nan, dtype=float)
    cutoff_mask = ~propagate
    kappa = np.sqrt(np.maximum(omega_pe[cutoff_mask] ** 2 - omega_probe**2, 0.0)) / C_LIGHT
    skin_depth[cutoff_mask] = 1.0 / np.maximum(kappa, 1e-300)

    return pd.DataFrame(
        {
            "n_e_m^-3": n_values,
            "omega_pe_rad_s^-1": omega_pe,
            "probe_frequency_Hz": probe_frequency_hz,
            "propagates": propagate,
            "skin_depth_m_if_cutoff": skin_depth,
        }
    )


def main() -> None:
    cfg = PlasmaConfig()

    omega_ref = plasma_angular_frequency(cfg.n_e)
    freq_ref = omega_ref / TWO_PI
    period_ref = 1.0 / freq_ref

    sim = simulate_cold_plasma_oscillation(cfg)
    x = np.asarray(sim["x"], dtype=float)
    energy = np.asarray(sim["energy"], dtype=float)
    dt = float(sim["dt"])

    start = cfg.steps // 5
    omega_est = estimate_angular_frequency_fft(x[start:], dt)
    rel_freq_err = abs(omega_est - omega_ref) / omega_ref

    energy_drift = abs(energy[-1] - energy[0]) / energy[0]

    n_values = np.geomspace(1.0e14, 1.0e20, 8)
    density_df = build_density_scan_table(n_values)
    cutoff_df = cutoff_analysis_table(n_values, cfg.probe_frequency_hz)

    n_crit = critical_density_for_frequency(cfg.probe_frequency_hz)
    classification_at_cfg = "propagates" if cfg.n_e < n_crit else "cutoff"

    print("== Plasma Frequency MVP ==")
    print(f"n_e = {cfg.n_e:.3e} m^-3")
    print(f"omega_pe (analytic) = {omega_ref:.6e} rad/s")
    print(f"f_pe (analytic) = {freq_ref:.6e} Hz")
    print(f"T_pe (analytic) = {period_ref:.6e} s")
    print()
    print("== Oscillation Validation ==")
    print(f"omega_pe (estimated from simulation) = {omega_est:.6e} rad/s")
    print(f"relative frequency error = {rel_freq_err:.3e}")
    print(f"normalized energy drift = {energy_drift:.3e}")
    print()
    print("== Density Scan (head) ==")
    print(density_df.to_string(index=False))
    print()
    print(f"== Cutoff Analysis at probe f = {cfg.probe_frequency_hz:.3e} Hz ==")
    print(f"critical density n_crit = {n_crit:.6e} m^-3")
    print(f"classification at n_e={cfg.n_e:.3e}: {classification_at_cfg}")
    print(cutoff_df.to_string(index=False))

    # Deterministic acceptance checks.
    monotonic = bool(np.all(np.diff(density_df["omega_pe_rad_s^-1"].to_numpy()) > 0.0))
    low_density_propagates = bool(cutoff_df["propagates"].iloc[0])
    high_density_cutoff = bool(~cutoff_df["propagates"].iloc[-1])

    assert rel_freq_err < 2.0e-2, f"Frequency error too large: {rel_freq_err:.3e}"
    assert energy_drift < 2.0e-2, f"Energy drift too large: {energy_drift:.3e}"
    assert monotonic, "omega_pe should increase monotonically with n_e."
    assert low_density_propagates, "Lowest-density sample should propagate for 1 GHz probe."
    assert high_density_cutoff, "Highest-density sample should be cutoff for 1 GHz probe."

    print()
    print("Validation: PASS")


if __name__ == "__main__":
    main()
