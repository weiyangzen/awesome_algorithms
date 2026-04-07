"""Minimal runnable MVP for Kerr effect (nonlinear index + SPM)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift


@dataclass(frozen=True)
class KerrConfig:
    """Configuration for a single-pass Kerr phase modulation simulation."""

    wavelength_m: float = 1.03e-6
    n0: float = 1.45
    n2_m2_per_w: float = 2.6e-20
    length_m: float = 0.01
    effective_area_m2: float = 80e-12
    pulse_fwhm_s: float = 120e-15
    pulse_energy_j: float = 1.2e-9
    time_window_s: float = 2.5e-12
    sample_points: int = 2**14


def _validate_config(cfg: KerrConfig) -> None:
    if cfg.wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if cfg.n0 <= 0:
        raise ValueError("n0 must be positive")
    if cfg.n2_m2_per_w <= 0:
        raise ValueError("n2_m2_per_w must be positive")
    if cfg.length_m <= 0:
        raise ValueError("length_m must be positive")
    if cfg.effective_area_m2 <= 0:
        raise ValueError("effective_area_m2 must be positive")
    if cfg.pulse_fwhm_s <= 0:
        raise ValueError("pulse_fwhm_s must be positive")
    if cfg.pulse_energy_j <= 0:
        raise ValueError("pulse_energy_j must be positive")
    if cfg.time_window_s <= 0:
        raise ValueError("time_window_s must be positive")
    if cfg.sample_points < 1024:
        raise ValueError("sample_points must be >= 1024")


def _peak_power_from_energy(pulse_energy_j: float, pulse_fwhm_s: float) -> float:
    """Compute Gaussian pulse peak power from pulse energy and FWHM duration."""

    integral_coeff = pulse_fwhm_s * math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
    return pulse_energy_j / integral_coeff


def _normalized_psd(field: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    spec = fftshift(fft(field))
    freqs_hz = fftshift(fftfreq(field.size, d=dt))
    power = np.abs(spec) ** 2
    power_sum = float(np.sum(power))
    if power_sum <= 0:
        raise ValueError("Degenerate spectrum with zero total power")
    return freqs_hz, power / power_sum


def _rms_bandwidth_hz(freqs_hz: np.ndarray, psd: np.ndarray) -> float:
    mean = float(np.sum(freqs_hz * psd))
    variance = float(np.sum(((freqs_hz - mean) ** 2) * psd))
    return math.sqrt(max(variance, 0.0))


def simulate_single_pass(cfg: KerrConfig) -> dict[str, object]:
    _validate_config(cfg)

    t = np.linspace(
        -0.5 * cfg.time_window_s,
        0.5 * cfg.time_window_s,
        cfg.sample_points,
        endpoint=False,
    )
    dt = float(t[1] - t[0])

    p_peak_w = _peak_power_from_energy(cfg.pulse_energy_j, cfg.pulse_fwhm_s)
    profile = np.exp(-4.0 * np.log(2.0) * (t / cfg.pulse_fwhm_s) ** 2)
    power_w = p_peak_w * profile
    intensity_w_m2 = power_w / cfg.effective_area_m2

    k0 = 2.0 * math.pi / cfg.wavelength_m
    delta_n = cfg.n2_m2_per_w * intensity_w_m2
    phi_nl = k0 * delta_n * cfg.length_m

    delta_omega_rad_s = -np.gradient(phi_nl, dt)

    e_in = np.sqrt(power_w)
    e_out = e_in * np.exp(1j * phi_nl)

    f_in_hz, psd_in = _normalized_psd(e_in, dt)
    f_out_hz, psd_out = _normalized_psd(e_out, dt)

    bw_in_hz = _rms_bandwidth_hz(f_in_hz, psd_in)
    bw_out_hz = _rms_bandwidth_hz(f_out_hz, psd_out)

    return {
        "config": cfg,
        "t_s": t,
        "power_w": power_w,
        "intensity_w_m2": intensity_w_m2,
        "delta_n": delta_n,
        "phi_nl": phi_nl,
        "delta_omega_rad_s": delta_omega_rad_s,
        "freq_hz": f_out_hz,
        "psd_in": psd_in,
        "psd_out": psd_out,
        "p_peak_w": float(np.max(power_w)),
        "i_peak_w_m2": float(np.max(intensity_w_m2)),
        "delta_n_peak": float(np.max(delta_n)),
        "phi_nl_peak": float(np.max(phi_nl)),
        "bw_in_hz": bw_in_hz,
        "bw_out_hz": bw_out_hz,
        "bw_ratio": bw_out_hz / bw_in_hz,
    }


def _build_energy_scan(base_cfg: KerrConfig, factors: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for factor in factors:
        cfg = replace(base_cfg, pulse_energy_j=base_cfg.pulse_energy_j * factor)
        result = simulate_single_pass(cfg)
        rows.append(
            {
                "energy_nj": cfg.pulse_energy_j * 1e9,
                "phi_nl_peak_rad": float(result["phi_nl_peak"]),
                "phi_per_energy": float(result["phi_nl_peak"]) / cfg.pulse_energy_j,
                "bw_ratio": float(result["bw_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def _build_temporal_sample(result: dict[str, object], sample_count: int = 9) -> pd.DataFrame:
    t = result["t_s"]
    power = result["power_w"]
    intensity = result["intensity_w_m2"]
    phi_nl = result["phi_nl"]
    delta_omega = result["delta_omega_rad_s"]

    idx = np.linspace(0, len(t) - 1, sample_count, dtype=int)
    return pd.DataFrame(
        {
            "t_fs": t[idx] * 1e15,
            "power_kW": power[idx] / 1e3,
            "intensity_TW_per_m2": intensity[idx] / 1e12,
            "phi_nl_rad": phi_nl[idx],
            "delta_omega_GHz": delta_omega[idx] / (2.0 * math.pi * 1e9),
        }
    )


def main() -> None:
    base_cfg = KerrConfig()
    base_result = simulate_single_pass(base_cfg)

    energy_scan = _build_energy_scan(base_cfg, factors=[0.5, 1.0, 2.0])
    temporal_sample = _build_temporal_sample(base_result)

    phi_per_energy = energy_scan["phi_per_energy"].to_numpy()
    ratio_spread = float((phi_per_energy.max() - phi_per_energy.min()) / phi_per_energy.mean())

    bw_ratios = energy_scan["bw_ratio"].to_numpy()
    bw_monotonic = bool(np.all(np.diff(bw_ratios) > 0.0))

    linearity_pass = ratio_spread < 1e-9
    validation_pass = linearity_pass and bw_monotonic

    print("=== Kerr Effect MVP: Single-pass SPM ===")
    print(
        "Core metrics:\n"
        f"  peak_power = {base_result['p_peak_w'] / 1e3:.3f} kW\n"
        f"  peak_intensity = {base_result['i_peak_w_m2'] / 1e12:.3f} TW/m^2\n"
        f"  delta_n_max = {base_result['delta_n_peak']:.6e}\n"
        f"  phi_nl_peak = {base_result['phi_nl_peak']:.6f} rad\n"
        f"  rms_bw_in = {base_result['bw_in_hz'] / 1e9:.3f} GHz\n"
        f"  rms_bw_out = {base_result['bw_out_hz'] / 1e9:.3f} GHz\n"
        f"  bw_broadening_ratio = {base_result['bw_ratio']:.6f}"
    )

    print("\nEnergy scaling check:")
    print(energy_scan.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6e}"))

    print("\nTemporal sample (auditing points):")
    print(temporal_sample.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6e}"))

    print("\nValidation checks:")
    print(f"  linear_phi_energy_spread = {ratio_spread:.3e} (threshold < 1e-9)")
    print(f"  bandwidth_monotonic = {bw_monotonic}")
    print(f"Validation: {'PASS' if validation_pass else 'FAIL'}")

    if not validation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
