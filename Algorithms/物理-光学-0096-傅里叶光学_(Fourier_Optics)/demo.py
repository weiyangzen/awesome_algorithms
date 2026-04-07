"""Minimal runnable MVP for Fourier Optics (PHYS-0096).

This script demonstrates two core Fourier-optics workflows:
1) Fraunhofer diffraction of a 1D single slit via FFT and comparison to sinc^2.
2) A simplified 4f low-pass spatial filtering experiment in frequency domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift


@dataclass
class FourierOpticsConfig:
    wavelength: float = 532e-9  # meter
    focal_length: float = 0.20  # meter
    slit_width: float = 120e-6  # meter
    aperture_half_size: float = 5e-3  # meter
    num_samples: int = 4096
    lowpass_cutoff: float = 1500.0  # cycles / meter
    object_low_freq: float = 600.0  # cycles / meter
    object_high_freq: float = 4200.0  # cycles / meter


@dataclass
class FraunhoferReport:
    rmse_center_band: float
    expected_first_zero_freq: float
    observed_first_zero_freq: float
    zero_relative_error: float


@dataclass
class FilterReport:
    low_freq_ratio: float
    high_freq_ratio: float
    high_freq_reduction_db: float


def make_grid(cfg: FourierOpticsConfig) -> tuple[np.ndarray, float]:
    """Generate aperture-plane sampling grid."""
    x = np.linspace(
        -cfg.aperture_half_size,
        cfg.aperture_half_size,
        cfg.num_samples,
        endpoint=False,
    )
    dx = float(x[1] - x[0])
    return x, dx


def build_single_slit_aperture(x: np.ndarray, slit_width: float) -> np.ndarray:
    """Unit-amplitude single-slit aperture field U0(x)."""
    return (np.abs(x) <= 0.5 * slit_width).astype(np.complex128)


def fourier_plane_field(u0: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fourier-plane field and frequency axis.

    Continuous Fourier integral is approximated by:
      Uf(fx) ~= FFT(U0) * dx
    """
    if u0.ndim != 1:
        raise ValueError("u0 must be 1D.")

    n = u0.size
    fx = fftshift(fftfreq(n, d=dx))
    uf = fftshift(fft(ifftshift(u0))) * dx
    return fx, uf


def normalized_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(field) ** 2
    peak = float(np.max(intensity))
    if peak <= 0.0:
        raise RuntimeError("Field intensity peak must be positive.")
    return intensity / peak


def analytic_single_slit_intensity(fx: np.ndarray, slit_width: float) -> np.ndarray:
    """Analytic Fraunhofer intensity for a rectangular aperture.

    Amplitude ~ slit_width * sinc(slit_width * fx)
    Intensity ~ |Amplitude|^2
    """
    amplitude = slit_width * np.sinc(slit_width * fx)
    intensity = np.abs(amplitude) ** 2
    return intensity / np.max(intensity)


def sensor_coordinate_mm(fx: np.ndarray, cfg: FourierOpticsConfig) -> np.ndarray:
    """Map spatial frequency to focal-plane coordinate: x' = lambda * f * fx."""
    x_sensor = cfg.wavelength * cfg.focal_length * fx
    return x_sensor * 1e3


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_first_zero_error(
    fx: np.ndarray,
    intensity: np.ndarray,
    slit_width: float,
) -> tuple[float, float, float]:
    """Estimate first positive zero position and compare with theory 1/slit_width."""
    expected = 1.0 / slit_width
    positive = fx > 0.0
    fx_p = fx[positive]
    i_p = intensity[positive]

    window = (fx_p >= 0.6 * expected) & (fx_p <= 1.4 * expected)
    if np.count_nonzero(window) < 3:
        raise RuntimeError("Insufficient samples near expected first zero.")

    fx_w = fx_p[window]
    i_w = i_p[window]
    observed = float(fx_w[int(np.argmin(i_w))])
    rel_error = abs(observed - expected) / expected
    return float(expected), observed, float(rel_error)


def build_object_signal(x: np.ndarray, cfg: FourierOpticsConfig) -> np.ndarray:
    """Create a deterministic object containing low/high spatial frequencies."""
    base = 0.6
    low = 0.30 * np.cos(2.0 * np.pi * cfg.object_low_freq * x)
    high = 0.15 * np.cos(2.0 * np.pi * cfg.object_high_freq * x)
    return base + low + high


def apply_4f_lowpass(
    signal: np.ndarray,
    dx: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a simplified 4f low-pass filter in spatial-frequency domain."""
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")

    n = signal.size
    fx = fftshift(fftfreq(n, d=dx))
    spectrum = fftshift(fft(ifftshift(signal)))
    mask = np.abs(fx) <= cutoff
    filtered_spectrum = spectrum * mask
    filtered = np.real(fftshift(ifft(ifftshift(filtered_spectrum))))
    return fx, spectrum, filtered


def sinusoid_amplitude(signal: np.ndarray, x: np.ndarray, freq: float) -> float:
    """Estimate cosine-component amplitude at a known frequency."""
    basis = np.exp(-1j * 2.0 * np.pi * freq * x)
    coeff = np.sum(signal * basis) / signal.size
    return float(2.0 * np.abs(coeff))


def build_fraunhofer_report(cfg: FourierOpticsConfig) -> tuple[FraunhoferReport, pd.DataFrame]:
    x, dx = make_grid(cfg)
    u0 = build_single_slit_aperture(x, cfg.slit_width)
    fx, uf = fourier_plane_field(u0, dx)

    numeric_i = normalized_intensity(uf)
    analytic_i = analytic_single_slit_intensity(fx, cfg.slit_width)

    center_band = np.abs(fx) <= (3.0 / cfg.slit_width)
    error_rmse = rmse(numeric_i[center_band], analytic_i[center_band])

    expected_zero, observed_zero, zero_rel = compute_first_zero_error(
        fx=fx,
        intensity=numeric_i,
        slit_width=cfg.slit_width,
    )

    profile = pd.DataFrame(
        {
            "fx_cycle_per_m": fx,
            "sensor_x_mm": sensor_coordinate_mm(fx, cfg),
            "I_numeric_norm": numeric_i,
            "I_analytic_norm": analytic_i,
        }
    )

    report = FraunhoferReport(
        rmse_center_band=error_rmse,
        expected_first_zero_freq=expected_zero,
        observed_first_zero_freq=observed_zero,
        zero_relative_error=zero_rel,
    )
    return report, profile


def build_filter_report(cfg: FourierOpticsConfig) -> FilterReport:
    x, dx = make_grid(cfg)
    obj = build_object_signal(x, cfg)
    _, _, filtered = apply_4f_lowpass(obj, dx, cfg.lowpass_cutoff)

    low_before = sinusoid_amplitude(obj, x, cfg.object_low_freq)
    low_after = sinusoid_amplitude(filtered, x, cfg.object_low_freq)
    high_before = sinusoid_amplitude(obj, x, cfg.object_high_freq)
    high_after = sinusoid_amplitude(filtered, x, cfg.object_high_freq)

    eps = 1e-15
    low_ratio = low_after / (low_before + eps)
    high_ratio = high_after / (high_before + eps)
    high_reduction_db = -20.0 * np.log10(max(high_ratio, 1e-12))

    return FilterReport(
        low_freq_ratio=float(low_ratio),
        high_freq_ratio=float(high_ratio),
        high_freq_reduction_db=float(high_reduction_db),
    )


def preview_profile_table(profile: pd.DataFrame, slit_width: float) -> pd.DataFrame:
    window = profile[np.abs(profile["fx_cycle_per_m"]) <= (2.2 / slit_width)]
    if window.empty:
        return profile.head(10)
    sample_idx = np.linspace(0, len(window) - 1, 11, dtype=int)
    return window.iloc[sample_idx]


def main() -> None:
    cfg = FourierOpticsConfig()

    fraunhofer_report, profile = build_fraunhofer_report(cfg)
    filter_report = build_filter_report(cfg)
    sample_table = preview_profile_table(profile, cfg.slit_width)

    checks = {
        "Fraunhofer central-band RMSE < 0.03": fraunhofer_report.rmse_center_band < 0.03,
        "First-zero relative error < 8%": fraunhofer_report.zero_relative_error < 0.08,
        "High-frequency ratio < 0.20": filter_report.high_freq_ratio < 0.20,
        "Low-frequency ratio in [0.85, 1.15]": 0.85 <= filter_report.low_freq_ratio <= 1.15,
    }

    print("=== Fourier Optics MVP (PHYS-0096) ===")
    print("Model: single-slit Fraunhofer diffraction + simplified 4f low-pass filtering")

    print("\n[Fraunhofer validation]")
    print(
        "RMSE(center band) = {rmse:.4e}, expected first zero = {fz_exp:.2f} cyc/m, "
        "observed first zero = {fz_obs:.2f} cyc/m, rel_error = {rel:.3e}".format(
            rmse=fraunhofer_report.rmse_center_band,
            fz_exp=fraunhofer_report.expected_first_zero_freq,
            fz_obs=fraunhofer_report.observed_first_zero_freq,
            rel=fraunhofer_report.zero_relative_error,
        )
    )

    print("\n[4f low-pass report]")
    print(
        "low-frequency ratio = {lr:.4f}, high-frequency ratio = {hr:.4f}, "
        "high-frequency reduction = {db:.2f} dB".format(
            lr=filter_report.low_freq_ratio,
            hr=filter_report.high_freq_ratio,
            db=filter_report.high_freq_reduction_db,
        )
    )

    print("\n[Sample Fourier-plane profile]")
    print(sample_table.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
        return

    print("\nValidation: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
