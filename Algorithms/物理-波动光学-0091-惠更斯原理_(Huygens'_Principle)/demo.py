"""Minimal runnable MVP for Huygens' Principle (PHYS-0091).

This script implements scalar 1D Huygens-wavelet propagation from an aperture
plane to a screen plane by direct superposition of secondary spherical waves.
It validates two canonical phenomena:
1) Single-slit diffraction envelope against Fraunhofer sinc^2 theory.
2) Double-slit interference fringe spacing against lambda*z/d prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class HuygensConfig:
    wavelength: float = 532e-9  # meter
    propagation_distance: float = 1.2  # meter
    aperture_half_span: float = 0.9e-3  # meter
    aperture_samples: int = 1201
    screen_half_span: float = 14e-3  # meter
    screen_samples: int = 1601
    single_slit_width: float = 120e-6  # meter
    double_slit_width: float = 60e-6  # meter
    double_slit_separation: float = 450e-6  # meter (center-to-center)


@dataclass
class SingleSlitReport:
    rmse_center_band: float
    expected_first_min_mm: float
    observed_first_min_mm: float
    first_min_relative_error: float


@dataclass
class DoubleSlitReport:
    expected_spacing_mm: float
    observed_spacing_mm: float
    spacing_relative_error: float
    fringe_visibility: float
    used_peak_count: int


def make_grids(cfg: HuygensConfig) -> tuple[np.ndarray, np.ndarray]:
    aperture_x = np.linspace(
        -cfg.aperture_half_span,
        cfg.aperture_half_span,
        cfg.aperture_samples,
        endpoint=True,
    )
    screen_x = np.linspace(
        -cfg.screen_half_span,
        cfg.screen_half_span,
        cfg.screen_samples,
        endpoint=True,
    )
    return aperture_x, screen_x


def build_single_slit_aperture(aperture_x: np.ndarray, slit_width: float) -> np.ndarray:
    return (np.abs(aperture_x) <= 0.5 * slit_width).astype(np.complex128)


def build_double_slit_aperture(
    aperture_x: np.ndarray,
    slit_width: float,
    slit_separation: float,
) -> np.ndarray:
    left = np.abs(aperture_x + 0.5 * slit_separation) <= 0.5 * slit_width
    right = np.abs(aperture_x - 0.5 * slit_separation) <= 0.5 * slit_width
    return (left | right).astype(np.complex128)


def huygens_propagate_1d(
    aperture_field: np.ndarray,
    aperture_x: np.ndarray,
    screen_x: np.ndarray,
    wavelength: float,
    distance: float,
    use_obliquity: bool = True,
) -> np.ndarray:
    """Direct 1D Huygens superposition with spherical-wave kernel."""
    if aperture_field.shape != aperture_x.shape:
        raise ValueError("aperture_field and aperture_x must have identical shape.")

    dx = float(aperture_x[1] - aperture_x[0])
    k = 2.0 * np.pi / wavelength

    r = np.sqrt((screen_x[None, :] - aperture_x[:, None]) ** 2 + distance**2)
    kernel = np.exp(1j * k * r) / r
    if use_obliquity:
        kernel = kernel * (distance / r)

    return np.sum(aperture_field[:, None] * kernel, axis=0) * dx


def normalize_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(field) ** 2
    peak = float(np.max(intensity))
    if peak <= 0.0:
        raise RuntimeError("Propagated intensity has non-positive peak.")
    return intensity / peak


def analytic_single_slit_intensity(
    screen_x: np.ndarray,
    slit_width: float,
    wavelength: float,
    distance: float,
) -> np.ndarray:
    argument = slit_width * screen_x / (wavelength * distance)
    analytic = np.sinc(argument) ** 2
    return analytic / np.max(analytic)


def estimate_first_minimum(
    screen_x: np.ndarray,
    intensity: np.ndarray,
    expected_position: float,
) -> tuple[float, float]:
    positive = screen_x > 0.0
    x_pos = screen_x[positive]
    i_pos = intensity[positive]

    band = (x_pos >= 0.55 * expected_position) & (x_pos <= 1.45 * expected_position)
    if np.count_nonzero(band) < 5:
        raise RuntimeError("Insufficient sampling around expected first minimum.")

    x_band = x_pos[band]
    i_band = i_pos[band]
    observed = float(x_band[int(np.argmin(i_band))])
    rel_error = abs(observed - expected_position) / expected_position
    return observed, float(rel_error)


def estimate_fringe_spacing(
    screen_x: np.ndarray,
    intensity: np.ndarray,
    expected_spacing: float,
) -> tuple[float, int]:
    center_window = np.abs(screen_x) <= 6.0 * expected_spacing
    x_win = screen_x[center_window]
    i_win = intensity[center_window]

    peaks, _ = find_peaks(i_win, height=0.20, prominence=0.08)
    if peaks.size < 5:
        raise RuntimeError("Not enough resolvable interference peaks for spacing estimate.")

    x_peaks = x_win[peaks]
    order = np.argsort(np.abs(x_peaks))
    chosen = np.sort(x_peaks[order[:9]])

    if chosen.size < 5:
        raise RuntimeError("Not enough central peaks after peak selection.")

    spacing = np.diff(chosen)
    observed = float(np.mean(spacing))
    return observed, int(chosen.size)


def estimate_visibility(
    screen_x: np.ndarray,
    intensity: np.ndarray,
    expected_spacing: float,
) -> float:
    window = np.abs(screen_x) <= 2.5 * expected_spacing
    local = intensity[window]
    i_max = float(np.max(local))
    i_min = float(np.min(local))
    return (i_max - i_min) / (i_max + i_min + 1e-15)


def build_single_slit_report(
    cfg: HuygensConfig,
    screen_x: np.ndarray,
    numeric_intensity: np.ndarray,
) -> tuple[SingleSlitReport, pd.DataFrame]:
    analytic = analytic_single_slit_intensity(
        screen_x=screen_x,
        slit_width=cfg.single_slit_width,
        wavelength=cfg.wavelength,
        distance=cfg.propagation_distance,
    )

    center_band = np.abs(screen_x) <= (2.4 * cfg.wavelength * cfg.propagation_distance / cfg.single_slit_width)
    rmse_center = float(np.sqrt(np.mean((numeric_intensity[center_band] - analytic[center_band]) ** 2)))

    expected_first_min = cfg.wavelength * cfg.propagation_distance / cfg.single_slit_width
    observed_first_min, rel_error = estimate_first_minimum(
        screen_x=screen_x,
        intensity=numeric_intensity,
        expected_position=expected_first_min,
    )

    profile = pd.DataFrame(
        {
            "screen_x_mm": screen_x * 1e3,
            "I_numeric_norm": numeric_intensity,
            "I_analytic_norm": analytic,
        }
    )

    report = SingleSlitReport(
        rmse_center_band=rmse_center,
        expected_first_min_mm=float(expected_first_min * 1e3),
        observed_first_min_mm=float(observed_first_min * 1e3),
        first_min_relative_error=rel_error,
    )
    return report, profile


def build_double_slit_report(
    cfg: HuygensConfig,
    screen_x: np.ndarray,
    numeric_intensity: np.ndarray,
) -> DoubleSlitReport:
    expected_spacing = cfg.wavelength * cfg.propagation_distance / cfg.double_slit_separation
    observed_spacing, used_peak_count = estimate_fringe_spacing(
        screen_x=screen_x,
        intensity=numeric_intensity,
        expected_spacing=expected_spacing,
    )
    spacing_rel_error = abs(observed_spacing - expected_spacing) / expected_spacing
    visibility = estimate_visibility(
        screen_x=screen_x,
        intensity=numeric_intensity,
        expected_spacing=expected_spacing,
    )

    return DoubleSlitReport(
        expected_spacing_mm=float(expected_spacing * 1e3),
        observed_spacing_mm=float(observed_spacing * 1e3),
        spacing_relative_error=float(spacing_rel_error),
        fringe_visibility=float(visibility),
        used_peak_count=used_peak_count,
    )


def preview_table(profile: pd.DataFrame, max_abs_x_mm: float = 7.5, sample_count: int = 11) -> pd.DataFrame:
    window = profile[np.abs(profile["screen_x_mm"]) <= max_abs_x_mm]
    if window.empty:
        return profile.head(sample_count)
    idx = np.linspace(0, len(window) - 1, sample_count, dtype=int)
    return window.iloc[idx].reset_index(drop=True)


def main() -> None:
    cfg = HuygensConfig()
    aperture_x, screen_x = make_grids(cfg)

    single_aperture = build_single_slit_aperture(aperture_x, cfg.single_slit_width)
    single_field = huygens_propagate_1d(
        aperture_field=single_aperture,
        aperture_x=aperture_x,
        screen_x=screen_x,
        wavelength=cfg.wavelength,
        distance=cfg.propagation_distance,
        use_obliquity=True,
    )
    single_intensity = normalize_intensity(single_field)
    single_report, profile = build_single_slit_report(cfg, screen_x, single_intensity)

    double_aperture = build_double_slit_aperture(
        aperture_x=aperture_x,
        slit_width=cfg.double_slit_width,
        slit_separation=cfg.double_slit_separation,
    )
    double_field = huygens_propagate_1d(
        aperture_field=double_aperture,
        aperture_x=aperture_x,
        screen_x=screen_x,
        wavelength=cfg.wavelength,
        distance=cfg.propagation_distance,
        use_obliquity=True,
    )
    double_intensity = normalize_intensity(double_field)
    double_report = build_double_slit_report(cfg, screen_x, double_intensity)

    checks = {
        "Single-slit center-band RMSE < 0.040": single_report.rmse_center_band < 0.040,
        "Single-slit first-min relative error < 0.10": single_report.first_min_relative_error < 0.10,
        "Double-slit spacing relative error < 0.10": double_report.spacing_relative_error < 0.10,
        "Double-slit fringe visibility > 0.55": double_report.fringe_visibility > 0.55,
    }

    print("=== Huygens Principle MVP (PHYS-0091) ===")
    print("Model: direct secondary-wavelet superposition in 1D scalar optics")

    print("\n[Single-slit diffraction validation]")
    print(
        "RMSE(center band) = {rmse:.4e}, expected first minimum = {xexp:.4f} mm, "
        "observed = {xobs:.4f} mm, rel_error = {rel:.3e}".format(
            rmse=single_report.rmse_center_band,
            xexp=single_report.expected_first_min_mm,
            xobs=single_report.observed_first_min_mm,
            rel=single_report.first_min_relative_error,
        )
    )

    print("\n[Double-slit interference validation]")
    print(
        "expected fringe spacing = {sexp:.4f} mm, observed = {sobs:.4f} mm, "
        "rel_error = {rel:.3e}, visibility = {vis:.4f}, used peaks = {n}".format(
            sexp=double_report.expected_spacing_mm,
            sobs=double_report.observed_spacing_mm,
            rel=double_report.spacing_relative_error,
            vis=double_report.fringe_visibility,
            n=double_report.used_peak_count,
        )
    )

    print("\n[Sample normalized profile (single slit)]")
    print(preview_table(profile).to_string(index=False))

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
