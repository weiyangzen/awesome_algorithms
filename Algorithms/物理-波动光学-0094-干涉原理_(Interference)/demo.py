"""Minimal runnable MVP for wave-optics interference (PHYS-0094).

This script builds a deterministic double-slit interference simulation,
then validates two core relations:
1) Fringe spacing near the optical axis.
2) Fringe visibility under unequal intensities and partial coherence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class InterferenceConfig:
    wavelength: float = 532e-9  # meter
    slit_separation: float = 0.25e-3  # meter
    screen_distance: float = 1.5  # meter
    amplitude_1: float = 1.0
    amplitude_2: float = 1.0
    coherence_gamma: float = 1.0
    phase_bias: float = 0.0  # rad
    slit_width: float = 0.0  # meter; <=0 means no diffraction envelope


@dataclass
class SpacingReport:
    predicted_spacing: float
    observed_spacing: float
    relative_error: float
    num_peaks_used: int


@dataclass
class VisibilityReport:
    predicted_visibility: float
    observed_visibility: float
    abs_error: float
    i_max: float
    i_min: float


def _sin_theta(y: np.ndarray, screen_distance: float) -> np.ndarray:
    """Exact geometry: sin(theta) = y / sqrt(y^2 + L^2)."""
    denom = np.sqrt(y * y + screen_distance * screen_distance)
    return y / denom


def phase_difference(y: np.ndarray, cfg: InterferenceConfig) -> np.ndarray:
    sin_theta = _sin_theta(y, cfg.screen_distance)
    geometric_phase = (2.0 * np.pi / cfg.wavelength) * cfg.slit_separation * sin_theta
    return geometric_phase + cfg.phase_bias


def envelope_factor(y: np.ndarray, cfg: InterferenceConfig) -> np.ndarray:
    """Single-slit diffraction envelope factor (sin beta / beta)^2.

    np.sinc(x) means sin(pi*x)/(pi*x), so we use:
      envelope = sinc((a/lambda) * sin(theta))^2
    """
    if cfg.slit_width <= 0.0:
        return np.ones_like(y)

    sin_theta = _sin_theta(y, cfg.screen_distance)
    x = (cfg.slit_width / cfg.wavelength) * sin_theta
    return np.sinc(x) ** 2


def intensity_profile(y: np.ndarray, cfg: InterferenceConfig) -> np.ndarray:
    i1 = cfg.amplitude_1 * cfg.amplitude_1
    i2 = cfg.amplitude_2 * cfg.amplitude_2
    delta = phase_difference(y, cfg)

    base = i1 + i2 + 2.0 * cfg.coherence_gamma * np.sqrt(i1 * i2) * np.cos(delta)
    env = envelope_factor(y, cfg)
    intensity = env * base

    if np.any(intensity < -1e-12):
        raise RuntimeError("Intensity became negative beyond numerical tolerance.")

    return np.clip(intensity, 0.0, None)


def build_profile_table(cfg: InterferenceConfig, y: np.ndarray) -> pd.DataFrame:
    delta = phase_difference(y, cfg)
    env = envelope_factor(y, cfg)
    intensity = intensity_profile(y, cfg)
    normalized = intensity / np.max(intensity)

    theta = np.arctan2(y, cfg.screen_distance)

    return pd.DataFrame(
        {
            "y_mm": y * 1e3,
            "theta_mrad": theta * 1e3,
            "delta_rad": delta,
            "envelope": env,
            "intensity": intensity,
            "intensity_norm": normalized,
        }
    )


def theoretical_spacing_small_angle(cfg: InterferenceConfig) -> float:
    return cfg.wavelength * cfg.screen_distance / cfg.slit_separation


def estimate_spacing_from_peaks(y: np.ndarray, intensity: np.ndarray, predicted_spacing: float) -> SpacingReport:
    if y.ndim != 1 or intensity.ndim != 1 or y.size != intensity.size:
        raise ValueError("y and intensity must be 1D arrays with equal length.")

    dy = float(y[1] - y[0])
    min_distance_samples = max(int(0.6 * predicted_spacing / dy), 5)

    peaks, _ = find_peaks(
        intensity,
        distance=min_distance_samples,
        prominence=0.02 * float(np.max(intensity)),
    )

    if peaks.size < 5:
        raise RuntimeError("Not enough fringe peaks detected for spacing estimation.")

    central_mask = np.abs(y[peaks]) < 0.75 * np.max(np.abs(y))
    central_peaks = peaks[central_mask]
    if central_peaks.size < 5:
        central_peaks = peaks

    peak_positions = y[central_peaks]
    spacings = np.diff(peak_positions)
    observed_spacing = float(np.median(spacings))
    rel_error = abs(observed_spacing - predicted_spacing) / predicted_spacing

    return SpacingReport(
        predicted_spacing=float(predicted_spacing),
        observed_spacing=observed_spacing,
        relative_error=float(rel_error),
        num_peaks_used=int(central_peaks.size),
    )


def theoretical_visibility(cfg: InterferenceConfig) -> float:
    i1 = cfg.amplitude_1 * cfg.amplitude_1
    i2 = cfg.amplitude_2 * cfg.amplitude_2
    return float(2.0 * cfg.coherence_gamma * np.sqrt(i1 * i2) / (i1 + i2))


def estimate_visibility(intensity: np.ndarray) -> tuple[float, float, float]:
    i_max = float(np.max(intensity))
    i_min = float(np.min(intensity))
    visibility = (i_max - i_min) / (i_max + i_min)
    return visibility, i_max, i_min


def build_visibility_report(cfg: InterferenceConfig, intensity: np.ndarray) -> VisibilityReport:
    observed_v, i_max, i_min = estimate_visibility(intensity)
    predicted_v = theoretical_visibility(cfg)
    return VisibilityReport(
        predicted_visibility=float(predicted_v),
        observed_visibility=float(observed_v),
        abs_error=float(abs(observed_v - predicted_v)),
        i_max=i_max,
        i_min=i_min,
    )


def print_profile_preview(df: pd.DataFrame) -> None:
    center = len(df) // 2
    preview = pd.concat([df.head(4), df.iloc[center - 2 : center + 2], df.tail(4)], axis=0)
    print(preview.to_string(index=False))


def main() -> None:
    num_points = 6001
    y = np.linspace(-0.015, 0.015, num_points)  # -15 mm to 15 mm

    spacing_cfg = InterferenceConfig(
        wavelength=532e-9,
        slit_separation=0.25e-3,
        screen_distance=1.5,
        amplitude_1=1.0,
        amplitude_2=1.0,
        coherence_gamma=1.0,
        phase_bias=0.0,
        slit_width=0.0,
    )

    vis_cfg = InterferenceConfig(
        wavelength=532e-9,
        slit_separation=0.25e-3,
        screen_distance=1.5,
        amplitude_1=1.0,
        amplitude_2=0.55,
        coherence_gamma=0.75,
        phase_bias=0.0,
        slit_width=0.0,
    )

    display_cfg = InterferenceConfig(
        wavelength=532e-9,
        slit_separation=0.25e-3,
        screen_distance=1.5,
        amplitude_1=1.0,
        amplitude_2=1.0,
        coherence_gamma=1.0,
        phase_bias=0.0,
        slit_width=45e-6,
    )

    spacing_intensity = intensity_profile(y, spacing_cfg)
    vis_intensity = intensity_profile(y, vis_cfg)

    predicted_spacing = theoretical_spacing_small_angle(spacing_cfg)
    spacing_report = estimate_spacing_from_peaks(y, spacing_intensity, predicted_spacing)
    visibility_report = build_visibility_report(vis_cfg, vis_intensity)

    display_df = build_profile_table(display_cfg, y)

    checks = {
        "fringe spacing relative error < 2%": spacing_report.relative_error < 0.02,
        "visibility absolute error < 0.02": visibility_report.abs_error < 0.02,
    }

    print("=== Interference Principle MVP (PHYS-0094) ===")
    print("Model: two-slit interference with optional single-slit envelope")

    print("\n[Spacing validation]")
    print(
        "predicted Δy = {pred_mm:.4f} mm, observed Δy = {obs_mm:.4f} mm, rel_error = {rel:.3e}, peaks = {n}".format(
            pred_mm=spacing_report.predicted_spacing * 1e3,
            obs_mm=spacing_report.observed_spacing * 1e3,
            rel=spacing_report.relative_error,
            n=spacing_report.num_peaks_used,
        )
    )

    print("\n[Visibility validation]")
    print(
        "predicted V = {pv:.4f}, observed V = {ov:.4f}, abs_error = {ae:.3e}, Imax = {imax:.4f}, Imin = {imin:.4f}".format(
            pv=visibility_report.predicted_visibility,
            ov=visibility_report.observed_visibility,
            ae=visibility_report.abs_error,
            imax=visibility_report.i_max,
            imin=visibility_report.i_min,
        )
    )

    print("\n[Sample profile with finite slit-width envelope]")
    print_profile_preview(display_df)

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
