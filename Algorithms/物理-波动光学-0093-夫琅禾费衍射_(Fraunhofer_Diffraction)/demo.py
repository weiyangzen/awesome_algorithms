"""Minimal runnable MVP for Fraunhofer diffraction (PHYS-0093).

Model: one-dimensional single-slit Fraunhofer diffraction.
We compute the far-field pattern in two ways:
1) Numerical Fourier transform of the aperture function.
2) Analytical sinc^2 reference formula.
Then we validate consistency via RMSE and first-minimum position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class FraunhoferConfig:
    wavelength: float = 532e-9  # meter
    slit_width: float = 120e-6  # meter
    x_extent: float = 8.0e-3  # half width of aperture-plane window, meter
    num_samples: int = 32768  # FFT samples in aperture plane
    compare_sin_theta_limit: float = 0.03  # use central angular range for RMSE
    rmse_threshold: float = 0.03
    minima_rel_error_threshold: float = 0.05


@dataclass
class ValidationReport:
    rmse: float
    predicted_first_min_sin_theta: float
    observed_first_min_sin_theta: float
    first_min_relative_error: float


def build_aperture(cfg: FraunhoferConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return aperture-plane coordinates and rectangular slit transmission."""
    x = np.linspace(-cfg.x_extent, cfg.x_extent, cfg.num_samples, endpoint=False)
    aperture = (np.abs(x) <= 0.5 * cfg.slit_width).astype(float)
    return x, aperture


def far_field_fft(cfg: FraunhoferConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized far-field intensity from FFT and analytical theory.

    Returns:
        sin_theta, theta, intensity_fft_norm, intensity_theory_norm
    """
    x, aperture = build_aperture(cfg)
    dx = float(x[1] - x[0])

    # Fraunhofer field is proportional to Fourier transform of aperture(x).
    field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(aperture))) * dx
    freq = np.fft.fftshift(np.fft.fftfreq(cfg.num_samples, d=dx))  # cycles / meter

    sin_theta = cfg.wavelength * freq
    valid = np.abs(sin_theta) <= 1.0

    sin_theta = sin_theta[valid]
    field = field[valid]

    intensity_fft = np.abs(field) ** 2
    intensity_fft_norm = intensity_fft / np.max(intensity_fft)

    # For rectangular slit: I(theta) ∝ sinc(a * f)^2, f = sin(theta)/lambda.
    intensity_theory_norm = np.sinc(cfg.slit_width * freq[valid]) ** 2

    theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    return sin_theta, theta, intensity_fft_norm, intensity_theory_norm


def rmse_in_central_region(
    sin_theta: np.ndarray,
    intensity_fft_norm: np.ndarray,
    intensity_theory_norm: np.ndarray,
    limit: float,
) -> float:
    mask = np.abs(sin_theta) <= limit
    if np.count_nonzero(mask) < 100:
        raise RuntimeError("Not enough points in central region for RMSE evaluation.")
    err = intensity_fft_norm[mask] - intensity_theory_norm[mask]
    return float(np.sqrt(np.mean(err * err)))


def estimate_first_minimum_sin_theta(
    sin_theta: np.ndarray,
    intensity_fft_norm: np.ndarray,
    predicted_first_min: float,
) -> float:
    """Estimate first positive-side diffraction minimum from numerical curve."""
    if predicted_first_min <= 0.0:
        raise ValueError("predicted_first_min must be positive.")

    mask = (sin_theta > 0.35 * predicted_first_min) & (sin_theta < 1.8 * predicted_first_min)
    sin_local = sin_theta[mask]
    i_local = intensity_fft_norm[mask]

    if sin_local.size < 20:
        raise RuntimeError("Insufficient samples near first minimum.")

    minima_idx, _ = find_peaks(-i_local, prominence=1e-4)
    if minima_idx.size == 0:
        raise RuntimeError("No local minimum detected near first diffraction minimum.")

    candidates = sin_local[minima_idx]
    best = candidates[np.argmin(np.abs(candidates - predicted_first_min))]
    return float(best)


def build_validation_report(
    cfg: FraunhoferConfig,
    sin_theta: np.ndarray,
    intensity_fft_norm: np.ndarray,
    intensity_theory_norm: np.ndarray,
) -> ValidationReport:
    predicted_first_min = cfg.wavelength / cfg.slit_width
    observed_first_min = estimate_first_minimum_sin_theta(
        sin_theta=sin_theta,
        intensity_fft_norm=intensity_fft_norm,
        predicted_first_min=predicted_first_min,
    )

    rmse = rmse_in_central_region(
        sin_theta=sin_theta,
        intensity_fft_norm=intensity_fft_norm,
        intensity_theory_norm=intensity_theory_norm,
        limit=cfg.compare_sin_theta_limit,
    )

    rel_error = abs(observed_first_min - predicted_first_min) / predicted_first_min

    return ValidationReport(
        rmse=rmse,
        predicted_first_min_sin_theta=float(predicted_first_min),
        observed_first_min_sin_theta=float(observed_first_min),
        first_min_relative_error=float(rel_error),
    )


def build_profile_table(
    sin_theta: np.ndarray,
    theta: np.ndarray,
    intensity_fft_norm: np.ndarray,
    intensity_theory_norm: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theta_mrad": theta * 1e3,
            "sin_theta": sin_theta,
            "intensity_fft": intensity_fft_norm,
            "intensity_theory": intensity_theory_norm,
            "abs_error": np.abs(intensity_fft_norm - intensity_theory_norm),
        }
    )


def preview_rows(df: pd.DataFrame, target_sin_theta: float, n: int = 3) -> pd.DataFrame:
    arr = df["sin_theta"].to_numpy()
    idx = np.argsort(np.abs(arr - target_sin_theta))[:n]
    return df.iloc[np.sort(idx)]


def print_profile_preview(df: pd.DataFrame, first_min_pred: float) -> None:
    center = preview_rows(df, 0.0, n=5)
    near_pos_min = preview_rows(df, first_min_pred, n=4)
    near_neg_min = preview_rows(df, -first_min_pred, n=4)

    preview = pd.concat([near_neg_min, center, near_pos_min], axis=0)
    preview = preview.drop_duplicates().sort_values("sin_theta")
    print(preview.to_string(index=False))


def main() -> None:
    cfg = FraunhoferConfig()

    sin_theta, theta, intensity_fft_norm, intensity_theory_norm = far_field_fft(cfg)

    report = build_validation_report(
        cfg=cfg,
        sin_theta=sin_theta,
        intensity_fft_norm=intensity_fft_norm,
        intensity_theory_norm=intensity_theory_norm,
    )

    df = build_profile_table(
        sin_theta=sin_theta,
        theta=theta,
        intensity_fft_norm=intensity_fft_norm,
        intensity_theory_norm=intensity_theory_norm,
    )

    checks = {
        "central RMSE < 0.03": report.rmse < cfg.rmse_threshold,
        "first-minimum relative error < 5%": report.first_min_relative_error < cfg.minima_rel_error_threshold,
    }

    predicted_theta = np.arcsin(report.predicted_first_min_sin_theta)
    observed_theta = np.arcsin(report.observed_first_min_sin_theta)

    print("=== Fraunhofer Diffraction MVP (PHYS-0093) ===")
    print("Model: 1D single-slit aperture, FFT far-field vs analytical sinc^2")

    print("\n[Parameters]")
    print(
        "wavelength = {lam_nm:.1f} nm, slit_width = {a_um:.1f} um, samples = {n}, x_extent = +/-{x_mm:.2f} mm".format(
            lam_nm=cfg.wavelength * 1e9,
            a_um=cfg.slit_width * 1e6,
            n=cfg.num_samples,
            x_mm=cfg.x_extent * 1e3,
        )
    )

    print("\n[Validation metrics]")
    print(f"central RMSE = {report.rmse:.4e}")
    print(
        "first minimum (pred): sin(theta) = {sp:.6e}, theta = {tp:.3f} mrad".format(
            sp=report.predicted_first_min_sin_theta,
            tp=predicted_theta * 1e3,
        )
    )
    print(
        "first minimum (obs) : sin(theta) = {so:.6e}, theta = {to:.3f} mrad, rel_error = {re:.3e}".format(
            so=report.observed_first_min_sin_theta,
            to=observed_theta * 1e3,
            re=report.first_min_relative_error,
        )
    )

    print("\n[Profile preview around center and first minima]")
    print_profile_preview(df, first_min_pred=report.predicted_first_min_sin_theta)

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
