"""Minimal runnable MVP for Holography (PHYS-0098).

This script demonstrates a compact digital off-axis holography pipeline:
1) Build a synthetic complex object field.
2) Propagate to sensor plane with angular-spectrum diffraction.
3) Record hologram intensity by interfering object and tilted reference waves.
4) Demodulate and low-pass isolate the first diffraction order.
5) Back-propagate to reconstruct the object field and validate errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift


@dataclass
class HolographyConfig:
    wavelength: float = 633e-9  # meter (He-Ne like)
    pixel_pitch: float = 8e-6  # meter
    grid_size: int = 320
    propagation_distance: float = 0.08  # meter
    reference_amplitude: float = 1.0
    reference_fx: float = 18_000.0  # cycles / meter
    reference_fy: float = 12_000.0  # cycles / meter
    sideband_cutoff: float = 7_500.0  # cycles / meter


@dataclass
class ReconstructionMetrics:
    amplitude_rmse: float
    amplitude_corr: float
    phase_mae_rad: float
    complex_relative_error: float
    carrier_to_dc_power_ratio: float
    fringe_contrast: float


def make_grid(grid_size: int, pixel_pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_span = 0.5 * grid_size * pixel_pitch
    x = np.linspace(-half_span, half_span, grid_size, endpoint=False)
    y = np.linspace(-half_span, half_span, grid_size, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return x, y, xx, yy


def build_object_field(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Create a deterministic complex object field U0(x, y)."""
    g1 = np.exp(-(((xx + 0.45e-3) ** 2 + (yy + 0.00e-3) ** 2) / (2.0 * (0.17e-3**2))))
    g2 = np.exp(-(((xx - 0.32e-3) ** 2 + (yy + 0.28e-3) ** 2) / (2.0 * (0.13e-3**2))))
    bar = ((np.abs(xx) < 0.10e-3) & (np.abs(yy - 0.42e-3) < 0.36e-3)).astype(np.float64)

    amp_raw = 0.10 + 0.65 * g1 + 0.58 * g2 + 0.40 * bar
    amp = amp_raw / np.max(amp_raw)

    phase = (
        0.90 * np.exp(-((xx**2 + yy**2) / (2.0 * (0.72e-3**2))))
        + 0.35 * np.sin(2.0 * np.pi * 620.0 * xx)
        + 0.22 * (yy / 1e-3)
    )
    return amp * np.exp(1j * phase)


def angular_spectrum_propagate(
    field: np.ndarray,
    wavelength: float,
    pixel_pitch: float,
    distance: float,
) -> np.ndarray:
    """Propagate a complex field using angular-spectrum transfer function."""
    ny, nx = field.shape
    fx = fftfreq(nx, d=pixel_pitch)
    fy = fftfreq(ny, d=pixel_pitch)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")

    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * fxx
    ky = 2.0 * np.pi * fyy
    kz = np.sqrt((k**2 - kx**2 - ky**2) + 0j)

    transfer = np.exp(1j * distance * kz)
    return ifft2(fft2(field) * transfer)


def build_reference_wave(
    xx: np.ndarray,
    yy: np.ndarray,
    amplitude: float,
    fx: float,
    fy: float,
) -> np.ndarray:
    return amplitude * np.exp(1j * 2.0 * np.pi * (fx * xx + fy * yy))


def record_hologram_intensity(object_sensor: np.ndarray, reference_wave: np.ndarray) -> np.ndarray:
    return np.abs(object_sensor + reference_wave) ** 2


def estimate_carrier_to_dc_ratio(
    hologram: np.ndarray,
    pixel_pitch: float,
    reference_fx: float,
    reference_fy: float,
    window_radius: float = 1800.0,
) -> float:
    """Rough separation metric: sideband power / zero-order power in spectrum."""
    spectrum = np.abs(fftshift(fft2(hologram))) ** 2
    n = hologram.shape[0]
    freqs = fftshift(fftfreq(n, d=pixel_pitch))
    fxx, fyy = np.meshgrid(freqs, freqs, indexing="xy")

    dc_mask = np.sqrt(fxx**2 + fyy**2) <= window_radius
    side_mask = np.sqrt((fxx - reference_fx) ** 2 + (fyy - reference_fy) ** 2) <= window_radius

    dc_power = float(np.sum(spectrum[dc_mask]))
    side_power = float(np.sum(spectrum[side_mask]))
    return side_power / (dc_power + 1e-15)


def demodulate_and_lowpass(
    hologram: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    pixel_pitch: float,
    reference_fx: float,
    reference_fy: float,
    cutoff: float,
) -> np.ndarray:
    """Isolate OR* term by demodulation and circular low-pass filtering."""
    demod_carrier = np.exp(1j * 2.0 * np.pi * (reference_fx * xx + reference_fy * yy))
    demodulated = hologram * demod_carrier

    spec = fftshift(fft2(demodulated))
    n = hologram.shape[0]
    freqs = fftshift(fftfreq(n, d=pixel_pitch))
    fxx, fyy = np.meshgrid(freqs, freqs, indexing="xy")
    lowpass = np.sqrt(fxx**2 + fyy**2) <= cutoff

    filtered = spec * lowpass
    return ifft2(ifftshift(filtered))


def align_global_complex_scale(reconstructed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, complex]:
    """Compensate global complex gain/phase ambiguity by least squares."""
    denom = np.vdot(reconstructed, reconstructed)
    if np.abs(denom) < 1e-15:
        raise RuntimeError("Degenerate reconstruction: zero norm.")
    alpha = np.vdot(reconstructed, target) / denom
    return alpha * reconstructed, alpha


def compute_metrics(
    target: np.ndarray,
    reconstructed: np.ndarray,
    carrier_to_dc_ratio: float,
    fringe_contrast: float,
) -> ReconstructionMetrics:
    target_amp = np.abs(target)
    recon_amp = np.abs(reconstructed)

    target_amp_norm = target_amp / (np.max(target_amp) + 1e-15)
    recon_amp_norm = recon_amp / (np.max(recon_amp) + 1e-15)

    amplitude_rmse = float(np.sqrt(np.mean((recon_amp_norm - target_amp_norm) ** 2)))
    amplitude_corr = float(np.corrcoef(target_amp_norm.ravel(), recon_amp_norm.ravel())[0, 1])

    phase_mask = target_amp_norm > 0.25
    if not np.any(phase_mask):
        raise RuntimeError("Phase mask is empty; object contrast is too weak.")
    phase_residual = np.angle(reconstructed[phase_mask] * np.conj(target[phase_mask]))
    phase_mae = float(np.mean(np.abs(phase_residual)))

    complex_relative_error = float(
        np.linalg.norm(reconstructed - target) / (np.linalg.norm(target) + 1e-15)
    )

    return ReconstructionMetrics(
        amplitude_rmse=amplitude_rmse,
        amplitude_corr=amplitude_corr,
        phase_mae_rad=phase_mae,
        complex_relative_error=complex_relative_error,
        carrier_to_dc_power_ratio=carrier_to_dc_ratio,
        fringe_contrast=fringe_contrast,
    )


def preview_centerline(
    target: np.ndarray,
    reconstructed: np.ndarray,
    x: np.ndarray,
    sample_count: int = 11,
) -> pd.DataFrame:
    row = target.shape[0] // 2
    target_amp = np.abs(target[row])
    recon_amp = np.abs(reconstructed[row])

    idx = np.linspace(0, target.shape[1] - 1, sample_count, dtype=int)
    return pd.DataFrame(
        {
            "x_mm": x[idx] * 1e3,
            "target_amp_norm": target_amp[idx] / (np.max(target_amp) + 1e-15),
            "recon_amp_norm": recon_amp[idx] / (np.max(recon_amp) + 1e-15),
        }
    )


def main() -> None:
    cfg = HolographyConfig()

    x, _, xx, yy = make_grid(cfg.grid_size, cfg.pixel_pitch)
    object_plane = build_object_field(xx, yy)

    object_sensor = angular_spectrum_propagate(
        field=object_plane,
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        distance=cfg.propagation_distance,
    )
    reference = build_reference_wave(
        xx=xx,
        yy=yy,
        amplitude=cfg.reference_amplitude,
        fx=cfg.reference_fx,
        fy=cfg.reference_fy,
    )

    hologram = record_hologram_intensity(object_sensor, reference)
    fringe_contrast = float(
        (np.max(hologram) - np.min(hologram)) / (np.max(hologram) + np.min(hologram) + 1e-15)
    )

    carrier_ratio = estimate_carrier_to_dc_ratio(
        hologram=hologram,
        pixel_pitch=cfg.pixel_pitch,
        reference_fx=cfg.reference_fx,
        reference_fy=cfg.reference_fy,
    )

    recovered_sensor = demodulate_and_lowpass(
        hologram=hologram,
        xx=xx,
        yy=yy,
        pixel_pitch=cfg.pixel_pitch,
        reference_fx=cfg.reference_fx,
        reference_fy=cfg.reference_fy,
        cutoff=cfg.sideband_cutoff,
    ) / cfg.reference_amplitude

    reconstructed_plane = angular_spectrum_propagate(
        field=recovered_sensor,
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        distance=-cfg.propagation_distance,
    )
    reconstructed_aligned, alpha = align_global_complex_scale(reconstructed_plane, object_plane)

    metrics = compute_metrics(
        target=object_plane,
        reconstructed=reconstructed_aligned,
        carrier_to_dc_ratio=carrier_ratio,
        fringe_contrast=fringe_contrast,
    )

    checks = {
        "Amplitude RMSE < 0.20": metrics.amplitude_rmse < 0.20,
        "Amplitude correlation > 0.90": metrics.amplitude_corr > 0.90,
        "Phase MAE < 0.80 rad": metrics.phase_mae_rad < 0.80,
        "Complex relative error < 0.70": metrics.complex_relative_error < 0.70,
        "Carrier/DC power ratio > 0.010": metrics.carrier_to_dc_power_ratio > 0.010,
        "Fringe contrast > 0.25": metrics.fringe_contrast > 0.25,
    }

    summary = pd.DataFrame(
        {
            "metric": [
                "amplitude_rmse",
                "amplitude_corr",
                "phase_mae_rad",
                "complex_relative_error",
                "carrier_to_dc_power_ratio",
                "fringe_contrast",
                "alignment_gain_abs",
            ],
            "value": [
                metrics.amplitude_rmse,
                metrics.amplitude_corr,
                metrics.phase_mae_rad,
                metrics.complex_relative_error,
                metrics.carrier_to_dc_power_ratio,
                metrics.fringe_contrast,
                float(np.abs(alpha)),
            ],
        }
    )

    print("=== Holography MVP (PHYS-0098) ===")
    print("Model: off-axis digital hologram recording + demodulation + back propagation")

    print("\n[Metrics]")
    print(summary.to_string(index=False))

    print("\n[Centerline amplitude samples]")
    print(preview_centerline(object_plane, reconstructed_aligned, x).to_string(index=False))

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
