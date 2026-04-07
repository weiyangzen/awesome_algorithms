"""Minimal runnable MVP for Transmission Electron Microscopy (TEM).

This script implements a compact, auditable forward model:
1) Build a synthetic projected potential of a crystalline sample.
2) Convert potential to an exit wave with a weak-phase style phase shift.
3) Propagate through objective-lens transfer function (defocus, Cs, aperture).
4) Form image intensity, compute diagnostics, and run deterministic checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import constants
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class TEMConfig:
    """Configuration for a small TEM image simulation MVP."""

    grid_size: int = 256
    pixel_size_angstrom: float = 0.25
    accelerating_voltage_kv: float = 200.0
    defocus_nm: float = -80.0
    spherical_aberration_mm: float = 1.2
    aperture_mrad: float = 18.0

    lattice_constant_angstrom: float = 2.5
    atom_sigma_angstrom: float = 0.45
    phase_gain_rad_per_norm_potential: float = 0.9

    dopant_amplitude: float = 1.7
    thickness_scale: float = 1.0


def electron_wavelength_relativistic(voltage_kv: float) -> float:
    """Return relativistic electron wavelength (meters)."""
    voltage_v = float(voltage_kv) * 1e3
    kinetic_term = 2.0 * constants.m_e * constants.e * voltage_v
    relativistic_correction = 1.0 + (constants.e * voltage_v) / (2.0 * constants.m_e * constants.c**2)
    return constants.h / np.sqrt(kinetic_term * relativistic_correction)


def build_projected_potential(cfg: TEMConfig) -> np.ndarray:
    """Construct a synthetic projected potential map on a square lattice."""
    n = cfg.grid_size
    pixel_a = cfg.pixel_size_angstrom
    lattice_a = cfg.lattice_constant_angstrom

    impulse = np.zeros((n, n), dtype=np.float64)
    step_px = max(1, int(round(lattice_a / pixel_a)))

    for iy in range(step_px // 2, n, step_px):
        for ix in range(step_px // 2, n, step_px):
            impulse[iy, ix] = 1.0

    # Add one dopant and one vacancy to create non-uniform local contrast.
    dopant_site = (n // 2 + step_px, n // 2 - 2 * step_px)
    vacancy_site = (n // 2 - step_px, n // 2 + step_px)
    impulse[dopant_site] = cfg.dopant_amplitude
    impulse[vacancy_site] = 0.0

    sigma_px = cfg.atom_sigma_angstrom / cfg.pixel_size_angstrom
    potential = gaussian_filter(impulse, sigma=sigma_px, mode="wrap")

    potential -= float(potential.min())
    potential /= max(float(potential.max()), 1e-12)
    potential *= cfg.thickness_scale
    return potential


def build_exit_wave(projected_potential: np.ndarray, cfg: TEMConfig) -> np.ndarray:
    """Map projected potential to a complex exit wave using a weak-phase style model."""
    phase = cfg.phase_gain_rad_per_norm_potential * projected_potential
    return np.exp(1j * phase)


def objective_transfer_function(
    shape: Tuple[int, int],
    pixel_size_m: float,
    wavelength_m: float,
    defocus_m: float,
    cs_m: float,
    aperture_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return coherent objective transfer function H(k) and spatial frequency radius."""
    ny, nx = shape
    fx = np.fft.fftfreq(nx, d=pixel_size_m)
    fy = np.fft.fftfreq(ny, d=pixel_size_m)
    kx, ky = np.meshgrid(fx, fy, indexing="xy")

    k2 = kx * kx + ky * ky
    k = np.sqrt(k2)
    scattering_angle = wavelength_m * k
    aperture = (scattering_angle <= aperture_rad).astype(np.float64)

    chi = np.pi * wavelength_m * defocus_m * k2 + 0.5 * np.pi * cs_m * (wavelength_m**3) * (k2**2)
    h = aperture * np.exp(-1j * chi)
    return h, k


def simulate_tem_image(cfg: TEMConfig) -> dict:
    """Run the full TEM forward simulation and collect intermediate products."""
    wavelength_m = electron_wavelength_relativistic(cfg.accelerating_voltage_kv)
    pixel_size_m = cfg.pixel_size_angstrom * 1e-10

    potential = build_projected_potential(cfg)
    exit_wave = build_exit_wave(potential, cfg)

    h, k_radius = objective_transfer_function(
        shape=exit_wave.shape,
        pixel_size_m=pixel_size_m,
        wavelength_m=wavelength_m,
        defocus_m=cfg.defocus_nm * 1e-9,
        cs_m=cfg.spherical_aberration_mm * 1e-3,
        aperture_rad=cfg.aperture_mrad * 1e-3,
    )

    img_wave = np.fft.ifft2(np.fft.fft2(exit_wave) * h)
    intensity = np.abs(img_wave) ** 2
    intensity /= max(float(intensity.mean()), 1e-12)

    centered = intensity - float(intensity.mean())
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2

    return {
        "wavelength_m": wavelength_m,
        "pixel_size_m": pixel_size_m,
        "projected_potential": potential,
        "exit_wave": exit_wave,
        "transfer_function": h,
        "k_radius": k_radius,
        "intensity": intensity,
        "power_spectrum": power_spectrum,
    }


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Compute radial average of a 2D image."""
    h, w = image.shape
    cy = 0.5 * (h - 1)
    cx = 0.5 * (w - 1)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rb = rr.astype(np.int32)

    sums = np.bincount(rb.ravel(), weights=image.ravel())
    counts = np.bincount(rb.ravel())
    return sums / np.maximum(counts, 1)


def estimate_primary_spatial_frequency(power_spectrum: np.ndarray, pixel_size_m: float) -> float:
    """Estimate dominant non-zero reciprocal-space peak frequency (1/m)."""
    n = power_spectrum.shape[0]
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_size_m))
    fy = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_size_m))
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    kr = np.sqrt(kx * kx + ky * ky)

    k_nyquist = 0.5 / pixel_size_m
    mask = (kr > 0.04 * k_nyquist) & (kr < 0.7 * k_nyquist)

    score = np.where(mask, power_spectrum, -1.0)
    peak_idx = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(kr[peak_idx])


def run_validations(cfg: TEMConfig, result: dict) -> pd.DataFrame:
    """Run deterministic checks and return a compact diagnostics table."""
    wavelength_pm = result["wavelength_m"] * 1e12
    assert 2.40 < wavelength_pm < 2.60, f"Unexpected 200kV wavelength: {wavelength_pm:.4f} pm"

    intensity = result["intensity"]
    assert np.isfinite(intensity).all(), "Intensity map contains non-finite values"
    assert float(intensity.min()) >= 0.0, "Intensity should be non-negative"

    std_contrast = float(intensity.std())
    assert std_contrast > 0.01, f"Image contrast too weak: std={std_contrast:.5f}"

    k_peak = estimate_primary_spatial_frequency(result["power_spectrum"], result["pixel_size_m"])
    expected_k = 1.0 / (cfg.lattice_constant_angstrom * 1e-10)
    rel_err = abs(k_peak - expected_k) / expected_k
    assert rel_err < 0.30, f"Primary reciprocal peak mismatch: rel_err={rel_err:.3f}"

    plus_cfg = TEMConfig(**{**cfg.__dict__, "defocus_nm": abs(cfg.defocus_nm)})
    minus_cfg = TEMConfig(**{**cfg.__dict__, "defocus_nm": -abs(cfg.defocus_nm)})
    i_plus = simulate_tem_image(plus_cfg)["intensity"].ravel()
    i_minus = simulate_tem_image(minus_cfg)["intensity"].ravel()
    corr = float(np.corrcoef(i_plus, i_minus)[0, 1])
    assert corr < 0.98, f"Defocus sign should change contrast transfer, corr={corr:.4f}"

    report = pd.DataFrame(
        [
            {"check": "200kV wavelength window", "value": wavelength_pm, "target": "2.40-2.60 pm", "pass": True},
            {
                "check": "primary reciprocal peak rel.error",
                "value": rel_err,
                "target": "<0.30",
                "pass": rel_err < 0.30,
            },
            {"check": "image contrast std", "value": std_contrast, "target": ">0.01", "pass": std_contrast > 0.01},
            {"check": "defocus +/- correlation", "value": corr, "target": "<0.98", "pass": corr < 0.98},
        ]
    )
    return report


def main() -> None:
    cfg = TEMConfig()
    result = simulate_tem_image(cfg)
    report = run_validations(cfg, result)

    intensity = result["intensity"]
    potential = result["projected_potential"]
    profile = radial_profile(result["power_spectrum"])
    k_peak = estimate_primary_spatial_frequency(result["power_spectrum"], result["pixel_size_m"])

    print("=== TEM MVP (weak-phase + objective transfer function) ===")
    print(f"accelerating_voltage_kV : {cfg.accelerating_voltage_kv:.1f}")
    print(f"electron_wavelength_pm  : {result['wavelength_m'] * 1e12:.6f}")
    print(f"pixel_size_A            : {cfg.pixel_size_angstrom:.3f}")
    print(f"lattice_constant_A      : {cfg.lattice_constant_angstrom:.3f}")
    print(f"defocus_nm              : {cfg.defocus_nm:.2f}")
    print(f"Cs_mm                   : {cfg.spherical_aberration_mm:.3f}")
    print(f"aperture_mrad           : {cfg.aperture_mrad:.2f}")

    summary = pd.DataFrame(
        [
            {"metric": "potential_mean", "value": float(potential.mean())},
            {"metric": "potential_std", "value": float(potential.std())},
            {"metric": "intensity_mean", "value": float(intensity.mean())},
            {"metric": "intensity_std", "value": float(intensity.std())},
            {"metric": "primary_k_1_per_A", "value": k_peak * 1e-10},
        ]
    )

    print("\nField statistics:")
    print(summary.to_string(index=False, formatters={"value": lambda x: f"{x:.6f}"}))

    print("\nPower-spectrum radial profile sample:")
    for i in range(10):
        print(f"r={i:2d} -> {profile[i]:.6e}")

    print("\nValidation checks:")
    print(report.to_string(index=False, formatters={"value": lambda x: f"{x:.6f}"}))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
