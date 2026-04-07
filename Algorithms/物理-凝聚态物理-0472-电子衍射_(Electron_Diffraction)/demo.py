"""Minimal runnable MVP for Electron Diffraction (TEM-style kinematic model).

The script implements a compact, auditable pipeline:
1) Compute relativistic electron wavelength from accelerating voltage.
2) Enumerate reciprocal-lattice reflections for a cubic fcc crystal.
3) Apply zone-law and structure-factor extinction rules.
4) Project reflections to detector coordinates and render a diffraction pattern.
5) Run built-in checks for physics consistency and geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import constants
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class DiffractionConfig:
    """Configuration for the electron diffraction MVP."""

    accelerating_voltage_kv: float = 200.0
    lattice_constant_angstrom: float = 3.615  # fcc Cu-like lattice spacing.
    camera_length_m: float = 1.0
    zone_axis: Tuple[int, int, int] = (0, 0, 1)
    max_miller_index: int = 6
    debye_waller_b_ang2: float = 0.35
    image_size: int = 512
    spot_sigma_px: float = 1.2


def electron_wavelength_relativistic(voltage_kv: float) -> float:
    """Return relativistic electron de Broglie wavelength in meters."""
    voltage_v = float(voltage_kv) * 1e3
    numerator = constants.h
    kinetic_term = 2.0 * constants.m_e * constants.e * voltage_v
    relativistic_correction = 1.0 + (constants.e * voltage_v) / (2.0 * constants.m_e * constants.c**2)
    denominator = np.sqrt(kinetic_term * relativistic_correction)
    return numerator / denominator


def enumerate_miller_indices(max_index: int) -> np.ndarray:
    """Enumerate all integer (h,k,l) with |index|<=max_index, excluding (0,0,0)."""
    idx = np.arange(-max_index, max_index + 1, dtype=np.int32)
    h, k, l = np.meshgrid(idx, idx, idx, indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1)
    keep = ~np.all(hkl == 0, axis=1)
    return hkl[keep]


def apply_zone_law(hkl: np.ndarray, zone_axis: Tuple[int, int, int]) -> np.ndarray:
    """Return reflections satisfying hu + kv + lw = 0 for the selected zone axis."""
    zone = np.asarray(zone_axis, dtype=np.int32)
    return hkl[(hkl @ zone) == 0]


def fcc_structure_factor(hkl: np.ndarray) -> np.ndarray:
    """Compute monoatomic fcc structure factor F(hkl)."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    phase = 2j * np.pi * (hkl @ basis.T)
    return np.exp(phase).sum(axis=1)


def detector_basis_from_zone(zone_axis: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal basis (e1,e2,ez) where e1/e2 span the detector plane."""
    ez = np.asarray(zone_axis, dtype=np.float64)
    ez /= np.linalg.norm(ez)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, ez))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(ez, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, ez


def build_reflection_table(cfg: DiffractionConfig) -> Tuple[pd.DataFrame, float]:
    """Generate allowed reflections and detector coordinates under kinematic approximation."""
    wavelength_m = electron_wavelength_relativistic(cfg.accelerating_voltage_kv)
    lattice_m = cfg.lattice_constant_angstrom * 1e-10

    hkl_all = enumerate_miller_indices(cfg.max_miller_index)
    hkl_zone = apply_zone_law(hkl_all, cfg.zone_axis)

    f_hkl = fcc_structure_factor(hkl_zone)
    intensity_raw = np.abs(f_hkl) ** 2

    g_cart = hkl_zone.astype(np.float64) / lattice_m
    g_mag = np.linalg.norm(g_cart, axis=1)

    b_m2 = cfg.debye_waller_b_ang2 * 1e-20
    intensity = intensity_raw * np.exp(-b_m2 * g_mag * g_mag)

    e1, e2, _ = detector_basis_from_zone(cfg.zone_axis)
    g_x = g_cart @ e1
    g_y = g_cart @ e2

    x_m = cfg.camera_length_m * wavelength_m * g_x
    y_m = cfg.camera_length_m * wavelength_m * g_y

    table = pd.DataFrame(
        {
            "h": hkl_zone[:, 0],
            "k": hkl_zone[:, 1],
            "l": hkl_zone[:, 2],
            "g_invA": g_mag * 1e-10,
            "x_mm": x_m * 1e3,
            "y_mm": y_m * 1e3,
            "intensity": intensity,
        }
    )

    table = table[table["intensity"] > 1e-10].copy()
    table["radius_mm"] = np.sqrt(table["x_mm"] ** 2 + table["y_mm"] ** 2)
    table["intensity_rel"] = table["intensity"] / float(table["intensity"].max())
    table = table.sort_values(["radius_mm", "intensity_rel"], ascending=[True, False]).reset_index(drop=True)
    return table, wavelength_m


def rasterize_pattern(
    reflections: pd.DataFrame,
    image_size: int,
    sigma_px: float,
) -> np.ndarray:
    """Rasterize diffraction spots into a normalized 2D image."""
    image = np.zeros((image_size, image_size), dtype=np.float64)
    center = 0.5 * (image_size - 1)

    radius_mm = reflections["radius_mm"].to_numpy(dtype=np.float64)
    max_radius = float(np.max(radius_mm))
    if max_radius <= 0.0:
        raise ValueError("max spot radius must be positive")

    scale = 0.46 * (image_size - 1) / max_radius
    x_px = center + reflections["x_mm"].to_numpy(dtype=np.float64) * scale
    y_px = center - reflections["y_mm"].to_numpy(dtype=np.float64) * scale
    w = reflections["intensity_rel"].to_numpy(dtype=np.float64)

    ix = np.rint(x_px).astype(np.int32)
    iy = np.rint(y_px).astype(np.int32)

    valid = (ix >= 0) & (ix < image_size) & (iy >= 0) & (iy < image_size)
    np.add.at(image, (iy[valid], ix[valid]), w[valid])

    image[int(center), int(center)] += 3.0
    blurred = gaussian_filter(image, sigma=float(sigma_px), mode="constant")
    blurred /= float(blurred.max())
    return blurred


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Compute simple radial average profile for diagnostic printing."""
    h, w = image.shape
    cy = 0.5 * (h - 1)
    cx = 0.5 * (w - 1)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rb = rr.astype(np.int32)

    sums = np.bincount(rb.ravel(), weights=image.ravel())
    counts = np.bincount(rb.ravel())
    return sums / np.maximum(counts, 1)


def run_validations(reflections: pd.DataFrame, wavelength_m: float, image: np.ndarray) -> None:
    """Run deterministic sanity checks for physics and geometry."""
    wavelength_pm = wavelength_m * 1e12
    assert 2.40 < wavelength_pm < 2.60, f"Unexpected 200kV wavelength: {wavelength_pm:.4f} pm"

    sf_test = fcc_structure_factor(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]], dtype=np.int32))
    assert abs(sf_test[0]) < 1e-10, "fcc extinction failed for (100)"
    assert abs(sf_test[1]) < 1e-10, "fcc extinction failed for (110)"
    assert abs(sf_test[2]) > 1.0, "fcc allowed reflection too weak for (111)"
    assert abs(sf_test[3]) > 1.0, "fcc allowed reflection too weak for (200)"

    mask_200 = (
        ((reflections["h"].abs() == 2) & (reflections["k"].abs() == 0) & (reflections["l"] == 0))
        | ((reflections["h"].abs() == 0) & (reflections["k"].abs() == 2) & (reflections["l"] == 0))
    )
    mask_220 = (reflections["h"].abs() == 2) & (reflections["k"].abs() == 2) & (reflections["l"] == 0)
    r200 = float(reflections.loc[mask_200, "radius_mm"].mean())
    r220 = float(reflections.loc[mask_220, "radius_mm"].mean())
    ratio = r220 / r200
    assert abs(ratio - np.sqrt(2.0)) < 0.03, f"Ring radius ratio mismatch, got {ratio:.4f}"

    symmetry_err = float(np.mean(np.abs(image - image[::-1, ::-1])))
    assert symmetry_err < 0.02, f"Pattern centrosymmetry residual too large: {symmetry_err:.5f}"


def main() -> None:
    cfg = DiffractionConfig()
    reflections, wavelength_m = build_reflection_table(cfg)
    pattern = rasterize_pattern(reflections, cfg.image_size, cfg.spot_sigma_px)
    profile = radial_profile(pattern)

    run_validations(reflections, wavelength_m, pattern)

    wavelength_pm = wavelength_m * 1e12
    print("=== Electron Diffraction MVP (fcc, kinematic, zone-axis pattern) ===")
    print(f"accelerating_voltage_kV : {cfg.accelerating_voltage_kv:.1f}")
    print(f"electron_wavelength_pm  : {wavelength_pm:.6f}")
    print(f"lattice_constant_A      : {cfg.lattice_constant_angstrom:.4f}")
    print(f"zone_axis               : {cfg.zone_axis}")
    print(f"num_reflections_kept    : {len(reflections)}")

    top = reflections.nlargest(12, "intensity_rel")[
        ["h", "k", "l", "radius_mm", "g_invA", "intensity_rel"]
    ].copy()
    top["radius_mm"] = top["radius_mm"].map(lambda x: f"{x:.4f}")
    top["g_invA"] = top["g_invA"].map(lambda x: f"{x:.4f}")
    top["intensity_rel"] = top["intensity_rel"].map(lambda x: f"{x:.4f}")

    print("\nTop reflections by relative intensity:")
    print(top.to_string(index=False))

    ring_table = reflections.copy()
    ring_table["s2"] = ring_table["h"] ** 2 + ring_table["k"] ** 2 + ring_table["l"] ** 2
    rings = (
        ring_table.groupby("s2", as_index=False)
        .agg(radius_mm=("radius_mm", "mean"), mean_intensity_rel=("intensity_rel", "mean"), count=("s2", "count"))
        .sort_values("radius_mm")
        .head(8)
    )
    rings["radius_mm"] = rings["radius_mm"].map(lambda x: f"{x:.4f}")
    rings["mean_intensity_rel"] = rings["mean_intensity_rel"].map(lambda x: f"{x:.4f}")

    print("\nFirst diffraction rings (grouped by h^2+k^2+l^2):")
    print(rings.to_string(index=False))

    sample_r = np.arange(min(12, profile.shape[0]))
    print("\nRadial profile sample (pixel radius -> average intensity):")
    for r in sample_r:
        print(f"r={int(r):2d} : {profile[r]:.6f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
