"""Minimal runnable MVP for Einstein Ring (point-mass gravitational lensing)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physical constants in SI units.
G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
MPC = 3.085677581e22
ARCSEC_PER_RAD = 206_265.0


@dataclass(frozen=True)
class LensSystem:
    """Simple lensing configuration for a point-mass lens."""

    name: str
    mass_msun: float
    d_l_mpc: float
    d_s_mpc: float
    beta_x_arcsec: float
    beta_y_arcsec: float
    source_sigma_arcsec: float


def validate_system(system: LensSystem) -> None:
    values = np.array(
        [
            system.mass_msun,
            system.d_l_mpc,
            system.d_s_mpc,
            system.beta_x_arcsec,
            system.beta_y_arcsec,
            system.source_sigma_arcsec,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("LensSystem contains non-finite values")
    if system.mass_msun <= 0.0:
        raise ValueError("Lens mass must be positive")
    if system.d_l_mpc <= 0.0:
        raise ValueError("Observer-lens distance must be positive")
    if system.d_s_mpc <= system.d_l_mpc:
        raise ValueError("Observer-source distance must be greater than observer-lens distance")
    if system.source_sigma_arcsec <= 0.0:
        raise ValueError("Source sigma must be positive")


def einstein_radius_arcsec(system: LensSystem) -> float:
    """Compute Einstein angular radius for a point-mass lens."""
    validate_system(system)

    mass_si = system.mass_msun * M_SUN
    d_l = system.d_l_mpc * MPC
    d_s = system.d_s_mpc * MPC
    d_ls = d_s - d_l

    theta_e_rad = np.sqrt((4.0 * G * mass_si / (C**2)) * (d_ls / (d_l * d_s)))
    theta_e_arcsec = float(theta_e_rad * ARCSEC_PER_RAD)
    if not np.isfinite(theta_e_arcsec) or theta_e_arcsec <= 0.0:
        raise RuntimeError("Einstein radius computation failed")
    return theta_e_arcsec


def build_theta_grid(npix: int = 401, fov_arcsec: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    if npix < 64:
        raise ValueError("npix must be >= 64")
    if not np.isfinite(fov_arcsec) or fov_arcsec <= 0.0:
        raise ValueError("fov_arcsec must be positive and finite")

    axis = np.linspace(-0.5 * fov_arcsec, 0.5 * fov_arcsec, npix)
    theta_x, theta_y = np.meshgrid(axis, axis, indexing="xy")
    return theta_x, theta_y


def lens_equation_point_mass(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    theta_e_arcsec: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Map image-plane coordinates theta to source-plane coordinates beta."""
    theta_sq = theta_x**2 + theta_y**2
    alpha_factor = (theta_e_arcsec**2) / (theta_sq + eps)

    beta_x = theta_x - alpha_factor * theta_x
    beta_y = theta_y - alpha_factor * theta_y
    return beta_x, beta_y


def gaussian_source_intensity(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    beta_x0: float,
    beta_y0: float,
    sigma_arcsec: float,
) -> np.ndarray:
    if sigma_arcsec <= 0.0:
        raise ValueError("sigma_arcsec must be positive")

    dist_sq = (beta_x - beta_x0) ** 2 + (beta_y - beta_y0) ** 2
    return np.exp(-dist_sq / (2.0 * sigma_arcsec**2))


def simulate_lensed_image(
    system: LensSystem,
    npix: int = 401,
    fov_arcsec: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta_x, theta_y = build_theta_grid(npix=npix, fov_arcsec=fov_arcsec)
    theta_e = einstein_radius_arcsec(system)

    beta_x, beta_y = lens_equation_point_mass(theta_x, theta_y, theta_e)
    image = gaussian_source_intensity(
        beta_x,
        beta_y,
        beta_x0=system.beta_x_arcsec,
        beta_y0=system.beta_y_arcsec,
        sigma_arcsec=system.source_sigma_arcsec,
    )
    return theta_x, theta_y, image, theta_e


def radial_profile(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    image: np.ndarray,
    n_bins: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape != theta_x.shape or image.shape != theta_y.shape:
        raise ValueError("theta and image shapes must match")
    if n_bins < 16:
        raise ValueError("n_bins must be >= 16")

    radius = np.sqrt(theta_x**2 + theta_y**2)
    r_max = float(radius.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    flat_idx = np.digitize(radius.ravel(), edges) - 1
    valid = (flat_idx >= 0) & (flat_idx < n_bins)

    counts = np.bincount(flat_idx[valid], minlength=n_bins)
    sums = np.bincount(flat_idx[valid], weights=image.ravel()[valid], minlength=n_bins)
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return centers, profile


def estimate_ring_radius(
    centers: np.ndarray,
    profile: np.ndarray,
    min_radius_arcsec: float = 0.2,
) -> float:
    if centers.shape != profile.shape:
        raise ValueError("centers and profile shapes must match")

    mask = centers >= min_radius_arcsec
    if not np.any(mask):
        raise RuntimeError("No radial bins satisfy minimum radius constraint")

    local_centers = centers[mask]
    local_profile = profile[mask]
    peak_idx = int(np.argmax(local_profile))
    return float(local_centers[peak_idx])


def azimuthal_uniformity_cv(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    image: np.ndarray,
    radius_arcsec: float,
    band_half_width_arcsec: float = 0.06,
    n_phi_bins: int = 72,
) -> float:
    radius = np.sqrt(theta_x**2 + theta_y**2)
    angles = np.arctan2(theta_y, theta_x)
    mask = np.abs(radius - radius_arcsec) <= band_half_width_arcsec

    if np.count_nonzero(mask) < n_phi_bins:
        raise RuntimeError("Insufficient pixels in annulus for azimuthal analysis")

    angle_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)
    phi_idx = np.digitize(angles[mask], angle_edges) - 1
    phi_idx = np.clip(phi_idx, 0, n_phi_bins - 1)

    vals = image[mask]
    counts = np.bincount(phi_idx, minlength=n_phi_bins)
    sums = np.bincount(phi_idx, weights=vals, minlength=n_phi_bins)
    mean_by_phi = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    valid_means = mean_by_phi[counts > 0]
    if valid_means.size == 0 or float(valid_means.mean()) <= 0.0:
        raise RuntimeError("Invalid azimuthal intensity statistics")

    return float(valid_means.std() / valid_means.mean())


def analyze_system(system: LensSystem, npix: int = 401, fov_arcsec: float = 6.0) -> dict[str, float | str]:
    theta_x, theta_y, image, theta_e = simulate_lensed_image(system, npix=npix, fov_arcsec=fov_arcsec)
    centers, profile = radial_profile(theta_x, theta_y, image)
    radius_hat = estimate_ring_radius(centers, profile)
    rel_err = abs(radius_hat - theta_e) / theta_e

    azimuthal_cv = azimuthal_uniformity_cv(
        theta_x,
        theta_y,
        image,
        radius_arcsec=theta_e,
        band_half_width_arcsec=max(0.04, 0.8 * system.source_sigma_arcsec),
    )

    ring_like = rel_err < 0.08 and azimuthal_cv < 0.25
    return {
        "case": system.name,
        "theta_E_arcsec": theta_e,
        "radius_hat_arcsec": radius_hat,
        "rel_error": rel_err,
        "azimuthal_cv": azimuthal_cv,
        "ring_like": ring_like,
        "beta_offset_arcsec": float(np.hypot(system.beta_x_arcsec, system.beta_y_arcsec)),
    }


def run_checks(results: list[dict[str, float | str]]) -> None:
    if len(results) != 2:
        raise RuntimeError("Expected exactly two benchmark cases")

    aligned = results[0]
    offset = results[1]

    if bool(aligned["ring_like"]) is not True:
        raise RuntimeError("Aligned case should be recognized as ring-like")
    if float(aligned["rel_error"]) >= 0.06:
        raise RuntimeError("Aligned case ring radius error is too large")
    if float(aligned["azimuthal_cv"]) >= 0.20:
        raise RuntimeError("Aligned case should have high azimuthal uniformity")

    if bool(offset["ring_like"]) is not False:
        raise RuntimeError("Offset case should not be recognized as full ring")
    if float(offset["azimuthal_cv"]) <= 0.80:
        raise RuntimeError("Offset case should have strong azimuthal asymmetry")


def main() -> None:
    base = LensSystem(
        name="near-aligned source (Einstein ring)",
        mass_msun=1.0e12,
        d_l_mpc=1200.0,
        d_s_mpc=2200.0,
        beta_x_arcsec=0.0,
        beta_y_arcsec=0.0,
        source_sigma_arcsec=0.08,
    )
    theta_e_base = einstein_radius_arcsec(base)

    offset = LensSystem(
        name="misaligned source (broken arc)",
        mass_msun=base.mass_msun,
        d_l_mpc=base.d_l_mpc,
        d_s_mpc=base.d_s_mpc,
        beta_x_arcsec=0.70 * theta_e_base,
        beta_y_arcsec=0.0,
        source_sigma_arcsec=base.source_sigma_arcsec,
    )

    cases = [base, offset]
    results = [analyze_system(case) for case in cases]
    run_checks(results)

    df = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    print("Einstein Ring MVP report (point-mass lens)")
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:0.6f}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
