"""Minimal runnable MVP for Gauss's Law (PHYS-0018)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import constants
from sklearn.linear_model import LinearRegression

# Coulomb constant: k = 1 / (4*pi*epsilon_0)
K_VACUUM = 1.0 / (4.0 * np.pi * constants.epsilon_0)
EPS0 = constants.epsilon_0


def electric_field_point_charge(
    points: np.ndarray,
    charge_c: float,
    charge_position: np.ndarray | None = None,
) -> np.ndarray:
    """Return E-field vectors (N/C) of one point charge at each query point."""
    if charge_position is None:
        charge_position = np.zeros(3, dtype=float)

    p = np.asarray(points, dtype=float)
    qpos = np.asarray(charge_position, dtype=float)
    r_vec = p - qpos
    r_norm = np.linalg.norm(r_vec, axis=1)

    if np.any(r_norm <= 0.0):
        raise ValueError("Query points must not coincide with the point charge.")

    return K_VACUUM * charge_c * r_vec / (r_norm[:, None] ** 3)


def sample_unit_sphere(n_samples: int, seed: int) -> np.ndarray:
    """Sample unit vectors uniformly on S^2 via normalized Gaussian vectors."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_samples, 3))
    norm = np.linalg.norm(vec, axis=1)
    return vec / norm[:, None]


def gauss_flux_sphere_mc(
    charge_c: float,
    radius_m: float,
    n_samples: int,
    seed: int,
    charge_position: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> tuple[float, float]:
    """Monte Carlo estimate of flux through a sphere: Phi = area * mean(E·n)."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")
    if center is None:
        center = np.zeros(3, dtype=float)

    dirs = sample_unit_sphere(n_samples=n_samples, seed=seed)
    points = np.asarray(center, dtype=float) + radius_m * dirs
    e_field = electric_field_point_charge(
        points=points,
        charge_c=charge_c,
        charge_position=charge_position,
    )

    integrand = np.sum(e_field * dirs, axis=1)
    area = 4.0 * np.pi * radius_m**2
    flux = float(area * np.mean(integrand))
    stderr = float(area * np.std(integrand, ddof=1) / np.sqrt(n_samples))
    return flux, stderr


def gauss_flux_sphere_torch(
    dirs: np.ndarray,
    charge_c: float,
    radius_m: float,
    charge_position: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> float:
    """Same sphere-flux estimator in PyTorch for consistency cross-check."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")
    if charge_position is None:
        charge_position = np.zeros(3, dtype=float)
    if center is None:
        center = np.zeros(3, dtype=float)

    dirs_t = torch.tensor(np.asarray(dirs, dtype=np.float64), dtype=torch.float64)
    qpos_t = torch.tensor(np.asarray(charge_position, dtype=np.float64), dtype=torch.float64)
    center_t = torch.tensor(np.asarray(center, dtype=np.float64), dtype=torch.float64)

    points_t = center_t + radius_m * dirs_t
    r_vec_t = points_t - qpos_t
    r_norm_t = torch.linalg.norm(r_vec_t, dim=1)

    if torch.any(r_norm_t <= 0):
        raise ValueError("Query points must not coincide with the point charge.")

    e_field_t = K_VACUUM * charge_c * r_vec_t / (r_norm_t[:, None] ** 3)
    integrand_t = torch.sum(e_field_t * dirs_t, dim=1)
    area = 4.0 * np.pi * radius_m**2
    flux_t = area * torch.mean(integrand_t)
    return float(flux_t.detach().cpu().item())


def gauss_flux_cube_mc(
    charge_c: float,
    edge_m: float,
    n_per_face: int,
    seed: int,
    charge_position: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> float:
    """Monte Carlo estimate of flux through an axis-aligned cube."""
    if edge_m <= 0:
        raise ValueError("edge_m must be positive.")
    if n_per_face <= 0:
        raise ValueError("n_per_face must be positive.")
    if charge_position is None:
        charge_position = np.zeros(3, dtype=float)
    if center is None:
        center = np.zeros(3, dtype=float)

    rng = np.random.default_rng(seed)
    half = edge_m / 2.0
    face_area = edge_m**2
    total_flux = 0.0

    # (axis index, sign)
    for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        uv = rng.uniform(-half, half, size=(n_per_face, 2))
        pts = np.zeros((n_per_face, 3), dtype=float)
        free_axes = [ax for ax in (0, 1, 2) if ax != axis]
        pts[:, free_axes[0]] = uv[:, 0]
        pts[:, free_axes[1]] = uv[:, 1]
        pts[:, axis] = sign * half
        pts += np.asarray(center, dtype=float)

        normal = np.zeros(3, dtype=float)
        normal[axis] = float(sign)

        e_field = electric_field_point_charge(
            points=pts,
            charge_c=charge_c,
            charge_position=charge_position,
        )
        flux_face = face_area * float(np.mean(e_field @ normal))
        total_flux += flux_face

    return float(total_flux)


def linear_fit_inside_uniform_sphere(
    rho_c_per_m3: float,
    radii_m: np.ndarray,
) -> tuple[float, float, float]:
    """Fit E(r)=a*r+b for a uniformly charged solid sphere interior field."""
    if rho_c_per_m3 == 0:
        raise ValueError("rho_c_per_m3 must be non-zero for this check.")

    r = np.asarray(radii_m, dtype=float)
    if np.any(r <= 0):
        raise ValueError("All radii must be positive.")

    e_vals = rho_c_per_m3 * r / (3.0 * EPS0)
    X = r.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, e_vals)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, e_vals))
    return slope, intercept, r2


def main() -> None:
    print("=== PHYS-0018 Gauss's Law MVP ===")
    print(f"epsilon_0 = {EPS0:.6e} F/m")

    q = 2.5e-9
    radii = np.array([0.05, 0.10, 0.20, 0.35], dtype=float)
    expected_flux = q / EPS0

    records: list[dict[str, float]] = []
    base_seed = 20260407
    n_samples = 120_000

    for idx, r in enumerate(radii):
        flux_np, stderr_np = gauss_flux_sphere_mc(
            charge_c=q,
            radius_m=float(r),
            n_samples=n_samples,
            seed=base_seed + idx,
            charge_position=np.zeros(3, dtype=float),
            center=np.zeros(3, dtype=float),
        )
        dirs = sample_unit_sphere(n_samples=n_samples, seed=base_seed + idx)
        flux_torch = gauss_flux_sphere_torch(
            dirs=dirs,
            charge_c=q,
            radius_m=float(r),
            charge_position=np.zeros(3, dtype=float),
            center=np.zeros(3, dtype=float),
        )

        rel_err_np = abs(flux_np - expected_flux) / abs(expected_flux)
        rel_err_torch = abs(flux_torch - expected_flux) / abs(expected_flux)

        records.append(
            {
                "radius_m": float(r),
                "flux_numpy": flux_np,
                "flux_torch": flux_torch,
                "flux_expected": expected_flux,
                "stderr_numpy": stderr_np,
                "rel_err_numpy": rel_err_np,
                "rel_err_torch": rel_err_torch,
            }
        )

    sphere_df = pd.DataFrame(records)

    print("\n[Sphere Gaussian surface: point charge at center]")
    print(sphere_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # Geometry independence check: same enclosed charge, different Gaussian surface.
    cube_flux = gauss_flux_cube_mc(
        charge_c=q,
        edge_m=0.30,
        n_per_face=60_000,
        seed=base_seed + 100,
        charge_position=np.zeros(3, dtype=float),
        center=np.zeros(3, dtype=float),
    )
    cube_rel_err = abs(cube_flux - expected_flux) / abs(expected_flux)

    print("\n[Cube Gaussian surface: same enclosed charge]")
    print(f"flux_cube = {cube_flux:.6e}")
    print(f"flux_expected = {expected_flux:.6e}")
    print(f"relative_error = {cube_rel_err:.6e}")

    # Zero enclosed charge check: charge outside a closed sphere -> net flux ~ 0.
    outside_flux, outside_stderr = gauss_flux_sphere_mc(
        charge_c=q,
        radius_m=0.10,
        n_samples=200_000,
        seed=base_seed + 200,
        charge_position=np.array([0.35, 0.0, 0.0], dtype=float),
        center=np.zeros(3, dtype=float),
    )

    print("\n[No enclosed charge check: source outside sphere]")
    print(f"flux_outside = {outside_flux:.6e}")
    print(f"stderr_outside = {outside_stderr:.6e}")

    # Inside uniformly charged solid sphere: E(r) should be linear in r.
    rho = 8.0e-6
    r_fit = np.linspace(0.01, 0.12, 12)
    slope, intercept, r2 = linear_fit_inside_uniform_sphere(rho_c_per_m3=rho, radii_m=r_fit)
    slope_theory = rho / (3.0 * EPS0)

    print("\n[Uniform volume charge interior-field linear fit: E(r)=a*r+b]")
    print(f"slope_fit = {slope:.6e}, slope_theory = {slope_theory:.6e}")
    print(f"intercept_fit = {intercept:.6e} (ideal: 0)")
    print(f"R^2 = {r2:.6f}")


if __name__ == "__main__":
    main()
