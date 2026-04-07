"""Minimal runnable MVP for two-flavor neutrino oscillation."""

from __future__ import annotations

import numpy as np

OSC_CONST = 1.267  # Unit conversion for dm2[eV^2], L[km], E[GeV]


def two_flavor_probabilities(
    theta_rad: float,
    delta_m2: float,
    l_km: float,
    e_gev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (survival, appearance) probabilities in the two-flavor vacuum model."""
    energies = np.asarray(e_gev, dtype=float)
    if energies.ndim != 1:
        raise ValueError("e_gev must be a 1D array")
    if np.any(energies <= 0.0):
        raise ValueError("all energies must be positive")

    sin2_2theta = np.sin(2.0 * theta_rad) ** 2
    phase = OSC_CONST * delta_m2 * l_km / energies
    p_appearance = sin2_2theta * np.sin(phase) ** 2
    p_survival = 1.0 - p_appearance
    return p_survival, p_appearance


def validate_physical_constraints(
    theta_rad: float,
    delta_m2: float,
    l_km: float,
    e_gev: np.ndarray,
) -> None:
    """Check probability bounds, unitarity, and the L=0 limit."""
    p_surv, p_app = two_flavor_probabilities(theta_rad, delta_m2, l_km, e_gev)

    if np.any((p_surv < -1e-12) | (p_surv > 1.0 + 1e-12)):
        raise ValueError("survival probability out of [0, 1]")
    if np.any((p_app < -1e-12) | (p_app > 1.0 + 1e-12)):
        raise ValueError("appearance probability out of [0, 1]")

    if not np.allclose(p_surv + p_app, 1.0, atol=1e-12):
        raise ValueError("unitarity check failed: P_surv + P_app != 1")

    p_surv_l0, p_app_l0 = two_flavor_probabilities(theta_rad, delta_m2, 0.0, e_gev)
    if not np.allclose(p_surv_l0, 1.0, atol=1e-12) or not np.allclose(p_app_l0, 0.0, atol=1e-12):
        raise ValueError("L=0 limit check failed")


def build_synthetic_dataset(
    e_gev: np.ndarray,
    l_km: float,
    theta_rad_true: float,
    delta_m2_true: float,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noiseless and noisy appearance probabilities."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    rng = np.random.default_rng(seed)
    _, p_true = two_flavor_probabilities(theta_rad_true, delta_m2_true, l_km, e_gev)
    noise = rng.normal(loc=0.0, scale=sigma, size=e_gev.shape[0])
    p_obs = np.clip(p_true + noise, 0.0, 1.0)
    sigma_vec = np.full(e_gev.shape[0], sigma, dtype=float)
    return p_true, p_obs, sigma_vec


def chi2_grid_search(
    e_gev: np.ndarray,
    l_km: float,
    observed: np.ndarray,
    sigma: np.ndarray,
    theta_grid_rad: np.ndarray,
    delta_m2_grid: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Grid-search best (theta, delta_m2) by minimizing chi-square."""
    energies = np.asarray(e_gev, dtype=float)
    obs = np.asarray(observed, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    theta_grid = np.asarray(theta_grid_rad, dtype=float)
    dm2_grid = np.asarray(delta_m2_grid, dtype=float)

    if energies.ndim != 1 or obs.ndim != 1 or sig.ndim != 1:
        raise ValueError("e_gev, observed, sigma must be 1D arrays")
    if energies.shape != obs.shape or energies.shape != sig.shape:
        raise ValueError("e_gev, observed, sigma must have identical shapes")
    if np.any(energies <= 0.0):
        raise ValueError("all energies must be positive")
    if np.any(sig <= 0.0):
        raise ValueError("all sigma values must be positive")
    if theta_grid.size == 0 or dm2_grid.size == 0:
        raise ValueError("theta_grid_rad and delta_m2_grid must be non-empty")

    sin2_2theta = np.sin(2.0 * theta_grid[:, None, None]) ** 2
    phase = OSC_CONST * dm2_grid[None, :, None] * l_km / energies[None, None, :]
    prediction = sin2_2theta * np.sin(phase) ** 2  # shape: (N_theta, N_dm2, N_energy)

    residual = (obs[None, None, :] - prediction) / sig[None, None, :]
    chi2_surface = np.sum(residual * residual, axis=2)

    flat_idx = int(np.argmin(chi2_surface))
    idx_theta, idx_dm2 = np.unravel_index(flat_idx, chi2_surface.shape)

    best_theta = float(theta_grid[idx_theta])
    best_delta_m2 = float(dm2_grid[idx_dm2])

    best_prediction = prediction[idx_theta, idx_dm2, :]
    return best_theta, best_delta_m2, best_prediction, chi2_surface


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def main() -> None:
    l_km = 295.0
    e_gev = np.linspace(0.35, 2.50, 14)

    theta_true_deg = 33.0
    delta_m2_true = 2.45e-3

    theta_true_rad = np.deg2rad(theta_true_deg)
    sigma = 0.03

    validate_physical_constraints(theta_true_rad, delta_m2_true, l_km, e_gev)

    p_true, p_obs, sigma_vec = build_synthetic_dataset(
        e_gev=e_gev,
        l_km=l_km,
        theta_rad_true=theta_true_rad,
        delta_m2_true=delta_m2_true,
        sigma=sigma,
        seed=7,
    )

    theta_grid_deg = np.linspace(20.0, 50.0, 241)
    theta_grid_rad = np.deg2rad(theta_grid_deg)
    delta_m2_grid = np.linspace(1.5e-3, 3.5e-3, 241)

    best_theta_rad, best_delta_m2, p_fit, chi2_surface = chi2_grid_search(
        e_gev=e_gev,
        l_km=l_km,
        observed=p_obs,
        sigma=sigma_vec,
        theta_grid_rad=theta_grid_rad,
        delta_m2_grid=delta_m2_grid,
    )

    chi2_min = float(np.min(chi2_surface))
    dof = e_gev.size - 2

    theta_err_deg = float(np.rad2deg(best_theta_rad - theta_true_rad))
    dm2_err = best_delta_m2 - delta_m2_true

    print("Neutrino Oscillation Demo (two-flavor, vacuum)")
    print(f"L = {l_km:.1f} km, energy points = {e_gev.size}")
    print(f"true_theta_deg = {theta_true_deg:.3f}, true_delta_m2 = {delta_m2_true:.6e} eV^2")
    print(
        "best_theta_deg = "
        f"{np.rad2deg(best_theta_rad):.3f}, best_delta_m2 = {best_delta_m2:.6e} eV^2"
    )
    print(f"theta_error_deg = {theta_err_deg:+.3f}, delta_m2_error = {dm2_err:+.3e} eV^2")
    print(f"chi2_min = {chi2_min:.3f}, dof = {dof}, chi2/dof = {chi2_min / dof:.3f}")
    print(f"RMSE(p_fit, p_obs) = {rmse(p_fit, p_obs):.4f}")
    print()
    print("E[GeV]   p_true   p_obs    p_fit")

    for energy, pt, po, pf in zip(e_gev, p_true, p_obs, p_fit):
        print(f"{energy:6.3f}  {pt:7.4f}  {po:7.4f}  {pf:7.4f}")


if __name__ == "__main__":
    main()
