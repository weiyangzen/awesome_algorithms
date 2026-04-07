"""Minimal runnable MVP for Chirp Mass (啁啾质量) estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

# SI constants
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_KG = 1.98847e30


@dataclass
class ChirpFitResult:
    chirp_mass_true_solar: float
    chirp_mass_est_point_solar: float
    chirp_mass_est_opt_solar: float
    point_rel_error: float
    opt_rel_error: float
    objective_value: float
    n_iter: int


def chirp_mass_from_component_masses(m1_solar: float, m2_solar: float) -> float:
    """Return chirp mass in solar masses from two component masses (solar masses)."""
    m1 = float(m1_solar)
    m2 = float(m2_solar)
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Component masses must be positive.")
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def fdot_from_chirp_mass(chirp_mass_solar: float, f_hz: np.ndarray) -> np.ndarray:
    """Leading-order inspiral chirp equation: df/dt as a function of chirp mass and frequency."""
    if chirp_mass_solar <= 0:
        raise ValueError("Chirp mass must be positive.")
    f = np.asarray(f_hz, dtype=float)
    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")

    mc_kg = chirp_mass_solar * M_SUN_KG
    prefactor = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * (G_SI * mc_kg / C_SI**3) ** (5.0 / 3.0)
    return prefactor * np.power(f, 11.0 / 3.0)


def chirp_mass_from_f_and_fdot(f_hz: np.ndarray, fdot_hz_per_s: np.ndarray) -> np.ndarray:
    """Invert leading-order chirp equation to estimate chirp mass pointwise."""
    f = np.asarray(f_hz, dtype=float)
    fdot = np.asarray(fdot_hz_per_s, dtype=float)
    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")
    if np.any(fdot <= 0):
        raise ValueError("Frequency derivatives must be positive.")

    base = (5.0 / 96.0) * np.pi ** (-8.0 / 3.0) * fdot * np.power(f, -11.0 / 3.0)
    base = np.maximum(base, 1e-40)
    mc_kg = (C_SI**3 / G_SI) * np.power(base, 3.0 / 5.0)
    return mc_kg / M_SUN_KG


def make_synthetic_observation(
    m1_solar: float,
    m2_solar: float,
    n_points: int = 160,
    f_min_hz: float = 25.0,
    f_max_hz: float = 180.0,
    rel_noise_std: float = 0.03,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Generate synthetic chirp observations (f, fdot) with multiplicative noise."""
    if n_points < 20:
        raise ValueError("n_points must be >= 20 for a stable demo.")
    if not (0 < f_min_hz < f_max_hz):
        raise ValueError("Need 0 < f_min_hz < f_max_hz.")
    if rel_noise_std < 0:
        raise ValueError("rel_noise_std must be non-negative.")

    rng = np.random.default_rng(seed)
    f = np.linspace(f_min_hz, f_max_hz, n_points)

    mc_true = chirp_mass_from_component_masses(m1_solar, m2_solar)
    fdot_true = fdot_from_chirp_mass(mc_true, f)

    # Multiplicative Gaussian noise approximates measurement uncertainty.
    noise = rng.normal(loc=0.0, scale=rel_noise_std, size=n_points)
    fdot_obs = fdot_true * (1.0 + noise)
    fdot_obs = np.clip(fdot_obs, 1e-15, None)

    mc_point = chirp_mass_from_f_and_fdot(f, fdot_obs)

    df = pd.DataFrame(
        {
            "f_hz": f,
            "fdot_true_hz_per_s": fdot_true,
            "fdot_obs_hz_per_s": fdot_obs,
            "chirp_mass_point_est_solar": mc_point,
        }
    )
    return mc_true, df


def estimate_chirp_mass_pointwise(df: pd.DataFrame, trim_ratio: float = 0.10) -> float:
    """Robust estimate via trimmed mean over pointwise inverted chirp-mass samples."""
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5).")
    mc_samples = df["chirp_mass_point_est_solar"].to_numpy(dtype=float)
    return float(stats.trim_mean(mc_samples, proportiontocut=trim_ratio))


def estimate_chirp_mass_nonlinear_ls(
    f_hz: np.ndarray,
    fdot_obs_hz_per_s: np.ndarray,
    init_mc_solar: float,
) -> tuple[float, float, int]:
    """Estimate chirp mass by minimizing relative squared error in df/dt model."""
    f = np.asarray(f_hz, dtype=float)
    fdot_obs = np.asarray(fdot_obs_hz_per_s, dtype=float)

    if np.any(f <= 0) or np.any(fdot_obs <= 0):
        raise ValueError("f and fdot observations must be positive.")
    if init_mc_solar <= 0:
        raise ValueError("init_mc_solar must be positive.")

    def objective(log_mc: np.ndarray) -> float:
        mc = float(np.exp(log_mc[0]))
        pred = fdot_from_chirp_mass(mc, f)
        resid = (pred - fdot_obs) / np.maximum(fdot_obs, 1e-20)
        return 0.5 * float(np.mean(resid**2))

    result = optimize.minimize(
        objective,
        x0=np.array([np.log(init_mc_solar)], dtype=float),
        method="L-BFGS-B",
        bounds=[(np.log(1e-3), np.log(300.0))],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    mc_hat = float(np.exp(result.x[0]))
    return mc_hat, float(result.fun), int(result.nit)


def fit_chirp_mass_demo() -> tuple[ChirpFitResult, pd.DataFrame]:
    """Run end-to-end simulation + estimation and return metrics/table."""
    m1_solar = 36.0
    m2_solar = 29.0

    mc_true, df = make_synthetic_observation(
        m1_solar=m1_solar,
        m2_solar=m2_solar,
        n_points=180,
        f_min_hz=25.0,
        f_max_hz=180.0,
        rel_noise_std=0.03,
        seed=2026,
    )

    mc_point = estimate_chirp_mass_pointwise(df, trim_ratio=0.10)
    mc_opt, obj, n_iter = estimate_chirp_mass_nonlinear_ls(
        df["f_hz"].to_numpy(),
        df["fdot_obs_hz_per_s"].to_numpy(),
        init_mc_solar=mc_point,
    )

    point_rel_error = abs(mc_point - mc_true) / mc_true
    opt_rel_error = abs(mc_opt - mc_true) / mc_true

    fit_result = ChirpFitResult(
        chirp_mass_true_solar=mc_true,
        chirp_mass_est_point_solar=mc_point,
        chirp_mass_est_opt_solar=mc_opt,
        point_rel_error=point_rel_error,
        opt_rel_error=opt_rel_error,
        objective_value=obj,
        n_iter=n_iter,
    )

    df = df.copy()
    df["fdot_pred_opt_hz_per_s"] = fdot_from_chirp_mass(mc_opt, df["f_hz"].to_numpy())
    df["point_abs_err_solar"] = np.abs(df["chirp_mass_point_est_solar"] - mc_true)
    return fit_result, df


def main() -> None:
    fit_result, df = fit_chirp_mass_demo()

    print("=== Chirp Mass MVP Demo ===")
    print(f"true_chirp_mass_solar      : {fit_result.chirp_mass_true_solar:.6f}")
    print(f"point_estimate_solar       : {fit_result.chirp_mass_est_point_solar:.6f}")
    print(f"nonlinear_ls_estimate_solar: {fit_result.chirp_mass_est_opt_solar:.6f}")
    print(f"point_relative_error       : {fit_result.point_rel_error:.4%}")
    print(f"ls_relative_error          : {fit_result.opt_rel_error:.4%}")
    print(f"optimizer_objective        : {fit_result.objective_value:.6e}")
    print(f"optimizer_iterations       : {fit_result.n_iter}")

    # Minimal quality gate for this synthetic demo.
    if fit_result.opt_rel_error > 0.08:
        raise RuntimeError(
            "Estimated chirp mass is too far from truth (>8%). "
            "Inspect noise level or model assumptions."
        )

    show_cols = [
        "f_hz",
        "fdot_obs_hz_per_s",
        "chirp_mass_point_est_solar",
        "point_abs_err_solar",
        "fdot_pred_opt_hz_per_s",
    ]
    print("\nSample observations (first 8 rows):")
    print(df.loc[:, show_cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
