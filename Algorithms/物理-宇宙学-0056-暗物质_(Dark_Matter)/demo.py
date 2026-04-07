"""Dark matter MVP: fit an NFW halo to synthetic galaxy rotation-curve data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# Gravitational constant in units: kpc * (km/s)^2 / Msun
G_KPC_KMS_MSU = 4.30091e-6


@dataclass(frozen=True)
class GalaxyConfig:
    n_points: int = 64
    r_min_kpc: float = 0.8
    r_max_kpc: float = 28.0
    noise_sigma_kms: float = 4.0
    true_log10_rho_s: float = 7.1  # Msun / kpc^3
    true_log10_r_s: float = 1.08  # kpc
    true_log10_upsilon: float = -0.05  # baryon mass-to-light proxy
    seed: int = 56


def baryon_velocity_base(r_kpc: np.ndarray) -> np.ndarray:
    """Simple baryonic rotation proxy from disk + bulge components (km/s)."""
    r = np.asarray(r_kpc, dtype=float)
    if np.any(r <= 0):
        raise ValueError("r_kpc must be strictly positive.")

    # Heuristic profiles: enough for a stable synthetic MVP.
    disk = 205.0 * (r / 3.3) * np.exp(-r / 6.6)
    bulge = 150.0 * (r / (r + 1.1))
    return np.sqrt(disk**2 + bulge**2)


def nfw_enclosed_mass(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    """NFW enclosed mass M(<r) in Msun."""
    if rho_s <= 0.0 or r_s <= 0.0:
        raise ValueError("rho_s and r_s must be positive.")

    x = np.asarray(r_kpc, dtype=float) / r_s
    shape = np.log1p(x) - x / (1.0 + x)
    return 4.0 * np.pi * rho_s * (r_s**3) * shape


def nfw_halo_velocity(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    """Circular velocity contributed by NFW halo in km/s."""
    r = np.asarray(r_kpc, dtype=float)
    m = nfw_enclosed_mass(r, rho_s=rho_s, r_s=r_s)
    return np.sqrt(G_KPC_KMS_MSU * m / r)


def total_velocity_from_params(
    r_kpc: np.ndarray, log10_upsilon: float, log10_rho_s: float, log10_r_s: float
) -> np.ndarray:
    """Total circular speed from baryons + NFW halo."""
    upsilon = 10.0**log10_upsilon
    rho_s = 10.0**log10_rho_s
    r_s = 10.0**log10_r_s

    v_bary = np.sqrt(upsilon) * baryon_velocity_base(r_kpc)
    v_halo = nfw_halo_velocity(r_kpc, rho_s=rho_s, r_s=r_s)
    return np.sqrt(v_bary**2 + v_halo**2)


def simulate_rotation_curve(cfg: GalaxyConfig) -> pd.DataFrame:
    """Create deterministic synthetic observations."""
    if cfg.n_points < 8:
        raise ValueError("n_points must be >= 8.")
    if cfg.r_min_kpc <= 0.0 or cfg.r_max_kpc <= cfg.r_min_kpc:
        raise ValueError("Invalid radial range.")
    if cfg.noise_sigma_kms <= 0.0:
        raise ValueError("noise_sigma_kms must be > 0.")

    rng = np.random.default_rng(cfg.seed)
    r = np.linspace(cfg.r_min_kpc, cfg.r_max_kpc, cfg.n_points)
    v_true = total_velocity_from_params(
        r,
        log10_upsilon=cfg.true_log10_upsilon,
        log10_rho_s=cfg.true_log10_rho_s,
        log10_r_s=cfg.true_log10_r_s,
    )
    v_obs = v_true + rng.normal(0.0, cfg.noise_sigma_kms, size=cfg.n_points)
    sigma = np.full_like(r, cfg.noise_sigma_kms, dtype=float)

    return pd.DataFrame(
        {
            "r_kpc": r,
            "v_obs_kms": v_obs,
            "sigma_kms": sigma,
            "v_true_kms": v_true,
            "v_bary_base_kms": baryon_velocity_base(r),
        }
    )


def residual_baryon_only(theta: np.ndarray, r: np.ndarray, v_obs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    log10_upsilon = float(theta[0])
    v_pred = np.sqrt(10.0**log10_upsilon) * baryon_velocity_base(r)
    return (v_pred - v_obs) / sigma


def residual_with_halo(theta: np.ndarray, r: np.ndarray, v_obs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    log10_upsilon, log10_rho_s, log10_r_s = [float(x) for x in theta]
    v_pred = total_velocity_from_params(r, log10_upsilon, log10_rho_s, log10_r_s)
    return (v_pred - v_obs) / sigma


def fit_baryon_only(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    r = df["r_kpc"].to_numpy()
    v_obs = df["v_obs_kms"].to_numpy()
    sigma = df["sigma_kms"].to_numpy()

    result = least_squares(
        residual_baryon_only,
        x0=np.array([-0.1], dtype=float),
        bounds=(np.array([-1.0]), np.array([0.8])),
        args=(r, v_obs, sigma),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"Baryon-only fit failed: {result.message}")

    log10_upsilon = float(result.x[0])
    v_pred = np.sqrt(10.0**log10_upsilon) * baryon_velocity_base(r)
    return {
        "params": result.x.copy(),
        "v_pred": v_pred,
        "chi2": float(np.sum(((v_pred - v_obs) / sigma) ** 2)),
        "k": 1.0,
    }


def fit_with_halo(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    r = df["r_kpc"].to_numpy()
    v_obs = df["v_obs_kms"].to_numpy()
    sigma = df["sigma_kms"].to_numpy()

    result = least_squares(
        residual_with_halo,
        x0=np.array([-0.1, 7.0, 1.0], dtype=float),
        bounds=(np.array([-1.0, 5.5, 0.2]), np.array([0.8, 9.8, 1.8])),
        args=(r, v_obs, sigma),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"Halo fit failed: {result.message}")

    log10_upsilon, log10_rho_s, log10_r_s = [float(x) for x in result.x]
    v_pred = total_velocity_from_params(r, log10_upsilon, log10_rho_s, log10_r_s)
    return {
        "params": result.x.copy(),
        "v_pred": v_pred,
        "chi2": float(np.sum(((v_pred - v_obs) / sigma) ** 2)),
        "k": 3.0,
    }


def model_metrics(v_obs: np.ndarray, v_pred: np.ndarray, sigma: np.ndarray, k: float) -> dict[str, float]:
    n = float(len(v_obs))
    chi2 = float(np.sum(((v_pred - v_obs) / sigma) ** 2))
    rmse = float(np.sqrt(mean_squared_error(v_obs, v_pred)))
    r2 = float(r2_score(v_obs, v_pred))
    aic = chi2 + 2.0 * k
    bic = chi2 + k * np.log(n)
    return {"chi2": chi2, "rmse": rmse, "r2": r2, "aic": float(aic), "bic": float(bic)}


def torch_consistency_check(log10_rho_s: float, log10_r_s: float) -> float:
    """Check NFW halo velocity formula in torch vs numpy."""
    r = np.linspace(0.9, 26.0, 40)
    rho_s = 10.0**log10_rho_s
    r_s = 10.0**log10_r_s
    v_np = nfw_halo_velocity(r, rho_s=rho_s, r_s=r_s)

    r_t = torch.tensor(r, dtype=torch.float64)
    rho_t = torch.tensor(rho_s, dtype=torch.float64)
    rs_t = torch.tensor(r_s, dtype=torch.float64)
    x_t = r_t / rs_t
    shape_t = torch.log1p(x_t) - x_t / (1.0 + x_t)
    m_t = 4.0 * torch.pi * rho_t * (rs_t**3) * shape_t
    v_t = torch.sqrt(G_KPC_KMS_MSU * m_t / r_t)
    return float(np.max(np.abs(v_np - v_t.detach().cpu().numpy())))


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    cfg = GalaxyConfig()
    df = simulate_rotation_curve(cfg)

    fit_b = fit_baryon_only(df)
    fit_h = fit_with_halo(df)

    v_obs = df["v_obs_kms"].to_numpy()
    sigma = df["sigma_kms"].to_numpy()
    metrics_b = model_metrics(v_obs, fit_b["v_pred"], sigma, k=float(fit_b["k"]))
    metrics_h = model_metrics(v_obs, fit_h["v_pred"], sigma, k=float(fit_h["k"]))

    # Parameter report
    halo_params = fit_h["params"]
    log10_upsilon_hat, log10_rho_s_hat, log10_r_s_hat = [float(x) for x in halo_params]
    print("=== Dark Matter Rotation-Curve Fit (Synthetic) ===")
    print(
        "True params:  "
        f"log10_upsilon={cfg.true_log10_upsilon:.3f}, "
        f"log10_rho_s={cfg.true_log10_rho_s:.3f}, "
        f"log10_r_s={cfg.true_log10_r_s:.3f}"
    )
    print(
        "Fitted params:"
        f" log10_upsilon={log10_upsilon_hat:.3f},"
        f" log10_rho_s={log10_rho_s_hat:.3f},"
        f" log10_r_s={log10_r_s_hat:.3f}"
    )

    # Metric table
    metric_table = pd.DataFrame(
        [
            {"model": "baryon_only", **metrics_b},
            {"model": "baryon_plus_nfw", **metrics_h},
        ]
    )
    print("\n=== Model Comparison ===")
    print(metric_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Show sample predictions
    preview = df[["r_kpc", "v_obs_kms", "v_true_kms"]].copy()
    preview["v_pred_baryon_only"] = fit_b["v_pred"]
    preview["v_pred_baryon_plus_nfw"] = fit_h["v_pred"]
    print("\n=== Prediction Preview (first 10 rows) ===")
    print(preview.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Torch/Numpy formula consistency
    torch_diff = torch_consistency_check(log10_rho_s_hat, log10_r_s_hat)
    print(f"\nTorch vs NumPy halo formula max |diff|: {torch_diff:.6e}")

    # Automated quality checks
    if not (metrics_h["rmse"] < 0.9 * metrics_b["rmse"]):
        raise AssertionError("Halo model should outperform baryon-only RMSE on this dataset.")
    if not (abs(log10_rho_s_hat - cfg.true_log10_rho_s) < 0.35):
        raise AssertionError("Recovered log10_rho_s is too far from synthetic ground truth.")
    if not (abs(log10_r_s_hat - cfg.true_log10_r_s) < 0.25):
        raise AssertionError("Recovered log10_r_s is too far from synthetic ground truth.")
    if not (torch_diff < 1e-9):
        raise AssertionError("Torch and NumPy halo implementations diverge unexpectedly.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
