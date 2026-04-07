"""Minimal runnable MVP for optical activity (Optical Rotatory Dispersion).

This script builds a transparent parameter-estimation pipeline for

    theta = l * c * A / (lambda^2 - lambda0^2) + b

where theta is measured rotation angle (deg), l is path length (dm),
c is concentration (g/mL), and b is instrument offset (deg).

Algorithm strategy (no black-box nonlinear optimizer):
1) Grid-search lambda0.
2) For each lambda0, solve linear least squares for (A, b).
3) Pick the candidate with minimal SSE.
4) Use fitted model to infer specific rotation at 589 nm and invert
   concentration of an unknown sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpticalActivityConfig:
    seed: int = 42
    n_samples: int = 48
    noise_std_deg: float = 0.08

    # Ground truth used to generate synthetic observations.
    true_A: float = 2.10e7
    true_lambda0_nm: float = 180.0
    true_offset_deg: float = 0.05

    # Search region for lambda0 (nm).
    lambda0_search_min: float = 130.0
    lambda0_search_max: float = 350.0
    lambda0_search_step: float = 0.2

    # Unknown-sample inversion setup.
    unknown_lambda_nm: float = 589.0
    unknown_path_dm: float = 1.5
    unknown_c_true: float = 0.11
    unknown_noise_std_deg: float = 0.03


def specific_rotation(a_param: float, lambda_nm: np.ndarray | float, lambda0_nm: float) -> np.ndarray | float:
    """Compute [alpha](lambda) using one-term Drude model."""
    return a_param / (np.asarray(lambda_nm) ** 2 - lambda0_nm**2)


def forward_rotation(
    lambda_nm: np.ndarray,
    path_dm: np.ndarray,
    concentration_g_ml: np.ndarray,
    a_param: float,
    lambda0_nm: float,
    offset_deg: float,
) -> np.ndarray:
    """Predict rotation angles theta from model parameters."""
    alpha = specific_rotation(a_param, lambda_nm, lambda0_nm)
    return path_dm * concentration_g_ml * alpha + offset_deg


def generate_synthetic_dataset(cfg: OpticalActivityConfig) -> pd.DataFrame:
    """Create a reproducible synthetic optical-activity dataset."""
    rng = np.random.default_rng(cfg.seed)

    wavelength_pool_nm = np.array([436.0, 486.0, 546.0, 589.0, 633.0], dtype=float)
    lambda_nm = rng.choice(wavelength_pool_nm, size=cfg.n_samples, replace=True)
    path_dm = rng.choice(np.array([0.5, 1.0, 2.0], dtype=float), size=cfg.n_samples, replace=True)
    concentration_g_ml = rng.uniform(0.04, 0.16, size=cfg.n_samples)

    theta_clean = forward_rotation(
        lambda_nm=lambda_nm,
        path_dm=path_dm,
        concentration_g_ml=concentration_g_ml,
        a_param=cfg.true_A,
        lambda0_nm=cfg.true_lambda0_nm,
        offset_deg=cfg.true_offset_deg,
    )
    noise = rng.normal(loc=0.0, scale=cfg.noise_std_deg, size=cfg.n_samples)
    theta_obs = theta_clean + noise

    return pd.DataFrame(
        {
            "lambda_nm": lambda_nm,
            "path_dm": path_dm,
            "concentration_g_ml": concentration_g_ml,
            "theta_obs_deg": theta_obs,
            "theta_clean_deg": theta_clean,
        }
    )


def design_feature(lambda_nm: np.ndarray, path_dm: np.ndarray, concentration_g_ml: np.ndarray, lambda0_nm: float) -> np.ndarray:
    """Feature x = l*c/(lambda^2-lambda0^2) under a fixed lambda0."""
    return path_dm * concentration_g_ml / (lambda_nm**2 - lambda0_nm**2)


def solve_linear_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Solve y ≈ A*x + b with least squares and return (A, b, y_hat, sse)."""
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_hat = float(beta[0])
    b_hat = float(beta[1])
    y_hat = X @ beta
    residual = y - y_hat
    sse = float(np.sum(residual**2))
    return a_hat, b_hat, y_hat, sse


def fit_optical_activity_model(df: pd.DataFrame, cfg: OpticalActivityConfig) -> dict[str, float | np.ndarray]:
    """Fit (A, lambda0, b) via grid-search lambda0 + linear least squares."""
    lambda_nm = df["lambda_nm"].to_numpy(dtype=float)
    path_dm = df["path_dm"].to_numpy(dtype=float)
    concentration = df["concentration_g_ml"].to_numpy(dtype=float)
    theta_obs = df["theta_obs_deg"].to_numpy(dtype=float)

    best: dict[str, float | np.ndarray] = {
        "a_hat": float("nan"),
        "lambda0_hat_nm": float("nan"),
        "offset_hat_deg": float("nan"),
        "sse": float("inf"),
        "y_hat": np.empty_like(theta_obs),
    }

    lambda0_candidates = np.arange(
        cfg.lambda0_search_min,
        cfg.lambda0_search_max + 0.5 * cfg.lambda0_search_step,
        cfg.lambda0_search_step,
    )

    min_lambda = float(np.min(lambda_nm))
    valid_upper = min_lambda - 5.0

    for lambda0 in lambda0_candidates:
        # Skip candidates too close to measured wavelengths.
        if lambda0 >= valid_upper:
            continue

        x = design_feature(lambda_nm=lambda_nm, path_dm=path_dm, concentration_g_ml=concentration, lambda0_nm=float(lambda0))
        a_hat, b_hat, y_hat, sse = solve_linear_least_squares(x=x, y=theta_obs)

        if sse < float(best["sse"]):
            best = {
                "a_hat": a_hat,
                "lambda0_hat_nm": float(lambda0),
                "offset_hat_deg": b_hat,
                "sse": sse,
                "y_hat": y_hat,
            }

    y = theta_obs
    y_hat_final = np.asarray(best["y_hat"], dtype=float)
    rmse = float(np.sqrt(np.mean((y - y_hat_final) ** 2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - float(best["sse"]) / sst) if sst > 1e-15 else 1.0

    best["rmse"] = rmse
    best["r2"] = r2
    return best


def invert_concentration(theta_deg: float, lambda_nm: float, path_dm: float, a_hat: float, lambda0_hat_nm: float, offset_hat_deg: float) -> float:
    """Infer concentration from observed theta and fitted model."""
    alpha = float(specific_rotation(a_hat, lambda_nm, lambda0_hat_nm))
    return (theta_deg - offset_hat_deg) / (path_dm * alpha)


def main() -> None:
    cfg = OpticalActivityConfig()
    df = generate_synthetic_dataset(cfg)
    fit = fit_optical_activity_model(df, cfg)

    a_hat = float(fit["a_hat"])
    lambda0_hat = float(fit["lambda0_hat_nm"])
    offset_hat = float(fit["offset_hat_deg"])
    rmse = float(fit["rmse"])
    r2 = float(fit["r2"])

    alpha_589_true = float(specific_rotation(cfg.true_A, cfg.unknown_lambda_nm, cfg.true_lambda0_nm))
    alpha_589_hat = float(specific_rotation(a_hat, cfg.unknown_lambda_nm, lambda0_hat))

    rng_unknown = np.random.default_rng(cfg.seed + 999)
    theta_unknown_clean = float(
        forward_rotation(
            lambda_nm=np.array([cfg.unknown_lambda_nm], dtype=float),
            path_dm=np.array([cfg.unknown_path_dm], dtype=float),
            concentration_g_ml=np.array([cfg.unknown_c_true], dtype=float),
            a_param=cfg.true_A,
            lambda0_nm=cfg.true_lambda0_nm,
            offset_deg=cfg.true_offset_deg,
        )[0]
    )
    theta_unknown_obs = theta_unknown_clean + float(rng_unknown.normal(0.0, cfg.unknown_noise_std_deg))
    c_unknown_hat = invert_concentration(
        theta_deg=theta_unknown_obs,
        lambda_nm=cfg.unknown_lambda_nm,
        path_dm=cfg.unknown_path_dm,
        a_hat=a_hat,
        lambda0_hat_nm=lambda0_hat,
        offset_hat_deg=offset_hat,
    )

    print("=== Optical Activity MVP (Drude ORD) ===")
    print("Synthetic dataset preview:")
    print(df.head(8).to_string(index=False, justify="center", float_format=lambda v: f"{v:8.4f}"))
    print()

    print("Fitted parameters:")
    print(f"  A_hat           = {a_hat:,.2f} (true {cfg.true_A:,.2f})")
    print(f"  lambda0_hat_nm  = {lambda0_hat:.3f} (true {cfg.true_lambda0_nm:.3f})")
    print(f"  offset_hat_deg  = {offset_hat:.4f} (true {cfg.true_offset_deg:.4f})")
    print(f"  RMSE            = {rmse:.4f} deg")
    print(f"  R^2             = {r2:.6f}")
    print()

    print("Specific rotation at 589 nm:")
    print(f"  alpha_589_hat   = {alpha_589_hat:.4f} deg/(dm*(g/mL))")
    print(f"  alpha_589_true  = {alpha_589_true:.4f} deg/(dm*(g/mL))")
    print()

    print("Unknown sample inversion:")
    print(f"  theta_obs_deg   = {theta_unknown_obs:.4f}")
    print(f"  c_hat           = {c_unknown_hat:.5f} g/mL")
    print(f"  c_true          = {cfg.unknown_c_true:.5f} g/mL")
    print(f"  abs_error       = {abs(c_unknown_hat - cfg.unknown_c_true):.5f} g/mL")

    # Basic quality gates for automated validation.
    assert abs(lambda0_hat - cfg.true_lambda0_nm) < 20.0
    assert abs(alpha_589_hat - alpha_589_true) / abs(alpha_589_true) < 0.08
    assert abs(c_unknown_hat - cfg.unknown_c_true) < 0.02


if __name__ == "__main__":
    main()
