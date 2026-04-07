"""Minimal runnable MVP for Faraday Effect.

Model:
    theta = V(lambda) * B * L + b
with a simple dispersion approximation
    V(lambda) = K / lambda^2

This script demonstrates:
1) Synthetic magneto-optic data generation.
2) Closed-form least-squares estimation of (K, b).
3) Inversion of unknown magnetic field from measured rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaradayConfig:
    seed: int = 476
    n_samples: int = 120
    noise_std_rad: float = 8.0e-4

    # Ground truth for synthetic data.
    true_k: float = 4.80e4  # rad * nm^2 / (T*m)
    true_offset_rad: float = 6.0e-4

    # Sampling ranges.
    wavelength_choices_nm: tuple[float, ...] = (405.0, 532.0, 633.0, 780.0)
    b_min_t: float = -1.8
    b_max_t: float = 1.8
    l_min_m: float = 0.02
    l_max_m: float = 0.10

    # Unknown-sample inversion target.
    unknown_lambda_nm: float = 532.0
    unknown_path_m: float = 0.03
    unknown_b_true_t: float = 0.82
    unknown_noise_std_rad: float = 4.0e-4


def verdet_constant(k_param: float, lambda_nm: np.ndarray | float) -> np.ndarray | float:
    """Return Verdet constant V(lambda)=K/lambda^2 in rad/(T*m)."""
    lam = np.asarray(lambda_nm, dtype=float)
    return k_param / (lam**2)


def forward_rotation_rad(
    lambda_nm: np.ndarray,
    magnetic_field_t: np.ndarray,
    path_length_m: np.ndarray,
    k_param: float,
    offset_rad: float,
) -> np.ndarray:
    """Compute rotation angles theta in radians."""
    v_lambda = verdet_constant(k_param, lambda_nm)
    return v_lambda * magnetic_field_t * path_length_m + offset_rad


def generate_synthetic_dataset(cfg: FaradayConfig) -> pd.DataFrame:
    """Create reproducible synthetic Faraday-effect observations."""
    rng = np.random.default_rng(cfg.seed)

    lambda_nm = rng.choice(np.array(cfg.wavelength_choices_nm, dtype=float), size=cfg.n_samples, replace=True)
    magnetic_field_t = rng.uniform(cfg.b_min_t, cfg.b_max_t, size=cfg.n_samples)
    path_length_m = rng.uniform(cfg.l_min_m, cfg.l_max_m, size=cfg.n_samples)

    theta_clean_rad = forward_rotation_rad(
        lambda_nm=lambda_nm,
        magnetic_field_t=magnetic_field_t,
        path_length_m=path_length_m,
        k_param=cfg.true_k,
        offset_rad=cfg.true_offset_rad,
    )
    noise = rng.normal(loc=0.0, scale=cfg.noise_std_rad, size=cfg.n_samples)
    theta_obs_rad = theta_clean_rad + noise

    return pd.DataFrame(
        {
            "lambda_nm": lambda_nm,
            "B_tesla": magnetic_field_t,
            "L_m": path_length_m,
            "theta_obs_rad": theta_obs_rad,
            "theta_clean_rad": theta_clean_rad,
            "theta_obs_deg": np.degrees(theta_obs_rad),
        }
    )


def build_feature(lambda_nm: np.ndarray, magnetic_field_t: np.ndarray, path_length_m: np.ndarray) -> np.ndarray:
    """Build linear feature x = B*L/lambda^2 for the K parameter."""
    return magnetic_field_t * path_length_m / (lambda_nm**2)


def solve_linear_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Solve y ≈ K*x + b and return (K_hat, b_hat, y_hat, sse)."""
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k_hat = float(beta[0])
    b_hat = float(beta[1])
    y_hat = X @ beta
    residual = y - y_hat
    sse = float(np.sum(residual**2))
    return k_hat, b_hat, y_hat, sse


def fit_faraday_model(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    """Estimate (K, b) and report fit quality metrics."""
    lambda_nm = df["lambda_nm"].to_numpy(dtype=float)
    magnetic_field_t = df["B_tesla"].to_numpy(dtype=float)
    path_length_m = df["L_m"].to_numpy(dtype=float)
    theta_obs = df["theta_obs_rad"].to_numpy(dtype=float)

    x = build_feature(lambda_nm=lambda_nm, magnetic_field_t=magnetic_field_t, path_length_m=path_length_m)
    k_hat, b_hat, y_hat, sse = solve_linear_least_squares(x=x, y=theta_obs)

    rmse = float(np.sqrt(np.mean((theta_obs - y_hat) ** 2)))
    sst = float(np.sum((theta_obs - np.mean(theta_obs)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-15 else 1.0

    return {
        "k_hat": k_hat,
        "offset_hat_rad": b_hat,
        "rmse": rmse,
        "r2": r2,
        "sse": sse,
        "y_hat": y_hat,
    }


def infer_magnetic_field(
    theta_obs_rad: float,
    lambda_nm: float,
    path_length_m: float,
    k_hat: float,
    offset_hat_rad: float,
) -> float:
    """Invert magnetic field from observed theta using fitted parameters."""
    v_lambda = float(verdet_constant(k_hat, lambda_nm))
    return (theta_obs_rad - offset_hat_rad) / (v_lambda * path_length_m)


def main() -> None:
    cfg = FaradayConfig()
    df = generate_synthetic_dataset(cfg)
    fit = fit_faraday_model(df)

    k_hat = float(fit["k_hat"])
    offset_hat = float(fit["offset_hat_rad"])
    rmse = float(fit["rmse"])
    r2 = float(fit["r2"])

    ref_lambda_nm = 532.0
    v_ref_true = float(verdet_constant(cfg.true_k, ref_lambda_nm))
    v_ref_hat = float(verdet_constant(k_hat, ref_lambda_nm))

    rng_unknown = np.random.default_rng(cfg.seed + 1000)
    theta_unknown_clean = float(
        forward_rotation_rad(
            lambda_nm=np.array([cfg.unknown_lambda_nm], dtype=float),
            magnetic_field_t=np.array([cfg.unknown_b_true_t], dtype=float),
            path_length_m=np.array([cfg.unknown_path_m], dtype=float),
            k_param=cfg.true_k,
            offset_rad=cfg.true_offset_rad,
        )[0]
    )
    theta_unknown_obs = theta_unknown_clean + float(rng_unknown.normal(0.0, cfg.unknown_noise_std_rad))
    b_unknown_hat = infer_magnetic_field(
        theta_obs_rad=theta_unknown_obs,
        lambda_nm=cfg.unknown_lambda_nm,
        path_length_m=cfg.unknown_path_m,
        k_hat=k_hat,
        offset_hat_rad=offset_hat,
    )

    print("=== Faraday Effect MVP ===")
    print("Synthetic dataset preview:")
    print(df.head(8).to_string(index=False, justify="center", float_format=lambda v: f"{v:9.5f}"))
    print()

    print("Fitted parameters:")
    print(f"  K_hat             = {k_hat:,.2f} (true {cfg.true_k:,.2f})")
    print(f"  offset_hat_rad    = {offset_hat:.6f} (true {cfg.true_offset_rad:.6f})")
    print(f"  RMSE              = {rmse:.6e} rad ({np.degrees(rmse):.4f} deg)")
    print(f"  R^2               = {r2:.6f}")
    print()

    print("Verdet constant at 532 nm:")
    print(f"  V_hat             = {v_ref_hat:.6e} rad/(T*m)")
    print(f"  V_true            = {v_ref_true:.6e} rad/(T*m)")
    print()

    print("Unknown magnetic-field inversion:")
    print(f"  theta_obs_rad     = {theta_unknown_obs:.6e}")
    print(f"  B_hat             = {b_unknown_hat:.5f} T")
    print(f"  B_true            = {cfg.unknown_b_true_t:.5f} T")
    print(f"  abs_error         = {abs(b_unknown_hat - cfg.unknown_b_true_t):.5f} T")

    # Quality gates for automated validation.
    assert abs(k_hat - cfg.true_k) / abs(cfg.true_k) < 0.12
    assert r2 > 0.90
    assert abs(b_unknown_hat - cfg.unknown_b_true_t) < 0.08


if __name__ == "__main__":
    main()
