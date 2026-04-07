"""Minimal runnable MVP for Hubble's Law (PHYS-0051).

This script creates a synthetic low-redshift galaxy sample and estimates the
Hubble constant from the linear relation v = H0 * d.

Design goals:
- Keep the pipeline small and auditable.
- Avoid black-box fitting by implementing weighted least squares explicitly.
- Provide deterministic threshold checks for quick validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-12
C_KM_S = 299_792.458


@dataclass
class HubbleConfig:
    n_samples: int = 90
    true_h0_kms_mpc: float = 70.0
    d_min_mpc: float = 20.0
    d_max_mpc: float = 520.0
    sigma_peculiar_kms: float = 220.0
    sigma_meas_frac: float = 0.02
    confidence_level: float = 0.95
    max_valid_z: float = 0.2
    intercept_guard_kms: float = 500.0
    seed: int = 51


@dataclass
class FitThroughOriginResult:
    h0_kms_mpc: float
    h0_se: float
    ci_low: float
    ci_high: float
    chi2: float
    reduced_chi2: float
    r2: float


@dataclass
class FitWithInterceptResult:
    h0_kms_mpc: float
    intercept_kms: float
    h0_se: float
    intercept_se: float
    chi2: float
    reduced_chi2: float


def make_synthetic_catalog(cfg: HubbleConfig) -> pd.DataFrame:
    """Generate a synthetic low-z catalog with heteroscedastic velocity noise."""
    rng = np.random.default_rng(cfg.seed)

    d_mpc = np.sort(rng.uniform(cfg.d_min_mpc, cfg.d_max_mpc, cfg.n_samples))
    v_true = cfg.true_h0_kms_mpc * d_mpc

    sigma_meas = cfg.sigma_meas_frac * v_true
    sigma_tot = np.sqrt(cfg.sigma_peculiar_kms**2 + sigma_meas**2)

    v_obs = v_true + rng.normal(0.0, sigma_tot)
    z_obs = v_obs / C_KM_S

    return pd.DataFrame(
        {
            "distance_mpc": d_mpc,
            "velocity_true_kms": v_true,
            "sigma_velocity_kms": sigma_tot,
            "velocity_obs_kms": v_obs,
            "redshift_obs": z_obs,
        }
    )


def fit_hubble_through_origin(
    distance_mpc: np.ndarray,
    velocity_kms: np.ndarray,
    sigma_velocity_kms: np.ndarray,
    confidence_level: float,
) -> FitThroughOriginResult:
    """Weighted least squares for v = H0 * d with explicit formulas."""
    x = np.asarray(distance_mpc, dtype=float)
    y = np.asarray(velocity_kms, dtype=float)
    sigma = np.clip(np.asarray(sigma_velocity_kms, dtype=float), EPS, None)

    if x.size < 3:
        raise ValueError("Need at least 3 samples for stable fitting")

    w = 1.0 / (sigma**2)
    denom = np.sum(w * x * x)
    if denom <= EPS:
        raise ValueError("Degenerate design: sum(w*x^2) is too small")

    h0 = float(np.sum(w * x * y) / denom)
    residual = y - h0 * x

    chi2 = float(np.sum((residual / sigma) ** 2))
    dof = x.size - 1
    reduced = chi2 / max(dof, 1)

    var_h0 = reduced / denom
    h0_se = float(np.sqrt(max(var_h0, 0.0)))

    alpha = 1.0 - confidence_level
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    ci_low = h0 - tcrit * h0_se
    ci_high = h0 + tcrit * h0_se

    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / max(sst, EPS)

    return FitThroughOriginResult(
        h0_kms_mpc=h0,
        h0_se=h0_se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        chi2=chi2,
        reduced_chi2=float(reduced),
        r2=float(r2),
    )


def fit_hubble_with_intercept(
    distance_mpc: np.ndarray,
    velocity_kms: np.ndarray,
    sigma_velocity_kms: np.ndarray,
) -> FitWithInterceptResult:
    """Weighted least squares for v = H0 * d + b (diagnostic model)."""
    x = np.asarray(distance_mpc, dtype=float)
    y = np.asarray(velocity_kms, dtype=float)
    sigma = np.clip(np.asarray(sigma_velocity_kms, dtype=float), EPS, None)

    if x.size < 4:
        raise ValueError("Need at least 4 samples for intercept model")

    w = 1.0 / (sigma**2)
    X = np.column_stack([x, np.ones_like(x)])

    # Solve weighted least squares via normal equations:
    # beta = (X^T W X)^(-1) X^T W y
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)
    beta = np.linalg.solve(XtWX, XtWy)

    h0 = float(beta[0])
    intercept = float(beta[1])

    residual = y - (h0 * x + intercept)
    chi2 = float(np.sum((residual / sigma) ** 2))
    dof = x.size - 2
    reduced = chi2 / max(dof, 1)

    cov = reduced * np.linalg.inv(XtWX)
    h0_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    intercept_se = float(np.sqrt(max(cov[1, 1], 0.0)))

    return FitWithInterceptResult(
        h0_kms_mpc=h0,
        intercept_kms=intercept,
        h0_se=h0_se,
        intercept_se=intercept_se,
        chi2=chi2,
        reduced_chi2=float(reduced),
    )


def format_sample_table(df: pd.DataFrame, n_rows: int = 8) -> str:
    cols = ["distance_mpc", "redshift_obs", "velocity_obs_kms", "sigma_velocity_kms"]
    view = df.loc[:, cols].head(n_rows).copy()
    return view.to_string(index=False)


def main() -> None:
    cfg = HubbleConfig()
    df = make_synthetic_catalog(cfg)

    fit_origin = fit_hubble_through_origin(
        distance_mpc=df["distance_mpc"].to_numpy(),
        velocity_kms=df["velocity_obs_kms"].to_numpy(),
        sigma_velocity_kms=df["sigma_velocity_kms"].to_numpy(),
        confidence_level=cfg.confidence_level,
    )

    fit_intercept = fit_hubble_with_intercept(
        distance_mpc=df["distance_mpc"].to_numpy(),
        velocity_kms=df["velocity_obs_kms"].to_numpy(),
        sigma_velocity_kms=df["sigma_velocity_kms"].to_numpy(),
    )

    z_abs_max = float(np.max(np.abs(df["redshift_obs"].to_numpy())))

    summary = pd.DataFrame(
        [
            {
                "model": "v = H0*d (through origin)",
                "H0_est_km_s_Mpc": fit_origin.h0_kms_mpc,
                "H0_se": fit_origin.h0_se,
                "CI_low": fit_origin.ci_low,
                "CI_high": fit_origin.ci_high,
                "chi2_reduced": fit_origin.reduced_chi2,
                "R2": fit_origin.r2,
                "intercept_km_s": np.nan,
            },
            {
                "model": "v = H0*d + b",
                "H0_est_km_s_Mpc": fit_intercept.h0_kms_mpc,
                "H0_se": fit_intercept.h0_se,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "chi2_reduced": fit_intercept.reduced_chi2,
                "R2": np.nan,
                "intercept_km_s": fit_intercept.intercept_kms,
            },
        ]
    )

    pd.set_option("display.float_format", lambda x: f"{x:.6g}")

    print("=== Hubble's Law MVP (PHYS-0051) ===")
    print(
        "Config: "
        f"N={cfg.n_samples}, true_H0={cfg.true_h0_kms_mpc:.2f} km/s/Mpc, "
        f"distance=[{cfg.d_min_mpc:.1f}, {cfg.d_max_mpc:.1f}] Mpc, "
        f"sigma_peculiar={cfg.sigma_peculiar_kms:.1f} km/s, "
        f"sigma_meas_frac={cfg.sigma_meas_frac:.3f}, seed={cfg.seed}"
    )
    print(f"Max |z| in sample: {z_abs_max:.5f}")

    print("\nSample observations (head):")
    print(format_sample_table(df))

    print("\nFit summary:")
    print(summary.to_string(index=False))

    checks = {
        "low-z regime respected (max |z| < 0.2)": z_abs_max < cfg.max_valid_z,
        "H0 estimate close to truth (|delta| < 3*SE)": abs(fit_origin.h0_kms_mpc - cfg.true_h0_kms_mpc)
        < 3.0 * fit_origin.h0_se,
        "confidence interval covers true H0": fit_origin.ci_low <= cfg.true_h0_kms_mpc <= fit_origin.ci_high,
        "diagnostic intercept magnitude < 500 km/s": abs(fit_intercept.intercept_kms) < cfg.intercept_guard_kms,
    }

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
