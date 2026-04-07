"""Minimal runnable MVP for Gibbs-Duhem equation in a binary mixture.

We use the isothermal-isobaric Gibbs-Duhem relation for activity coefficients:
    x1 * d(ln gamma1) + x2 * d(ln gamma2) = 0,  where x2 = 1 - x1.

MVP idea:
1) Generate synthetic ln(gamma1) data from a Margules model + measurement noise.
2) Smooth ln(gamma1) and compute d(ln gamma1)/dx1.
3) Reconstruct d(ln gamma2)/dx1 from Gibbs-Duhem.
4) Integrate to recover ln(gamma2) profile.
5) Compare against known ground truth and verify residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline


@dataclass(frozen=True)
class BinarySystemConfig:
    """Configuration for the binary-mixture Gibbs-Duhem demo."""

    margules_a: float = 2.0
    noise_std: float = 0.015
    n_points: int = 240
    x_min: float = 0.01
    x_max: float = 0.99
    seed: int = 2026



def validate_composition_grid(x1: np.ndarray) -> None:
    if x1.ndim != 1:
        raise ValueError(f"x1 must be 1D, got shape={x1.shape}")
    if len(x1) < 5:
        raise ValueError("x1 must contain at least 5 points")
    if not np.all(np.isfinite(x1)):
        raise ValueError("x1 must be finite")
    if np.any(x1 <= 0.0) or np.any(x1 >= 1.0):
        raise ValueError("x1 must lie strictly inside (0, 1)")
    if not np.all(np.diff(x1) > 0.0):
        raise ValueError("x1 must be strictly increasing")



def margules_ln_gamma(x1: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """One-parameter Margules model.

    ln(gamma1) = A * x2^2
    ln(gamma2) = A * x1^2
    """
    x2 = 1.0 - x1
    ln_gamma1 = a * x2 * x2
    ln_gamma2 = a * x1 * x1
    return ln_gamma1, ln_gamma2



def reconstruct_ln_gamma2_from_gibbs_duhem(
    x1: np.ndarray,
    dln_gamma1_dx1: np.ndarray,
    ln_gamma2_at_xmin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ln(gamma2) from ln(gamma1) derivative via Gibbs-Duhem.

    x1*dln(gamma1) + x2*dln(gamma2) = 0
    => dln(gamma2)/dx1 = -(x1/x2) * dln(gamma1)/dx1
    """
    validate_composition_grid(x1)
    if dln_gamma1_dx1.shape != x1.shape:
        raise ValueError("dln_gamma1_dx1 must have the same shape as x1")

    x2 = 1.0 - x1
    dln_gamma2_dx1 = -(x1 / x2) * dln_gamma1_dx1
    integrated = cumulative_trapezoid(dln_gamma2_dx1, x1, initial=0.0)
    ln_gamma2 = ln_gamma2_at_xmin + integrated
    return ln_gamma2, dln_gamma2_dx1



def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))



def main() -> None:
    cfg = BinarySystemConfig()

    x1 = np.linspace(cfg.x_min, cfg.x_max, cfg.n_points, dtype=float)
    validate_composition_grid(x1)

    ln_gamma1_true, ln_gamma2_true = margules_ln_gamma(x1, a=cfg.margules_a)

    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=cfg.n_points)
    ln_gamma1_obs = ln_gamma1_true + noise

    # Practical smoothing scale: s = lambda * N * sigma^2 (lambda=2 here).
    smooth_s = 2.0 * cfg.n_points * (cfg.noise_std ** 2)
    spline = UnivariateSpline(x1, ln_gamma1_obs, s=smooth_s, k=3)
    ln_gamma1_smooth = spline(x1)
    dln_gamma1_dx1 = spline.derivative(1)(x1)

    ln_gamma2_recon, dln_gamma2_dx1_from_gd = reconstruct_ln_gamma2_from_gibbs_duhem(
        x1=x1,
        dln_gamma1_dx1=dln_gamma1_dx1,
        ln_gamma2_at_xmin=0.0,
    )

    # Numerical derivative of reconstructed profile for an independent residual check.
    dln_gamma2_dx1_num = np.gradient(ln_gamma2_recon, x1)

    x2 = 1.0 - x1
    gd_residual_num = x1 * dln_gamma1_dx1 + x2 * dln_gamma2_dx1_num
    gd_residual_formula = x1 * dln_gamma1_dx1 + x2 * dln_gamma2_dx1_from_gd

    rmse_ln_gamma2 = rms(ln_gamma2_recon - ln_gamma2_true)
    mae_ln_gamma1_fit = float(np.mean(np.abs(ln_gamma1_smooth - ln_gamma1_true)))
    max_abs_residual_num = float(np.max(np.abs(gd_residual_num)))
    max_abs_residual_formula = float(np.max(np.abs(gd_residual_formula)))

    summary = pd.DataFrame(
        [
            {
                "metric": "rmse_ln_gamma2_reconstruction",
                "value": rmse_ln_gamma2,
            },
            {
                "metric": "mae_ln_gamma1_smoothing",
                "value": mae_ln_gamma1_fit,
            },
            {
                "metric": "max_abs_gd_residual_formula",
                "value": max_abs_residual_formula,
            },
            {
                "metric": "max_abs_gd_residual_numeric",
                "value": max_abs_residual_num,
            },
        ]
    )

    profile = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "ln_gamma1_true": ln_gamma1_true,
            "ln_gamma1_obs": ln_gamma1_obs,
            "ln_gamma1_smooth": ln_gamma1_smooth,
            "ln_gamma2_true": ln_gamma2_true,
            "ln_gamma2_reconstructed": ln_gamma2_recon,
            "gd_residual_numeric": gd_residual_num,
        }
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("Gibbs-Duhem Equation MVP (binary mixture, constant T/P)")
    print(f"config: A={cfg.margules_a}, noise_std={cfg.noise_std}, points={cfg.n_points}, seed={cfg.seed}")
    print()
    print("=== Summary Metrics ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.10f}"))
    print()
    print("=== Profile Head (first 8 rows) ===")
    print(profile.head(8).to_string(index=False, float_format=lambda v: f"{v:.10f}"))
    print()
    print("=== Profile Tail (last 8 rows) ===")
    print(profile.tail(8).to_string(index=False, float_format=lambda v: f"{v:.10f}"))

    # Deterministic acceptance checks for this MVP.
    assert rmse_ln_gamma2 < 0.035, f"ln_gamma2 reconstruction RMSE too large: {rmse_ln_gamma2}"
    assert mae_ln_gamma1_fit < 0.020, f"ln_gamma1 smoothing MAE too large: {mae_ln_gamma1_fit}"
    assert max_abs_residual_formula < 1e-10, f"formula residual too large: {max_abs_residual_formula}"
    assert max_abs_residual_num < 0.06, f"numeric residual too large: {max_abs_residual_num}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
