"""Minimal runnable MVP for dielectric polarization (linear isotropic model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON_0 = 8.854_187_8128e-12  # Vacuum permittivity (F/m).


@dataclass
class PolarizationResult:
    """Container for estimation outputs and diagnostics."""

    E: np.ndarray
    P_true: np.ndarray
    P_measured: np.ndarray
    P_fitted: np.ndarray
    D_measured: np.ndarray
    D_fitted: np.ndarray
    chi_true: float
    chi_est: float
    rmse: float
    r2: float


def generate_synthetic_dataset(
    chi_true: float,
    n_points: int,
    e_min: float,
    e_max: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic static-field data for a linear dielectric."""
    if chi_true <= 0.0:
        raise ValueError("chi_true must be positive.")
    if n_points < 5:
        raise ValueError("n_points must be >= 5.")
    if not np.isfinite(e_min) or not np.isfinite(e_max) or e_min >= e_max:
        raise ValueError("Require finite bounds with e_min < e_max.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")

    rng = np.random.default_rng(seed)
    E = np.linspace(e_min, e_max, n_points, dtype=float)
    P_true = EPSILON_0 * chi_true * E
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
    P_measured = P_true + noise
    return E, P_true, P_measured


def estimate_chi_linear(E: np.ndarray, P_measured: np.ndarray) -> tuple[float, np.ndarray]:
    """Estimate susceptibility via least squares constrained through origin."""
    if E.ndim != 1 or P_measured.ndim != 1 or E.shape != P_measured.shape:
        raise ValueError("E and P_measured must be 1D arrays with same shape.")
    if E.size < 2:
        raise ValueError("At least 2 points are required.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(P_measured)):
        raise ValueError("E and P_measured must be finite.")

    denom = float(np.dot(E, E))
    if np.isclose(denom, 0.0):
        raise ValueError("Degenerate E values: sum(E^2) is zero.")

    numer = float(np.dot(E, P_measured))
    chi_est = numer / (EPSILON_0 * denom)
    P_fitted = EPSILON_0 * chi_est * E
    return chi_est, P_fitted


def evaluate_fit(P_measured: np.ndarray, P_fitted: np.ndarray) -> tuple[float, float]:
    """Compute RMSE and R^2 for fitted polarization."""
    residual = P_measured - P_fitted
    rmse = float(np.sqrt(np.mean(residual**2)))

    sse = float(np.sum(residual**2))
    centered = P_measured - float(np.mean(P_measured))
    sst = float(np.sum(centered**2))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else 1.0
    return rmse, r2


def run_dielectric_polarization_mvp(
    chi_true: float = 2.7,
    n_points: int = 61,
    e_min: float = -5.0e5,
    e_max: float = 5.0e5,
    noise_std: float = 8.0e-8,
    seed: int = 157,
) -> PolarizationResult:
    """Simulate data, estimate susceptibility, and return diagnostics."""
    E, P_true, P_measured = generate_synthetic_dataset(
        chi_true=chi_true,
        n_points=n_points,
        e_min=e_min,
        e_max=e_max,
        noise_std=noise_std,
        seed=seed,
    )
    chi_est, P_fitted = estimate_chi_linear(E=E, P_measured=P_measured)
    rmse, r2 = evaluate_fit(P_measured=P_measured, P_fitted=P_fitted)

    D_measured = EPSILON_0 * E + P_measured
    D_fitted = EPSILON_0 * E + P_fitted

    return PolarizationResult(
        E=E,
        P_true=P_true,
        P_measured=P_measured,
        P_fitted=P_fitted,
        D_measured=D_measured,
        D_fitted=D_fitted,
        chi_true=float(chi_true),
        chi_est=float(chi_est),
        rmse=rmse,
        r2=r2,
    )


def run_checks(result: PolarizationResult) -> None:
    """Fail fast if recovery quality is unexpectedly poor."""
    rel_err = abs(result.chi_est - result.chi_true) / result.chi_true
    max_abs_d_residual = float(np.max(np.abs(result.D_measured - result.D_fitted)))

    if not np.all(np.isfinite(result.P_measured)):
        raise AssertionError("P_measured contains non-finite values.")
    if result.r2 < 0.985:
        raise AssertionError(f"R^2 too low: {result.r2:.6f}")
    if rel_err > 0.03:
        raise AssertionError(f"chi relative error too large: {rel_err:.3%}")
    if max_abs_d_residual > 6.0e-7:
        raise AssertionError(f"D residual unexpectedly large: {max_abs_d_residual:.3e}")


def preview_table(result: PolarizationResult, n_head: int = 5) -> pd.DataFrame:
    """Create a compact table preview for terminal output."""
    df = pd.DataFrame(
        {
            "E_V_per_m": result.E,
            "P_true_C_per_m2": result.P_true,
            "P_measured_C_per_m2": result.P_measured,
            "P_fitted_C_per_m2": result.P_fitted,
            "D_measured_C_per_m2": result.D_measured,
            "D_fitted_C_per_m2": result.D_fitted,
        }
    )
    if 2 * n_head >= len(df):
        return df

    top = df.head(n_head)
    bottom = df.tail(n_head)
    ellipsis_row = pd.DataFrame([{col: np.nan for col in df.columns}])
    return pd.concat([top, ellipsis_row, bottom], ignore_index=True)


def main() -> None:
    result = run_dielectric_polarization_mvp()
    run_checks(result)

    rel_err = abs(result.chi_est - result.chi_true) / result.chi_true
    max_abs_p_residual = float(np.max(np.abs(result.P_measured - result.P_fitted)))
    max_abs_d_residual = float(np.max(np.abs(result.D_measured - result.D_fitted)))

    print("Dielectric Polarization MVP report")
    print(f"epsilon_0 (F/m)                 : {EPSILON_0:.10e}")
    print(f"chi_true                        : {result.chi_true:.6f}")
    print(f"chi_est                         : {result.chi_est:.6f}")
    print(f"chi_relative_error              : {rel_err:.3%}")
    print(f"RMSE(P) (C/m^2)                 : {result.rmse:.3e}")
    print(f"R^2(P fit)                      : {result.r2:.6f}")
    print(f"max|P_measured - P_fitted|      : {max_abs_p_residual:.3e}")
    print(f"max|D_measured - D_fitted|      : {max_abs_d_residual:.3e}")

    print("\nPreview (top/bottom rows):")
    preview = preview_table(result=result, n_head=4)
    print(preview.to_string(index=False, float_format=lambda x: f"{x: .3e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
