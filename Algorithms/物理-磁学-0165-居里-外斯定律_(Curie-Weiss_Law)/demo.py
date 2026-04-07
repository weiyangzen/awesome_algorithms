"""Minimal runnable MVP for Curie-Weiss Law.

This script demonstrates how to recover Curie constant C and Weiss temperature
Theta from susceptibility-vs-temperature data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class CurieWeissConfig:
    """Configuration for synthetic-data generation and fitting."""

    curie_constant_true: float = 1.8
    weiss_temperature_true: float = 120.0
    temperature_min: float = 150.0
    temperature_max: float = 420.0
    n_points: int = 80
    relative_noise_std: float = 0.02
    random_seed: int = 7


def curie_weiss_model(temperature: np.ndarray, curie_constant: float, weiss_temperature: float) -> np.ndarray:
    """Curie-Weiss law: chi(T) = C / (T - Theta)."""

    return curie_constant / (temperature - weiss_temperature)


def generate_synthetic_data(cfg: CurieWeissConfig) -> pd.DataFrame:
    """Generate deterministic noisy susceptibility data."""

    temperatures = np.linspace(cfg.temperature_min, cfg.temperature_max, cfg.n_points, dtype=np.float64)
    if np.min(temperatures) <= cfg.weiss_temperature_true:
        raise ValueError("temperature_min must be strictly larger than weiss_temperature_true.")

    chi_clean = curie_weiss_model(
        temperatures,
        curie_constant=cfg.curie_constant_true,
        weiss_temperature=cfg.weiss_temperature_true,
    )

    rng = np.random.default_rng(cfg.random_seed)
    multiplicative_noise = rng.normal(loc=0.0, scale=cfg.relative_noise_std, size=temperatures.size)
    chi_noisy = chi_clean * (1.0 + multiplicative_noise)
    chi_noisy = np.clip(chi_noisy, 1e-12, None)

    return pd.DataFrame(
        {
            "temperature": temperatures,
            "chi_clean": chi_clean,
            "chi_observed": chi_noisy,
        }
    )


def fit_linearized_sklearn(temperatures: np.ndarray, chi_observed: np.ndarray) -> dict[str, float]:
    """Fit 1/chi = (1/C) * T - Theta/C using scikit-learn linear regression."""

    inv_chi = 1.0 / chi_observed
    model = LinearRegression(fit_intercept=True)
    model.fit(temperatures.reshape(-1, 1), inv_chi)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    curie_constant_est = 1.0 / slope
    weiss_temperature_est = -intercept / slope

    inv_chi_pred = model.predict(temperatures.reshape(-1, 1))
    r2_inv = float(r2_score(inv_chi, inv_chi_pred))

    return {
        "slope": slope,
        "intercept": intercept,
        "curie_constant": curie_constant_est,
        "weiss_temperature": weiss_temperature_est,
        "r2_inverse_space": r2_inv,
    }


def fit_linearized_torch(temperatures: np.ndarray, chi_observed: np.ndarray) -> dict[str, float]:
    """Solve the same linearized system with torch.linalg.lstsq for cross-check."""

    t = torch.tensor(temperatures, dtype=torch.float64)
    y = 1.0 / torch.tensor(chi_observed, dtype=torch.float64)

    design = torch.stack([t, torch.ones_like(t)], dim=1)
    solution = torch.linalg.lstsq(design, y).solution

    slope = float(solution[0].item())
    intercept = float(solution[1].item())

    curie_constant_est = 1.0 / slope
    weiss_temperature_est = -intercept / slope

    return {
        "slope": slope,
        "intercept": intercept,
        "curie_constant": curie_constant_est,
        "weiss_temperature": weiss_temperature_est,
    }


def fit_nonlinear_scipy(
    temperatures: np.ndarray,
    chi_observed: np.ndarray,
    initial_curie_constant: float,
    initial_weiss_temperature: float,
) -> dict[str, float]:
    """Refine parameters with constrained non-linear least squares (SciPy)."""

    theta_upper = float(np.min(temperatures) - 1e-6)
    lower_bounds = [1e-12, -1e6]
    upper_bounds = [1e6, theta_upper]

    popt, _ = curve_fit(
        curie_weiss_model,
        temperatures,
        chi_observed,
        p0=[initial_curie_constant, initial_weiss_temperature],
        bounds=(lower_bounds, upper_bounds),
        maxfev=50_000,
    )

    curie_constant_est = float(popt[0])
    weiss_temperature_est = float(popt[1])

    return {
        "curie_constant": curie_constant_est,
        "weiss_temperature": weiss_temperature_est,
    }


def summarize_results(df: pd.DataFrame, cfg: CurieWeissConfig, linear_fit: dict[str, float], nonlinear_fit: dict[str, float]) -> None:
    """Print key metrics and perform non-interactive sanity checks."""

    chi_linear = curie_weiss_model(
        df["temperature"].to_numpy(),
        linear_fit["curie_constant"],
        linear_fit["weiss_temperature"],
    )
    chi_nonlinear = curie_weiss_model(
        df["temperature"].to_numpy(),
        nonlinear_fit["curie_constant"],
        nonlinear_fit["weiss_temperature"],
    )

    df_out = df.copy()
    df_out["chi_fit_linear"] = chi_linear
    df_out["chi_fit_nonlinear"] = chi_nonlinear

    mae_linear = float(mean_absolute_error(df_out["chi_observed"], df_out["chi_fit_linear"]))
    mae_nonlinear = float(mean_absolute_error(df_out["chi_observed"], df_out["chi_fit_nonlinear"]))
    rmse_nonlinear = float(
        np.sqrt(mean_squared_error(df_out["chi_observed"], df_out["chi_fit_nonlinear"]))
    )

    rel_err_c = abs(nonlinear_fit["curie_constant"] - cfg.curie_constant_true) / cfg.curie_constant_true
    abs_err_theta = abs(nonlinear_fit["weiss_temperature"] - cfg.weiss_temperature_true)

    magnetic_tendency = (
        "ferromagnetic-correlation-like (Theta > 0)"
        if nonlinear_fit["weiss_temperature"] > 0
        else "antiferromagnetic-correlation-like (Theta < 0)"
    )

    print("=== Curie-Weiss Law MVP ===")
    print(f"True parameters: C={cfg.curie_constant_true:.6f}, Theta={cfg.weiss_temperature_true:.6f}")
    print(
        "Linearized fit (sklearn): "
        f"C={linear_fit['curie_constant']:.6f}, Theta={linear_fit['weiss_temperature']:.6f}, "
        f"R2(1/chi~T)={linear_fit['r2_inverse_space']:.6f}"
    )
    print(
        "Nonlinear refinement (scipy): "
        f"C={nonlinear_fit['curie_constant']:.6f}, Theta={nonlinear_fit['weiss_temperature']:.6f}"
    )
    print(f"MAE linear={mae_linear:.6e}, MAE nonlinear={mae_nonlinear:.6e}, RMSE nonlinear={rmse_nonlinear:.6e}")
    print(f"Physical interpretation: {magnetic_tendency}")

    print("\nSample rows (head):")
    with pd.option_context("display.max_columns", 10, "display.width", 120):
        print(df_out.head(6).to_string(index=False))

    print("\nSample rows (tail):")
    with pd.option_context("display.max_columns", 10, "display.width", 120):
        print(df_out.tail(6).to_string(index=False))

    # Deterministic quality checks for validation.
    assert linear_fit["r2_inverse_space"] > 0.99, "Linearized inverse susceptibility fit is unexpectedly poor."
    assert rel_err_c < 0.07, "Recovered Curie constant is too far from ground truth."
    assert abs_err_theta < 9.0, "Recovered Weiss temperature is too far from ground truth."
    assert mae_nonlinear <= mae_linear, "Nonlinear refinement should not be worse than linearized fit here."



def main() -> None:
    cfg = CurieWeissConfig()
    df = generate_synthetic_data(cfg)

    temperatures = df["temperature"].to_numpy(dtype=np.float64)
    chi_observed = df["chi_observed"].to_numpy(dtype=np.float64)

    linear_fit = fit_linearized_sklearn(temperatures, chi_observed)
    torch_fit = fit_linearized_torch(temperatures, chi_observed)

    # Ensure independent linear solvers are numerically consistent.
    assert abs(linear_fit["slope"] - torch_fit["slope"]) < 1e-9
    assert abs(linear_fit["intercept"] - torch_fit["intercept"]) < 1e-9

    nonlinear_fit = fit_nonlinear_scipy(
        temperatures=temperatures,
        chi_observed=chi_observed,
        initial_curie_constant=linear_fit["curie_constant"],
        initial_weiss_temperature=linear_fit["weiss_temperature"],
    )

    summarize_results(df=df, cfg=cfg, linear_fit=linear_fit, nonlinear_fit=nonlinear_fit)


if __name__ == "__main__":
    main()
