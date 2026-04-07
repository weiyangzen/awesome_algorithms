"""Minimal runnable MVP for Curie's Law.

This script demonstrates how to recover the Curie constant C from
susceptibility-vs-temperature data based on chi(T) = C / T.
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
class CurieLawConfig:
    """Configuration for synthetic-data generation and fitting."""

    curie_constant_true: float = 1.75
    temperature_min: float = 80.0
    temperature_max: float = 420.0
    n_points: int = 90
    relative_noise_std: float = 0.02
    random_seed: int = 11


def curie_law_model(temperature: np.ndarray, curie_constant: float) -> np.ndarray:
    """Curie's Law: chi(T) = C / T."""

    return curie_constant / temperature


def generate_synthetic_data(cfg: CurieLawConfig) -> pd.DataFrame:
    """Generate deterministic noisy susceptibility data."""

    temperatures = np.linspace(cfg.temperature_min, cfg.temperature_max, cfg.n_points, dtype=np.float64)
    if np.min(temperatures) <= 0.0:
        raise ValueError("temperature_min must be strictly positive in Kelvin.")

    chi_clean = curie_law_model(temperatures, cfg.curie_constant_true)

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
    """Fit 1/chi = (1/C)*T with zero intercept using scikit-learn."""

    inv_chi = 1.0 / chi_observed
    model = LinearRegression(fit_intercept=False)
    model.fit(temperatures.reshape(-1, 1), inv_chi)

    slope = float(model.coef_[0])
    if slope <= 0.0:
        raise ValueError("Estimated slope must be positive for Curie's law.")

    curie_constant_est = 1.0 / slope
    inv_chi_pred = model.predict(temperatures.reshape(-1, 1))
    r2_inverse_space = float(r2_score(inv_chi, inv_chi_pred))

    return {
        "slope": slope,
        "curie_constant": curie_constant_est,
        "r2_inverse_space": r2_inverse_space,
    }


def fit_linearized_torch(temperatures: np.ndarray, chi_observed: np.ndarray) -> dict[str, float]:
    """Solve the same linearized least squares system with torch."""

    t = torch.tensor(temperatures, dtype=torch.float64).unsqueeze(1)
    y = (1.0 / torch.tensor(chi_observed, dtype=torch.float64)).unsqueeze(1)

    solution = torch.linalg.lstsq(t, y).solution
    slope = float(solution[0, 0].item())
    if slope <= 0.0:
        raise ValueError("Torch-estimated slope must be positive for Curie's law.")

    curie_constant_est = 1.0 / slope
    return {
        "slope": slope,
        "curie_constant": curie_constant_est,
    }


def fit_nonlinear_scipy(
    temperatures: np.ndarray,
    chi_observed: np.ndarray,
    initial_curie_constant: float,
) -> dict[str, float]:
    """Refine C with bounded non-linear least squares in original chi-space."""

    popt, _ = curve_fit(
        curie_law_model,
        temperatures,
        chi_observed,
        p0=[initial_curie_constant],
        bounds=([1e-12], [1e6]),
        maxfev=20_000,
    )

    return {"curie_constant": float(popt[0])}


def summarize_results(
    df: pd.DataFrame,
    cfg: CurieLawConfig,
    linear_fit: dict[str, float],
    nonlinear_fit: dict[str, float],
) -> None:
    """Print metrics, example rows, and run deterministic quality checks."""

    chi_fit_linear = curie_law_model(df["temperature"].to_numpy(), linear_fit["curie_constant"])
    chi_fit_nonlinear = curie_law_model(df["temperature"].to_numpy(), nonlinear_fit["curie_constant"])

    df_out = df.copy()
    df_out["chi_fit_linear"] = chi_fit_linear
    df_out["chi_fit_nonlinear"] = chi_fit_nonlinear

    mae_linear = float(mean_absolute_error(df_out["chi_observed"], df_out["chi_fit_linear"]))
    mae_nonlinear = float(mean_absolute_error(df_out["chi_observed"], df_out["chi_fit_nonlinear"]))
    rmse_nonlinear = float(np.sqrt(mean_squared_error(df_out["chi_observed"], df_out["chi_fit_nonlinear"])))

    rel_err_linear = abs(linear_fit["curie_constant"] - cfg.curie_constant_true) / cfg.curie_constant_true
    rel_err_nonlinear = abs(nonlinear_fit["curie_constant"] - cfg.curie_constant_true) / cfg.curie_constant_true

    print("=== Curie's Law MVP ===")
    print(f"True Curie constant: C_true={cfg.curie_constant_true:.6f}")
    print(
        "Linearized fit (sklearn): "
        f"C={linear_fit['curie_constant']:.6f}, R2(1/chi~T)={linear_fit['r2_inverse_space']:.6f}"
    )
    print(f"Linearized fit (torch):   C={1.0 / linear_fit['slope']:.6f}")
    print(f"Nonlinear refinement:     C={nonlinear_fit['curie_constant']:.6f}")
    print(
        f"MAE linear={mae_linear:.6e}, MAE nonlinear={mae_nonlinear:.6e}, "
        f"RMSE nonlinear={rmse_nonlinear:.6e}"
    )
    print(f"Relative error linear={rel_err_linear:.4%}, nonlinear={rel_err_nonlinear:.4%}")

    print("\nSample rows (head):")
    with pd.option_context("display.max_columns", 10, "display.width", 120):
        print(df_out.head(6).to_string(index=False))

    print("\nSample rows (tail):")
    with pd.option_context("display.max_columns", 10, "display.width", 120):
        print(df_out.tail(6).to_string(index=False))

    assert linear_fit["r2_inverse_space"] > 0.99, "Linearized inverse fit is unexpectedly poor."
    assert rel_err_nonlinear < 0.05, "Recovered Curie constant is too far from ground truth."
    assert abs(mae_nonlinear - mae_linear) < 1e-3, "Linear and nonlinear fits diverged unexpectedly."



def main() -> None:
    cfg = CurieLawConfig()
    df = generate_synthetic_data(cfg)

    temperatures = df["temperature"].to_numpy(dtype=np.float64)
    chi_observed = df["chi_observed"].to_numpy(dtype=np.float64)

    linear_fit = fit_linearized_sklearn(temperatures, chi_observed)
    torch_fit = fit_linearized_torch(temperatures, chi_observed)

    assert abs(linear_fit["slope"] - torch_fit["slope"]) < 1e-9

    nonlinear_fit = fit_nonlinear_scipy(
        temperatures=temperatures,
        chi_observed=chi_observed,
        initial_curie_constant=linear_fit["curie_constant"],
    )

    summarize_results(df=df, cfg=cfg, linear_fit=linear_fit, nonlinear_fit=nonlinear_fit)


if __name__ == "__main__":
    main()
