"""Minimal runnable MVP for Microlensing (Paczynski single-lens light curve)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
except ImportError:  # pragma: no cover - torch is in workspace deps, fallback keeps script robust.
    torch = None


@dataclass(frozen=True)
class MicrolensingParams:
    """Point-source point-lens (PSPL) parameters."""

    t0: float  # Time of closest approach.
    u0: float  # Impact parameter in Einstein-radius units.
    tE: float  # Einstein crossing time.
    fs: float  # Source flux.
    fb: float  # Blend flux.


def validate_params(params: MicrolensingParams) -> None:
    values = np.array([params.t0, params.u0, params.tE, params.fs, params.fb], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Microlensing parameters contain non-finite values")
    if params.u0 <= 0.0:
        raise ValueError("u0 must be positive")
    if params.tE <= 0.0:
        raise ValueError("tE must be positive")
    if params.fs <= 0.0:
        raise ValueError("fs must be positive")
    if params.fb < 0.0:
        raise ValueError("fb must be non-negative")


def impact_parameter(time: np.ndarray, params: MicrolensingParams) -> np.ndarray:
    tau = (time - params.t0) / params.tE
    return np.sqrt(params.u0**2 + tau**2)


def magnification_pspl(u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Paczynski magnification A(u) for a point-source point-lens event."""
    u = np.asarray(u, dtype=float)
    u_safe = np.maximum(u, eps)
    return (u_safe**2 + 2.0) / (u_safe * np.sqrt(u_safe**2 + 4.0))


def flux_model(time: np.ndarray, params: MicrolensingParams) -> np.ndarray:
    validate_params(params)
    u = impact_parameter(time, params)
    return params.fs * magnification_pspl(u) + params.fb


def synthesize_dataset(
    true_params: MicrolensingParams,
    n_points: int = 240,
    span_factor: float = 4.0,
    noise_sigma: float = 0.02,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time-series data around one microlensing event."""
    if n_points < 32:
        raise ValueError("n_points must be >= 32")
    if span_factor <= 0.0:
        raise ValueError("span_factor must be positive")
    if noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be positive")

    validate_params(true_params)
    rng = np.random.default_rng(seed)
    t_min = true_params.t0 - span_factor * true_params.tE
    t_max = true_params.t0 + span_factor * true_params.tE
    time = np.linspace(t_min, t_max, n_points, dtype=float)

    flux_clean = flux_model(time, true_params)
    flux_obs = flux_clean + rng.normal(loc=0.0, scale=noise_sigma, size=n_points)
    flux_err = np.full(n_points, noise_sigma, dtype=float)
    return time, flux_obs, flux_err, flux_clean


def initial_guess(time: np.ndarray, flux_obs: np.ndarray) -> MicrolensingParams:
    if time.ndim != 1 or flux_obs.ndim != 1 or time.size != flux_obs.size:
        raise ValueError("time and flux_obs must be 1-D arrays with equal length")
    if time.size < 16:
        raise ValueError("Need at least 16 points to build initial guess")

    edge = max(8, time.size // 10)
    baseline = float(np.median(np.concatenate([flux_obs[:edge], flux_obs[-edge:]])))
    peak_idx = int(np.argmax(flux_obs))
    t0_guess = float(time[peak_idx])

    fs_guess = max(0.7 * baseline, 1e-3)
    fb_guess = max(baseline - fs_guess, 0.0)
    tE_guess = max((float(time.max()) - float(time.min())) / 8.0, 1e-2)
    return MicrolensingParams(t0=t0_guess, u0=0.2, tE=tE_guess, fs=fs_guess, fb=fb_guess)


def fit_microlensing_curve(
    time: np.ndarray,
    flux_obs: np.ndarray,
    flux_err: np.ndarray,
    guess: MicrolensingParams,
) -> tuple[MicrolensingParams, object]:
    validate_params(guess)
    if time.shape != flux_obs.shape or time.shape != flux_err.shape:
        raise ValueError("time, flux_obs and flux_err must share the same shape")
    if np.any(flux_err <= 0.0):
        raise ValueError("flux_err must be strictly positive")

    x0 = np.array([guess.t0, guess.u0, guess.tE, guess.fs, guess.fb], dtype=float)
    lower = np.array([time.min() - 10.0, 1e-3, 0.1, 1e-3, 0.0], dtype=float)
    upper = np.array([time.max() + 10.0, 3.0, 500.0, 50.0, 50.0], dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        params = MicrolensingParams(t0=x[0], u0=x[1], tE=x[2], fs=x[3], fb=x[4])
        model = flux_model(time, params)
        return (model - flux_obs) / flux_err

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=5000,
    )
    best = MicrolensingParams(
        t0=float(result.x[0]),
        u0=float(result.x[1]),
        tE=float(result.x[2]),
        fs=float(result.x[3]),
        fb=float(result.x[4]),
    )
    return best, result


def evaluate_fit(
    time: np.ndarray,
    flux_obs: np.ndarray,
    flux_err: np.ndarray,
    true_params: MicrolensingParams,
    fit_params: MicrolensingParams,
) -> dict[str, float | str]:
    flux_pred = flux_model(time, fit_params)
    rmse = float(np.sqrt(mean_squared_error(flux_obs, flux_pred)))
    mae = float(mean_absolute_error(flux_obs, flux_pred))
    chi2 = float(np.sum(((flux_obs - flux_pred) / flux_err) ** 2))
    dof = max(time.size - 5, 1)
    reduced_chi2 = chi2 / dof

    rel_t0 = abs(fit_params.t0 - true_params.t0) / max(abs(true_params.t0), 1e-9)
    rel_u0 = abs(fit_params.u0 - true_params.u0) / true_params.u0
    rel_tE = abs(fit_params.tE - true_params.tE) / true_params.tE
    rel_fs = abs(fit_params.fs - true_params.fs) / true_params.fs
    rel_fb = abs(fit_params.fb - true_params.fb) / max(true_params.fb, 1e-9)
    max_rel_param_error = max(rel_t0, rel_u0, rel_tE, rel_fs, rel_fb)

    peak_amp_true = float(flux_model(np.array([true_params.t0]), true_params)[0])
    peak_amp_fit = float(flux_model(np.array([fit_params.t0]), fit_params)[0])

    return {
        "rmse_flux": rmse,
        "mae_flux": mae,
        "reduced_chi2": reduced_chi2,
        "rel_err_t0": rel_t0,
        "rel_err_u0": rel_u0,
        "rel_err_tE": rel_tE,
        "rel_err_fs": rel_fs,
        "rel_err_fb": rel_fb,
        "max_rel_param_error": max_rel_param_error,
        "peak_flux_true": peak_amp_true,
        "peak_flux_fit": peak_amp_fit,
    }


def torch_formula_consistency_check() -> float:
    """Cross-check the Paczynski formula with torch tensor computation."""
    if torch is None:
        return float("nan")

    u = np.linspace(0.05, 3.0, 256, dtype=float)
    mag_np = magnification_pspl(u)
    u_tensor = torch.tensor(u, dtype=torch.float64)
    mag_torch = ((u_tensor**2 + 2.0) / (u_tensor * torch.sqrt(u_tensor**2 + 4.0))).cpu().numpy()
    return float(np.max(np.abs(mag_np - mag_torch)))


def run_checks(fit_metrics: dict[str, float | str], optimize_result: object) -> None:
    if not bool(getattr(optimize_result, "success", False)):
        raise RuntimeError("Least-squares fitting did not converge")
    if float(fit_metrics["rmse_flux"]) >= 0.08:
        raise RuntimeError("RMSE is too high for the synthetic data quality")
    if float(fit_metrics["reduced_chi2"]) >= 3.0:
        raise RuntimeError("Reduced chi-square indicates poor fit")
    if float(fit_metrics["max_rel_param_error"]) >= 0.15:
        raise RuntimeError("Recovered parameters deviate too much from truth")


def main() -> None:
    true_params = MicrolensingParams(t0=60.0, u0=0.23, tE=24.0, fs=1.2, fb=0.35)
    time, flux_obs, flux_err, flux_clean = synthesize_dataset(
        true_params=true_params,
        n_points=240,
        span_factor=4.0,
        noise_sigma=0.02,
        seed=7,
    )

    guess = initial_guess(time, flux_obs)
    fit_params, optimize_result = fit_microlensing_curve(time, flux_obs, flux_err, guess)
    fit_metrics = evaluate_fit(time, flux_obs, flux_err, true_params, fit_params)
    run_checks(fit_metrics, optimize_result)

    torch_diff = torch_formula_consistency_check()
    torch_diff_text = "NA(torch unavailable)" if np.isnan(torch_diff) else f"{torch_diff:.3e}"

    summary_rows = [
        {
            "param": "t0",
            "true": true_params.t0,
            "fitted": fit_params.t0,
            "rel_error": fit_metrics["rel_err_t0"],
        },
        {
            "param": "u0",
            "true": true_params.u0,
            "fitted": fit_params.u0,
            "rel_error": fit_metrics["rel_err_u0"],
        },
        {
            "param": "tE",
            "true": true_params.tE,
            "fitted": fit_params.tE,
            "rel_error": fit_metrics["rel_err_tE"],
        },
        {
            "param": "fs",
            "true": true_params.fs,
            "fitted": fit_params.fs,
            "rel_error": fit_metrics["rel_err_fs"],
        },
        {
            "param": "fb",
            "true": true_params.fb,
            "fitted": fit_params.fb,
            "rel_error": fit_metrics["rel_err_fb"],
        },
    ]

    df_params = pd.DataFrame(summary_rows)
    df_metrics = pd.DataFrame(
        [
            {
                "rmse_flux": fit_metrics["rmse_flux"],
                "mae_flux": fit_metrics["mae_flux"],
                "reduced_chi2": fit_metrics["reduced_chi2"],
                "max_rel_param_error": fit_metrics["max_rel_param_error"],
                "peak_flux_true": fit_metrics["peak_flux_true"],
                "peak_flux_fit": fit_metrics["peak_flux_fit"],
                "torch_formula_max_abs_diff": torch_diff_text,
                "n_points": time.size,
                "noise_sigma": float(flux_err[0]),
                "optimizer_nfev": int(getattr(optimize_result, "nfev", -1)),
            }
        ]
    )

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)

    print("Microlensing MVP report (PSPL / Paczynski curve)")
    print("\nRecovered parameters:")
    print(df_params.to_string(index=False, justify="center", float_format=lambda x: f"{x:0.6f}"))
    print("\nFit diagnostics:")
    print(df_metrics.to_string(index=False))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
