"""Minimal runnable MVP for Scalar Spectral Index estimation.

This script estimates primordial power-spectrum parameters (A_s, n_s)
from synthetic data under the power-law model:
P_s(k) = A_s * (k / k0)^(n_s - 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    """Configuration for synthetic data and fitting."""

    k_min: float = 1.0e-3
    k_max: float = 3.0e-1
    n_points: int = 80
    k0: float = 0.05
    a_s_true: float = 2.1e-9
    n_s_true: float = 0.965
    base_sigma_ln: float = 0.04
    seed: int = 42
    torch_steps: int = 1200
    torch_lr: float = 0.05


def scalar_power_spectrum(
    k_mpc_inv: FloatArray,
    a_s: float,
    n_s: float,
    k0_mpc_inv: float,
) -> FloatArray:
    """Primordial scalar power spectrum P_s(k) under a power-law model."""
    return a_s * np.power(k_mpc_inv / k0_mpc_inv, n_s - 1.0)


def generate_synthetic_data(cfg: Config) -> dict[str, FloatArray]:
    """Generate heteroscedastic synthetic observations in log-space."""
    rng = np.random.default_rng(cfg.seed)

    k = np.logspace(
        np.log10(cfg.k_min),
        np.log10(cfg.k_max),
        cfg.n_points,
        dtype=np.float64,
    )
    p_true = scalar_power_spectrum(k, cfg.a_s_true, cfg.n_s_true, cfg.k0)

    x = np.log(k / cfg.k0)
    sigma_ln = cfg.base_sigma_ln * (1.0 + 0.3 * np.abs(x))

    ln_p_obs = np.log(np.clip(p_true, 1e-300, None)) + rng.normal(
        loc=0.0,
        scale=sigma_ln,
        size=cfg.n_points,
    )
    p_obs = np.exp(ln_p_obs)
    p_err = np.clip(p_obs * sigma_ln, 1e-30, None)

    return {
        "k": k,
        "p_true": p_true,
        "p_obs": p_obs,
        "sigma_ln": sigma_ln,
        "p_err": p_err,
    }


def _linear_covariance(
    x: FloatArray,
    y: FloatArray,
    w: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return weighted linear-fit coefficients and covariance.

    Model: y = b0 + b1*x.
    """
    x1 = np.asarray(x, dtype=np.float64)
    y1 = np.asarray(y, dtype=np.float64)
    w1 = np.asarray(w, dtype=np.float64)

    x_design = np.column_stack([np.ones_like(x1), x1])
    wx = x_design * w1[:, None]
    xtwx = x_design.T @ wx
    xtwy = x_design.T @ (w1 * y1)

    beta = np.linalg.solve(xtwx, xtwy)
    residual = y1 - x_design @ beta

    dof = max(y1.size - 2, 1)
    chi2 = float(np.sum(w1 * residual**2))
    s2 = chi2 / dof
    cov = s2 * np.linalg.inv(xtwx)

    return beta, cov


def evaluate_fit(
    observed: FloatArray,
    predicted: FloatArray,
    sigma: FloatArray,
    dof: int,
) -> tuple[float, float]:
    """Compute chi-square and reduced chi-square for one fitted curve."""
    sigma_safe = np.clip(sigma, 1e-30, None)
    chi2 = float(np.sum(np.square((observed - predicted) / sigma_safe)))
    return chi2, chi2 / max(dof, 1)


def fit_with_sklearn(
    k: FloatArray,
    p_obs: FloatArray,
    sigma_ln: FloatArray,
    k0: float,
) -> dict[str, float | FloatArray]:
    """Estimate (A_s, n_s) by weighted linear regression in log-space."""
    x = np.log(k / k0)
    y = np.log(np.clip(p_obs, 1e-300, None))
    w = 1.0 / np.square(np.clip(sigma_ln, 1e-12, None))

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y, sample_weight=w)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    beta, cov = _linear_covariance(x, y, w)
    ln_a_hat = float(beta[0])
    slope_hat = float(beta[1])

    ln_a_std = float(np.sqrt(max(cov[0, 0], 0.0)))
    slope_std = float(np.sqrt(max(cov[1, 1], 0.0)))

    a_s_hat = float(np.exp(ln_a_hat))
    a_s_std = float(a_s_hat * ln_a_std)
    n_s_hat = slope_hat + 1.0
    n_s_std = slope_std

    y_fit = intercept + slope * x
    p_fit = np.exp(y_fit)
    dof = max(y.size - 2, 1)
    chi2, _ = evaluate_fit(y, y_fit, np.clip(sigma_ln, 1e-12, None), dof)

    return {
        "a_s": a_s_hat,
        "a_s_std": a_s_std,
        "n_s": n_s_hat,
        "n_s_std": n_s_std,
        "chi2": chi2,
        "y_fit": y_fit,
        "p_fit": p_fit,
    }


def fit_with_scipy(
    k: FloatArray,
    p_obs: FloatArray,
    p_err: FloatArray,
    k0: float,
    a_s_init: float,
    n_s_init: float,
) -> dict[str, float | FloatArray]:
    """Estimate (A_s, n_s) by weighted nonlinear least squares in P-space."""

    def model(k_vals: FloatArray, a_s: float, n_s: float) -> FloatArray:
        return scalar_power_spectrum(k_vals, a_s, n_s, k0)

    popt, pcov = curve_fit(
        model,
        k,
        p_obs,
        p0=np.array([a_s_init, n_s_init], dtype=np.float64),
        sigma=np.clip(p_err, 1e-30, None),
        absolute_sigma=True,
        maxfev=20000,
    )

    a_s_hat = float(popt[0])
    n_s_hat = float(popt[1])
    a_s_std = float(np.sqrt(max(pcov[0, 0], 0.0)))
    n_s_std = float(np.sqrt(max(pcov[1, 1], 0.0)))

    p_fit = model(k, a_s_hat, n_s_hat)
    dof = max(k.size - 2, 1)
    chi2, _ = evaluate_fit(p_obs, p_fit, p_err, dof)

    return {
        "a_s": a_s_hat,
        "a_s_std": a_s_std,
        "n_s": n_s_hat,
        "n_s_std": n_s_std,
        "chi2": chi2,
        "p_fit": p_fit,
    }


def fit_with_torch(
    k: FloatArray,
    p_obs: FloatArray,
    sigma_ln: FloatArray,
    k0: float,
    steps: int,
    lr: float,
) -> dict[str, float | FloatArray]:
    """Estimate (A_s, n_s) in log-space with PyTorch gradient descent."""
    torch.manual_seed(123)

    dtype = torch.float64
    x = torch.tensor(np.log(k / k0), dtype=dtype)
    y = torch.tensor(np.log(np.clip(p_obs, 1e-300, None)), dtype=dtype)
    w = torch.tensor(1.0 / np.square(np.clip(sigma_ln, 1e-12, None)), dtype=dtype)

    ln_a = torch.tensor(np.log(np.median(p_obs)), dtype=dtype, requires_grad=True)
    slope = torch.tensor(-0.04, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam([ln_a, slope], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        y_hat = ln_a + slope * x
        loss = torch.sum(w * (y_hat - y) ** 2) / torch.sum(w)
        loss.backward()
        optimizer.step()

    ln_a_hat = float(ln_a.detach().cpu().item())
    slope_hat = float(slope.detach().cpu().item())

    y_fit = ln_a_hat + slope_hat * np.log(k / k0)
    p_fit = np.exp(y_fit)
    dof = max(k.size - 2, 1)
    log_obs = np.log(np.clip(p_obs, 1e-300, None))
    chi2, _ = evaluate_fit(log_obs, y_fit, sigma_ln, dof)

    return {
        "a_s": float(np.exp(ln_a_hat)),
        "a_s_std": float("nan"),
        "n_s": slope_hat + 1.0,
        "n_s_std": float("nan"),
        "chi2": chi2,
        "y_fit": y_fit,
        "p_fit": p_fit,
    }


def format_table(df: pd.DataFrame) -> str:
    """Deterministic table formatter."""
    return df.to_string(index=False, float_format=lambda x: f"{x:.6e}")


def main() -> None:
    cfg = Config()
    data = generate_synthetic_data(cfg)

    k = data["k"]
    p_true = data["p_true"]
    p_obs = data["p_obs"]
    sigma_ln = data["sigma_ln"]
    p_err = data["p_err"]

    if np.any(p_obs <= 0.0):
        raise RuntimeError("Observed spectrum must stay positive for log-space fitting.")

    fit_lr = fit_with_sklearn(k, p_obs, sigma_ln, cfg.k0)
    fit_cf = fit_with_scipy(k, p_obs, p_err, cfg.k0, cfg.a_s_true, cfg.n_s_true)
    fit_th = fit_with_torch(
        k,
        p_obs,
        sigma_ln,
        cfg.k0,
        steps=cfg.torch_steps,
        lr=cfg.torch_lr,
    )

    dof = max(cfg.n_points - 2, 1)

    method_rows = [
        {
            "method": "sklearn_weighted_loglin",
            "A_s_hat": float(fit_lr["a_s"]),
            "n_s_hat": float(fit_lr["n_s"]),
            "n_s_bias": float(fit_lr["n_s"]) - cfg.n_s_true,
            "chi2/dof": float(fit_lr["chi2"]) / dof,
        },
        {
            "method": "scipy_curve_fit",
            "A_s_hat": float(fit_cf["a_s"]),
            "n_s_hat": float(fit_cf["n_s"]),
            "n_s_bias": float(fit_cf["n_s"]) - cfg.n_s_true,
            "chi2/dof": float(fit_cf["chi2"]) / dof,
        },
        {
            "method": "torch_weighted_gd",
            "A_s_hat": float(fit_th["a_s"]),
            "n_s_hat": float(fit_th["n_s"]),
            "n_s_bias": float(fit_th["n_s"]) - cfg.n_s_true,
            "chi2/dof": float(fit_th["chi2"]) / dof,
        },
    ]

    df_methods = pd.DataFrame(method_rows)

    sample_idx = np.linspace(0, cfg.n_points - 1, 8, dtype=int)
    df_sample = pd.DataFrame(
        {
            "k[Mpc^-1]": k[sample_idx],
            "P_true": p_true[sample_idx],
            "P_obs": p_obs[sample_idx],
            "sigma_lnP": sigma_ln[sample_idx],
        }
    )

    ns_spread = float(df_methods["n_s_hat"].max() - df_methods["n_s_hat"].min())

    print("Scalar Spectral Index MVP")
    print("=" * 72)
    print(
        f"True params: A_s={cfg.a_s_true:.6e}, n_s={cfg.n_s_true:.6f}, "
        f"k0={cfg.k0:.3f} Mpc^-1"
    )
    print(
        f"k-range: [{cfg.k_min:.1e}, {cfg.k_max:.1e}] Mpc^-1, "
        f"N={cfg.n_points}, seed={cfg.seed}"
    )
    print()
    print("Synthetic data sample:")
    print(format_table(df_sample))
    print()
    print("Fit summary:")
    print(format_table(df_methods))
    print()
    print(f"Cross-method n_s spread = {ns_spread:.6e}")


if __name__ == "__main__":
    main()
