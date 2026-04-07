"""Minimal runnable MVP for Fermi transitions in nuclear physics.

This script demonstrates a compact analysis workflow for allowed Fermi transitions:
1) Compute the statistical rate function f from an explicit phase-space integral.
2) Build synthetic transition measurements and evaluate ft and corrected Ft values.
3) Test the constant-Ft hypothesis with weighted statistics, sklearn, and PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

ALPHA = 1.0 / 137.035999084
M_E_MEV = 0.51099895


@dataclass(frozen=True)
class TransitionSpec:
    """Input specification for one idealized allowed Fermi transition."""

    label: str
    z_daughter: int
    endpoint_kinetic_mev: float


def fermi_function(z_daughter: int, w: np.ndarray, beta_sign: int) -> np.ndarray:
    """Approximate Fermi function F(±Z, W).

    Parameters
    ----------
    z_daughter:
        Daughter nuclear charge Z.
    w:
        Total lepton energy in electron-mass units, W = E / (m_e c^2).
    beta_sign:
        +1 for beta- and -1 for beta+.
    """
    p = np.sqrt(np.clip(w * w - 1.0, 1e-16, None))
    eta = beta_sign * ALPHA * z_daughter * w / p
    x = 2.0 * math.pi * eta

    # Stable evaluation of x / (1 - exp(-x)).
    denom = -np.expm1(-x)
    small = np.abs(x) < 1e-8
    ratio = np.where(small, 1.0 + 0.5 * x, x / denom)
    return np.clip(ratio, 1e-12, None)


def statistical_rate_function(
    endpoint_kinetic_mev: float,
    z_daughter: int,
    beta_sign: int = 1,
    n_grid: int = 8000,
) -> float:
    """Compute dimensionless statistical rate function f for allowed transitions.

    f = integral p * W * (W0 - W)^2 * F(±Z, W) dW

    where W0 = 1 + T0 / m_e and T0 is endpoint kinetic energy in MeV.
    """
    if endpoint_kinetic_mev <= 0.0:
        raise ValueError("Endpoint kinetic energy must be positive.")

    w0 = 1.0 + endpoint_kinetic_mev / M_E_MEV
    w = np.linspace(1.0 + 1e-8, w0, n_grid)
    p = np.sqrt(np.clip(w * w - 1.0, 0.0, None))
    fermi = fermi_function(z_daughter=z_daughter, w=w, beta_sign=beta_sign)
    integrand = p * w * np.clip(w0 - w, 0.0, None) ** 2 * fermi
    return float(simpson(integrand, x=w))


def build_synthetic_dataset(seed: int = 20260407, ft_ref_s: float = 3072.27) -> pd.DataFrame:
    """Generate reproducible synthetic transition measurements.

    The generated data follows the corrected-Ft relation with small measurement noise:
    Ft = ft * (1 + delta_R') * (1 + delta_NS - delta_C)
    """
    rng = np.random.default_rng(seed)

    specs = [
        TransitionSpec("F-01", 8, 2.10),
        TransitionSpec("F-02", 10, 2.55),
        TransitionSpec("F-03", 12, 3.00),
        TransitionSpec("F-04", 14, 3.45),
        TransitionSpec("F-05", 16, 3.90),
        TransitionSpec("F-06", 18, 4.30),
        TransitionSpec("F-07", 20, 4.75),
        TransitionSpec("F-08", 22, 5.20),
        TransitionSpec("F-09", 24, 5.75),
        TransitionSpec("F-10", 26, 6.30),
    ]

    n = len(specs)
    delta_r_pct = np.linspace(1.30, 1.62, n) + rng.normal(0.0, 0.015, size=n)
    delta_ns_pct = np.linspace(0.04, 0.34, n) + rng.normal(0.0, 0.020, size=n)
    delta_c_pct = np.linspace(0.18, 0.72, n) + rng.normal(0.0, 0.020, size=n)
    sigma_t_pct = rng.uniform(0.08, 0.22, size=n)

    rows: list[dict[str, float | int | str]] = []
    for i, spec in enumerate(specs):
        f_value = statistical_rate_function(
            endpoint_kinetic_mev=spec.endpoint_kinetic_mev,
            z_daughter=spec.z_daughter,
            beta_sign=1,
            n_grid=8000,
        )

        corr_factor = (1.0 + delta_r_pct[i] / 100.0) * (
            1.0 + (delta_ns_pct[i] - delta_c_pct[i]) / 100.0
        )
        t_partial_true_s = ft_ref_s / (f_value * corr_factor)
        t_partial_obs_s = t_partial_true_s * (1.0 + rng.normal(0.0, sigma_t_pct[i] / 100.0))

        ft_obs_s = f_value * t_partial_obs_s
        ft_corr_s = ft_obs_s * corr_factor
        sigma_ft_corr_s = ft_corr_s * sigma_t_pct[i] / 100.0

        rows.append(
            {
                "transition": spec.label,
                "z_daughter": spec.z_daughter,
                "endpoint_kinetic_mev": spec.endpoint_kinetic_mev,
                "f_stat": f_value,
                "delta_r_pct": float(delta_r_pct[i]),
                "delta_ns_pct": float(delta_ns_pct[i]),
                "delta_c_pct": float(delta_c_pct[i]),
                "t_partial_obs_s": float(t_partial_obs_s),
                "ft_obs_s": float(ft_obs_s),
                "Ft_corr_s": float(ft_corr_s),
                "sigma_Ft_corr_s": float(sigma_ft_corr_s),
            }
        )

    return pd.DataFrame(rows)


def weighted_ft_estimate(ft_corr_s: np.ndarray, sigma_ft_corr_s: np.ndarray) -> tuple[float, float, float, float]:
    """Return weighted mean Ft, uncertainty, chi2/ndf and p-value."""
    y = np.asarray(ft_corr_s, dtype=float)
    sigma = np.asarray(sigma_ft_corr_s, dtype=float)
    w = 1.0 / np.clip(sigma, 1e-12, None) ** 2

    ft_mean = float(np.sum(w * y) / np.sum(w))
    ft_unc = float(math.sqrt(1.0 / np.sum(w)))

    chi2_val = float(np.sum(((y - ft_mean) / sigma) ** 2))
    dof = max(1, y.size - 1)
    chi2_per_dof = chi2_val / dof
    p_val = float(chi2.sf(chi2_val, dof))
    return ft_mean, ft_unc, chi2_per_dof, p_val


def fit_ft_constant_torch(
    ft_corr_s: np.ndarray,
    sigma_ft_corr_s: np.ndarray,
    steps: int = 1200,
    lr: float = 0.08,
) -> float:
    """Fit one-parameter constant Ft model with weighted MSE using PyTorch."""
    y = torch.tensor(ft_corr_s, dtype=torch.float64)
    sigma = torch.tensor(sigma_ft_corr_s, dtype=torch.float64)
    w = 1.0 / torch.clamp(sigma, min=1e-12) ** 2

    theta = torch.tensor(float(torch.mean(y)), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.mean(w * (theta - y) ** 2)
        loss.backward()
        optimizer.step()

    return float(theta.detach().cpu().item())


def run_linear_trend_test(df: pd.DataFrame) -> tuple[float, float]:
    """Check whether corrected Ft has residual linear trend versus Z."""
    x = df[["z_daughter"]].to_numpy(dtype=float)
    y = df["Ft_corr_s"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    r2 = float(model.score(x, y))
    return slope, r2


def main() -> None:
    seed = 20260407
    torch.manual_seed(seed)

    df = build_synthetic_dataset(seed=seed, ft_ref_s=3072.27)

    ft_mean, ft_unc, chi2_ndf, chi2_p = weighted_ft_estimate(
        df["Ft_corr_s"].to_numpy(),
        df["sigma_Ft_corr_s"].to_numpy(),
    )
    ft_torch = fit_ft_constant_torch(
        df["Ft_corr_s"].to_numpy(),
        df["sigma_Ft_corr_s"].to_numpy(),
    )
    slope_z, r2_z = run_linear_trend_test(df)

    df = df.copy()
    df["Ft_residual_s"] = df["Ft_corr_s"] - ft_torch
    df["rel_residual_pct"] = 100.0 * df["Ft_residual_s"] / ft_torch

    print("=== Fermi Transition MVP ===")
    print(f"seed={seed}, transitions={len(df)}")
    print("\n[Per-transition summary]")
    cols = [
        "transition",
        "z_daughter",
        "endpoint_kinetic_mev",
        "f_stat",
        "t_partial_obs_s",
        "ft_obs_s",
        "Ft_corr_s",
        "sigma_Ft_corr_s",
        "rel_residual_pct",
    ]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    print("\n[Global diagnostics]")
    print(f"Weighted Ft (analytic): {ft_mean:.6f} +/- {ft_unc:.6f} s")
    print(f"Weighted Ft (PyTorch):  {ft_torch:.6f} s")
    print(f"Difference (torch-analytic): {ft_torch - ft_mean:.6e} s")
    print(f"chi2/ndf: {chi2_ndf:.6f}")
    print(f"chi2 p-value: {chi2_p:.6f}")
    print(f"Linear trend slope dFt/dZ: {slope_z:.6f} s per charge")
    print(f"Linear trend R^2: {r2_z:.6f}")

    # Minimal sanity checks for automated validation.
    assert (df["f_stat"] > 0.0).all()
    assert (df["t_partial_obs_s"] > 0.0).all()
    assert abs(ft_torch - 3072.27) < 80.0
    assert abs(slope_z) < 3.0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
