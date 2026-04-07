"""Minimal runnable MVP for Quantum Hall Effect (QHE).

This script builds a small but transparent numerical pipeline:
1) Landau-level DOS with Gaussian disorder broadening.
2) Chemical-potential solving at fixed 2DEG density.
3) Hall and longitudinal conductivity/resistivity curves vs magnetic field.
4) A tiny PyTorch inverse step to recover transition width parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.constants import e, h, hbar, k as k_B, m_e
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

E2_OVER_H = e**2 / h
H_OVER_E2 = h / e**2
K_B_EV = k_B / e


@dataclass(frozen=True)
class QHEParams:
    electron_density_m2: float = 3.2e15
    effective_mass_ratio: float = 0.067
    temperature_K: float = 1.6

    B_min_T: float = 2.0
    B_max_T: float = 14.0
    n_B: int = 180

    max_landau_level: int = 28

    energy_min_meV: float = -6.0
    energy_max_meV: float = 190.0
    n_energy: int = 2400

    gamma_dos_meV: float = 1.8
    gamma_xy_meV: float = 0.55
    gamma_xx_meV: float = 1.6
    sigma_xx0_quantum: float = 0.33

    torch_epochs: int = 500
    torch_lr: float = 0.035

    obs_noise_sigma_xy: float = 0.012
    obs_noise_rho_xx: float = 0.004
    seed: int = 7


def meV_to_eV(x_meV: float | np.ndarray) -> float | np.ndarray:
    return 1e-3 * x_meV


def fermi_dirac(E_eV: np.ndarray, mu_eV: float, temperature_K: float) -> np.ndarray:
    x = (E_eV - mu_eV) / max(K_B_EV * temperature_K, 1e-12)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(x))


def landau_levels_eV(B_T: float, m_eff_ratio: float, n_max: int) -> np.ndarray:
    omega_c = e * B_T / (m_eff_ratio * m_e)
    n = np.arange(n_max + 1, dtype=float)
    return (hbar * omega_c / e) * (n + 0.5)


def dos_from_landau(E_eV: np.ndarray, levels_eV: np.ndarray, degeneracy_m2: float, gamma_dos_eV: float) -> np.ndarray:
    z = (E_eV[None, :] - levels_eV[:, None]) / gamma_dos_eV
    gauss = np.exp(-0.5 * z * z) / (gamma_dos_eV * np.sqrt(2.0 * np.pi))
    return degeneracy_m2 * np.sum(gauss, axis=0)


def solve_mu_for_density(
    E_eV: np.ndarray,
    dos: np.ndarray,
    target_density_m2: float,
    temperature_K: float,
) -> float:
    def density_error(mu: float) -> float:
        occ = fermi_dirac(E_eV, mu, temperature_K)
        density = trapezoid(dos * occ, E_eV)
        return density - target_density_m2

    mu_lo = float(E_eV[0] - 0.06)
    mu_hi = float(E_eV[-1] + 0.06)

    f_lo = density_error(mu_lo)
    f_hi = density_error(mu_hi)
    if f_lo > 0 or f_hi < 0:
        raise RuntimeError(
            "Chemical potential root not bracketed. "
            f"f(mu_lo)={f_lo:.3e}, f(mu_hi)={f_hi:.3e}"
        )

    return float(brentq(density_error, mu_lo, mu_hi, xtol=1e-12, rtol=1e-10, maxiter=200))


def qhe_forward(params: QHEParams) -> pd.DataFrame:
    B_grid = np.linspace(params.B_min_T, params.B_max_T, params.n_B)
    E_grid = np.linspace(meV_to_eV(params.energy_min_meV), meV_to_eV(params.energy_max_meV), params.n_energy)

    gamma_dos_eV = float(meV_to_eV(params.gamma_dos_meV))
    gamma_xy_eV = float(meV_to_eV(params.gamma_xy_meV))
    gamma_xx_eV = float(meV_to_eV(params.gamma_xx_meV))

    records: list[dict[str, float]] = []
    for B_T in B_grid:
        levels = landau_levels_eV(B_T, params.effective_mass_ratio, params.max_landau_level)
        degeneracy = e * B_T / h
        dos = dos_from_landau(E_grid, levels, degeneracy, gamma_dos_eV)
        mu = solve_mu_for_density(E_grid, dos, params.electron_density_m2, params.temperature_K)

        x_xy = np.clip((mu - levels) / gamma_xy_eV, -80.0, 80.0)
        sigma_xy_q = float(np.sum(1.0 / (1.0 + np.exp(-x_xy))))

        x_xx = (mu - levels) / gamma_xx_eV
        sigma_xx_q = float(params.sigma_xx0_quantum * np.sum(np.exp(-0.5 * x_xx * x_xx)))

        sigma_xy = sigma_xy_q * E2_OVER_H
        sigma_xx = sigma_xx_q * E2_OVER_H
        denom = sigma_xy * sigma_xy + sigma_xx * sigma_xx + 1e-30
        rho_xy = sigma_xy / denom
        rho_xx = sigma_xx / denom

        nu_classical = params.electron_density_m2 * h / (e * B_T)

        records.append(
            {
                "B_T": float(B_T),
                "mu_meV": float(mu * 1e3),
                "nu_classical": float(nu_classical),
                "sigma_xy_q": sigma_xy_q,
                "sigma_xx_q": sigma_xx_q,
                "rho_xy_ohm": float(rho_xy),
                "rho_xx_ohm": float(rho_xx),
                "rho_xy_q": float(rho_xy / H_OVER_E2),
                "rho_xx_q": float(rho_xx / H_OVER_E2),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["plateau_integer"] = np.rint(df["sigma_xy_q"]).clip(lower=0.0)
    df["plateau_abs_error"] = (df["sigma_xy_q"] - df["plateau_integer"]).abs()
    return df


def fit_low_field_density(df: pd.DataFrame) -> dict[str, float]:
    n_fit = max(10, int(0.25 * len(df)))
    low_B = df.nsmallest(n_fit, "B_T")

    X = low_B[["B_T"]].to_numpy()
    y = low_B["rho_xy_ohm"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    density_est = 1.0 / max(e * slope, 1e-30)

    return {
        "low_field_slope_ohm_per_T": slope,
        "low_field_intercept_ohm": intercept,
        "low_field_r2": float(r2),
        "density_est_m2": float(density_est),
    }


def torch_fit_transition(df: pd.DataFrame, params: QHEParams) -> dict[str, float]:
    rng = np.random.default_rng(params.seed)
    torch.manual_seed(params.seed)

    B = df["B_T"].to_numpy(dtype=np.float64)
    mu_eV = (df["mu_meV"].to_numpy(dtype=np.float64)) * 1e-3
    sigma_xy_target = df["sigma_xy_q"].to_numpy(dtype=np.float64)
    rho_xx_target = df["rho_xx_q"].to_numpy(dtype=np.float64)

    sigma_xy_obs = sigma_xy_target + rng.normal(0.0, params.obs_noise_sigma_xy, size=sigma_xy_target.shape)
    rho_xx_obs = np.clip(
        rho_xx_target + rng.normal(0.0, params.obs_noise_rho_xx, size=rho_xx_target.shape),
        0.0,
        None,
    )

    n_levels = params.max_landau_level + 1
    levels = np.stack([landau_levels_eV(float(b), params.effective_mass_ratio, params.max_landau_level) for b in B], axis=0)

    mu_t = torch.tensor(mu_eV, dtype=torch.float64).unsqueeze(1)
    levels_t = torch.tensor(levels, dtype=torch.float64)
    sigma_xy_obs_t = torch.tensor(sigma_xy_obs, dtype=torch.float64)
    rho_xx_obs_t = torch.tensor(rho_xx_obs, dtype=torch.float64)

    gamma_xx = float(meV_to_eV(params.gamma_xx_meV))
    x_xx = (mu_t - levels_t) / gamma_xx
    ll_xx_kernel = torch.exp(-0.5 * x_xx * x_xx).sum(dim=1)

    raw_gamma_xy = torch.nn.Parameter(torch.tensor(np.log(np.expm1(meV_to_eV(0.9))), dtype=torch.float64))
    raw_sigma_xx0 = torch.nn.Parameter(torch.tensor(np.log(np.expm1(0.25)), dtype=torch.float64))

    optimizer = torch.optim.Adam([raw_gamma_xy, raw_sigma_xx0], lr=params.torch_lr)

    loss_value = np.nan
    for _ in range(params.torch_epochs):
        optimizer.zero_grad()

        gamma_xy = torch.nn.functional.softplus(raw_gamma_xy) + 1e-9
        sigma_xx0 = torch.nn.functional.softplus(raw_sigma_xx0) + 1e-9

        x_xy = torch.clamp((mu_t - levels_t) / gamma_xy, min=-80.0, max=80.0)
        sigma_xy_pred = torch.sigmoid(x_xy).sum(dim=1)

        sigma_xx_pred = sigma_xx0 * ll_xx_kernel
        rho_xx_pred = sigma_xx_pred / (sigma_xy_pred * sigma_xy_pred + sigma_xx_pred * sigma_xx_pred + 1e-12)

        loss = torch.mean((sigma_xy_pred - sigma_xy_obs_t) ** 2) + torch.mean((rho_xx_pred - rho_xx_obs_t) ** 2)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())

    gamma_fit_meV = float((torch.nn.functional.softplus(raw_gamma_xy) * 1e3).detach().cpu().item())
    sigma_xx0_fit = float(torch.nn.functional.softplus(raw_sigma_xx0).detach().cpu().item())

    return {
        "torch_levels_used": float(n_levels),
        "torch_gamma_xy_fit_meV": gamma_fit_meV,
        "torch_sigma_xx0_fit_q": sigma_xx0_fit,
        "torch_final_loss": float(loss_value),
        "torch_gamma_xy_true_meV": float(params.gamma_xy_meV),
        "torch_sigma_xx0_true_q": float(params.sigma_xx0_quantum),
    }


def validate(df: pd.DataFrame, params: QHEParams, low_field_fit: dict[str, float], torch_fit: dict[str, float]) -> dict[str, float]:
    plateau_mask = df["rho_xx_q"] < 0.045
    plateau_count = int(plateau_mask.sum())

    if plateau_count > 0:
        plateau_mae = float(df.loc[plateau_mask, "plateau_abs_error"].mean())
    else:
        plateau_mae = float(df["plateau_abs_error"].mean())

    trend = np.diff(df["sigma_xy_q"].to_numpy())
    positive_jumps = float(np.mean(trend > 0.0))

    density_rel_err = abs(low_field_fit["density_est_m2"] - params.electron_density_m2) / params.electron_density_m2
    gamma_fit_err_meV = abs(torch_fit["torch_gamma_xy_fit_meV"] - params.gamma_xy_meV)

    assert not df.isna().any().any(), "NaN detected in simulation table"
    assert plateau_count >= 25, f"Not enough plateau-like points: {plateau_count}"
    assert plateau_mae <= 0.12, f"Plateau MAE too large: {plateau_mae:.4f}"
    assert positive_jumps <= 0.06, f"sigma_xy is not mostly decreasing with B: {positive_jumps:.3f}"
    assert density_rel_err <= 0.18, f"Low-field density fit error too large: {density_rel_err:.3f}"
    assert low_field_fit["low_field_r2"] >= 0.97, f"Low-field Hall linear fit R2 too low: {low_field_fit['low_field_r2']:.3f}"
    assert torch_fit["torch_final_loss"] <= 2.5e-2, f"Torch inverse fit loss too high: {torch_fit['torch_final_loss']:.4e}"
    assert gamma_fit_err_meV <= 0.45, f"Torch gamma_xy fit error too large: {gamma_fit_err_meV:.3f} meV"

    return {
        "plateau_count": float(plateau_count),
        "plateau_mae": float(plateau_mae),
        "sigma_xy_positive_jump_ratio": float(positive_jumps),
        "density_relative_error": float(density_rel_err),
        "gamma_fit_error_meV": float(gamma_fit_err_meV),
    }


def main() -> None:
    params = QHEParams()

    df = qhe_forward(params)
    low_field_fit = fit_low_field_density(df)
    torch_fit = torch_fit_transition(df, params)
    checks = validate(df, params, low_field_fit, torch_fit)

    summary = {
        "n_B_points": int(len(df)),
        "B_range_T": (float(df["B_T"].min()), float(df["B_T"].max())),
        "mu_meV_range": (float(df["mu_meV"].min()), float(df["mu_meV"].max())),
        "sigma_xy_q_range": (float(df["sigma_xy_q"].min()), float(df["sigma_xy_q"].max())),
        "rho_xx_q_peak": float(df["rho_xx_q"].max()),
        **low_field_fit,
        **torch_fit,
        **checks,
    }

    print("=== Quantum Hall MVP Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Curve Samples (first 10 rows) ===")
    cols = [
        "B_T",
        "nu_classical",
        "mu_meV",
        "sigma_xy_q",
        "sigma_xx_q",
        "rho_xy_q",
        "rho_xx_q",
        "plateau_integer",
        "plateau_abs_error",
    ]
    print(df[cols].head(10).to_string(index=False))

    print("\n=== Plateau-region Samples (rho_xx_q < 0.045, first 10 rows) ===")
    plateau_view = df.loc[df["rho_xx_q"] < 0.045, cols].head(10)
    print(plateau_view.to_string(index=False))


if __name__ == "__main__":
    main()
