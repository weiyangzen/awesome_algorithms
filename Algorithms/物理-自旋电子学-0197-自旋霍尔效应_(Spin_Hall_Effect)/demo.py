"""Minimal runnable MVP for Spin Hall Effect (SHE).

Pipeline in this demo:
1) Forward simulation from 1D spin-diffusion with SHE source term.
2) scipy curve_fit inversion on thickness-dependent edge accumulation.
3) sklearn linear fit on inverse-SHE voltage vs charge current density.
4) PyTorch joint fitting of (theta_SH, lambda_sf) from noisy profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class SHEParams:
    conductivity_S_per_m: float = 3.6e6
    resistivity_ohm_m: float = 2.7777777777777777e-7  # 1 / conductivity
    theta_sh: float = 0.11
    lambda_sf_nm: float = 1.7

    thickness_nm: float = 8.0
    length_um: float = 20.0

    n_y: int = 121
    n_j: int = 9
    j_min_A_m2: float = 2.0e10
    j_max_A_m2: float = 1.0e11

    thickness_scan_min_nm: float = 1.5
    thickness_scan_max_nm: float = 18.0
    n_thickness_scan: int = 26

    noise_edge_uV: float = 0.012
    noise_ishe_uV: float = 0.015
    noise_profile_uV: float = 0.020

    torch_epochs: int = 650
    torch_lr: float = 0.04
    seed: int = 11


def nm_to_m(x_nm: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(x_nm) * 1e-9


def um_to_m(x_um: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(x_um) * 1e-6


def edge_accumulation_model_V(
    thickness_m: np.ndarray,
    theta_sh: float,
    lambda_sf_m: float,
    j_c_A_m2: float,
    conductivity_S_per_m: float,
) -> np.ndarray:
    """Spin accumulation magnitude at film edge from 1D diffusion solution.

    Model in this MVP (voltage-like spin accumulation):
        mu_edge = 2*lambda*theta*j_c/sigma * tanh(t/(2*lambda))
    """
    ratio = thickness_m / (2.0 * lambda_sf_m)
    return (2.0 * lambda_sf_m * theta_sh * j_c_A_m2 / conductivity_S_per_m) * np.tanh(ratio)


def spin_profile_V_and_current(
    y_m: np.ndarray,
    thickness_m: float,
    j_c_A_m2: float,
    theta_sh: float,
    lambda_sf_m: float,
    conductivity_S_per_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spin accumulation profile mu_s(y) and spin current j_s(y).

    Boundary condition: j_s(y=+-t/2)=0 (open spin-current boundaries).
    Constitutive relation used in this MVP:
        j_s = theta*j_c - (sigma/2) * d(mu_s)/dy
    """
    scale = 2.0 * lambda_sf_m * theta_sh * j_c_A_m2 / conductivity_S_per_m
    denom = np.cosh(thickness_m / (2.0 * lambda_sf_m))

    y_over_l = y_m / lambda_sf_m
    mu_s = scale * np.sinh(y_over_l) / denom

    dmu_dy = (2.0 * theta_sh * j_c_A_m2 / conductivity_S_per_m) * np.cosh(y_over_l) / denom
    j_s = theta_sh * j_c_A_m2 - 0.5 * conductivity_S_per_m * dmu_dy
    return mu_s, j_s


def mean_spin_current_factor(thickness_m: float, lambda_sf_m: float) -> float:
    """Return <j_s>/j_c/theta factor for ISHE conversion."""
    x = thickness_m / (2.0 * lambda_sf_m)
    return 1.0 - (np.tanh(x) / max(x, 1e-15))


def simulate_spin_profiles(params: SHEParams) -> pd.DataFrame:
    thickness_m = float(nm_to_m(params.thickness_nm))
    y_m = np.linspace(-0.5 * thickness_m, 0.5 * thickness_m, params.n_y)
    j_grid = np.linspace(params.j_min_A_m2, params.j_max_A_m2, params.n_j)

    records: list[dict[str, float]] = []
    lambda_m = float(nm_to_m(params.lambda_sf_nm))

    for j_c in j_grid:
        mu_s, j_s = spin_profile_V_and_current(
            y_m=y_m,
            thickness_m=thickness_m,
            j_c_A_m2=float(j_c),
            theta_sh=params.theta_sh,
            lambda_sf_m=lambda_m,
            conductivity_S_per_m=params.conductivity_S_per_m,
        )

        for yi, mu_i, js_i in zip(y_m, mu_s, j_s):
            records.append(
                {
                    "j_c_A_m2": float(j_c),
                    "y_nm": float(yi * 1e9),
                    "mu_s_uV": float(mu_i * 1e6),
                    "j_s_A_m2": float(js_i),
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def fit_edge_scan_with_scipy(params: SHEParams) -> dict[str, float]:
    rng = np.random.default_rng(params.seed)
    t_scan_nm = np.linspace(params.thickness_scan_min_nm, params.thickness_scan_max_nm, params.n_thickness_scan)
    t_scan_m = nm_to_m(t_scan_nm)

    j_ref = 6.5e10
    lambda_true_m = float(nm_to_m(params.lambda_sf_nm))
    mu_true_V = edge_accumulation_model_V(
        thickness_m=t_scan_m,
        theta_sh=params.theta_sh,
        lambda_sf_m=lambda_true_m,
        j_c_A_m2=j_ref,
        conductivity_S_per_m=params.conductivity_S_per_m,
    )
    mu_true_uV = mu_true_V * 1e6
    mu_obs_uV = mu_true_uV + rng.normal(0.0, params.noise_edge_uV, size=mu_true_uV.shape)

    def model_for_fit_uV(thickness_nm: np.ndarray, theta: float, lambda_nm: float) -> np.ndarray:
        mu_v = edge_accumulation_model_V(
            thickness_m=nm_to_m(thickness_nm),
            theta_sh=theta,
            lambda_sf_m=float(nm_to_m(lambda_nm)),
            j_c_A_m2=j_ref,
            conductivity_S_per_m=params.conductivity_S_per_m,
        )
        return mu_v * 1e6

    p0 = np.array([0.08, 1.2], dtype=float)
    popt, _ = curve_fit(
        f=model_for_fit_uV,
        xdata=t_scan_nm,
        ydata=mu_obs_uV,
        p0=p0,
        bounds=([0.01, 0.2], [0.5, 10.0]),
        maxfev=12000,
    )

    theta_fit, lambda_fit_nm = float(popt[0]), float(popt[1])
    mu_fit_uV = model_for_fit_uV(t_scan_nm, theta_fit, lambda_fit_nm)

    mae_uV = float(np.mean(np.abs(mu_fit_uV - mu_true_uV)))
    rel_theta_err = abs(theta_fit - params.theta_sh) / params.theta_sh
    rel_lambda_err = abs(lambda_fit_nm - params.lambda_sf_nm) / params.lambda_sf_nm

    return {
        "edge_theta_fit": theta_fit,
        "edge_lambda_fit_nm": lambda_fit_nm,
        "edge_mae_uV": mae_uV,
        "edge_theta_rel_err": float(rel_theta_err),
        "edge_lambda_rel_err": float(rel_lambda_err),
        "edge_scan_j_ref_A_m2": float(j_ref),
    }


def fit_ishe_line_with_sklearn(params: SHEParams, lambda_for_eta_nm: float) -> dict[str, float]:
    rng = np.random.default_rng(params.seed + 1)

    j_grid = np.linspace(params.j_min_A_m2, params.j_max_A_m2, 14)
    thickness_m = float(nm_to_m(params.thickness_nm))
    lambda_m = float(nm_to_m(lambda_for_eta_nm))
    length_m = float(um_to_m(params.length_um))

    eta = mean_spin_current_factor(thickness_m=thickness_m, lambda_sf_m=lambda_m)
    slope_true = length_m * params.resistivity_ohm_m * (params.theta_sh**2) * eta

    v_true_V = slope_true * j_grid
    v_obs_V = v_true_V + rng.normal(0.0, params.noise_ishe_uV * 1e-6, size=v_true_V.shape)

    model = LinearRegression()
    X = j_grid.reshape(-1, 1)
    y = v_obs_V
    model.fit(X, y)
    r2 = float(model.score(X, y))

    slope_fit = float(model.coef_[0])
    intercept_fit = float(model.intercept_)

    theta_est = float(np.sqrt(max(slope_fit / max(length_m * params.resistivity_ohm_m * eta, 1e-30), 1e-30)))

    return {
        "ishe_slope_true_V_per_A_m2": float(slope_true),
        "ishe_slope_fit_V_per_A_m2": float(slope_fit),
        "ishe_intercept_fit_V": intercept_fit,
        "ishe_r2": r2,
        "ishe_theta_est": theta_est,
        "ishe_theta_rel_err": float(abs(theta_est - params.theta_sh) / params.theta_sh),
        "ishe_eta": float(eta),
    }


def torch_fit_profiles(params: SHEParams, df_profiles: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(params.seed + 2)
    torch.manual_seed(params.seed)

    y_m = (df_profiles["y_nm"].to_numpy(dtype=np.float64)) * 1e-9
    j_c = df_profiles["j_c_A_m2"].to_numpy(dtype=np.float64)
    mu_true_uV = df_profiles["mu_s_uV"].to_numpy(dtype=np.float64)
    mu_obs_uV = mu_true_uV + rng.normal(0.0, params.noise_profile_uV, size=mu_true_uV.shape)

    thickness_m = float(nm_to_m(params.thickness_nm))
    sigma = float(params.conductivity_S_per_m)

    y_t = torch.tensor(y_m, dtype=torch.float64)
    j_t = torch.tensor(j_c, dtype=torch.float64)
    mu_obs_t = torch.tensor(mu_obs_uV, dtype=torch.float64)

    raw_theta = torch.nn.Parameter(torch.tensor(np.log(np.expm1(0.09)), dtype=torch.float64))
    raw_lambda = torch.nn.Parameter(torch.tensor(np.log(np.expm1(1.1e-9)), dtype=torch.float64))

    optimizer = torch.optim.Adam([raw_theta, raw_lambda], lr=params.torch_lr)

    loss_value = np.nan
    for _ in range(params.torch_epochs):
        optimizer.zero_grad()

        theta = torch.nn.functional.softplus(raw_theta) + 1e-12
        lambda_m = torch.nn.functional.softplus(raw_lambda) + 1e-12

        denom = torch.cosh(thickness_m / (2.0 * lambda_m))
        mu_pred = (2.0 * lambda_m * theta * j_t / sigma) * torch.sinh(y_t / lambda_m) / denom
        mu_pred_uV = mu_pred * 1e6

        loss = torch.mean((mu_pred_uV - mu_obs_t) ** 2)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())

    theta_fit = float(torch.nn.functional.softplus(raw_theta).detach().cpu().item())
    lambda_fit_m = float(torch.nn.functional.softplus(raw_lambda).detach().cpu().item())

    return {
        "torch_theta_fit": theta_fit,
        "torch_lambda_fit_nm": float(lambda_fit_m * 1e9),
        "torch_final_loss_V2": float(loss_value),
        "torch_theta_rel_err": float(abs(theta_fit - params.theta_sh) / params.theta_sh),
        "torch_lambda_rel_err": float(abs(lambda_fit_m * 1e9 - params.lambda_sf_nm) / params.lambda_sf_nm),
    }


def validate(
    params: SHEParams,
    df_profiles: pd.DataFrame,
    edge_fit: dict[str, float],
    ishe_fit: dict[str, float],
    torch_fit: dict[str, float],
) -> dict[str, float]:
    grouped = df_profiles.groupby("j_c_A_m2", sort=True)
    boundary_residuals = []

    for _, part in grouped:
        left_js = float(part.iloc[0]["j_s_A_m2"])
        right_js = float(part.iloc[-1]["j_s_A_m2"])
        j_c = float(part.iloc[0]["j_c_A_m2"])
        boundary_residuals.append(abs(left_js) / max(abs(j_c), 1e-30))
        boundary_residuals.append(abs(right_js) / max(abs(j_c), 1e-30))

    boundary_max_rel = float(np.max(boundary_residuals))

    mu_max = float(df_profiles["mu_s_uV"].max())
    mu_min = float(df_profiles["mu_s_uV"].min())
    center_mask = np.isclose(df_profiles["y_nm"].to_numpy(), 0.0)
    center_mu_mean = float(np.mean(np.abs(df_profiles.loc[center_mask, "mu_s_uV"])))

    assert boundary_max_rel < 5e-11, f"Boundary spin-current condition violated: {boundary_max_rel:.3e}"
    assert mu_max > 0.0 and mu_min < 0.0, "Spin accumulation must be antisymmetric with opposite signs"
    assert center_mu_mean < 2e-12, f"Center spin accumulation should be near zero, got {center_mu_mean:.3e} uV"

    assert edge_fit["edge_theta_rel_err"] < 0.20, "scipy edge fit theta error too large"
    assert edge_fit["edge_lambda_rel_err"] < 0.20, "scipy edge fit lambda error too large"
    assert edge_fit["edge_mae_uV"] < 0.020, "scipy edge fit MAE too large"

    assert ishe_fit["ishe_r2"] > 0.995, "ISHE linear fit R2 too low"
    assert ishe_fit["ishe_theta_rel_err"] < 0.14, "ISHE theta estimate error too large"
    assert abs(ishe_fit["ishe_intercept_fit_V"]) < 6e-8, "ISHE intercept is unexpectedly large"

    assert torch_fit["torch_theta_rel_err"] < 0.12, "Torch theta fit error too large"
    assert torch_fit["torch_lambda_rel_err"] < 0.12, "Torch lambda fit error too large"
    assert torch_fit["torch_final_loss_V2"] < 1.5e-3, "Torch profile fit loss too large"

    return {
        "boundary_max_rel": boundary_max_rel,
        "mu_s_uV_min": mu_min,
        "mu_s_uV_max": mu_max,
        "center_mu_abs_mean_uV": center_mu_mean,
    }


def build_summary(
    params: SHEParams,
    df_profiles: pd.DataFrame,
    edge_fit: dict[str, float],
    ishe_fit: dict[str, float],
    torch_fit: dict[str, float],
    validation: dict[str, float],
) -> dict[str, float]:
    return {
        "theta_sh_true": float(params.theta_sh),
        "lambda_sf_true_nm": float(params.lambda_sf_nm),
        "thickness_nm": float(params.thickness_nm),
        "conductivity_S_per_m": float(params.conductivity_S_per_m),
        "n_profile_rows": float(len(df_profiles)),
        "edge_theta_fit": edge_fit["edge_theta_fit"],
        "edge_lambda_fit_nm": edge_fit["edge_lambda_fit_nm"],
        "edge_mae_uV": edge_fit["edge_mae_uV"],
        "ishe_r2": ishe_fit["ishe_r2"],
        "ishe_theta_est": ishe_fit["ishe_theta_est"],
        "torch_theta_fit": torch_fit["torch_theta_fit"],
        "torch_lambda_fit_nm": torch_fit["torch_lambda_fit_nm"],
        "torch_final_loss_V2": torch_fit["torch_final_loss_V2"],
        "boundary_max_rel": validation["boundary_max_rel"],
    }


def main() -> None:
    params = SHEParams()

    df_profiles = simulate_spin_profiles(params)
    edge_fit = fit_edge_scan_with_scipy(params)
    ishe_fit = fit_ishe_line_with_sklearn(params, lambda_for_eta_nm=edge_fit["edge_lambda_fit_nm"])
    torch_fit = torch_fit_profiles(params, df_profiles)
    validation = validate(params, df_profiles, edge_fit, ishe_fit, torch_fit)
    summary = build_summary(params, df_profiles, edge_fit, ishe_fit, torch_fit, validation)

    print("=== Spin Hall Effect MVP Summary ===")
    for k, v in summary.items():
        print(f"{k:>24s}: {v:.10g}")

    print("\n=== Profile sample (head) ===")
    print(df_profiles.head(10).to_string(index=False))

    print("\n=== Edge/ISHE/Torch key metrics ===")
    merged = {
        **edge_fit,
        **ishe_fit,
        **torch_fit,
        **validation,
    }
    for k in sorted(merged):
        print(f"{k:>24s}: {merged[k]:.10g}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
