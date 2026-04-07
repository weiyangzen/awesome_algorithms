"""Minimal runnable MVP for the Magnetoelectric Effect.

The script builds synthetic magnetoelectric measurements with temperature
dependence, estimates the ME tensor per temperature by linear regression,
fits a critical-law envelope with SciPy, and performs global parameter
refinement with PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class MEParams:
    """Physical and numerical settings for the ME MVP."""

    seed: int = 7
    n_temps: int = 24
    samples_per_temp: int = 36

    temp_min_K: float = 120.0
    temp_max_K: float = 350.0
    tn_true_K: float = 282.0
    beta_true: float = 0.46

    e_field_max_MVm: float = 2.0
    h_field_max_T: float = 1.6

    noise_P: float = 0.018
    noise_M: float = 0.015

    # Base tensor alpha0 (arbitrary units in this MVP), alpha(T)=alpha0*s(T).
    alpha0_true: tuple[tuple[float, float], tuple[float, float]] = (
        (2.80, -1.00),
        (0.60, 2.20),
    )
    p0_true: tuple[float, float] = (0.030, -0.020)
    m0_true: tuple[float, float] = (0.012, 0.018)

    torch_epochs: int = 500
    torch_lr: float = 0.05
    torch_weight_decay: float = 2e-4


def _alpha0_true_array(params: MEParams) -> np.ndarray:
    return np.array(params.alpha0_true, dtype=float)


def _p0_true_array(params: MEParams) -> np.ndarray:
    return np.array(params.p0_true, dtype=float)


def _m0_true_array(params: MEParams) -> np.ndarray:
    return np.array(params.m0_true, dtype=float)


def _inv_softplus(x: float) -> float:
    x = max(x, 1e-8)
    return float(np.log(np.expm1(x)))


def check_params(params: MEParams) -> None:
    if params.n_temps < 8:
        raise ValueError("n_temps must be >= 8")
    if params.samples_per_temp < 16:
        raise ValueError("samples_per_temp must be >= 16")
    if params.temp_max_K <= params.temp_min_K:
        raise ValueError("temperature range is invalid")
    if not (params.temp_min_K < params.tn_true_K < params.temp_max_K + 120.0):
        raise ValueError("tn_true_K is outside a valid fitting range")
    if params.beta_true <= 0.0:
        raise ValueError("beta_true must be positive")
    if params.e_field_max_MVm <= 0.0 or params.h_field_max_T <= 0.0:
        raise ValueError("field maxima must be positive")
    if params.noise_P <= 0.0 or params.noise_M <= 0.0:
        raise ValueError("noise levels must be positive")
    if params.torch_epochs < 100:
        raise ValueError("torch_epochs too small")


def order_parameter_scale(temperature_K: np.ndarray, tn_K: float, beta: float) -> np.ndarray:
    """Landau-like scale factor s(T)=max(1-T/TN,0)^beta."""

    x = np.clip(1.0 - temperature_K / tn_K, 0.0, None)
    return np.power(x, beta)


def simulate_dataset(params: MEParams) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate synthetic measurements for (E,H,P,M,T)."""

    rng = np.random.default_rng(params.seed)
    alpha0 = _alpha0_true_array(params)
    p0 = _p0_true_array(params)
    m0 = _m0_true_array(params)

    temperatures = np.linspace(params.temp_min_K, params.temp_max_K, params.n_temps)
    rows: list[dict[str, float]] = []

    for temp in temperatures:
        s_t = float(order_parameter_scale(np.array([temp]), params.tn_true_K, params.beta_true)[0])
        alpha_t = alpha0 * s_t

        for _ in range(params.samples_per_temp):
            e_vec = rng.uniform(-params.e_field_max_MVm, params.e_field_max_MVm, size=2)
            h_vec = rng.uniform(-params.h_field_max_T, params.h_field_max_T, size=2)

            p_vec = p0 + alpha_t @ h_vec + rng.normal(0.0, params.noise_P, size=2)
            m_vec = m0 + alpha_t.T @ e_vec + rng.normal(0.0, params.noise_M, size=2)

            rows.append(
                {
                    "temperature_K": float(temp),
                    "alpha_scale_true": s_t,
                    "E_x_MVm": float(e_vec[0]),
                    "E_y_MVm": float(e_vec[1]),
                    "H_x_T": float(h_vec[0]),
                    "H_y_T": float(h_vec[1]),
                    "P_x": float(p_vec[0]),
                    "P_y": float(p_vec[1]),
                    "M_x": float(m_vec[0]),
                    "M_y": float(m_vec[1]),
                }
            )

    df = pd.DataFrame(rows)
    return temperatures, df


def estimate_tensor_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate alpha(T) from P~H and M~E regressions, then compare reciprocity."""

    out_rows: list[dict[str, float]] = []
    grouped = df.groupby("temperature_K", sort=True)

    for temp, g in grouped:
        h = g[["H_x_T", "H_y_T"]].to_numpy()
        p = g[["P_x", "P_y"]].to_numpy()
        e = g[["E_x_MVm", "E_y_MVm"]].to_numpy()
        m = g[["M_x", "M_y"]].to_numpy()

        reg_p = LinearRegression(fit_intercept=True)
        reg_p.fit(h, p)
        alpha_ph = reg_p.coef_  # shape: (2 outputs, 2 features)
        p0_hat = reg_p.intercept_
        r2_p = float(reg_p.score(h, p))

        reg_m = LinearRegression(fit_intercept=True)
        reg_m.fit(e, m)
        alpha_me = reg_m.coef_
        m0_hat = reg_m.intercept_
        r2_m = float(reg_m.score(e, m))

        reciprocity_fro = float(np.linalg.norm(alpha_ph - alpha_me.T, ord="fro"))
        alpha_sym = 0.5 * (alpha_ph + alpha_me.T)
        alpha_eff = float(np.linalg.norm(alpha_sym, ord="fro") / np.sqrt(alpha_sym.size))

        out_rows.append(
            {
                "temperature_K": float(temp),
                "alpha_scale_true": float(g["alpha_scale_true"].iloc[0]),
                "a11_est": float(alpha_sym[0, 0]),
                "a12_est": float(alpha_sym[0, 1]),
                "a21_est": float(alpha_sym[1, 0]),
                "a22_est": float(alpha_sym[1, 1]),
                "alpha_eff": alpha_eff,
                "r2_P_given_H": r2_p,
                "r2_M_given_E": r2_m,
                "reciprocity_fro": reciprocity_fro,
                "p0_x_est": float(p0_hat[0]),
                "p0_y_est": float(p0_hat[1]),
                "m0_x_est": float(m0_hat[0]),
                "m0_y_est": float(m0_hat[1]),
            }
        )

    return pd.DataFrame(out_rows).sort_values("temperature_K").reset_index(drop=True)


def critical_law(temp_K: np.ndarray, amplitude: float, tn_K: float, beta: float, offset: float) -> np.ndarray:
    """Scalar critical law for effective coupling amplitude."""

    return amplitude * order_parameter_scale(temp_K, tn_K, beta) + offset


def fit_critical_envelope(temp_df: pd.DataFrame, params: MEParams) -> dict[str, float]:
    """Fit alpha_eff(T) to a critical-law envelope via SciPy least squares."""

    t = temp_df["temperature_K"].to_numpy()
    y = temp_df["alpha_eff"].to_numpy()

    amp0 = max(float(np.max(y) - np.min(y)), 0.1)
    x0 = np.array([amp0, params.tn_true_K * 0.95, 0.5, float(np.min(y))], dtype=float)

    lower = np.array([0.0, params.temp_min_K + 1.0, 0.05, -2.0], dtype=float)
    upper = np.array([10.0, params.temp_max_K + 140.0, 2.0, 2.0], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        amp, tn, beta, offset = x
        return critical_law(t, amp, tn, beta, offset) - y

    result = least_squares(
        fun=residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=4000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    amp_fit, tn_fit, beta_fit, offset_fit = [float(v) for v in result.x]
    y_pred = critical_law(t, amp_fit, tn_fit, beta_fit, offset_fit)

    return {
        "amp_fit": amp_fit,
        "tn_fit_K": tn_fit,
        "beta_fit": beta_fit,
        "offset_fit": offset_fit,
        "fit_r2": float(r2_score(y, y_pred)),
        "fit_mae": float(mean_absolute_error(y, y_pred)),
        "success": float(result.success),
    }


def torch_global_refinement(df: pd.DataFrame, params: MEParams) -> dict[str, float | np.ndarray]:
    """Jointly refine alpha0, TN, beta from all samples using PyTorch autograd."""

    torch.manual_seed(params.seed)

    e_obs = torch.tensor(df[["E_x_MVm", "E_y_MVm"]].to_numpy(), dtype=torch.float64)
    h_obs = torch.tensor(df[["H_x_T", "H_y_T"]].to_numpy(), dtype=torch.float64)
    t_obs = torch.tensor(df["temperature_K"].to_numpy(), dtype=torch.float64)
    p_obs = torch.tensor(df[["P_x", "P_y"]].to_numpy(), dtype=torch.float64)
    m_obs = torch.tensor(df[["M_x", "M_y"]].to_numpy(), dtype=torch.float64)

    alpha0 = torch.nn.Parameter(torch.tensor([[1.8, -0.4], [0.2, 1.4]], dtype=torch.float64))
    p0 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    m0 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    raw_tn = torch.nn.Parameter(torch.tensor(_inv_softplus(params.tn_true_K - params.temp_min_K + 15.0)))
    raw_beta = torch.nn.Parameter(torch.tensor(_inv_softplus(0.7)))

    optimizer = torch.optim.Adam(
        [alpha0, p0, m0, raw_tn, raw_beta],
        lr=params.torch_lr,
        weight_decay=params.torch_weight_decay,
    )

    mse_fn = torch.nn.MSELoss()
    temp_floor = float(params.temp_min_K + 5.0)

    for _ in range(params.torch_epochs):
        optimizer.zero_grad(set_to_none=True)

        tn = torch.nn.functional.softplus(raw_tn) + temp_floor
        beta = torch.nn.functional.softplus(raw_beta) + 0.05

        scale = torch.clamp(1.0 - t_obs / tn, min=0.0) ** beta
        alpha_t = scale[:, None, None] * alpha0[None, :, :]

        p_pred = p0[None, :] + torch.einsum("nij,nj->ni", alpha_t, h_obs)
        m_pred = m0[None, :] + torch.einsum("nij,nj->ni", alpha_t.transpose(1, 2), e_obs)

        loss_p = mse_fn(p_pred, p_obs)
        loss_m = mse_fn(m_pred, m_obs)
        reg = 1e-4 * torch.mean(alpha0**2)
        loss = loss_p + loss_m + reg

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        tn = torch.nn.functional.softplus(raw_tn) + temp_floor
        beta = torch.nn.functional.softplus(raw_beta) + 0.05

        scale = torch.clamp(1.0 - t_obs / tn, min=0.0) ** beta
        alpha_t = scale[:, None, None] * alpha0[None, :, :]
        p_pred = p0[None, :] + torch.einsum("nij,nj->ni", alpha_t, h_obs)
        m_pred = m0[None, :] + torch.einsum("nij,nj->ni", alpha_t.transpose(1, 2), e_obs)

        p_pred_np = p_pred.cpu().numpy()
        m_pred_np = m_pred.cpu().numpy()

    p_obs_np = p_obs.cpu().numpy()
    m_obs_np = m_obs.cpu().numpy()

    return {
        "alpha0_fit": alpha0.detach().cpu().numpy(),
        "p0_fit": p0.detach().cpu().numpy(),
        "m0_fit": m0.detach().cpu().numpy(),
        "tn_fit_K": float(tn.item()),
        "beta_fit": float(beta.item()),
        "joint_mse": float(np.mean((p_pred_np - p_obs_np) ** 2 + (m_pred_np - m_obs_np) ** 2)),
        "r2_P": float(r2_score(p_obs_np, p_pred_np, multioutput="variance_weighted")),
        "r2_M": float(r2_score(m_obs_np, m_pred_np, multioutput="variance_weighted")),
    }


def main() -> None:
    params = MEParams()
    check_params(params)

    _, data_df = simulate_dataset(params)
    temp_df = estimate_tensor_by_temperature(data_df)
    critical_fit = fit_critical_envelope(temp_df, params)
    torch_fit = torch_global_refinement(data_df, params)
    ordered_df = temp_df[temp_df["alpha_scale_true"] > 0.08].copy()

    alpha0_true = _alpha0_true_array(params)
    alpha0_fit = torch_fit["alpha0_fit"]
    assert isinstance(alpha0_fit, np.ndarray)

    summary = {
        "n_samples_total": float(len(data_df)),
        "n_temps": float(params.n_temps),
        "mean_r2_P_given_H": float(temp_df["r2_P_given_H"].mean()),
        "mean_r2_M_given_E": float(temp_df["r2_M_given_E"].mean()),
        "mean_r2_P_given_H_ordered": float(ordered_df["r2_P_given_H"].mean()),
        "mean_r2_M_given_E_ordered": float(ordered_df["r2_M_given_E"].mean()),
        "mean_reciprocity_fro": float(temp_df["reciprocity_fro"].mean()),
        "critical_fit_r2": float(critical_fit["fit_r2"]),
        "critical_fit_mae": float(critical_fit["fit_mae"]),
        "critical_tn_fit_K": float(critical_fit["tn_fit_K"]),
        "critical_beta_fit": float(critical_fit["beta_fit"]),
        "torch_tn_fit_K": float(torch_fit["tn_fit_K"]),
        "torch_beta_fit": float(torch_fit["beta_fit"]),
        "torch_joint_mse": float(torch_fit["joint_mse"]),
        "torch_r2_P": float(torch_fit["r2_P"]),
        "torch_r2_M": float(torch_fit["r2_M"]),
        "alpha0_mae": float(mean_absolute_error(alpha0_true.ravel(), alpha0_fit.ravel())),
        "tn_abs_error_K": float(abs(torch_fit["tn_fit_K"] - params.tn_true_K)),
        "beta_abs_error": float(abs(torch_fit["beta_fit"] - params.beta_true)),
    }

    print("Summary:")
    for k, v in summary.items():
        print(f"{k}={v:.6f}")

    print("\nEstimated alpha(T) head:")
    print(
        temp_df[
            [
                "temperature_K",
                "alpha_scale_true",
                "alpha_eff",
                "r2_P_given_H",
                "r2_M_given_E",
                "reciprocity_fro",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    print("\nTorch alpha0_fit:")
    print(pd.DataFrame(alpha0_fit, index=["P_x", "P_y"], columns=["H_x", "H_y"]).to_string())

    # MVP quality gates.
    assert summary["mean_r2_P_given_H_ordered"] > 0.995
    assert summary["mean_r2_M_given_E_ordered"] > 0.995
    assert summary["mean_reciprocity_fro"] < 0.25
    assert summary["critical_fit_r2"] > 0.90
    assert summary["tn_abs_error_K"] < 18.0
    assert summary["beta_abs_error"] < 0.25
    assert summary["torch_joint_mse"] < 0.020
    assert summary["torch_r2_P"] > 0.97
    assert summary["torch_r2_M"] > 0.97
    assert summary["alpha0_mae"] < 0.35


if __name__ == "__main__":
    main()
