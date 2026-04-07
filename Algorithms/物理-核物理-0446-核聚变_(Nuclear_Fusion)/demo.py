"""Minimal runnable MVP for Nuclear Fusion (D-T plasma).

This script builds a transparent computation chain:
1) fit a compact D-T reactivity model from anchor data,
2) compute alpha-heating vs bremsstrahlung balance,
3) derive Lawson-style n*tau_E and n*T*tau_E diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

# Physical constants
EV_TO_J = 1.602176634e-19
KEV_TO_J = 1.0e3 * EV_TO_J
MEV_TO_J = 1.0e6 * EV_TO_J

E_FUSION_J = 17.6 * MEV_TO_J
E_ALPHA_J = 3.5 * MEV_TO_J

# Simple bremsstrahlung coefficient (SI-style practical fit)
C_BREM = 5.35e-37  # W * m^3 / (m^-6 * eV^0.5)


def build_anchor_dataset() -> pd.DataFrame:
    """Return a compact D-T reactivity anchor table.

    Reactivity units: m^3/s, Temperature units: keV.
    """
    t_keV = np.array([4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 90], dtype=float)
    sv = np.array(
        [
            3.0e-24,
            1.2e-23,
            3.0e-23,
            6.0e-23,
            1.0e-22,
            1.9e-22,
            3.8e-22,
            5.5e-22,
            7.0e-22,
            9.0e-22,
            1.05e-21,
            1.15e-21,
            1.10e-21,
        ],
        dtype=float,
    )
    return pd.DataFrame({"T_keV": t_keV, "reactivity_m3_per_s": sv})


def dt_reactivity_model(t_keV: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Gamow-style compact surrogate:

    <sigma v>(T) = a*T^2*exp(-b/T^(1/3)) / (1 + c*T + d*T^2)
    params = [a, b, c, d], all positive.
    """
    a, b, c, d = params
    t = np.asarray(t_keV, dtype=float)
    t_safe = np.clip(t, 1e-6, None)

    numerator = a * (t_safe**2) * np.exp(-b / np.cbrt(t_safe))
    denominator = 1.0 + c * t_safe + d * (t_safe**2)
    return numerator / denominator


def _relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ratio = (y_pred - y_true) / np.clip(y_true, 1e-40, None)
    return float(np.sqrt(np.mean(ratio * ratio)))


def fit_reactivity_scipy(t_keV: np.ndarray, sv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit parametric reactivity with SciPy least-squares on log residuals."""

    def residual(theta_log: np.ndarray) -> np.ndarray:
        params = np.exp(theta_log)
        pred = dt_reactivity_model(t_keV, params)
        return np.log(np.clip(pred, 1e-40, None)) - np.log(np.clip(sv, 1e-40, None))

    init = np.log(np.array([2.0e-22, 4.0, 0.01, 0.001], dtype=float))
    lower = np.log(np.array([1e-30, 0.05, 1e-7, 1e-8], dtype=float))
    upper = np.log(np.array([1e-18, 40.0, 2.0, 1.0], dtype=float))

    result = least_squares(
        residual,
        x0=init,
        bounds=(lower, upper),
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    fitted_params = np.exp(result.x)
    pred = dt_reactivity_model(t_keV, fitted_params)
    rrmse = _relative_rmse(sv, pred)
    return fitted_params, pred, rrmse


def fit_reactivity_sklearn(t_keV: np.ndarray, sv: np.ndarray) -> tuple[LinearRegression, np.ndarray, float]:
    """Fit a log-space linear baseline model with sklearn."""
    t = np.asarray(t_keV, dtype=float)
    y = np.log(np.clip(np.asarray(sv, dtype=float), 1e-40, None))

    x = np.column_stack(
        [
            np.log(t),
            1.0 / np.cbrt(t),
            np.log1p(t),
            t,
        ]
    )

    model = LinearRegression()
    model.fit(x, y)

    pred = np.exp(model.predict(x))
    rrmse = _relative_rmse(sv, pred)
    return model, pred, rrmse


def _dt_reactivity_model_torch(t_keV: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Torch version of the same parametric model."""
    a, b, c, d = params[0], params[1], params[2], params[3]
    t_safe = torch.clamp(t_keV, min=1e-6)
    numerator = a * t_safe * t_safe * torch.exp(-b / torch.pow(t_safe, 1.0 / 3.0))
    denominator = 1.0 + c * t_safe + d * t_safe * t_safe
    return numerator / denominator


def fit_reactivity_torch(
    t_keV: np.ndarray,
    sv: np.ndarray,
    init_params: np.ndarray,
    steps: int = 2500,
    lr: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine the parametric fit via PyTorch autograd on log-MSE."""
    t = torch.tensor(t_keV, dtype=torch.float64)
    y = torch.tensor(sv, dtype=torch.float64)

    raw = torch.tensor(np.log(np.clip(init_params, 1e-40, None)), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        params = torch.exp(raw)
        pred = _dt_reactivity_model_torch(t, params)
        loss = torch.mean((torch.log(torch.clamp(pred, min=1e-40)) - torch.log(y)) ** 2)
        loss.backward()
        optimizer.step()

    fitted = torch.exp(raw).detach().cpu().numpy()
    pred_np = dt_reactivity_model(t_keV, fitted)
    rrmse = _relative_rmse(sv, pred_np)
    return fitted, pred_np, rrmse


def build_fit_comparison_table(
    t_keV: np.ndarray,
    sv_true: np.ndarray,
    sv_scipy: np.ndarray,
    sv_sklearn: np.ndarray,
    sv_torch: np.ndarray,
) -> pd.DataFrame:
    """Per-anchor comparison table for all fitting tracks."""
    df = pd.DataFrame(
        {
            "T_keV": t_keV,
            "sv_anchor": sv_true,
            "sv_scipy": sv_scipy,
            "sv_sklearn": sv_sklearn,
            "sv_torch": sv_torch,
        }
    )
    df["err_scipy_pct"] = 100.0 * (df["sv_scipy"] - df["sv_anchor"]) / df["sv_anchor"]
    df["err_sklearn_pct"] = 100.0 * (df["sv_sklearn"] - df["sv_anchor"]) / df["sv_anchor"]
    df["err_torch_pct"] = 100.0 * (df["sv_torch"] - df["sv_anchor"]) / df["sv_anchor"]
    return df


def build_operating_scan(
    t_grid_keV: np.ndarray,
    reactivity: np.ndarray,
    n_e: float = 1.0e20,
    z_eff: float = 1.0,
) -> pd.DataFrame:
    """Compute power balance and Lawson-style diagnostics over temperature grid."""
    t = np.asarray(t_grid_keV, dtype=float)
    sv = np.asarray(reactivity, dtype=float)

    rate = 0.25 * n_e * n_e * sv
    p_fus = rate * E_FUSION_J
    p_alpha = rate * E_ALPHA_J

    t_e_ev = np.clip(t * 1.0e3, 1e-12, None)
    p_brem = C_BREM * z_eff * n_e * n_e * np.sqrt(t_e_ev)
    p_net = p_alpha - p_brem

    thermal_energy_density = 3.0 * n_e * t * KEV_TO_J
    tau_e_min = np.where(p_net > 0.0, thermal_energy_density / p_net, np.inf)

    n_tau = n_e * tau_e_min
    n_t_tau = n_e * t * tau_e_min

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_over_brem = np.where(p_brem > 0.0, p_alpha / p_brem, np.inf)

    return pd.DataFrame(
        {
            "T_keV": t,
            "reactivity_m3_per_s": sv,
            "P_fus_W_m3": p_fus,
            "P_alpha_W_m3": p_alpha,
            "P_brem_W_m3": p_brem,
            "P_net_W_m3": p_net,
            "alpha_over_brem": alpha_over_brem,
            "tau_E_min_s": tau_e_min,
            "n_tau_E_m3s": n_tau,
            "nTtau_keV_s_m3": n_t_tau,
        }
    )


def extract_key_points(scan_df: pd.DataFrame) -> dict[str, float]:
    """Extract key operating points from the scan."""
    mask = np.isfinite(scan_df["tau_E_min_s"].to_numpy())
    if not np.any(mask):
        raise RuntimeError("No positive-net-heating points found in scan.")

    first_idx = int(np.argmax(mask))
    feasible_df = scan_df.loc[mask].reset_index(drop=True)

    idx_min_n_tau = int(np.argmin(feasible_df["n_tau_E_m3s"].to_numpy()))
    idx_min_triple = int(np.argmin(feasible_df["nTtau_keV_s_m3"].to_numpy()))
    idx_best_ratio = int(np.argmax(scan_df["alpha_over_brem"].to_numpy()))

    return {
        "break_even_T_keV": float(scan_df.iloc[first_idx]["T_keV"]),
        "break_even_P_net_W_m3": float(scan_df.iloc[first_idx]["P_net_W_m3"]),
        "min_n_tau_T_keV": float(feasible_df.iloc[idx_min_n_tau]["T_keV"]),
        "min_n_tau_value": float(feasible_df.iloc[idx_min_n_tau]["n_tau_E_m3s"]),
        "min_triple_T_keV": float(feasible_df.iloc[idx_min_triple]["T_keV"]),
        "min_triple_value": float(feasible_df.iloc[idx_min_triple]["nTtau_keV_s_m3"]),
        "max_alpha_over_brem_T_keV": float(scan_df.iloc[idx_best_ratio]["T_keV"]),
        "max_alpha_over_brem": float(scan_df.iloc[idx_best_ratio]["alpha_over_brem"]),
    }


def main() -> None:
    seed = 20260407
    np.random.seed(seed)
    torch.manual_seed(seed)

    anchor_df = build_anchor_dataset()
    t_anchor = anchor_df["T_keV"].to_numpy(dtype=float)
    sv_anchor = anchor_df["reactivity_m3_per_s"].to_numpy(dtype=float)

    scipy_params, sv_scipy, rrmse_scipy = fit_reactivity_scipy(t_anchor, sv_anchor)
    sklearn_model, sv_sklearn, rrmse_sklearn = fit_reactivity_sklearn(t_anchor, sv_anchor)
    torch_params, sv_torch, rrmse_torch = fit_reactivity_torch(t_anchor, sv_anchor, init_params=scipy_params)

    fit_df = build_fit_comparison_table(
        t_keV=t_anchor,
        sv_true=sv_anchor,
        sv_scipy=sv_scipy,
        sv_sklearn=sv_sklearn,
        sv_torch=sv_torch,
    )

    t_grid = np.linspace(4.0, 100.0, 240)
    sv_scan = dt_reactivity_model(t_grid, torch_params)
    scan_df = build_operating_scan(t_grid, sv_scan, n_e=1.0e20, z_eff=1.0)
    key = extract_key_points(scan_df)

    print("=== Nuclear Fusion MVP (D-T, 0D power-balance) ===")
    print(f"seed={seed}")

    print("\n[Reactivity fit quality]")
    print(f"SciPy relative RMSE:   {rrmse_scipy:.4%}")
    print(f"Sklearn relative RMSE: {rrmse_sklearn:.4%}")
    print(f"PyTorch relative RMSE: {rrmse_torch:.4%}")

    print("\n[Reactivity parameters (a, b, c, d)]")
    print("SciPy :", np.array2string(scipy_params, precision=6, separator=", "))
    print("Torch :", np.array2string(torch_params, precision=6, separator=", "))
    print("Sklearn intercept:", float(sklearn_model.intercept_))

    print("\n[Anchor-point comparison]")
    cols = [
        "T_keV",
        "sv_anchor",
        "sv_scipy",
        "sv_sklearn",
        "sv_torch",
        "err_torch_pct",
    ]
    print(fit_df[cols].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    sample_idx = np.linspace(0, len(scan_df) - 1, 10, dtype=int)
    sample_scan = scan_df.iloc[sample_idx][
        [
            "T_keV",
            "P_alpha_W_m3",
            "P_brem_W_m3",
            "P_net_W_m3",
            "tau_E_min_s",
            "nTtau_keV_s_m3",
        ]
    ]

    print("\n[Operating scan sample]")
    print(sample_scan.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    print("\n[Key operating points]")
    for k, v in key.items():
        print(f"{k}: {v:.6g}")

    # Sanity checks for validation workflow
    assert rrmse_torch < 0.20, f"Torch fit too inaccurate: {rrmse_torch:.3f}"
    feasible_mask = np.isfinite(scan_df["tau_E_min_s"].to_numpy())
    assert feasible_mask.sum() > 5, "Too few feasible operating points."
    assert math.isfinite(key["min_triple_value"]) and key["min_triple_value"] > 0.0
    assert 4.0 <= key["min_triple_T_keV"] <= 100.0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
