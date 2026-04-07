"""Minimal runnable MVP for Onsager Reciprocal Relations.

The script synthesizes near-equilibrium linear transport data and verifies:
1) Onsager symmetry at zero magnetic field:      L(0) ~= L(0)^T
2) Onsager-Casimir symmetry with magnetic field: L(B) ~= L(-B)^T

It uses three estimation routes:
- scikit-learn linear regression per magnetic field;
- SciPy constrained fit for the B=0 symmetric matrix;
- PyTorch global fit for symmetric + antisymmetric transport parts.
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
class OnsagerParams:
    """Physical and numerical settings for the MVP."""

    seed: int = 305
    n_samples_per_field: int = 480
    force_max: float = 1.25
    noise_std: float = 0.035

    magnetic_fields: tuple[float, float, float] = (-1.0, 0.0, 1.0)

    l_sym_true: tuple[tuple[float, float], tuple[float, float]] = (
        (1.35, 0.52),
        (0.52, 1.05),
    )
    hall_coeff_true: float = 0.36

    torch_epochs: int = 700
    torch_lr: float = 0.04
    torch_weight_decay: float = 3e-4


def check_params(params: OnsagerParams) -> None:
    if params.n_samples_per_field < 80:
        raise ValueError("n_samples_per_field must be >= 80")
    if params.force_max <= 0.0:
        raise ValueError("force_max must be positive")
    if params.noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    if len(params.magnetic_fields) != 3:
        raise ValueError("magnetic_fields must have exactly three values")
    if set(params.magnetic_fields) != {-1.0, 0.0, 1.0}:
        raise ValueError("magnetic_fields must be exactly (-1, 0, +1)")
    if params.torch_epochs < 200:
        raise ValueError("torch_epochs too small")


def l_sym_true_array(params: OnsagerParams) -> np.ndarray:
    return np.array(params.l_sym_true, dtype=float)


def l_asym_true_array(params: OnsagerParams) -> np.ndarray:
    k = float(params.hall_coeff_true)
    return np.array([[0.0, k], [-k, 0.0]], dtype=float)


def true_l_matrix(params: OnsagerParams, b_field: float) -> np.ndarray:
    return l_sym_true_array(params) + b_field * l_asym_true_array(params)


def simulate_dataset(params: OnsagerParams) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Generate synthetic flux-force pairs for each magnetic field value."""

    rng = np.random.default_rng(params.seed)
    rows: list[dict[str, float]] = []
    true_mats: dict[float, np.ndarray] = {}

    for b in params.magnetic_fields:
        l_mat = true_l_matrix(params, b)
        true_mats[b] = l_mat

        x = rng.uniform(-params.force_max, params.force_max, size=(params.n_samples_per_field, 2))
        j_clean = x @ l_mat.T
        j_noisy = j_clean + rng.normal(0.0, params.noise_std, size=j_clean.shape)

        for i in range(params.n_samples_per_field):
            rows.append(
                {
                    "B": float(b),
                    "X1": float(x[i, 0]),
                    "X2": float(x[i, 1]),
                    "J1": float(j_noisy[i, 0]),
                    "J2": float(j_noisy[i, 1]),
                }
            )

    df = pd.DataFrame(rows)
    return df, true_mats


def estimate_l_by_field(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[float, float], dict[float, float]]:
    """Estimate L for each B using independent multi-output linear regressions."""

    table_rows: list[dict[str, float]] = []
    l_hat: dict[float, np.ndarray] = {}
    r2_map: dict[float, float] = {}
    mae_map: dict[float, float] = {}

    for b, g in df.groupby("B", sort=True):
        x = g[["X1", "X2"]].to_numpy()
        y = g[["J1", "J2"]].to_numpy()

        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)

        l_est = reg.coef_.copy()
        y_pred = reg.predict(x)

        l_hat[float(b)] = l_est
        r2_map[float(b)] = float(r2_score(y, y_pred, multioutput="variance_weighted"))
        mae_map[float(b)] = float(mean_absolute_error(y, y_pred))

        table_rows.append(
            {
                "B": float(b),
                "L11": float(l_est[0, 0]),
                "L12": float(l_est[0, 1]),
                "L21": float(l_est[1, 0]),
                "L22": float(l_est[1, 1]),
                "r2": r2_map[float(b)],
                "mae": mae_map[float(b)],
            }
        )

    table = pd.DataFrame(table_rows).sort_values("B").reset_index(drop=True)
    return table, l_hat, r2_map, mae_map


def fit_symmetric_b0_scipy(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    """Fit a symmetric 2x2 matrix for B=0 using SciPy least squares."""

    g = df[np.isclose(df["B"], 0.0)]
    x = g[["X1", "X2"]].to_numpy()
    y = g[["J1", "J2"]].to_numpy()

    reg0 = LinearRegression(fit_intercept=False)
    reg0.fit(x, y)
    l0 = reg0.coef_
    x0 = np.array([l0[0, 0], 0.5 * (l0[0, 1] + l0[1, 0]), l0[1, 1]], dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        a, b, c = theta
        l_mat = np.array([[a, b], [b, c]], dtype=float)
        y_pred = x @ l_mat.T
        return (y_pred - y).ravel()

    result = least_squares(
        fun=residual,
        x0=x0,
        method="trf",
        max_nfev=3000,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )

    a_fit, b_fit, c_fit = [float(v) for v in result.x]
    l_sym = np.array([[a_fit, b_fit], [b_fit, c_fit]], dtype=float)
    y_fit = x @ l_sym.T

    return {
        "L_sym_fit": l_sym,
        "r2": float(r2_score(y, y_fit, multioutput="variance_weighted")),
        "mae": float(mean_absolute_error(y, y_fit)),
        "cost": float(result.cost),
        "success": float(result.success),
    }


def torch_joint_fit(df: pd.DataFrame, params: OnsagerParams) -> dict[str, float | np.ndarray]:
    """Jointly fit L_sym and Hall coefficient using all B in one model."""

    torch.manual_seed(params.seed)

    x = torch.tensor(df[["X1", "X2"]].to_numpy(), dtype=torch.float64)
    y = torch.tensor(df[["J1", "J2"]].to_numpy(), dtype=torch.float64)
    b = torch.tensor(df["B"].to_numpy(), dtype=torch.float64)

    theta = torch.nn.Parameter(torch.tensor([1.0, 0.4, 1.0, 0.2], dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=params.torch_lr, weight_decay=params.torch_weight_decay)
    mse_fn = torch.nn.MSELoss()

    for _ in range(params.torch_epochs):
        optimizer.zero_grad()

        a, b12, c, k = theta[0], theta[1], theta[2], theta[3]
        l_sym = torch.stack([torch.stack([a, b12]), torch.stack([b12, c])])
        l_asym = torch.stack([torch.stack([torch.tensor(0.0, dtype=torch.float64), k]), torch.stack([-k, torch.tensor(0.0, dtype=torch.float64)])])

        l_all = l_sym.unsqueeze(0) + b.view(-1, 1, 1) * l_asym.unsqueeze(0)
        y_pred = torch.einsum("nij,nj->ni", l_all, x)

        loss_data = mse_fn(y_pred, y)
        loss_reg = 8e-5 * torch.sum(theta**2)
        loss = loss_data + loss_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        a, b12, c, k = [float(v) for v in theta.detach().cpu().numpy()]

    l_sym_fit = np.array([[a, b12], [b12, c]], dtype=float)
    l_asym_fit = np.array([[0.0, k], [-k, 0.0]], dtype=float)

    matrices: dict[float, np.ndarray] = {}
    for b_value in params.magnetic_fields:
        matrices[b_value] = l_sym_fit + b_value * l_asym_fit

    y_pred_np = np.zeros((len(df), 2), dtype=float)
    for idx, row in df.reset_index(drop=True).iterrows():
        l_mat = matrices[float(row["B"])]
        x_vec = np.array([row["X1"], row["X2"]], dtype=float)
        y_pred_np[idx] = l_mat @ x_vec

    y_true_np = df[["J1", "J2"]].to_numpy()

    return {
        "L_sym_fit": l_sym_fit,
        "L_asym_fit": l_asym_fit,
        "hall_coeff_fit": float(k),
        "joint_r2": float(r2_score(y_true_np, y_pred_np, multioutput="variance_weighted")),
        "joint_mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "joint_mse": float(np.mean((y_true_np - y_pred_np) ** 2)),
        "matrices": matrices,
    }


def fmt_matrix(mat: np.ndarray) -> str:
    return np.array2string(mat, precision=4, suppress_small=False)


def main() -> None:
    params = OnsagerParams()
    check_params(params)

    df, true_mats = simulate_dataset(params)
    table, l_hat, r2_map, mae_map = estimate_l_by_field(df)
    scipy_fit = fit_symmetric_b0_scipy(df)
    torch_fit = torch_joint_fit(df, params)

    l_neg = l_hat[-1.0]
    l_zero = l_hat[0.0]
    l_pos = l_hat[1.0]

    reg_zero_symmetry_fro = float(np.linalg.norm(l_zero - l_zero.T, ord="fro"))
    reg_casimir_fro = float(np.linalg.norm(l_pos - l_neg.T, ord="fro"))
    reg_l0_mae_vs_true = float(np.mean(np.abs(l_zero - true_mats[0.0])))

    scipy_l0 = scipy_fit["L_sym_fit"]
    scipy_l0_mae_vs_true = float(np.mean(np.abs(scipy_l0 - true_mats[0.0])))

    torch_mats = torch_fit["matrices"]
    torch_zero_symmetry_fro = float(np.linalg.norm(torch_mats[0.0] - torch_mats[0.0].T, ord="fro"))
    torch_casimir_fro = float(np.linalg.norm(torch_mats[1.0] - torch_mats[-1.0].T, ord="fro"))
    torch_l0_mae_vs_true = float(np.mean(np.abs(torch_mats[0.0] - true_mats[0.0])))
    hall_abs_err = abs(float(torch_fit["hall_coeff_fit"]) - params.hall_coeff_true)

    summary = {
        "n_samples_total": int(len(df)),
        "r2_B=-1": r2_map[-1.0],
        "r2_B=0": r2_map[0.0],
        "r2_B=+1": r2_map[1.0],
        "mae_B=-1": mae_map[-1.0],
        "mae_B=0": mae_map[0.0],
        "mae_B=+1": mae_map[1.0],
        "reg_zero_symmetry_fro": reg_zero_symmetry_fro,
        "reg_casimir_fro": reg_casimir_fro,
        "reg_l0_mae_vs_true": reg_l0_mae_vs_true,
        "scipy_r2_B0": float(scipy_fit["r2"]),
        "scipy_l0_mae_vs_true": scipy_l0_mae_vs_true,
        "torch_joint_r2": float(torch_fit["joint_r2"]),
        "torch_joint_mse": float(torch_fit["joint_mse"]),
        "torch_zero_symmetry_fro": torch_zero_symmetry_fro,
        "torch_casimir_fro": torch_casimir_fro,
        "torch_l0_mae_vs_true": torch_l0_mae_vs_true,
        "hall_abs_err": hall_abs_err,
    }

    print("Onsager Reciprocal Relations MVP")
    print(f"n_samples_per_field={params.n_samples_per_field}, total={len(df)}")
    print("\nEstimated L by scikit-learn (per magnetic field):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nTrue L(B=0):")
    print(fmt_matrix(true_mats[0.0]))
    print("Regression L(B=0):")
    print(fmt_matrix(l_hat[0.0]))
    print("SciPy symmetric L(B=0):")
    print(fmt_matrix(scipy_l0))
    print("PyTorch L(B=0):")
    print(fmt_matrix(torch_mats[0.0]))

    print("\nSummary metrics:")
    for key, value in summary.items():
        if isinstance(value, int):
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: {value:.6f}")

    # Quality gates for the MVP.
    assert min(r2_map.values()) > 0.985, f"R2 too low: {r2_map}"
    assert reg_zero_symmetry_fro < 0.090, f"L(0) symmetry error too large: {reg_zero_symmetry_fro}"
    assert reg_casimir_fro < 0.100, f"Casimir error too large: {reg_casimir_fro}"
    assert float(scipy_fit["r2"]) > 0.985, f"SciPy fit quality too low: {scipy_fit['r2']}"
    assert float(torch_fit["joint_r2"]) > 0.985, f"Torch fit quality too low: {torch_fit['joint_r2']}"
    assert hall_abs_err < 0.060, f"Hall coefficient recovery failed: abs err={hall_abs_err}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
