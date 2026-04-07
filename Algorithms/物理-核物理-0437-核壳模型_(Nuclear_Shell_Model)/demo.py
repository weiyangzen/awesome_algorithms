"""Minimal runnable MVP for the Nuclear Shell Model.

This script demonstrates a transparent shell-model pipeline:
1) Build a compact single-particle orbital table with (N, l, j, capacity).
2) Fit a phenomenological mean-field formula with spin-orbit coupling.
3) Sort orbitals by fitted energy and extract shell closures from large energy gaps.

Model formula used in this MVP:
    E = b0 + bN*N + bL2*l(l+1) + bLS*<l·s>
where <l·s> = 0.5 * (j(j+1)-l(l+1)-3/4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

SEED = 20260407
EMPIRICAL_MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]


@dataclass(frozen=True)
class OrbitalLevel:
    """Single-particle orbital information for shell-model ordering."""

    name: str
    major_n: int
    l: int
    j: float
    capacity: int
    reference_energy_mev: float


def ls_expectation(l: int, j: float) -> float:
    """Return the expectation value of l·s for s=1/2."""
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def build_orbitals() -> list[OrbitalLevel]:
    """Construct a compact phenomenological orbital table.

    Energies are reference targets used to fit the simple model coefficients.
    They are intentionally compact and pedagogical (not a full nuclear data table).
    """

    return [
        OrbitalLevel("1s1/2", 0, 0, 0.5, 2, -40.0),
        OrbitalLevel("1p3/2", 1, 1, 1.5, 4, -30.0),
        OrbitalLevel("1p1/2", 1, 1, 0.5, 2, -26.5),
        OrbitalLevel("1d5/2", 2, 2, 2.5, 6, -20.5),
        OrbitalLevel("2s1/2", 2, 0, 0.5, 2, -17.0),
        OrbitalLevel("1d3/2", 2, 2, 1.5, 4, -15.0),
        OrbitalLevel("1f7/2", 3, 3, 3.5, 8, -11.0),
        OrbitalLevel("2p3/2", 3, 1, 1.5, 4, -9.5),
        OrbitalLevel("1f5/2", 3, 3, 2.5, 6, -7.0),
        OrbitalLevel("2p1/2", 3, 1, 0.5, 2, -4.8),
        OrbitalLevel("1g9/2", 4, 4, 4.5, 10, -4.5),
        OrbitalLevel("1g7/2", 4, 4, 3.5, 8, 1.8),
        OrbitalLevel("2d5/2", 4, 2, 2.5, 6, 3.2),
        OrbitalLevel("2d3/2", 4, 2, 1.5, 4, 4.3),
        OrbitalLevel("3s1/2", 4, 0, 0.5, 2, 4.9),
        OrbitalLevel("1h11/2", 5, 5, 5.5, 12, 6.7),
        OrbitalLevel("1h9/2", 5, 5, 4.5, 10, 8.7),
        OrbitalLevel("2f7/2", 5, 3, 3.5, 8, 10.0),
        OrbitalLevel("1i13/2", 6, 6, 6.5, 14, 12.4),
        OrbitalLevel("3p3/2", 5, 1, 1.5, 4, 13.2),
        OrbitalLevel("2f5/2", 5, 3, 2.5, 6, 14.1),
        OrbitalLevel("3p1/2", 5, 1, 0.5, 2, 14.8),
        OrbitalLevel("1j15/2", 7, 7, 7.5, 16, 20.0),
    ]


def build_feature_matrix(orbitals: list[OrbitalLevel]) -> tuple[np.ndarray, np.ndarray]:
    """Build linear-model features [1, N, l(l+1), <l·s>] and targets."""

    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for orb in orbitals:
        x_rows.append(
            [
                1.0,
                float(orb.major_n),
                float(orb.l * (orb.l + 1)),
                float(ls_expectation(orb.l, orb.j)),
            ]
        )
        y_vals.append(float(orb.reference_energy_mev))

    return np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, R^2, and Spearman rank correlation."""

    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    spearman = float(stats.spearmanr(y_true, y_pred).statistic)
    if not np.isfinite(spearman):
        spearman = 1.0

    return {"rmse_mev": rmse, "r2": float(r2), "spearman_rho": spearman}


def pack_model_row(model_name: str, theta: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    """Pack parameters and metrics into one summary row."""

    metrics = evaluate_regression(y_true, y_pred)
    return {
        "model": model_name,
        "b0": float(theta[0]),
        "bN": float(theta[1]),
        "bL2": float(theta[2]),
        "bLS": float(theta[3]),
        **metrics,
    }


def fit_sklearn_linear(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    """Fit shell-model coefficients with scikit-learn linear regression."""

    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    theta = model.coef_.astype(float)
    pred = model.predict(x)
    return pack_model_row("sklearn_linear", theta, y, pred), theta, pred


def fit_scipy_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    """Fit the same linear model using SciPy least-squares optimization."""

    theta0 = np.linalg.lstsq(x, y, rcond=None)[0]

    def residual(theta: np.ndarray) -> np.ndarray:
        return x @ theta - y

    result = optimize.least_squares(residual, x0=theta0, method="trf")
    theta = result.x.astype(float)
    pred = x @ theta
    return pack_model_row("scipy_least_squares", theta, y, pred), theta, pred


def fit_torch_adam(
    x: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    steps: int = 4000,
    lr: float = 0.03,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    """Fit coefficients with PyTorch + Adam (explicit gradient route)."""

    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred_t = x_t @ theta
        # Tiny L2 regularization keeps updates stable without changing the fit materially.
        loss = torch.mean((pred_t - y_t) ** 2) + 1e-8 * torch.sum(theta[1:] ** 2)
        loss.backward()
        optimizer.step()

    theta_np = theta.detach().cpu().numpy().astype(float)
    pred_np = (x @ theta_np).astype(float)
    return pack_model_row("torch_adam", theta_np, y, pred_np), theta_np, pred_np


def extract_shell_closures(
    orbitals: list[OrbitalLevel],
    energies_mev: np.ndarray,
    top_k: int,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """Sort orbitals by energy and pick top shell gaps as closure candidates."""

    order = np.argsort(energies_mev)
    sorted_orbitals = [orbitals[i] for i in order]
    e_sorted = energies_mev[order]

    capacities = np.asarray([orb.capacity for orb in sorted_orbitals], dtype=int)
    cumulative = np.cumsum(capacities)

    gaps = np.diff(e_sorted)
    gap_to_next = np.append(gaps, np.nan)

    ordered_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(sorted_orbitals) + 1, dtype=int),
            "orbital": [orb.name for orb in sorted_orbitals],
            "N": [orb.major_n for orb in sorted_orbitals],
            "l": [orb.l for orb in sorted_orbitals],
            "j": [orb.j for orb in sorted_orbitals],
            "2j+1": capacities,
            "E_pred_MeV": e_sorted,
            "gap_to_next_MeV": gap_to_next,
            "cum_nucleons": cumulative,
        }
    )

    k = min(top_k, len(gaps))
    top_gap_idx = np.argsort(gaps)[-k:][::-1]
    magic_candidates = sorted(int(cumulative[i]) for i in top_gap_idx)

    gap_table = pd.DataFrame(
        {
            "gap_rank": np.arange(1, k + 1, dtype=int),
            "after_orbital": [sorted_orbitals[i].name for i in top_gap_idx],
            "gap_MeV": gaps[top_gap_idx],
            "magic_candidate": [int(cumulative[i]) for i in top_gap_idx],
        }
    )

    return magic_candidates, ordered_table, gap_table


def compare_magic(predicted: list[int], empirical: list[int]) -> pd.DataFrame:
    """Build a compact comparison report for magic numbers."""

    pred_set = set(predicted)
    emp_set = set(empirical)
    matched = sorted(pred_set & emp_set)

    precision = len(matched) / len(pred_set) if pred_set else 0.0
    recall = len(matched) / len(emp_set) if emp_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    return pd.DataFrame(
        {
            "predicted_magic": [predicted],
            "empirical_magic": [empirical],
            "matched": [matched],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
        }
    )


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    orbitals = build_orbitals()
    x, y = build_feature_matrix(orbitals)

    sk_row, sk_theta, pred_sk = fit_sklearn_linear(x, y)
    sp_row, _sp_theta, pred_sp = fit_scipy_least_squares(x, y)
    th_row, _th_theta, pred_th = fit_torch_adam(x, y, theta0=sk_theta)

    summary = pd.DataFrame([sk_row, sp_row, th_row])
    pred_map = {
        "sklearn_linear": pred_sk,
        "scipy_least_squares": pred_sp,
        "torch_adam": pred_th,
    }

    best_model = str(summary.sort_values("rmse_mev", ascending=True).iloc[0]["model"])
    best_pred = pred_map[best_model]

    orbital_table = pd.DataFrame(
        {
            "orbital": [orb.name for orb in orbitals],
            "N": [orb.major_n for orb in orbitals],
            "l": [orb.l for orb in orbitals],
            "j": [orb.j for orb in orbitals],
            "2j+1": [orb.capacity for orb in orbitals],
            "E_ref_MeV": y,
            "E_pred_sklearn_MeV": pred_sk,
            "E_pred_scipy_MeV": pred_sp,
            "E_pred_torch_MeV": pred_th,
        }
    )

    predicted_magic, ordered_table, gap_table = extract_shell_closures(
        orbitals=orbitals,
        energies_mev=best_pred,
        top_k=len(EMPIRICAL_MAGIC_NUMBERS),
    )
    magic_report = compare_magic(predicted_magic, EMPIRICAL_MAGIC_NUMBERS)

    print("=== Nuclear Shell Model MVP ===")
    print("Energy model: E = b0 + bN*N + bL2*l(l+1) + bLS*<l·s>")
    print(f"seed={SEED}")
    print(f"best_model_by_rmse={best_model}\n")

    print("[Fit summary]")
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    print("\n[Orbital energies: reference vs fitted]")
    print(orbital_table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    print("\n[Best-model sorted orbitals and shell gaps]")
    print(ordered_table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    print("\n[Top shell gaps => closure candidates]")
    print(gap_table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    print("\n[Magic number comparison]")
    print(magic_report.to_string(index=False, float_format=lambda v: f"{v: .6f}"))


if __name__ == "__main__":
    main()
