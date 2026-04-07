"""Minimal runnable MVP for the nuclear collective model.

This script compares two classic collective limits for even-even nuclei:
1) Rotational model: E(J) = E0 + A*J(J+1) [optionally with centrifugal stretching].
2) Vibrational model: E(n) = E0 + hw*n, with n=J/2 on even-J yrast states.

It demonstrates transparent, non-black-box fitting via:
- scikit-learn linear regression,
- PyTorch gradient optimization,
- SciPy nonlinear curve fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

SEED = 20260407


@dataclass(frozen=True)
class NucleusLevels:
    """Level data for one even-even nucleus (ground-state band style)."""

    name: str
    j_values: np.ndarray
    energies_kev: np.ndarray


def j_term(j_values: np.ndarray) -> np.ndarray:
    """Return J(J+1) for each angular momentum J."""
    j = np.asarray(j_values, dtype=float)
    return j * (j + 1.0)


def rotor_energy(j_values: np.ndarray, e0: float, a_kev: float, b_kev: float = 0.0) -> np.ndarray:
    """Rotational energy model.

    E(J) = E0 + A*x - B*x^2, x=J(J+1)
    b_kev defaults to 0 for pure rigid-rotor limit.
    """
    x = j_term(j_values)
    return e0 + a_kev * x - b_kev * x * x


def vibrator_energy(j_values: np.ndarray, e0: float, hw_kev: float) -> np.ndarray:
    """Vibrational energy model on even-J yrast proxy n=J/2.

    E(n) = E0 + hw*n with n = J/2.
    """
    n = 0.5 * np.asarray(j_values, dtype=float)
    return e0 + hw_kev * n


def fit_rotor_sklearn(levels: NucleusLevels) -> tuple[dict[str, float], np.ndarray]:
    """Fit rigid-rotor parameters (E0, A) with linear regression."""
    x = j_term(levels.j_values).reshape(-1, 1)
    y = levels.energies_kev

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    pred = model.predict(x)
    rmse = math.sqrt(mean_squared_error(y, pred))
    r2 = float(model.score(x, y))

    out = {
        "e0_kev": float(model.intercept_),
        "a_kev": float(model.coef_[0]),
        "rmse_kev": float(rmse),
        "r2": r2,
    }
    return out, pred


def fit_vibrator_sklearn(levels: NucleusLevels) -> tuple[dict[str, float], np.ndarray]:
    """Fit vibrational parameters (E0, hw) with linear regression."""
    n = (0.5 * levels.j_values).reshape(-1, 1)
    y = levels.energies_kev

    model = LinearRegression(fit_intercept=True)
    model.fit(n, y)

    pred = model.predict(n)
    rmse = math.sqrt(mean_squared_error(y, pred))
    r2 = float(model.score(n, y))

    out = {
        "e0_kev": float(model.intercept_),
        "hw_kev": float(model.coef_[0]),
        "rmse_kev": float(rmse),
        "r2": r2,
    }
    return out, pred


def fit_rotor_stretching_scipy(levels: NucleusLevels) -> tuple[dict[str, float], np.ndarray]:
    """Fit E(J)=E0+A*x-B*x^2 with SciPy curve_fit (nonlinear in parameters due to bounds)."""

    x = j_term(levels.j_values)
    y = levels.energies_kev

    def model(x_in: np.ndarray, e0: float, a_kev: float, b_kev: float) -> np.ndarray:
        return e0 + a_kev * x_in - b_kev * x_in * x_in

    initial_a = max(1e-6, float((y[-1] - y[0]) / max(x[-1] - x[0], 1e-6)))
    p0 = np.array([float(y[0]), initial_a, 1e-3], dtype=float)

    popt, _pcov = optimize.curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([-500.0, 0.0, 0.0], [500.0, 500.0, 5.0]),
        maxfev=20_000,
    )

    pred = model(x, *popt)
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    out = {
        "e0_kev": float(popt[0]),
        "a_kev": float(popt[1]),
        "b_kev": float(popt[2]),
        "rmse_kev": float(math.sqrt(sse / len(y))),
        "r2": float(r2),
    }
    return out, pred


def fit_rotor_torch(levels: NucleusLevels, steps: int = 2000, lr: float = 0.03) -> tuple[dict[str, float], np.ndarray]:
    """Fit rigid-rotor parameters via gradient descent in PyTorch.

    Parameterization uses softplus(raw_a) to keep A positive.
    """

    x_np = j_term(levels.j_values)
    y_np = levels.energies_kev

    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    e0 = torch.tensor(float(y_np.min()), dtype=torch.float64, requires_grad=True)
    raw_a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([e0, raw_a], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        a_pos = torch.nn.functional.softplus(raw_a)
        pred = e0 + a_pos * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        a_final = float(torch.nn.functional.softplus(raw_a).cpu().item())
        e0_final = float(e0.cpu().item())

    pred_np = rotor_energy(levels.j_values, e0_final, a_final)
    rmse = float(math.sqrt(mean_squared_error(y_np, pred_np)))
    sst = float(np.sum((y_np - np.mean(y_np)) ** 2))
    sse = float(np.sum((y_np - pred_np) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    out = {
        "e0_kev": e0_final,
        "a_kev": a_final,
        "rmse_kev": rmse,
        "r2": float(r2),
    }
    return out, pred_np


def compute_r42(levels: NucleusLevels) -> float:
    """Compute R4/2 = E(4+)/E(2+) if available, else NaN."""
    j = levels.j_values.astype(int)
    e = levels.energies_kev

    idx2 = np.where(j == 2)[0]
    idx4 = np.where(j == 4)[0]
    if idx2.size == 0 or idx4.size == 0:
        return float("nan")

    e2 = float(e[idx2[0]])
    e4 = float(e[idx4[0]])
    if e2 <= 0.0:
        return float("nan")
    return e4 / e2


def classify_collectivity(r42: float) -> str:
    """Simple rule-of-thumb classification by R4/2 ratio."""
    if not np.isfinite(r42):
        return "unknown"
    if r42 >= 3.0:
        return "rotational-like"
    if r42 <= 2.2:
        return "vibrational-like"
    return "transitional"


def build_samples() -> list[NucleusLevels]:
    """Return two compact datasets (illustrative, near-textbook limits)."""
    return [
        NucleusLevels(
            name="168Er-like (deformed rotor)",
            j_values=np.array([0, 2, 4, 6, 8], dtype=float),
            energies_kev=np.array([0.0, 79.8, 264.7, 548.6, 911.2], dtype=float),
        ),
        NucleusLevels(
            name="112Cd-like (quadrupole vibrator)",
            j_values=np.array([0, 2, 4, 6], dtype=float),
            energies_kev=np.array([0.0, 617.0, 1228.0, 1844.0], dtype=float),
        ),
    ]


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    samples = build_samples()

    model_rows: list[dict[str, float | str]] = []

    print("=== Collective Model MVP ===")
    print("Models: rigid-rotor, rotor+stretching, harmonic vibrator")
    print(f"seed={SEED}\n")

    for levels in samples:
        rotor_lr, pred_rotor_lr = fit_rotor_sklearn(levels)
        vibrator_lr, pred_vibrator_lr = fit_vibrator_sklearn(levels)
        rotor_torch, pred_rotor_torch = fit_rotor_torch(levels)
        rotor_stretch, pred_rotor_stretch = fit_rotor_stretching_scipy(levels)

        r42 = compute_r42(levels)
        tag = classify_collectivity(r42)

        best_model = min(
            [
                ("rotor_lr", rotor_lr["rmse_kev"]),
                ("vibrator_lr", vibrator_lr["rmse_kev"]),
                ("rotor_stretch", rotor_stretch["rmse_kev"]),
                ("rotor_torch", rotor_torch["rmse_kev"]),
            ],
            key=lambda item: item[1],
        )[0]

        inertia_rel = float("nan")
        if rotor_lr["a_kev"] > 0.0:
            inertia_rel = 1.0 / (2.0 * rotor_lr["a_kev"])

        level_table = pd.DataFrame(
            {
                "J": levels.j_values.astype(int),
                "E_obs_keV": levels.energies_kev,
                "E_rotor_lr_keV": pred_rotor_lr,
                "E_rotor_torch_keV": pred_rotor_torch,
                "E_rotor_stretch_keV": pred_rotor_stretch,
                "E_vibrator_keV": pred_vibrator_lr,
            }
        )

        print(f"[Nucleus] {levels.name}")
        print(level_table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print(
            "Diagnostics: "
            f"R4/2={r42:.4f}, tag={tag}, best_model={best_model}, "
            f"rotor_RMSE={rotor_lr['rmse_kev']:.3f} keV, "
            f"vibrator_RMSE={vibrator_lr['rmse_kev']:.3f} keV\n"
        )

        model_rows.append(
            {
                "nucleus": levels.name,
                "R4_over_2": r42,
                "collective_tag_by_R42": tag,
                "best_model_by_rmse": best_model,
                "rotor_A_kev": rotor_lr["a_kev"],
                "rotor_E0_kev": rotor_lr["e0_kev"],
                "rotor_rmse_kev": rotor_lr["rmse_kev"],
                "rotor_r2": rotor_lr["r2"],
                "rotor_torch_A_kev": rotor_torch["a_kev"],
                "rotor_torch_rmse_kev": rotor_torch["rmse_kev"],
                "rotor_stretch_A_kev": rotor_stretch["a_kev"],
                "rotor_stretch_B_kev": rotor_stretch["b_kev"],
                "rotor_stretch_rmse_kev": rotor_stretch["rmse_kev"],
                "vibrator_hw_kev": vibrator_lr["hw_kev"],
                "vibrator_rmse_kev": vibrator_lr["rmse_kev"],
                "vibrator_r2": vibrator_lr["r2"],
                "inertia_relative_hbar2_per_kev": inertia_rel,
            }
        )

    summary = pd.DataFrame(model_rows)

    print("[Model summary]")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))


if __name__ == "__main__":
    main()
