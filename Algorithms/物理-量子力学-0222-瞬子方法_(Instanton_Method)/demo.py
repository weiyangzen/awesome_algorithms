"""Instanton Method MVP for quantum tunneling in a 1D double-well potential.

Model:
- Potential: V(x) = lambda_ * (x^2 - a^2)^2
- Euclidean instanton: x_I(tau) = a * tanh(k * (tau - tau0)), k = a * sqrt(2 * lambda_ / m)
- Semiclassical splitting estimate (dilute instanton gas, one-loop scale):
  DeltaE_inst ~ 2 * omega0 * sqrt(S0 / (2*pi*hbar)) * exp(-S0 / hbar)

This script compares:
1) Instanton-based energy splitting estimate
2) Numerical splitting from finite-difference Hamiltonian diagonalization

The run is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LinearRegression


@dataclass
class InstantonResult:
    omega0: float
    s0_analytic: float
    s0_numeric: float
    s0_torch: float
    torch_interior_grad_norm: float
    table: pd.DataFrame
    log_fit_slope: float
    log_fit_intercept: float
    log_fit_r2: float


def validate_parameters(mass: float, a: float, lambda_: float) -> None:
    if mass <= 0.0 or a <= 0.0 or lambda_ <= 0.0:
        raise ValueError("mass, a, lambda_ must all be positive.")


def potential(x: np.ndarray, a: float, lambda_: float) -> np.ndarray:
    return lambda_ * (x**2 - a**2) ** 2


def potential_torch(x: torch.Tensor, a: float, lambda_: float) -> torch.Tensor:
    return lambda_ * (x**2 - a**2) ** 2


def omega_small_oscillation(mass: float, a: float, lambda_: float) -> float:
    # Around minima x = ±a: V''(±a) = 8 * lambda_ * a^2, so omega0 = sqrt(V''/m)
    return float(np.sqrt(8.0 * lambda_ * a**2 / mass))


def instanton_profile(tau: np.ndarray, mass: float, a: float, lambda_: float, tau0: float = 0.0) -> np.ndarray:
    k = a * np.sqrt(2.0 * lambda_ / mass)
    return a * np.tanh(k * (tau - tau0))


def instanton_action_analytic(mass: float, a: float, lambda_: float) -> float:
    # S0 = ∫_{-a}^{a} dx * sqrt(2 m V(x)) = (4/3) * a^3 * sqrt(2 m lambda_)
    return float((4.0 / 3.0) * a**3 * np.sqrt(2.0 * mass * lambda_))


def instanton_action_numeric(tau: np.ndarray, x_tau: np.ndarray, mass: float, a: float, lambda_: float) -> float:
    xdot = np.gradient(x_tau, tau)
    lag = 0.5 * mass * xdot**2 + potential(x_tau, a=a, lambda_=lambda_)
    return float(simpson(lag, x=tau))


def instanton_action_torch_and_grad(
    tau: np.ndarray,
    x_tau: np.ndarray,
    mass: float,
    a: float,
    lambda_: float,
) -> tuple[float, float]:
    dt = float(tau[1] - tau[0])
    x = torch.tensor(x_tau, dtype=torch.float64, requires_grad=True)

    dx = (x[1:] - x[:-1]) / dt
    kinetic = 0.5 * mass * dx**2
    v_mid = potential_torch(0.5 * (x[1:] + x[:-1]), a=a, lambda_=lambda_)
    action = torch.sum((kinetic + v_mid) * dt)

    grad = torch.autograd.grad(action, x)[0]
    interior_grad_norm = torch.linalg.norm(grad[1:-1]).item()
    return float(action.item()), float(interior_grad_norm)


def instanton_splitting_estimate(hbar: float, omega0: float, s0: float) -> float:
    if hbar <= 0.0:
        raise ValueError("hbar must be positive.")
    prefactor = 2.0 * omega0 * np.sqrt(s0 / (2.0 * np.pi * hbar))
    return float(prefactor * np.exp(-s0 / hbar))


def numerical_splitting_fdm(
    hbar: float,
    mass: float,
    a: float,
    lambda_: float,
    x_max: float = 3.0,
    n_grid: int = 1200,
) -> tuple[float, float, float]:
    if hbar <= 0.0:
        raise ValueError("hbar must be positive.")
    if n_grid < 100:
        raise ValueError("n_grid too small for stable eigenvalue estimation.")

    x = np.linspace(-x_max, x_max, n_grid)
    dx = float(x[1] - x[0])

    v = potential(x, a=a, lambda_=lambda_)
    kinetic_pref = hbar**2 / (2.0 * mass * dx**2)

    diagonal = 2.0 * kinetic_pref + v
    off_diagonal = -kinetic_pref * np.ones(n_grid - 1, dtype=float)

    h_mat = diags([off_diagonal, diagonal, off_diagonal], offsets=[-1, 0, 1], format="csr")
    e_vals = eigsh(h_mat, k=2, which="SA", return_eigenvectors=False, tol=1e-9)
    e_vals = np.sort(e_vals)

    e0 = float(e_vals[0])
    e1 = float(e_vals[1])
    return e0, e1, float(e1 - e0)


def build_comparison_table(
    hbar_values: Sequence[float],
    mass: float,
    a: float,
    lambda_: float,
    omega0: float,
    s0: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    for hbar in hbar_values:
        e0, e1, delta_num = numerical_splitting_fdm(
            hbar=hbar,
            mass=mass,
            a=a,
            lambda_=lambda_,
            x_max=3.0,
            n_grid=1200,
        )
        delta_inst = instanton_splitting_estimate(hbar=hbar, omega0=omega0, s0=s0)
        rel_err = abs(delta_num - delta_inst) / max(abs(delta_num), 1e-15)
        rows.append(
            {
                "hbar": float(hbar),
                "E0_numeric": e0,
                "E1_numeric": e1,
                "DeltaE_numeric": delta_num,
                "DeltaE_instanton": delta_inst,
                "relative_error": float(rel_err),
            }
        )

    table = pd.DataFrame(rows).sort_values("hbar", ascending=False).reset_index(drop=True)
    return table


def fit_log_splitting_line(table: pd.DataFrame) -> tuple[float, float, float]:
    x = (1.0 / table["hbar"].to_numpy(dtype=float)).reshape(-1, 1)
    y = np.log(table["DeltaE_numeric"].to_numpy(dtype=float))

    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x, y))
    return slope, intercept, r2


def run_instanton_mvp() -> InstantonResult:
    mass = 1.0
    a = 1.4
    lambda_ = 0.65

    validate_parameters(mass=mass, a=a, lambda_=lambda_)

    omega0 = omega_small_oscillation(mass=mass, a=a, lambda_=lambda_)
    s0_analytic = instanton_action_analytic(mass=mass, a=a, lambda_=lambda_)

    tau = np.linspace(-8.0, 8.0, 2401)
    x_tau = instanton_profile(tau=tau, mass=mass, a=a, lambda_=lambda_)
    s0_numeric = instanton_action_numeric(tau=tau, x_tau=x_tau, mass=mass, a=a, lambda_=lambda_)
    s0_torch, grad_norm = instanton_action_torch_and_grad(
        tau=tau,
        x_tau=x_tau,
        mass=mass,
        a=a,
        lambda_=lambda_,
    )

    hbar_values = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30], dtype=float)
    table = build_comparison_table(
        hbar_values=hbar_values,
        mass=mass,
        a=a,
        lambda_=lambda_,
        omega0=omega0,
        s0=s0_analytic,
    )

    slope, intercept, r2 = fit_log_splitting_line(table)

    return InstantonResult(
        omega0=omega0,
        s0_analytic=s0_analytic,
        s0_numeric=s0_numeric,
        s0_torch=s0_torch,
        torch_interior_grad_norm=grad_norm,
        table=table,
        log_fit_slope=slope,
        log_fit_intercept=intercept,
        log_fit_r2=r2,
    )


def main() -> None:
    result = run_instanton_mvp()

    print("Instanton Method MVP: 1D Double-Well Quantum Tunneling")
    print(f"omega0 = {result.omega0:.8f}")
    print(f"S0 (analytic) = {result.s0_analytic:.8f}")
    print(f"S0 (numeric, scipy) = {result.s0_numeric:.8f}")
    print(f"S0 (numeric, torch) = {result.s0_torch:.8f}")
    print(f"|grad S_E| on interior nodes (torch) = {result.torch_interior_grad_norm:.3e}")

    print("\nComparison: numerical splitting vs instanton estimate")
    print(result.table.to_string(index=False, float_format=lambda v: f"{v:.8e}"))

    print("\nlog(DeltaE_numeric) ~ slope * (1/hbar) + intercept")
    print(f"slope = {result.log_fit_slope:.8f}  (expected about -S0 = {-result.s0_analytic:.8f})")
    print(f"intercept = {result.log_fit_intercept:.8f}")
    print(f"R^2 = {result.log_fit_r2:.8f}")

    all_finite = bool(np.isfinite(result.table.to_numpy(dtype=float)).all())
    positive_splitting = bool((result.table["DeltaE_numeric"] > 0.0).all())
    print(f"\ncheck_all_finite = {all_finite}")
    print(f"check_positive_splitting = {positive_splitting}")


if __name__ == "__main__":
    main()
