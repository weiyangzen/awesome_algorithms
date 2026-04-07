"""Feynman Path Integral MVP for a 1D harmonic oscillator in Euclidean time.

The script computes the transition kernel K_E(x_f, x_i; beta) using:
1) Discrete time-sliced path integral
2) Analytic Gaussian integration over intermediate coordinates
3) Comparison against the known exact kernel

The run is deterministic and needs no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import LinAlgError, cholesky, cho_solve
from sklearn.linear_model import LinearRegression


@dataclass
class PathIntegralResult:
    mass: float
    omega: float
    beta: float
    hbar: float
    x_initial: float
    x_final: float
    exact_kernel: float
    convergence_table: pd.DataFrame
    fitted_order: float
    fitted_log_intercept: float
    fitted_r2: float
    torch_reference_slices: int
    torch_kernel: float
    numpy_kernel_at_reference: float
    numpy_torch_abs_diff: float


def validate_parameters(
    mass: float,
    omega: float,
    beta: float,
    hbar: float,
    x_initial: float,
    x_final: float,
    n_slices_list: Sequence[int],
) -> None:
    if mass <= 0.0 or omega <= 0.0 or beta <= 0.0 or hbar <= 0.0:
        raise ValueError("mass, omega, beta, and hbar must be positive.")
    if not np.isfinite([mass, omega, beta, hbar, x_initial, x_final]).all():
        raise ValueError("All scalar parameters must be finite.")
    if len(n_slices_list) < 3:
        raise ValueError("At least three discretization levels are required for convergence fitting.")
    if any(n < 2 for n in n_slices_list):
        raise ValueError("Each n_slices must be >= 2.")


def exact_euclidean_kernel(
    x_initial: float,
    x_final: float,
    mass: float,
    omega: float,
    beta: float,
    hbar: float,
) -> float:
    sinh_term = np.sinh(omega * beta)
    prefactor = np.sqrt(mass * omega / (2.0 * np.pi * hbar * sinh_term))
    exponent = -(
        mass
        * omega
        * ((x_final**2 + x_initial**2) * np.cosh(omega * beta) - 2.0 * x_final * x_initial)
        / (2.0 * hbar * sinh_term)
    )
    return float(prefactor * np.exp(exponent))


def build_discrete_action_system(
    n_slices: int,
    mass: float,
    omega: float,
    beta: float,
    x_initial: float,
    x_final: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Build S_E = 0.5*y^T A y - b^T y + c0 for interior coordinates y."""
    dtau = beta / float(n_slices)
    dim = n_slices - 1

    c_edge = mass / (2.0 * dtau) + 0.25 * dtau * mass * omega * omega
    # Interior x_k appears in two neighboring slices, so the quadratic
    # coefficient in S_E is doubled before mapping to 0.5*y^T*A*y.
    diag = 4.0 * c_edge
    off = -mass / dtau

    a_mat = np.zeros((dim, dim), dtype=np.float64)
    np.fill_diagonal(a_mat, diag)
    if dim > 1:
        idx = np.arange(dim - 1)
        a_mat[idx, idx + 1] = off
        a_mat[idx + 1, idx] = off

    b_vec = np.zeros(dim, dtype=np.float64)
    b_vec[0] = (mass / dtau) * x_initial
    b_vec[-1] = (mass / dtau) * x_final

    c0 = c_edge * (x_initial * x_initial + x_final * x_final)
    return dtau, a_mat, b_vec, float(c0)


def discrete_euclidean_kernel_numpy(
    n_slices: int,
    mass: float,
    omega: float,
    beta: float,
    hbar: float,
    x_initial: float,
    x_final: float,
) -> float:
    dtau, a_mat, b_vec, c0 = build_discrete_action_system(
        n_slices=n_slices,
        mass=mass,
        omega=omega,
        beta=beta,
        x_initial=x_initial,
        x_final=x_final,
    )

    try:
        chol = cholesky(a_mat, lower=True, check_finite=True)
    except LinAlgError as exc:
        raise ValueError("Action Hessian is not positive definite; check parameters.") from exc

    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    solve_vec = cho_solve((chol, True), b_vec, check_finite=True)
    quad_term = float(b_vec @ solve_vec)

    log_prefactor = 0.5 * n_slices * np.log(mass / (2.0 * np.pi * hbar * dtau))
    log_integral = (
        0.5 * (n_slices - 1) * np.log(2.0 * np.pi * hbar)
        - 0.5 * logdet
        + (0.5 * quad_term - c0) / hbar
    )
    return float(np.exp(log_prefactor + log_integral))


def discrete_euclidean_kernel_torch(
    n_slices: int,
    mass: float,
    omega: float,
    beta: float,
    hbar: float,
    x_initial: float,
    x_final: float,
) -> float:
    dtau, a_np, b_np, c0 = build_discrete_action_system(
        n_slices=n_slices,
        mass=mass,
        omega=omega,
        beta=beta,
        x_initial=x_initial,
        x_final=x_final,
    )

    a_mat = torch.tensor(a_np, dtype=torch.float64)
    b_vec = torch.tensor(b_np, dtype=torch.float64)

    sign, logabsdet = torch.linalg.slogdet(a_mat)
    sign_val = float(sign.item())
    if sign_val <= 0.0:
        raise ValueError("Torch check failed: Hessian is not positive definite.")

    solve_vec = torch.linalg.solve(a_mat, b_vec)
    quad_term = float(torch.dot(b_vec, solve_vec).item())

    log_prefactor = 0.5 * n_slices * np.log(mass / (2.0 * np.pi * hbar * dtau))
    log_integral = (
        0.5 * (n_slices - 1) * np.log(2.0 * np.pi * hbar)
        - 0.5 * float(logabsdet.item())
        + (0.5 * quad_term - c0) / hbar
    )
    return float(np.exp(log_prefactor + log_integral))


def build_convergence_table(
    n_slices_list: Sequence[int],
    mass: float,
    omega: float,
    beta: float,
    hbar: float,
    x_initial: float,
    x_final: float,
) -> tuple[pd.DataFrame, float]:
    exact_k = exact_euclidean_kernel(
        x_initial=x_initial,
        x_final=x_final,
        mass=mass,
        omega=omega,
        beta=beta,
        hbar=hbar,
    )

    rows: list[dict[str, float]] = []
    for n_slices in n_slices_list:
        dtau = beta / float(n_slices)
        k_discrete = discrete_euclidean_kernel_numpy(
            n_slices=n_slices,
            mass=mass,
            omega=omega,
            beta=beta,
            hbar=hbar,
            x_initial=x_initial,
            x_final=x_final,
        )
        abs_error = abs(k_discrete - exact_k)
        rel_error = abs_error / max(abs(exact_k), 1e-15)
        rows.append(
            {
                "n_slices": float(n_slices),
                "dtau": float(dtau),
                "kernel_discrete": float(k_discrete),
                "kernel_exact": float(exact_k),
                "abs_error": float(abs_error),
                "rel_error": float(rel_error),
            }
        )

    table = pd.DataFrame(rows)
    return table, exact_k


def fit_convergence_order(convergence_table: pd.DataFrame) -> tuple[float, float, float]:
    # Fit log(abs_error) = c - p * log(n_slices), so convergence order is p.
    n_vals = convergence_table["n_slices"].to_numpy(dtype=np.float64)
    err_vals = convergence_table["abs_error"].to_numpy(dtype=np.float64)

    safe_err = np.maximum(err_vals, 1e-18)
    x = np.log(n_vals).reshape(-1, 1)
    y = np.log(safe_err)

    model = LinearRegression()
    model.fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x, y))
    order = -slope
    return order, intercept, r2


def run_path_integral_mvp() -> PathIntegralResult:
    mass = 1.0
    omega = 1.25
    beta = 1.10
    hbar = 1.0
    x_initial = 0.35
    x_final = -0.20
    n_slices_list = [4, 6, 8, 12, 16, 24, 32, 48, 64]

    validate_parameters(
        mass=mass,
        omega=omega,
        beta=beta,
        hbar=hbar,
        x_initial=x_initial,
        x_final=x_final,
        n_slices_list=n_slices_list,
    )

    convergence_table, exact_k = build_convergence_table(
        n_slices_list=n_slices_list,
        mass=mass,
        omega=omega,
        beta=beta,
        hbar=hbar,
        x_initial=x_initial,
        x_final=x_final,
    )
    fitted_order, fitted_log_intercept, fitted_r2 = fit_convergence_order(convergence_table)

    torch_reference_slices = 32
    torch_kernel = discrete_euclidean_kernel_torch(
        n_slices=torch_reference_slices,
        mass=mass,
        omega=omega,
        beta=beta,
        hbar=hbar,
        x_initial=x_initial,
        x_final=x_final,
    )
    numpy_reference = float(
        convergence_table.loc[
            convergence_table["n_slices"] == float(torch_reference_slices), "kernel_discrete"
        ].iloc[0]
    )

    return PathIntegralResult(
        mass=mass,
        omega=omega,
        beta=beta,
        hbar=hbar,
        x_initial=x_initial,
        x_final=x_final,
        exact_kernel=float(exact_k),
        convergence_table=convergence_table,
        fitted_order=fitted_order,
        fitted_log_intercept=fitted_log_intercept,
        fitted_r2=fitted_r2,
        torch_reference_slices=torch_reference_slices,
        torch_kernel=float(torch_kernel),
        numpy_kernel_at_reference=numpy_reference,
        numpy_torch_abs_diff=abs(float(torch_kernel) - numpy_reference),
    )


def main() -> None:
    result = run_path_integral_mvp()

    print("Feynman Path Integral MVP (Euclidean Harmonic Oscillator)")
    print(
        f"mass={result.mass:.3f}, omega={result.omega:.3f}, beta={result.beta:.3f}, "
        f"hbar={result.hbar:.3f}, x_i={result.x_initial:.3f}, x_f={result.x_final:.3f}"
    )
    print(f"exact_kernel = {result.exact_kernel:.10e}")
    print(
        f"convergence_fit: order={result.fitted_order:.6f}, "
        f"log_intercept={result.fitted_log_intercept:.6f}, R2={result.fitted_r2:.6f}"
    )

    print("\nDiscretized path-integral convergence table:")
    print(result.convergence_table.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    print(
        "\nNumPy vs PyTorch cross-check "
        f"(n_slices={result.torch_reference_slices}): "
        f"numpy={result.numpy_kernel_at_reference:.10e}, "
        f"torch={result.torch_kernel:.10e}, "
        f"abs_diff={result.numpy_torch_abs_diff:.3e}"
    )

    table_all_finite = bool(np.isfinite(result.convergence_table.to_numpy(dtype=float)).all())
    print(f"check_convergence_table_all_finite = {table_all_finite}")


if __name__ == "__main__":
    main()
