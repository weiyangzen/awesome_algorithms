"""Minimal, runnable SOCP MVP.

Problem:
    minimize_{x,t} t
    subject to ||A x - b||_2 <= t
               t >= 0

This is a standard second-order cone program (SOCP).
The script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class SOCPResult:
    x: np.ndarray
    t: float
    success: bool
    status: int
    message: str
    n_iter: int
    objective: float
    cone_slack: float


def validate_problem_data(a: np.ndarray, b: np.ndarray) -> None:
    if a.ndim != 2:
        raise ValueError(f"A must be 2D, got shape={a.shape}.")
    if b.ndim != 1:
        raise ValueError(f"b must be 1D, got shape={b.shape}.")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Row mismatch: A has {a.shape[0]} rows, b has {b.shape[0]}.")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("A must have non-zero rows and columns.")
    if not np.all(np.isfinite(a)):
        raise ValueError("A contains non-finite values.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b contains non-finite values.")


def build_socp_problem(
    seed: int = 2026,
    m: int = 40,
    n: int = 8,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    b = a @ x_true + noise_std * rng.normal(size=m)

    validate_problem_data(a, b)
    return a, b, x_true


def soc_constraint_value(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    x = z[:-1]
    t = z[-1]
    residual = a @ x - b
    return float(t - np.linalg.norm(residual))


def soc_constraint_jacobian(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = z[:-1]
    residual = a @ x - b
    residual_norm = float(np.linalg.norm(residual))

    jac = np.zeros_like(z)
    if residual_norm > 1e-12:
        jac[:-1] = -(a.T @ residual) / residual_norm
    jac[-1] = 1.0
    return jac


def solve_socp_least_residual(
    a: np.ndarray,
    b: np.ndarray,
    ftol: float = 1e-12,
    maxiter: int = 500,
) -> SOCPResult:
    validate_problem_data(a, b)
    _, n = a.shape

    def objective(z: np.ndarray) -> float:
        return float(z[-1])

    def objective_grad(z: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(z)
        grad[-1] = 1.0
        return grad

    z0 = np.zeros(n + 1, dtype=float)
    z0[-1] = float(np.linalg.norm(b) + 1.0)

    bounds = [(None, None)] * n + [(0.0, None)]
    constraints = [
        {
            "type": "ineq",
            "fun": lambda z: soc_constraint_value(z, a, b),
            "jac": lambda z: soc_constraint_jacobian(z, a, b),
        }
    ]

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            "SLSQP failed on SOCP instance: "
            f"status={result.status}, message={result.message}"
        )

    z_star = result.x
    x_star = z_star[:-1]
    t_star = float(z_star[-1])
    slack = soc_constraint_value(z_star, a, b)

    return SOCPResult(
        x=x_star,
        t=t_star,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        n_iter=int(result.nit),
        objective=float(result.fun),
        cone_slack=float(slack),
    )


def compute_least_squares_reference(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    x_ls, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual_norm = float(np.linalg.norm(a @ x_ls - b))
    return x_ls, residual_norm


def summarize_and_assert(
    a: np.ndarray,
    b: np.ndarray,
    result: SOCPResult,
    x_ls: np.ndarray,
    ls_residual_norm: float,
    feas_tol: float = 1e-7,
    value_tol: float = 1e-6,
) -> None:
    residual_socp = float(np.linalg.norm(a @ result.x - b))
    obj_gap = abs(result.t - ls_residual_norm)
    x_gap = float(np.linalg.norm(result.x - x_ls))

    print("=== SOCP MVP: minimize t s.t. ||A x - b||_2 <= t ===")
    print(f"success: {result.success}")
    print(f"status: {result.status}")
    print(f"message: {result.message}")
    print(f"iterations: {result.n_iter}")
    print(f"objective t*: {result.t:.12f}")
    print(f"||A x* - b||_2: {residual_socp:.12f}")
    print(f"cone slack (t - ||A x* - b||): {result.cone_slack:.12e}")
    print(f"LS residual norm: {ls_residual_norm:.12f}")
    print(f"|t* - LS|: {obj_gap:.12e}")
    print(f"||x_socp - x_ls||_2: {x_gap:.12e}")

    if result.cone_slack < -feas_tol:
        raise AssertionError(
            f"SOCP solution violates cone feasibility: slack={result.cone_slack:.3e}"
        )
    if obj_gap > value_tol:
        raise AssertionError(
            f"SOCP objective mismatches least-squares residual too much: gap={obj_gap:.3e}"
        )


def main() -> None:
    a, b, x_true = build_socp_problem(seed=2026, m=40, n=8, noise_std=0.05)

    result = solve_socp_least_residual(a, b, ftol=1e-12, maxiter=500)
    x_ls, ls_residual_norm = compute_least_squares_reference(a, b)

    summarize_and_assert(
        a=a,
        b=b,
        result=result,
        x_ls=x_ls,
        ls_residual_norm=ls_residual_norm,
        feas_tol=1e-7,
        value_tol=1e-6,
    )

    true_x_error = float(np.linalg.norm(result.x - x_true))
    print(f"||x_socp - x_true||_2: {true_x_error:.12e}")


if __name__ == "__main__":
    main()
