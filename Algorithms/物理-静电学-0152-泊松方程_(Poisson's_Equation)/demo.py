"""泊松方程（静电学）最小可运行示例。

模型方程（均匀介质）：
    ∇²φ = -ρ / ε

令 f = ρ/ε，则等价于：
    -∇²φ = f

本示例在单位方形 [0,1]x[0,1] 上采用齐次 Dirichlet 边界 φ=0，
使用五点有限差分 + 稀疏直接法（SciPy SuperLU）求解，
并用制造解进行误差与收敛阶验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class PoissonCaseResult:
    """Single grid case diagnostics."""

    n_inner: int
    h: float
    l2_error: float
    linf_error: float
    residual_rel: float
    phi_center_num: float
    phi_center_true: float


def build_poisson_matrix(n_inner: int, h: float) -> sp.csc_matrix:
    """Build sparse matrix A for -∇² on interior grid (five-point stencil).

    Unknown ordering uses Fortran-style flattening (`order='F'`), consistent with
    the Kronecker product construction:
        A = (I ⊗ T + T ⊗ I) / h^2,
    where T = tridiag(-1, 2, -1).
    """
    if n_inner < 2:
        raise ValueError("n_inner must be >= 2")
    if h <= 0.0:
        raise ValueError("h must be positive")

    e = np.ones(n_inner, dtype=np.float64)
    t = sp.diags(
        diagonals=[-e, 2.0 * e, -e],
        offsets=[-1, 0, 1],
        shape=(n_inner, n_inner),
        format="csc",
    )
    eye = sp.eye(n_inner, format="csc")
    a = (sp.kron(eye, t, format="csc") + sp.kron(t, eye, format="csc")) / (h * h)
    return a


def manufactured_solution_and_rhs(
    n_inner: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create manufactured solution φ and corresponding f=ρ/ε on interior points.

    Chosen exact solution:
        φ(x,y) = sin(πx) sin(πy)
    Then:
        -∇²φ = 2π² sin(πx) sin(πy) = f
        ρ = ε f
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    h = 1.0 / (n_inner + 1)
    x = np.linspace(0.0, 1.0, n_inner + 2)
    y = np.linspace(0.0, 1.0, n_inner + 2)
    x_inner = x[1:-1]
    y_inner = y[1:-1]

    xx, yy = np.meshgrid(x_inner, y_inner, indexing="ij")
    phi_true = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    f = 2.0 * (np.pi ** 2) * phi_true
    rho = epsilon * f

    return x_inner, y_inner, phi_true, f, rho


def solve_poisson_dirichlet(
    n_inner: int,
    epsilon: float = 1.0,
) -> PoissonCaseResult:
    """Solve one Poisson case and report diagnostics."""
    h = 1.0 / (n_inner + 1)
    a = build_poisson_matrix(n_inner=n_inner, h=h)

    x_inner, y_inner, phi_true, f, _rho = manufactured_solution_and_rhs(
        n_inner=n_inner,
        epsilon=epsilon,
    )

    b = f.ravel(order="F")
    phi_num_vec = spsolve(a, b)

    residual = a @ phi_num_vec - b
    residual_rel = float(np.linalg.norm(residual) / np.linalg.norm(b))

    phi_num = phi_num_vec.reshape((n_inner, n_inner), order="F")
    err = phi_num - phi_true
    l2_error = float(np.sqrt(np.mean(err * err)))
    linf_error = float(np.max(np.abs(err)))

    center_idx = n_inner // 2
    phi_center_num = float(phi_num[center_idx, center_idx])

    x_c = x_inner[center_idx]
    y_c = y_inner[center_idx]
    phi_center_true = float(np.sin(np.pi * x_c) * np.sin(np.pi * y_c))

    return PoissonCaseResult(
        n_inner=n_inner,
        h=h,
        l2_error=l2_error,
        linf_error=linf_error,
        residual_rel=residual_rel,
        phi_center_num=phi_center_num,
        phi_center_true=phi_center_true,
    )


def estimate_orders(hs: List[float], errs: List[float]) -> List[float]:
    """Estimate observed order p from consecutive grid refinements."""
    if len(hs) != len(errs):
        raise ValueError("hs and errs must have the same length")

    orders = [float("nan")]
    for i in range(1, len(hs)):
        p = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
        orders.append(float(p))
    return orders


def main() -> None:
    epsilon = 1.0
    grid_levels = [16, 32, 64]

    results = [solve_poisson_dirichlet(n_inner=n, epsilon=epsilon) for n in grid_levels]

    hs = [r.h for r in results]
    l2s = [r.l2_error for r in results]
    orders = estimate_orders(hs, l2s)

    table = pd.DataFrame(
        {
            "n_inner": [r.n_inner for r in results],
            "h": hs,
            "L2_error": l2s,
            "Linf_error": [r.linf_error for r in results],
            "residual_rel": [r.residual_rel for r in results],
            "order_L2": orders,
        }
    )

    finest = results[-1]
    center_abs_err = abs(finest.phi_center_num - finest.phi_center_true)

    print("=== Poisson Equation MVP (Electrostatics) ===")
    print("PDE         : ∇²φ = -ρ/ε  (equiv. -∇²φ = f, f=ρ/ε)")
    print("Domain      : [0,1] x [0,1]")
    print("Boundary    : φ = 0 on boundary (Dirichlet)")
    print("Manufactured: φ = sin(πx)sin(πy)")
    print(f"epsilon     : {epsilon}")
    print("--- Grid Refinement Diagnostics ---")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("--- Finest Grid Checks ---")
    print(f"n_inner         : {finest.n_inner}")
    print(f"phi_center_num  : {finest.phi_center_num:.10f}")
    print(f"phi_center_true : {finest.phi_center_true:.10f}")
    print(f"center_abs_err  : {center_abs_err:.6e}")

    if np.max(table["residual_rel"].to_numpy()) > 1e-9:
        raise RuntimeError("Relative residual is too large; linear solve quality check failed.")

    if np.isfinite(orders[-1]) and orders[-1] < 1.7:
        raise RuntimeError("Observed convergence order is too low for a 2nd-order stencil.")


if __name__ == "__main__":
    main()
