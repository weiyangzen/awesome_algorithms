"""Geometric multigrid MVP for 1D FEM Poisson problem.

Problem:
    -u''(x) = pi^2 sin(pi x), x in (0, 1)
    u(0) = u(1) = 0
Exact solution:
    u(x) = sin(pi x)

The discrete operator is assembled with linear finite elements.
Multigrid components are implemented explicitly (not black-box):
- weighted Jacobi smoother
- full-weighting restriction
- linear interpolation prolongation
- recursive V-cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class LevelData:
    """Data for one multigrid level."""

    n: int
    h: float
    A: np.ndarray


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def forcing_term(x: np.ndarray) -> np.ndarray:
    return (np.pi**2) * np.sin(np.pi * x)


def assemble_fem_system(num_intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble interior FEM system for 1D Poisson equation.

    Returns:
        A: (n, n) stiffness matrix for interior DoFs.
        b: (n,) load vector for interior DoFs.
        x_interior: interior coordinates.
    """
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2")

    h = 1.0 / num_intervals
    nodes = np.linspace(0.0, 1.0, num_intervals + 1)

    # Global system (including two Dirichlet boundary nodes).
    size_full = num_intervals + 1
    A_full = np.zeros((size_full, size_full), dtype=float)
    b_full = np.zeros(size_full, dtype=float)

    # Two-point Gauss rule on reference element [-1, 1].
    gauss_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    gauss_w = np.array([1.0, 1.0], dtype=float)

    # Local stiffness for linear element on interval length h.
    k_local = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

    for e in range(num_intervals):
        x_left = nodes[e]
        x_right = nodes[e + 1]
        jac = 0.5 * (x_right - x_left)
        mid = 0.5 * (x_left + x_right)

        # Local load by quadrature: b_i += integral f(x) * N_i(x) dx.
        b_local = np.zeros(2, dtype=float)
        for q in range(2):
            xi = gauss_xi[q]
            wq = gauss_w[q]
            xq = mid + jac * xi
            f_val = forcing_term(np.array([xq], dtype=float))[0]
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            b_local[0] += wq * f_val * n1 * jac
            b_local[1] += wq * f_val * n2 * jac

        g0, g1 = e, e + 1
        A_full[g0 : g1 + 1, g0 : g1 + 1] += k_local
        b_full[g0 : g1 + 1] += b_local

    # Remove Dirichlet boundaries (u(0)=u(1)=0).
    A = A_full[1:-1, 1:-1]
    b = b_full[1:-1]
    x_interior = nodes[1:-1]
    return A, b, x_interior


def build_hierarchy(fine_intervals: int, min_coarse_n: int = 3) -> List[LevelData]:
    """Build geometric multigrid hierarchy from fine to coarse."""
    if fine_intervals & (fine_intervals - 1):
        raise ValueError("fine_intervals must be a power of two")

    levels: List[LevelData] = []
    intervals = fine_intervals
    while True:
        A, _, _ = assemble_fem_system(intervals)
        n = A.shape[0]
        levels.append(LevelData(n=n, h=1.0 / intervals, A=A))

        # Stop when coarse grid is small enough for direct solve.
        if n <= min_coarse_n:
            break

        if intervals % 2 != 0:
            raise ValueError("interval hierarchy requires divisibility by 2")
        intervals //= 2

    return levels


def restrict_full_weighting(res_fine: np.ndarray) -> np.ndarray:
    """Restrict residual from fine to coarse grid.

    For this FEM formulation we use the transpose of prolongation (P^T),
    equivalent to 2x classical full-weighting on interior residuals.
    """
    n_f = res_fine.size
    if n_f < 3 or (n_f - 1) % 2 != 0:
        raise ValueError("fine residual size must be 2*n_coarse+1")

    n_c = (n_f - 1) // 2
    res_coarse = np.zeros(n_c, dtype=float)
    for j in range(n_c):
        i = 2 * j + 1
        res_coarse[j] = (
            0.5 * res_fine[i - 1] + 1.0 * res_fine[i] + 0.5 * res_fine[i + 1]
        )
    return res_coarse


def prolong_linear(err_coarse: np.ndarray) -> np.ndarray:
    """Linear interpolation from coarse-grid error to fine grid."""
    n_c = err_coarse.size
    n_f = 2 * n_c + 1

    ext = np.zeros(n_c + 2, dtype=float)
    ext[1:-1] = err_coarse

    err_fine = np.zeros(n_f, dtype=float)
    for I in range(1, n_f + 1):
        if I % 2 == 0:
            # Fine point aligns with coarse interior node.
            err_fine[I - 1] = ext[I // 2]
        else:
            # Midpoint between two coarse nodes.
            left = (I - 1) // 2
            err_fine[I - 1] = 0.5 * (ext[left] + ext[left + 1])
    return err_fine


def matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product via explicit Einstein summation."""
    return np.einsum("ij,j->i", A, x, optimize=True)


def weighted_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    omega: float,
    steps: int,
) -> np.ndarray:
    """Perform weighted Jacobi smoothing iterations."""
    x = x0.copy()
    d_inv = 1.0 / np.diag(A)
    for _ in range(steps):
        r = b - matvec(A, x)
        x = x + omega * d_inv * r
    return x


def v_cycle(
    levels: List[LevelData],
    level_idx: int,
    b: np.ndarray,
    x: np.ndarray,
    omega: float,
    nu_pre: int,
    nu_post: int,
) -> np.ndarray:
    """Recursive V-cycle."""
    level = levels[level_idx]
    A = level.A

    # Coarsest level: solve exactly.
    if level_idx == len(levels) - 1:
        return np.linalg.solve(A, b)

    # Pre-smoothing.
    x = weighted_jacobi(A, b, x, omega=omega, steps=nu_pre)

    # Residual and restriction.
    residual = b - matvec(A, x)
    residual_coarse = restrict_full_weighting(residual)

    # Coarse-grid error solve.
    err_coarse0 = np.zeros_like(residual_coarse)
    err_coarse = v_cycle(
        levels,
        level_idx + 1,
        residual_coarse,
        err_coarse0,
        omega=omega,
        nu_pre=nu_pre,
        nu_post=nu_post,
    )

    # Correction and post-smoothing.
    x = x + prolong_linear(err_coarse)
    x = weighted_jacobi(A, b, x, omega=omega, steps=nu_post)
    return x


def run_multigrid(
    levels: List[LevelData],
    b_fine: np.ndarray,
    max_cycles: int = 20,
    tol: float = 1e-10,
    omega: float = 2.0 / 3.0,
    nu_pre: int = 2,
    nu_post: int = 2,
) -> Dict[str, np.ndarray]:
    """Run repeated V-cycles on finest level until tolerance or max cycles."""
    A_fine = levels[0].A
    x = np.zeros_like(b_fine)

    residual_history = []
    for _ in range(max_cycles):
        r = b_fine - matvec(A_fine, x)
        residual_history.append(np.linalg.norm(r, ord=2))
        if residual_history[-1] < tol:
            break

        x = v_cycle(
            levels,
            level_idx=0,
            b=b_fine,
            x=x,
            omega=omega,
            nu_pre=nu_pre,
            nu_post=nu_post,
        )

    # Record final residual after the last cycle.
    r = b_fine - matvec(A_fine, x)
    residual_history.append(np.linalg.norm(r, ord=2))

    return {
        "u": x,
        "residual_history": np.array(residual_history, dtype=float),
    }


def relative_l2_error(u_num: np.ndarray, x_interior: np.ndarray) -> float:
    u_ex = exact_solution(x_interior)
    h = x_interior[1] - x_interior[0]
    num = np.sqrt(h * np.sum((u_num - u_ex) ** 2))
    den = np.sqrt(h * np.sum(u_ex**2))
    return float(num / den)


def main() -> None:
    fine_intervals = 128
    levels = build_hierarchy(fine_intervals=fine_intervals, min_coarse_n=3)

    A_fine, b_fine, x_interior = assemble_fem_system(fine_intervals)

    result = run_multigrid(
        levels,
        b_fine,
        max_cycles=20,
        tol=1e-10,
        omega=2.0 / 3.0,
        nu_pre=2,
        nu_post=2,
    )

    u_mg = result["u"]
    residual_history = result["residual_history"]

    # Reference direct solve for sanity check.
    u_direct = np.linalg.solve(A_fine, b_fine)

    rel_l2_mg = relative_l2_error(u_mg, x_interior)
    rel_l2_direct = relative_l2_error(u_direct, x_interior)
    max_abs_mg = float(np.max(np.abs(u_mg - exact_solution(x_interior))))

    reduction_factors = residual_history[1:] / residual_history[:-1]

    print("=== Geometric Multigrid (FEM) MVP ===")
    print(f"fine_intervals={fine_intervals}, finest_unknowns={A_fine.shape[0]}")
    print(f"levels={len(levels)} (fine -> coarse unknowns: {[lv.n for lv in levels]})")
    print()

    print("Residual history (L2 norm):")
    for i, rn in enumerate(residual_history):
        if i == 0:
            print(f"  iter {i:02d}: {rn:.6e}")
        else:
            print(f"  iter {i:02d}: {rn:.6e}, factor={reduction_factors[i-1]:.6f}")
    print()

    print("Accuracy summary:")
    print(f"  MG final residual L2        : {residual_history[-1]:.6e}")
    print(f"  MG relative L2 error        : {rel_l2_mg:.6e}")
    print(f"  MG max abs nodal error      : {max_abs_mg:.6e}")
    print(f"  Direct-solve relative L2    : {rel_l2_direct:.6e}")
    print(f"  ||u_mg - u_direct||_inf     : {np.max(np.abs(u_mg - u_direct)):.6e}")


if __name__ == "__main__":
    main()
