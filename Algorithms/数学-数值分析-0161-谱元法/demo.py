"""Spectral Element Method (SEM) MVP for a 1D Poisson problem.

We solve
    -u''(x) = f(x), x in (0, 1)
    u(0) = u(1) = 0

with a manufactured exact solution u*(x)=sin(pi x).
The SEM discretization uses:
- Gauss-Lobatto-Legendre (GLL) nodes in each element,
- Lagrange nodal basis on GLL nodes,
- GLL quadrature for mass/load integration,
- assembled global stiffness system with C0 continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SemResult:
    num_elements: int
    poly_degree: int
    num_nodes: int
    residual_l2: float
    rel_l2_error: float
    max_abs_error: float


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def rhs_function(x: np.ndarray) -> np.ndarray:
    # For u*=sin(pi x), we have -u*'' = pi^2 sin(pi x).
    return (np.pi**2) * np.sin(np.pi * x)


def gll_nodes_weights(poly_degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Lobatto-Legendre nodes/weights for degree p (p+1 nodes)."""
    if poly_degree < 1:
        raise ValueError("poly_degree must be >= 1")

    if poly_degree == 1:
        nodes = np.array([-1.0, 1.0], dtype=float)
        weights = np.array([1.0, 1.0], dtype=float)
        return nodes, weights

    leg_poly = np.polynomial.legendre.Legendre.basis(poly_degree)
    internal = np.sort(leg_poly.deriv().roots())
    nodes = np.concatenate((np.array([-1.0]), internal, np.array([1.0]))).astype(float)

    p_vals = leg_poly(nodes)
    weights = 2.0 / (poly_degree * (poly_degree + 1) * (p_vals**2))
    return nodes, weights.astype(float)


def differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build D_ij = d l_j / d xi evaluated at xi_i for nodal Lagrange basis."""
    n = nodes.size
    bary = np.ones(n, dtype=float)

    for j in range(n):
        prod = 1.0
        for k in range(n):
            if k != j:
                prod *= nodes[j] - nodes[k]
        bary[j] = 1.0 / prod

    dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dmat[i, j] = bary[j] / (bary[i] * (nodes[i] - nodes[j]))
        dmat[i, i] = -np.sum(dmat[i, :])

    return dmat


def reference_stiffness(poly_degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xi, w, K_ref) on reference element [-1,1]."""
    xi, w = gll_nodes_weights(poly_degree)
    dmat = differentiation_matrix(xi)
    wmat = np.diag(w)
    k_ref = dmat.T @ wmat @ dmat
    return xi, w, k_ref


def assemble_global_system(num_elements: int, poly_degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global K u = b for 1D SEM with uniform mesh on [0,1]."""
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    if poly_degree < 1:
        raise ValueError("poly_degree must be >= 1")

    xi, w, k_ref = reference_stiffness(poly_degree)

    p = poly_degree
    n_local = p + 1
    n_global = num_elements * p + 1
    h = 1.0 / num_elements

    k_global = np.zeros((n_global, n_global), dtype=float)
    b_global = np.zeros(n_global, dtype=float)
    x_global = np.zeros(n_global, dtype=float)

    for elem in range(num_elements):
        x_left = elem * h
        jacobian = 0.5 * h

        x_local = x_left + jacobian * (xi + 1.0)
        k_local = (1.0 / jacobian) * k_ref
        b_local = jacobian * w * rhs_function(x_local)

        for i_local in range(n_local):
            i_global = elem * p + i_local
            x_global[i_global] = x_local[i_local]
            b_global[i_global] += b_local[i_local]
            for j_local in range(n_local):
                j_global = elem * p + j_local
                k_global[i_global, j_global] += k_local[i_local, j_local]

    return k_global, b_global, x_global, xi, w


def sem_l2_relative_error(
    u: np.ndarray,
    x: np.ndarray,
    num_elements: int,
    poly_degree: int,
    gll_weights: np.ndarray,
) -> float:
    """Compute relative L2 error using per-element GLL quadrature."""
    p = poly_degree
    jacobian = 0.5 / num_elements

    err_sq = 0.0
    ref_sq = 0.0

    for elem in range(num_elements):
        ids = elem * p + np.arange(p + 1)
        x_e = x[ids]
        u_e = u[ids]
        u_true_e = exact_solution(x_e)
        diff = u_e - u_true_e

        err_sq += jacobian * float(np.sum(gll_weights * diff * diff))
        ref_sq += jacobian * float(np.sum(gll_weights * u_true_e * u_true_e))

    return float(np.sqrt(err_sq / ref_sq))


def solve_sem_poisson(num_elements: int, poly_degree: int) -> SemResult:
    """Solve the model Poisson problem with SEM and return diagnostics."""
    k_global, b_global, x_global, _xi, w = assemble_global_system(num_elements, poly_degree)

    n = x_global.size
    u = np.zeros(n, dtype=float)

    interior = np.arange(1, n - 1)
    k_ii = k_global[np.ix_(interior, interior)]
    b_i = b_global[interior]
    u[interior] = np.linalg.solve(k_ii, b_i)

    # NOTE: use dot() instead of @ to avoid spurious matmul runtime warnings
    # on some BLAS backends for this dense-by-vector multiply.
    residual = k_global.dot(u) - b_global
    residual_l2 = float(np.linalg.norm(residual[interior], ord=2))

    rel_l2_error = sem_l2_relative_error(
        u=u,
        x=x_global,
        num_elements=num_elements,
        poly_degree=poly_degree,
        gll_weights=w,
    )
    max_abs_error = float(np.max(np.abs(u - exact_solution(x_global))))

    return SemResult(
        num_elements=num_elements,
        poly_degree=poly_degree,
        num_nodes=n,
        residual_l2=residual_l2,
        rel_l2_error=rel_l2_error,
        max_abs_error=max_abs_error,
    )


def print_table(title: str, results: list[SemResult]) -> None:
    print(title)
    print("E   p   nodes   residual_l2        rel_l2_error      max_abs_error")
    print("-" * 72)
    for r in results:
        print(
            f"{r.num_elements:<3d} {r.poly_degree:<3d} {r.num_nodes:<7d} "
            f"{r.residual_l2:>14.6e}   {r.rel_l2_error:>14.6e}   {r.max_abs_error:>14.6e}"
        )
    print()


def main() -> None:
    print("Spectral Element Method MVP: 1D Poisson with homogeneous Dirichlet BC")
    print("Exact solution: u*(x)=sin(pi x)")
    print()

    p_refinement_results = [
        solve_sem_poisson(num_elements=4, poly_degree=p) for p in (1, 2, 3, 4, 5, 6)
    ]
    print_table("[Case A] p-refinement (fixed elements E=4)", p_refinement_results)

    h_refinement_results = [
        solve_sem_poisson(num_elements=e, poly_degree=4) for e in (2, 4, 8, 16)
    ]
    print_table("[Case B] h-refinement (fixed polynomial degree p=4)", h_refinement_results)


if __name__ == "__main__":
    main()
