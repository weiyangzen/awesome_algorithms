"""Adaptive FEM (1D, P1) minimal runnable MVP.

Model problem:
    -u''(x) = pi^2 sin(pi x), x in (0, 1)
    u(0) = u(1) = 0
Exact solution:
    u(x) = sin(pi x)

Adaptive loop:
    SOLVE -> ESTIMATE -> MARK (Doerfler) -> REFINE (bisection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class IterationStat:
    iteration: int
    elements: int
    nodes: int
    dof: int
    l2_error: float
    h1_semi_error: float
    estimator: float
    marked: int
    max_eta: float


def rhs_f(x: np.ndarray) -> np.ndarray:
    return (math.pi**2) * np.sin(math.pi * x)


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


def exact_du(x: np.ndarray) -> np.ndarray:
    return math.pi * np.cos(math.pi * x)


def solve_poisson_p1(nodes: np.ndarray) -> np.ndarray:
    """Assemble and solve 1D P1 FEM with homogeneous Dirichlet BC."""
    n_nodes = nodes.size
    if n_nodes < 3:
        raise ValueError("Need at least 3 nodes for interior DOFs.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    f_global = np.zeros(n_nodes, dtype=float)

    xi_q, w_q = np.polynomial.legendre.leggauss(4)

    for e in range(n_nodes - 1):
        xl = nodes[e]
        xr = nodes[e + 1]
        h = xr - xl

        k_local = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float) / h

        x_q = 0.5 * (xl + xr) + 0.5 * h * xi_q
        phi_l = 0.5 * (1.0 - xi_q)
        phi_r = 0.5 * (1.0 + xi_q)
        f_q = rhs_f(x_q)
        f_local = np.array(
            [
                np.sum(f_q * phi_l * w_q) * 0.5 * h,
                np.sum(f_q * phi_r * w_q) * 0.5 * h,
            ],
            dtype=float,
        )

        dofs = (e, e + 1)
        for a, ia in enumerate(dofs):
            f_global[ia] += f_local[a]
            for b, ib in enumerate(dofs):
                rows.append(ia)
                cols.append(ib)
                data.append(float(k_local[a, b]))

    free = np.arange(1, n_nodes - 1, dtype=int)
    rhs = f_global[free]
    u = np.zeros(n_nodes, dtype=float)

    if SCIPY_AVAILABLE:
        k_global = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        k_ff = k_global[free][:, free]
        u[free] = spsolve(k_ff, rhs)
    else:
        k_dense = np.zeros((n_nodes, n_nodes), dtype=float)
        for r, c, v in zip(rows, cols, data):
            k_dense[r, c] += v
        u[free] = np.linalg.solve(k_dense[np.ix_(free, free)], rhs)

    return u


def residual_estimator(nodes: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float]:
    """Residual-type a posteriori estimator for 1D Poisson.

    eta_K^2 = h_K^2 ||f||^2_{L2(K)} + 0.5 h_K (J_left^2 + J_right^2)
    where J is the derivative jump at an interior node.
    """
    n_nodes = nodes.size
    n_elem = n_nodes - 1
    eta2 = np.zeros(n_elem, dtype=float)

    h = nodes[1:] - nodes[:-1]
    slopes = (u[1:] - u[:-1]) / h

    jump_at_node = np.zeros(n_nodes, dtype=float)
    for i in range(1, n_nodes - 1):
        jump_at_node[i] = slopes[i] - slopes[i - 1]

    xi_q, w_q = np.polynomial.legendre.leggauss(5)

    for e in range(n_elem):
        xl = nodes[e]
        xr = nodes[e + 1]
        he = xr - xl

        x_q = 0.5 * (xl + xr) + 0.5 * he * xi_q
        f2_l2 = np.sum((rhs_f(x_q) ** 2) * w_q) * 0.5 * he

        left_jump = jump_at_node[e] if e > 0 else 0.0
        right_jump = jump_at_node[e + 1] if (e + 1) < (n_nodes - 1) else 0.0

        eta2[e] = (he**2) * f2_l2 + 0.5 * he * (left_jump**2 + right_jump**2)

    total_eta = float(np.sqrt(np.sum(eta2)))
    return eta2, total_eta


def mark_doerfler(eta2: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Bulk marking: choose minimal set M with sum_{K in M} eta_K^2 >= theta * total."""
    if eta2.size == 0:
        return np.zeros(0, dtype=bool)

    total = float(np.sum(eta2))
    if total <= 0.0:
        mask = np.zeros_like(eta2, dtype=bool)
        mask[int(np.argmax(eta2))] = True
        return mask

    order = np.argsort(-eta2)
    target = theta * total
    accum = 0.0
    chosen: list[int] = []

    for idx in order:
        chosen.append(int(idx))
        accum += float(eta2[idx])
        if accum >= target:
            break

    mask = np.zeros_like(eta2, dtype=bool)
    mask[chosen] = True
    return mask


def bisect_marked_elements(nodes: np.ndarray, marked: np.ndarray) -> np.ndarray:
    """Refine marked elements by midpoint bisection."""
    new_nodes = [float(nodes[0])]

    for e in range(nodes.size - 1):
        xl = float(nodes[e])
        xr = float(nodes[e + 1])

        if marked[e]:
            mid = 0.5 * (xl + xr)
            if mid > new_nodes[-1] + 1e-14:
                new_nodes.append(mid)

        if xr > new_nodes[-1] + 1e-14:
            new_nodes.append(xr)

    return np.array(new_nodes, dtype=float)


def exact_errors(nodes: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Compute exact L2 and H1-seminorm errors with numerical quadrature."""
    xi_q, w_q = np.polynomial.legendre.leggauss(6)

    l2_sq = 0.0
    h1_sq = 0.0

    for e in range(nodes.size - 1):
        xl = nodes[e]
        xr = nodes[e + 1]
        h = xr - xl

        x_q = 0.5 * (xl + xr) + 0.5 * h * xi_q
        phi_l = 0.5 * (1.0 - xi_q)
        phi_r = 0.5 * (1.0 + xi_q)

        uh_q = u[e] * phi_l + u[e + 1] * phi_r
        ue_q = exact_u(x_q)
        diff = uh_q - ue_q
        l2_sq += np.sum((diff**2) * w_q) * 0.5 * h

        duh = (u[e + 1] - u[e]) / h
        due_q = exact_du(x_q)
        h1_sq += np.sum(((duh - due_q) ** 2) * w_q) * 0.5 * h

    return float(np.sqrt(l2_sq)), float(np.sqrt(h1_sq))


def adaptive_fem_1d(
    initial_elements: int = 4,
    max_iterations: int = 10,
    theta: float = 0.5,
    estimator_tol: float = 1e-3,
    max_elements: int = 4096,
) -> tuple[list[IterationStat], np.ndarray, np.ndarray]:
    """Run adaptive h-FEM loop and return iteration statistics + final mesh/solution."""
    if initial_elements < 2:
        raise ValueError("initial_elements must be >= 2")
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must be in (0, 1]")

    nodes = np.linspace(0.0, 1.0, initial_elements + 1)
    history: list[IterationStat] = []

    for it in range(max_iterations):
        u = solve_poisson_p1(nodes)
        eta2, estimator = residual_estimator(nodes, u)
        l2_error, h1_error = exact_errors(nodes, u)

        marked_mask = mark_doerfler(eta2, theta=theta)
        marked_count = int(np.sum(marked_mask))

        history.append(
            IterationStat(
                iteration=it,
                elements=nodes.size - 1,
                nodes=nodes.size,
                dof=nodes.size - 2,
                l2_error=l2_error,
                h1_semi_error=h1_error,
                estimator=estimator,
                marked=marked_count,
                max_eta=float(np.sqrt(np.max(eta2))) if eta2.size > 0 else 0.0,
            )
        )

        if estimator <= estimator_tol:
            break
        if (nodes.size - 1) >= max_elements:
            break

        nodes = bisect_marked_elements(nodes, marked_mask)

    final_u = solve_poisson_p1(nodes)
    return history, nodes, final_u


def main() -> None:
    history, final_nodes, final_u = adaptive_fem_1d(
        initial_elements=4,
        max_iterations=10,
        theta=0.5,
        estimator_tol=1e-3,
        max_elements=4096,
    )

    backend = "scipy.sparse + spsolve" if SCIPY_AVAILABLE else "numpy dense fallback"

    print("Adaptive FEM demo: -u'' = pi^2 sin(pi x), u(0)=u(1)=0")
    print(f"Linear solver backend: {backend}")
    print("-" * 96)
    print(
        f"{'it':>3} | {'elem':>5} | {'nodes':>5} | {'dof':>5} | "
        f"{'L2 error':>12} | {'H1-semi err':>12} | {'estimator':>12} | {'marked':>6}"
    )
    print("-" * 96)

    for row in history:
        print(
            f"{row.iteration:>3d} | {row.elements:>5d} | {row.nodes:>5d} | {row.dof:>5d} | "
            f"{row.l2_error:>12.4e} | {row.h1_semi_error:>12.4e} | "
            f"{row.estimator:>12.4e} | {row.marked:>6d}"
        )

    l2_final, h1_final = exact_errors(final_nodes, final_u)
    print("-" * 96)
    print(
        "Final mesh summary: "
        f"elements={final_nodes.size - 1}, nodes={final_nodes.size}, dof={final_nodes.size - 2}"
    )
    print(f"Final exact errors: L2={l2_final:.6e}, H1-semi={h1_final:.6e}")


if __name__ == "__main__":
    main()
