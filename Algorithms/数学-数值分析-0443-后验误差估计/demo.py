"""Minimal runnable MVP for a posteriori error estimation (MATH-0443).

Model problem:
    -u''(x) = pi^2 sin(pi x), x in (0,1)
    u(0) = u(1) = 0
Exact solution:
    u(x) = sin(pi x)

We use 1D P1 finite elements on uniform meshes and evaluate a residual-type
a posteriori estimator:
    eta_K^2 = h_K^2 ||f||^2_{L2(K)} + 0.5 h_K (J_left^2 + J_right^2)
where J is the jump of element slope across interior nodes.
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


@dataclass(frozen=True)
class Config:
    """Non-interactive configuration for this MVP."""

    element_counts: tuple[int, ...] = (8, 16, 32, 64, 128)
    assemble_quadrature_order: int = 4
    estimator_quadrature_order: int = 5
    error_quadrature_order: int = 6


@dataclass(frozen=True)
class CaseResult:
    """Diagnostics for one mesh size."""

    n_elements: int
    n_nodes: int
    dof: int
    h_max: float
    l2_error: float
    energy_error: float
    eta: float
    eta_over_error: float
    max_indicator: float


def rhs_f(x: np.ndarray) -> np.ndarray:
    return (math.pi**2) * np.sin(math.pi * x)


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


def exact_du(x: np.ndarray) -> np.ndarray:
    return math.pi * np.cos(math.pi * x)


def make_uniform_mesh(n_elements: int) -> np.ndarray:
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")
    return np.linspace(0.0, 1.0, n_elements + 1)


def solve_poisson_p1(nodes: np.ndarray, quadrature_order: int) -> np.ndarray:
    """Solve -u''=f on [0,1] with P1 FEM and homogeneous Dirichlet BC."""
    n_nodes = nodes.size
    if n_nodes < 3:
        raise ValueError("mesh must contain at least 3 nodes")

    xi_q, w_q = np.polynomial.legendre.leggauss(quadrature_order)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    f_global = np.zeros(n_nodes, dtype=float)

    for e in range(n_nodes - 1):
        xl = nodes[e]
        xr = nodes[e + 1]
        h = xr - xl

        k_local = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float) / h

        x_q = 0.5 * (xl + xr) + 0.5 * h * xi_q
        phi_l = 0.5 * (1.0 - xi_q)
        phi_r = 0.5 * (1.0 + xi_q)
        fq = rhs_f(x_q)

        f_local = np.array(
            [
                np.sum(fq * phi_l * w_q) * 0.5 * h,
                np.sum(fq * phi_r * w_q) * 0.5 * h,
            ],
            dtype=float,
        )

        dofs = (e, e + 1)
        for a, ia in enumerate(dofs):
            f_global[ia] += float(f_local[a])
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


def residual_estimator(
    nodes: np.ndarray,
    u: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, float]:
    """Residual-type estimator per element and globally.

    In 1D with P1 elements, u_h'' = 0 on each element, so the volume residual is
    directly driven by f. Jump residual uses slope jumps at interior nodes.
    """
    n_nodes = nodes.size
    n_elem = n_nodes - 1
    h = nodes[1:] - nodes[:-1]
    slopes = (u[1:] - u[:-1]) / h

    jump_at_node = np.zeros(n_nodes, dtype=float)
    for i in range(1, n_nodes - 1):
        jump_at_node[i] = slopes[i] - slopes[i - 1]

    xi_q, w_q = np.polynomial.legendre.leggauss(quadrature_order)

    eta2 = np.zeros(n_elem, dtype=float)
    for e in range(n_elem):
        xl = nodes[e]
        xr = nodes[e + 1]
        he = xr - xl

        x_q = 0.5 * (xl + xr) + 0.5 * he * xi_q
        f2_l2 = np.sum((rhs_f(x_q) ** 2) * w_q) * 0.5 * he

        jump_left = jump_at_node[e] if e > 0 else 0.0
        jump_right = jump_at_node[e + 1] if (e + 1) < (n_nodes - 1) else 0.0

        eta2[e] = (he**2) * f2_l2 + 0.5 * he * (jump_left**2 + jump_right**2)

    eta = float(np.sqrt(np.sum(eta2)))
    return eta2, eta


def exact_errors(nodes: np.ndarray, u: np.ndarray, quadrature_order: int) -> tuple[float, float]:
    """Compute exact L2 error and energy error ||u'-u_h'||_{L2}."""
    xi_q, w_q = np.polynomial.legendre.leggauss(quadrature_order)

    l2_sq = 0.0
    energy_sq = 0.0

    for e in range(nodes.size - 1):
        xl = nodes[e]
        xr = nodes[e + 1]
        h = xr - xl

        x_q = 0.5 * (xl + xr) + 0.5 * h * xi_q
        phi_l = 0.5 * (1.0 - xi_q)
        phi_r = 0.5 * (1.0 + xi_q)

        uh_q = u[e] * phi_l + u[e + 1] * phi_r
        ue_q = exact_u(x_q)
        l2_sq += np.sum(((uh_q - ue_q) ** 2) * w_q) * 0.5 * h

        duh = (u[e + 1] - u[e]) / h
        due_q = exact_du(x_q)
        energy_sq += np.sum(((duh - due_q) ** 2) * w_q) * 0.5 * h

    return float(np.sqrt(l2_sq)), float(np.sqrt(energy_sq))


def run_case(n_elements: int, cfg: Config) -> CaseResult:
    nodes = make_uniform_mesh(n_elements)
    u = solve_poisson_p1(nodes, quadrature_order=cfg.assemble_quadrature_order)

    eta2, eta = residual_estimator(
        nodes=nodes,
        u=u,
        quadrature_order=cfg.estimator_quadrature_order,
    )
    l2_error, energy_error = exact_errors(
        nodes=nodes,
        u=u,
        quadrature_order=cfg.error_quadrature_order,
    )

    ratio = float(eta / energy_error) if energy_error > 0.0 else float("inf")

    return CaseResult(
        n_elements=n_elements,
        n_nodes=nodes.size,
        dof=nodes.size - 2,
        h_max=float(np.max(nodes[1:] - nodes[:-1])),
        l2_error=l2_error,
        energy_error=energy_error,
        eta=eta,
        eta_over_error=ratio,
        max_indicator=float(np.sqrt(np.max(eta2))),
    )


def run_suite(cfg: Config) -> list[CaseResult]:
    return [run_case(n_elements=n, cfg=cfg) for n in cfg.element_counts]


def validate_results(results: list[CaseResult]) -> None:
    if not results:
        raise AssertionError("no experiment results")

    energy = np.array([r.energy_error for r in results], dtype=float)
    eta = np.array([r.eta for r in results], dtype=float)
    ratios = np.array([r.eta_over_error for r in results], dtype=float)

    if energy.size >= 2:
        assert np.all(np.diff(energy) < 0.0), "energy error is not monotonically decreasing"
    if eta.size >= 2:
        assert np.all(np.diff(eta) < 0.0), "estimator is not monotonically decreasing"

    assert energy[-1] < 3.0e-2, "final mesh is unexpectedly inaccurate"

    for ratio in ratios:
        assert np.isfinite(ratio), "non-finite eta/error ratio"
        assert 3.5 <= ratio <= 6.5, "estimator ratio out of expected range"


def print_report(results: list[CaseResult]) -> None:
    print("A Posteriori Error Estimation MVP (MATH-0443)")
    print("PDE: -u''(x)=pi^2 sin(pi x), u(0)=u(1)=0, exact u=sin(pi x)")
    print("Estimator: residual volume + jump terms (1D P1 FEM)")
    print("-" * 118)
    print(
        "{:<8} {:<8} {:<8} {:<10} {:<14} {:<14} {:<12} {:<12} {:<12}".format(
            "Elems",
            "Nodes",
            "DOF",
            "h_max",
            "L2_error",
            "EnergyErr",
            "eta",
            "eta/error",
            "max_etaK",
        )
    )

    for r in results:
        print(
            "{:<8d} {:<8d} {:<8d} {:<10.4e} {:<14.6e} {:<14.6e} {:<12.6e} {:<12.6f} {:<12.6e}".format(
                r.n_elements,
                r.n_nodes,
                r.dof,
                r.h_max,
                r.l2_error,
                r.energy_error,
                r.eta,
                r.eta_over_error,
                r.max_indicator,
            )
        )

    if len(results) >= 2:
        print("\nObserved rates (energy error vs h halving):")
        for i in range(1, len(results)):
            e0 = results[i - 1]
            e1 = results[i]
            rate = math.log(e0.energy_error / e1.energy_error) / math.log(2.0)
            print(f"Elems={e0.n_elements:>3d} -> {e1.n_elements:>3d}: rate={rate:.3f}")


def main() -> None:
    cfg = Config()
    results = run_suite(cfg)
    print_report(results)
    validate_results(results)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
