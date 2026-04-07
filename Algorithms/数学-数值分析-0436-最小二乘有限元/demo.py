"""Minimal runnable MVP for Least-Squares Finite Element Method (MATH-0436).

We solve the 1D Poisson problem on [0, 1]:
    -u''(x) = f(x),   u(0)=u(1)=0
by rewriting it as a first-order system:
    p - u' = 0,
    p' + f = 0.

A least-squares functional is minimized over piecewise-linear FE spaces:
    J(u_h, p_h) = ||p_h - u_h'||^2 + ||p_h' + f||^2.

The demo prints residual norms, L2 errors, and convergence rates for mesh
refinement levels. No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Sequence

import numpy as np


RealFn = Callable[[float], float]


@dataclass
class ExperimentConfig:
    """Configuration for deterministic mesh-refinement experiments."""

    mesh_sizes: Sequence[int] = (8, 16, 32, 64)

    def validate(self) -> None:
        if len(self.mesh_sizes) < 2:
            raise ValueError("mesh_sizes must contain at least two levels")
        for n in self.mesh_sizes:
            if not isinstance(n, int) or n < 2:
                raise ValueError("every mesh size must be an integer >= 2")


@dataclass
class SolveResult:
    """Numerical result for one mesh resolution."""

    n_elements: int
    nodes: np.ndarray
    u_nodal: np.ndarray
    p_nodal: np.ndarray
    l2_u_error: float
    l2_p_error: float
    h1_u_semi_error: float
    residual_r1_l2: float
    residual_r2_l2: float


def exact_u(x: float) -> float:
    """Analytical solution u(x) for verification."""
    return math.sin(math.pi * x)


def exact_p(x: float) -> float:
    """Exact first-order variable p(x) = u'(x)."""
    return math.pi * math.cos(math.pi * x)


def forcing_f(x: float) -> float:
    """Right-hand side f(x) in -u''=f for the chosen exact solution."""
    return (math.pi ** 2) * math.sin(math.pi * x)


def gauss_legendre_2() -> tuple[np.ndarray, np.ndarray]:
    """Two-point Gauss quadrature on [-1, 1]."""
    xi = 1.0 / math.sqrt(3.0)
    points = np.array([-xi, xi], dtype=float)
    weights = np.array([1.0, 1.0], dtype=float)
    return points, weights


def idx_u(node: int, n_elements: int) -> int | None:
    """Global index for unknown u(node), excluding Dirichlet endpoints."""
    if node == 0 or node == n_elements:
        return None
    return node - 1


def idx_p(node: int, n_elements: int) -> int:
    """Global index for unknown p(node), all nodes are free."""
    n_u = n_elements - 1
    return n_u + node


def assemble_normal_equations(n_elements: int, forcing: RealFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble normal equations Mx=b from least-squares residuals.

    Unknown vector x packs:
      - interior u nodal values (size n_elements - 1)
      - all p nodal values      (size n_elements + 1)
    so total size is 2*n_elements.
    """
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")

    nodes = np.linspace(0.0, 1.0, n_elements + 1)
    n_u = n_elements - 1
    n_total = 2 * n_elements

    mat = np.zeros((n_total, n_total), dtype=float)
    rhs = np.zeros(n_total, dtype=float)

    q_points, q_weights = gauss_legendre_2()

    for e in range(n_elements):
        left = nodes[e]
        right = nodes[e + 1]
        h = right - left
        element_nodes = (e, e + 1)

        dshape = (-1.0 / h, 1.0 / h)

        for xi, wi in zip(q_points, q_weights):
            shape = (0.5 * (1.0 - xi), 0.5 * (1.0 + xi))
            xq = 0.5 * (left + right) + 0.5 * h * xi
            weight = 0.5 * h * wi

            # Residual r1 = p_h - u_h'
            a = np.zeros(n_total, dtype=float)
            for local_id, node in enumerate(element_nodes):
                n_val = shape[local_id]
                dn_val = dshape[local_id]

                a[idx_p(node, n_elements)] += n_val

                u_global = idx_u(node, n_elements)
                if u_global is not None:
                    a[u_global] += -dn_val
                # Dirichlet values are zero, so no constant-term contribution.

            mat += weight * np.outer(a, a)

            # Residual r2 = p_h' + f
            a = np.zeros(n_total, dtype=float)
            for local_id, node in enumerate(element_nodes):
                dn_val = dshape[local_id]
                a[idx_p(node, n_elements)] += dn_val

            b_const = forcing(xq)
            mat += weight * np.outer(a, a)
            rhs += -weight * a * b_const

    return mat, rhs, nodes


def solve_lsfem_poisson_1d(n_elements: int, forcing: RealFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the least-squares FE normal equations and return (nodes, u, p)."""
    mat, rhs, nodes = assemble_normal_equations(n_elements, forcing)

    if not np.all(np.isfinite(mat)) or not np.all(np.isfinite(rhs)):
        raise RuntimeError("assembly produced non-finite entries")

    # Least-squares normal matrix should be symmetric positive definite here.
    cond = float(np.linalg.cond(mat))
    if not math.isfinite(cond) or cond > 1e13:
        raise RuntimeError(f"normal matrix is ill-conditioned: cond={cond:.3e}")

    sol = np.linalg.solve(mat, rhs)

    n_u = n_elements - 1
    u = np.zeros(n_elements + 1, dtype=float)
    u[1:-1] = sol[:n_u]
    p = sol[n_u:]

    return nodes, u, p


def compute_errors(
    nodes: np.ndarray,
    u_nodal: np.ndarray,
    p_nodal: np.ndarray,
    u_exact: RealFn,
    p_exact: RealFn,
) -> tuple[float, float, float]:
    """Compute L2(u), L2(p), and H1-semi(u) errors via quadrature."""
    n_elements = nodes.size - 1
    q_points, q_weights = gauss_legendre_2()

    u_l2_acc = 0.0
    p_l2_acc = 0.0
    u_h1_semi_acc = 0.0

    for e in range(n_elements):
        left = nodes[e]
        right = nodes[e + 1]
        h = right - left

        u_left = float(u_nodal[e])
        u_right = float(u_nodal[e + 1])
        p_left = float(p_nodal[e])
        p_right = float(p_nodal[e + 1])

        uh_prime = (u_right - u_left) / h

        for xi, wi in zip(q_points, q_weights):
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            xq = 0.5 * (left + right) + 0.5 * h * xi
            weight = 0.5 * h * wi

            uh = n1 * u_left + n2 * u_right
            ph = n1 * p_left + n2 * p_right

            ue = u_exact(xq)
            pe = p_exact(xq)

            u_l2_acc += weight * (uh - ue) ** 2
            p_l2_acc += weight * (ph - pe) ** 2
            u_h1_semi_acc += weight * (uh_prime - pe) ** 2

    return math.sqrt(u_l2_acc), math.sqrt(p_l2_acc), math.sqrt(u_h1_semi_acc)


def compute_residual_norms(nodes: np.ndarray, u_nodal: np.ndarray, p_nodal: np.ndarray, forcing: RealFn) -> tuple[float, float]:
    """Compute L2 norms of r1=(p-u') and r2=(p'+f)."""
    n_elements = nodes.size - 1
    q_points, q_weights = gauss_legendre_2()

    r1_acc = 0.0
    r2_acc = 0.0

    for e in range(n_elements):
        left = nodes[e]
        right = nodes[e + 1]
        h = right - left

        u_left = float(u_nodal[e])
        u_right = float(u_nodal[e + 1])
        p_left = float(p_nodal[e])
        p_right = float(p_nodal[e + 1])

        uh_prime = (u_right - u_left) / h
        ph_prime = (p_right - p_left) / h

        for xi, wi in zip(q_points, q_weights):
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            xq = 0.5 * (left + right) + 0.5 * h * xi
            weight = 0.5 * h * wi

            ph = n1 * p_left + n2 * p_right

            r1 = ph - uh_prime
            r2 = ph_prime + forcing(xq)

            r1_acc += weight * r1 * r1
            r2_acc += weight * r2 * r2

    return math.sqrt(r1_acc), math.sqrt(r2_acc)


def run_refinement_experiment(cfg: ExperimentConfig) -> List[SolveResult]:
    """Run LSFEM solves on multiple meshes and collect diagnostics."""
    cfg.validate()

    results: List[SolveResult] = []
    for n in cfg.mesh_sizes:
        nodes, u_nodal, p_nodal = solve_lsfem_poisson_1d(n, forcing_f)
        l2_u, l2_p, h1_u = compute_errors(nodes, u_nodal, p_nodal, exact_u, exact_p)
        r1_l2, r2_l2 = compute_residual_norms(nodes, u_nodal, p_nodal, forcing_f)

        results.append(
            SolveResult(
                n_elements=n,
                nodes=nodes,
                u_nodal=u_nodal,
                p_nodal=p_nodal,
                l2_u_error=l2_u,
                l2_p_error=l2_p,
                h1_u_semi_error=h1_u,
                residual_r1_l2=r1_l2,
                residual_r2_l2=r2_l2,
            )
        )

    return results


def estimated_orders(errors: Sequence[float]) -> List[float]:
    """Compute log2(e_h / e_h/2) convergence orders."""
    orders: List[float] = []
    for i in range(1, len(errors)):
        prev = errors[i - 1]
        curr = errors[i]
        if prev <= 0.0 or curr <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(math.log(prev / curr, 2.0))
    return orders


def run_checks(results: Sequence[SolveResult]) -> None:
    """Sanity and quality checks for this MVP."""
    if len(results) < 2:
        raise AssertionError("need at least two mesh levels for checks")

    u_errors = [r.l2_u_error for r in results]
    p_errors = [r.l2_p_error for r in results]

    if not np.all(np.isfinite(u_errors)) or not np.all(np.isfinite(p_errors)):
        raise AssertionError("non-finite errors encountered")

    # Basic mesh-refinement expectation: errors should decrease.
    for i in range(1, len(results)):
        if not (u_errors[i] < u_errors[i - 1]):
            raise AssertionError("L2(u) error did not decrease under refinement")
        if not (p_errors[i] < p_errors[i - 1]):
            raise AssertionError("L2(p) error did not decrease under refinement")

    # Ensure final accuracy is in a reasonable range for this small demo.
    if u_errors[-1] > 2e-3:
        raise AssertionError("final L2(u) error is unexpectedly large")
    if p_errors[-1] > 6e-2:
        raise AssertionError("final L2(p) error is unexpectedly large")


def print_report(results: Sequence[SolveResult]) -> None:
    """Print a compact, deterministic summary table."""
    print("n_el   h         ||u-uh||_L2    ||p-ph||_L2    |u-uh|_H1    ||r1||_L2    ||r2||_L2")
    for r in results:
        h = 1.0 / r.n_elements
        print(
            f"{r.n_elements:4d}  {h:0.5f}   {r.l2_u_error:0.6e}  {r.l2_p_error:0.6e}  "
            f"{r.h1_u_semi_error:0.6e}  {r.residual_r1_l2:0.6e}  {r.residual_r2_l2:0.6e}"
        )

    u_orders = estimated_orders([r.l2_u_error for r in results])
    p_orders = estimated_orders([r.l2_p_error for r in results])

    print("\nEstimated refinement orders (log2 ratio):")
    for i, (ou, op) in enumerate(zip(u_orders, p_orders), start=1):
        prev_n = results[i - 1].n_elements
        curr_n = results[i].n_elements
        print(f"  {prev_n:>3d} -> {curr_n:<3d}: order_u={ou:0.3f}, order_p={op:0.3f}")


def main() -> None:
    print("Least-Squares Finite Element Method MVP (MATH-0436)")
    print("Model: 1D Poisson, first-order system, linear FE, normal equations")
    print("=" * 88)

    cfg = ExperimentConfig()
    results = run_refinement_experiment(cfg)
    print_report(results)
    run_checks(results)

    print("=" * 88)
    print("All checks passed.")


if __name__ == "__main__":
    main()
