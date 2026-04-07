"""h-version FEM MVP: solve 1D Poisson equation on [0, 1].

PDE:
    -u''(x) = f(x), x in (0, 1)
    u(0) = u(1) = 0

Choose exact solution u(x) = sin(pi*x), then f(x) = pi^2*sin(pi*x).
Use piecewise linear finite elements and refine mesh size h.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


def source_f(x: np.ndarray) -> np.ndarray:
    return (math.pi**2) * np.sin(math.pi * x)


@dataclass
class FEMResult:
    n_elem: int
    h: float
    l2_error: float
    h1_semi_error: float
    l2_rate: float
    h1_rate: float


def assemble_system(n_elem: int) -> tuple[np.ndarray, np.ndarray]:
    """Assemble dense global stiffness matrix A and load vector b."""
    nodes = np.linspace(0.0, 1.0, n_elem + 1)
    n_node = n_elem + 1
    a = np.zeros((n_node, n_node), dtype=float)
    b = np.zeros(n_node, dtype=float)

    # 2-point Gauss quadrature on reference interval [-1, 1]
    gauss_xi = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=float)
    gauss_w = np.array([1.0, 1.0], dtype=float)

    for e in range(n_elem):
        i, j = e, e + 1
        x_l, x_r = nodes[i], nodes[j]
        h = x_r - x_l

        # Local stiffness for linear element
        k_local = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float) / h
        f_local = np.zeros(2, dtype=float)

        # Local load via numerical quadrature
        for xi, w in zip(gauss_xi, gauss_w):
            x = 0.5 * (x_l + x_r) + 0.5 * h * xi
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            f_val = source_f(np.array([x], dtype=float))[0]
            jac = 0.5 * h
            f_local[0] += w * f_val * n1 * jac
            f_local[1] += w * f_val * n2 * jac

        # Scatter to global system
        a[i, i] += k_local[0, 0]
        a[i, j] += k_local[0, 1]
        a[j, i] += k_local[1, 0]
        a[j, j] += k_local[1, 1]
        b[i] += f_local[0]
        b[j] += f_local[1]

    return a, b


def solve_poisson_fem(n_elem: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve FEM system with Dirichlet boundary values fixed to zero."""
    nodes = np.linspace(0.0, 1.0, n_elem + 1)
    a, b = assemble_system(n_elem)

    # Apply homogeneous Dirichlet BC by solving only interior unknowns.
    # Boundary values are both zero in this benchmark.
    a_ii = a[1:-1, 1:-1]
    b_i = b[1:-1].copy()
    u_i = np.linalg.solve(a_ii, b_i)

    u = np.zeros(n_elem + 1, dtype=float)
    u[1:-1] = u_i
    return nodes, u


def exact_u_prime(x: np.ndarray) -> np.ndarray:
    return math.pi * np.cos(math.pi * x)


def compute_errors(nodes: np.ndarray, u_num: np.ndarray) -> tuple[float, float]:
    """Return (L2 error, H1 seminorm error) using element-wise quadrature."""
    # 3-point Gauss quadrature on reference interval [-1, 1]
    gauss_xi = np.array(
        [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)],
        dtype=float,
    )
    gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)

    l2_acc = 0.0
    h1_acc = 0.0
    n_elem = len(nodes) - 1
    for e in range(n_elem):
        i, j = e, e + 1
        x_l, x_r = nodes[i], nodes[j]
        h = x_r - x_l
        u_i, u_j = u_num[i], u_num[j]
        du_h = (u_j - u_i) / h

        for xi, w in zip(gauss_xi, gauss_w):
            x = 0.5 * (x_l + x_r) + 0.5 * h * xi
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            u_h = n1 * u_i + n2 * u_j

            u_ex = exact_u(np.array([x], dtype=float))[0]
            du_ex = exact_u_prime(np.array([x], dtype=float))[0]
            jac = 0.5 * h

            l2_acc += w * ((u_h - u_ex) ** 2) * jac
            h1_acc += w * ((du_h - du_ex) ** 2) * jac

    return math.sqrt(l2_acc), math.sqrt(h1_acc)


def run_h_refinement_study(n_list: list[int]) -> list[FEMResult]:
    results: list[FEMResult] = []
    prev_l2 = None
    prev_h1 = None

    for n_elem in n_list:
        nodes, u_num = solve_poisson_fem(n_elem)
        l2_error, h1_semi_error = compute_errors(nodes, u_num)
        h = 1.0 / n_elem

        if prev_l2 is None:
            l2_rate = float("nan")
            h1_rate = float("nan")
        else:
            l2_rate = math.log(prev_l2 / l2_error, 2.0)
            h1_rate = math.log(prev_h1 / h1_semi_error, 2.0)

        results.append(
            FEMResult(
                n_elem=n_elem,
                h=h,
                l2_error=l2_error,
                h1_semi_error=h1_semi_error,
                l2_rate=l2_rate,
                h1_rate=h1_rate,
            )
        )
        prev_l2 = l2_error
        prev_h1 = h1_semi_error

    return results


def main() -> None:
    n_list = [8, 16, 32, 64, 128]
    results = run_h_refinement_study(n_list)

    print("h-version FEM demo for 1D Poisson equation")
    print("Exact solution: u(x)=sin(pi*x), Dirichlet boundary u(0)=u(1)=0")
    print()
    print(
        f"{'n_elem':>8} {'h':>10} {'L2_error':>14} {'L2_rate':>10} "
        f"{'H1_semi_err':>14} {'H1_rate':>10}"
    )
    for r in results:
        l2_rate_text = "-" if math.isnan(r.l2_rate) else f"{r.l2_rate:.4f}"
        h1_rate_text = "-" if math.isnan(r.h1_rate) else f"{r.h1_rate:.4f}"
        print(
            f"{r.n_elem:8d} {r.h:10.6f} {r.l2_error:14.6e} {l2_rate_text:>10} "
            f"{r.h1_semi_error:14.6e} {h1_rate_text:>10}"
        )

    final = results[-1]
    print()
    print(
        "Final mesh summary:"
        f" n_elem={final.n_elem}, h={final.h:.6f}, "
        f"L2_error={final.l2_error:.6e}, H1_semi_error={final.h1_semi_error:.6e}"
    )


if __name__ == "__main__":
    main()
