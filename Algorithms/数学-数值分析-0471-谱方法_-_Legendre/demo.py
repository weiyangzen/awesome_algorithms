"""Legendre spectral (Galerkin) MVP for a 1D elliptic problem.

Problem:
    -u''(x) + u(x) = f(x), x in [-1, 1]
    u(-1) = u(1) = 0

Exact solution used for validation:
    u(x) = exp(x) * (1 - x^2)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    import pandas as pd  # Optional: prettier convergence table output.
except Exception:  # pragma: no cover - fallback is handled in runtime output.
    pd = None


Array = np.ndarray


def exact_solution(x: Array) -> Array:
    return np.exp(x) * (1.0 - x * x)


def forcing_term(x: Array) -> Array:
    # For u = exp(x) * (1 - x^2): -u'' + u = exp(x) * (4x + 2)
    return np.exp(x) * (4.0 * x + 2.0)


def legendre_values_and_derivatives(max_degree: int, x: Array) -> tuple[Array, Array]:
    """Return P_k(x), P'_k(x) for k=0..max_degree at points x."""
    values = np.empty((max_degree + 1, x.size), dtype=float)
    derivs = np.empty((max_degree + 1, x.size), dtype=float)
    for k in range(max_degree + 1):
        basis = np.polynomial.legendre.Legendre.basis(k)
        values[k] = basis(x)
        derivs[k] = basis.deriv()(x)
    return values, derivs


def solve_legendre_galerkin(n_modes: int, n_quad: int | None = None) -> Array:
    """Solve for coefficients c_k in u_N = sum_k c_k * (P_k - P_{k+2})."""
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    quad_n = max(2 * n_modes + 16, 64) if n_quad is None else max(n_quad, 16)
    xq, wq = np.polynomial.legendre.leggauss(quad_n)

    max_degree = n_modes + 1
    P, dP = legendre_values_and_derivatives(max_degree, xq)

    # Basis satisfying Dirichlet boundary conditions on [-1, 1].
    phi = P[:n_modes] - P[2 : n_modes + 2]
    dphi = dP[:n_modes] - dP[2 : n_modes + 2]

    stiffness = np.einsum("iq,jq,q->ij", dphi, dphi, wq)
    mass = np.einsum("iq,jq,q->ij", phi, phi, wq)
    A = stiffness + mass

    rhs = np.einsum("iq,q,q->i", phi, wq, forcing_term(xq))
    coeffs = np.linalg.solve(A, rhs)
    return coeffs


def evaluate_legendre_series(coeffs: Array, x: Array) -> Array:
    n_modes = coeffs.size
    max_degree = n_modes + 1
    P, _ = legendre_values_and_derivatives(max_degree, x)
    phi = P[:n_modes] - P[2 : n_modes + 2]
    return np.dot(coeffs, phi)


def compute_errors(coeffs: Array) -> tuple[float, float]:
    x_dense = np.linspace(-1.0, 1.0, 1201)
    u_true_dense = exact_solution(x_dense)
    u_num_dense = evaluate_legendre_series(coeffs, x_dense)
    l_inf = float(np.max(np.abs(u_num_dense - u_true_dense)))

    # High-order quadrature for L2 norm.
    xq, wq = np.polynomial.legendre.leggauss(300)
    err = evaluate_legendre_series(coeffs, xq) - exact_solution(xq)
    l2 = float(math.sqrt(np.sum(wq * err * err)))
    return l_inf, l2


def convergence_study(modes: Iterable[int]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    prev_err: float | None = None
    prev_n: int | None = None

    for n in modes:
        coeffs = solve_legendre_galerkin(n_modes=n)
        _, l2 = compute_errors(coeffs)

        row: dict[str, float] = {"n_modes": float(n), "L2_error": l2}
        if prev_err is not None and prev_n is not None and l2 > 0.0:
            ratio = prev_err / l2
            p_eff = math.log(ratio) / math.log(float(n) / float(prev_n))
            row["effective_order"] = p_eff
        else:
            row["effective_order"] = float("nan")

        rows.append(row)
        prev_err = l2
        prev_n = n

    return rows


def print_convergence(rows: list[dict[str, float]]) -> None:
    if pd is not None:
        df = pd.DataFrame(rows)
        print("\nConvergence table (Legendre Galerkin):")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
        return

    print("\nConvergence table (Legendre Galerkin):")
    header = f"{'n_modes':>10} {'L2_error':>14} {'effective_order':>18}"
    print(header)
    print("-" * len(header))
    for row in rows:
        n = int(row["n_modes"])
        l2 = row["L2_error"]
        p = row["effective_order"]
        p_txt = "nan" if math.isnan(p) else f"{p:.6e}"
        print(f"{n:>10d} {l2:>14.6e} {p_txt:>18}")


def main() -> None:
    n_modes = 14
    coeffs = solve_legendre_galerkin(n_modes=n_modes)
    l_inf, l2 = compute_errors(coeffs)

    print("Legendre Spectral Method MVP")
    print(f"n_modes = {n_modes}")
    print(f"L_inf error = {l_inf:.6e}")
    print(f"L2 error    = {l2:.6e}")
    print("First 6 coefficients:")
    for idx, c in enumerate(coeffs[:6]):
        print(f"  c[{idx}] = {c:.6e}")

    rows = convergence_study([4, 6, 8, 10, 12, 14])
    print_convergence(rows)


if __name__ == "__main__":
    main()
