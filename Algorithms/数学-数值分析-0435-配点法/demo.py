"""Minimal runnable MVP for collocation method (MATH-0435)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class SolveResult:
    """Container for one collocation solve."""

    degree: int
    coeffs: np.ndarray
    collocation_points: np.ndarray
    residual_on_points: np.ndarray
    grid: np.ndarray
    y_num: np.ndarray
    y_exact: np.ndarray
    max_abs_error: float


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Analytic solution for y'' + y = sin(pi x), y(0)=y(1)=0."""
    return np.sin(np.pi * x) / (1.0 - np.pi**2)


def rhs_function(x: np.ndarray) -> np.ndarray:
    """Right-hand side f(x)=sin(pi x)."""
    return np.sin(np.pi * x)


def basis_function(k: int, x: np.ndarray) -> np.ndarray:
    """Basis phi_k(x)=x^k(1-x), k>=1, automatically satisfies boundary values."""
    return (x**k) * (1.0 - x)


def basis_second_derivative(k: int, x: np.ndarray) -> np.ndarray:
    """Second derivative of phi_k(x)=x^k-x^(k+1)."""
    x = np.asarray(x, dtype=float)

    term1 = np.zeros_like(x, dtype=float)
    if k >= 2:
        term1 = k * (k - 1) * (x ** (k - 2))

    term2 = (k + 1) * k * (x ** (k - 1))
    return term1 - term2


def collocation_points(m: int) -> np.ndarray:
    """Uniform interior collocation points x_i=i/(m+1), i=1..m."""
    if m < 1:
        raise ValueError(f"degree m must be >= 1, got {m}")
    idx = np.arange(1, m + 1, dtype=float)
    return idx / (m + 1)


def build_linear_system(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build A c = b from residual equations at collocation points."""
    x_col = collocation_points(m)
    a = np.zeros((m, m), dtype=float)
    b = rhs_function(x_col)

    for i, xi in enumerate(x_col):
        xi_arr = np.asarray([xi], dtype=float)
        for j in range(m):
            k = j + 1
            a[i, j] = (
                basis_second_derivative(k, xi_arr)[0] + basis_function(k, xi_arr)[0]
            )

    return a, b, x_col


def evaluate_trial_solution(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate y_m(x)=sum_{k=1}^m c_k phi_k(x)."""
    y = np.zeros_like(x, dtype=float)
    for j, ck in enumerate(coeffs):
        y += ck * basis_function(j + 1, x)
    return y


def evaluate_residual(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate residual r(x)=y_m''(x)+y_m(x)-f(x)."""
    y = evaluate_trial_solution(x, coeffs)
    y_dd = np.zeros_like(x, dtype=float)
    for j, ck in enumerate(coeffs):
        y_dd += ck * basis_second_derivative(j + 1, x)
    return y_dd + y - rhs_function(x)


def solve_collocation_bvp(m: int, grid_size: int = 400) -> SolveResult:
    """Solve the BVP with degree-m polynomial collocation."""
    if grid_size < 10:
        raise ValueError(f"grid_size must be >= 10, got {grid_size}")

    a, b, x_col = build_linear_system(m)
    coeffs = np.linalg.solve(a, b)

    residual_col = evaluate_residual(x_col, coeffs)
    grid = np.linspace(0.0, 1.0, grid_size + 1)
    y_num = evaluate_trial_solution(grid, coeffs)
    y_ex = exact_solution(grid)
    max_abs_error = float(np.max(np.abs(y_num - y_ex)))

    return SolveResult(
        degree=m,
        coeffs=coeffs,
        collocation_points=x_col,
        residual_on_points=residual_col,
        grid=grid,
        y_num=y_num,
        y_exact=y_ex,
        max_abs_error=max_abs_error,
    )


def run_convergence_demo(degrees: Sequence[int]) -> List[SolveResult]:
    """Run multiple degrees and print convergence table."""
    results: List[SolveResult] = []

    print("Collocation method for BVP: y'' + y = sin(pi x), x in [0, 1]")
    print("Boundary: y(0)=0, y(1)=0")
    print("Exact: y(x)=sin(pi x)/(1-pi^2)")
    print("=" * 86)
    print(" m    max_abs_error         collocation_residual_inf      prev/cur error")
    print("=" * 86)

    prev_err = None
    for m in degrees:
        res = solve_collocation_bvp(m)
        res_inf = float(np.max(np.abs(res.residual_on_points)))
        ratio_str = "-"
        if prev_err is not None and res.max_abs_error > 0.0:
            ratio_str = f"{prev_err / res.max_abs_error:.6f}"

        print(
            f"{m:2d}   {res.max_abs_error:14.6e}      {res_inf:14.6e}            {ratio_str}"
        )

        prev_err = res.max_abs_error
        results.append(res)

    return results


def print_solution_sample(result: SolveResult, rows: int = 8) -> None:
    """Print several grid points for one selected degree."""
    print("-" * 86)
    print(f"Sample solution values (degree m={result.degree}, first {rows} points)")
    print(" idx      x           y_num           y_exact         abs_error")
    print("-" * 86)

    rows = min(rows, len(result.grid))
    for i in range(rows):
        x = result.grid[i]
        y_n = result.y_num[i]
        y_e = result.y_exact[i]
        err = abs(y_n - y_e)
        print(f"{i:3d}   {x:8.4f}   {y_n:13.8f}   {y_e:13.8f}   {err:11.4e}")


def main() -> None:
    degrees = [2, 3, 4, 5, 6, 7]
    results = run_convergence_demo(degrees)

    # Basic self-checks for numerical behavior.
    residual_bounds = [float(np.max(np.abs(r.residual_on_points))) for r in results]
    assert max(residual_bounds) < 1e-10, residual_bounds

    errors = [r.max_abs_error for r in results]
    assert errors[-1] < errors[0], errors
    assert errors[-1] < 3e-5, errors[-1]

    print_solution_sample(results[-1], rows=10)
    print("=" * 86)
    print("All checks passed.")


if __name__ == "__main__":
    main()
