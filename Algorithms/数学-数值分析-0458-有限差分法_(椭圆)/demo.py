"""2D elliptic PDE solved by finite differences (five-point stencil).

Problem:
    -Delta u = f  in (0, 1) x (0, 1)
    u = g         on boundary

We choose an analytic solution u*(x, y) = exp(x + y), therefore:
    f(x, y) = -2 * exp(x + y), g = u*|boundary
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency path
    pd = None


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(x + y)


def rhs_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -2.0 * np.exp(x + y)


def boundary_function(x: float, y: float) -> float:
    return float(math.exp(x + y))


def apply_operator(u_vec: np.ndarray, n: int, h: float) -> np.ndarray:
    """Matrix-free application of five-point operator for -Delta."""
    u = u_vec.reshape(n, n)
    au = 4.0 * u.copy()
    au[1:, :] -= u[:-1, :]
    au[:-1, :] -= u[1:, :]
    au[:, 1:] -= u[:, :-1]
    au[:, :-1] -= u[:, 1:]
    au /= h * h
    return au.reshape(-1)


def build_rhs(
    n: int,
    h: float,
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary: Callable[[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-hand side including boundary contributions."""
    x = np.linspace(0.0, 1.0, n + 2)
    y = np.linspace(0.0, 1.0, n + 2)
    x_inner = x[1:-1]
    y_inner = y[1:-1]

    xx_inner, yy_inner = np.meshgrid(x_inner, y_inner, indexing="ij")
    b = rhs(xx_inner, yy_inner).reshape(-1).astype(float)

    inv_h2 = 1.0 / (h * h)
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if i == 0:
                b[idx] += inv_h2 * boundary(0.0, y_inner[j])
            if i == n - 1:
                b[idx] += inv_h2 * boundary(1.0, y_inner[j])
            if j == 0:
                b[idx] += inv_h2 * boundary(x_inner[i], 0.0)
            if j == n - 1:
                b[idx] += inv_h2 * boundary(x_inner[i], 1.0)
    return b, x, y


def conjugate_gradient(
    n: int,
    h: float,
    b: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, int, float]:
    """Solve A x = b using matrix-free Conjugate Gradient."""
    x = np.zeros_like(b)
    r = b - apply_operator(x, n, h)
    p = r.copy()
    rs_old = float(np.dot(r, r))
    b_norm = float(np.linalg.norm(b))
    b_norm = 1.0 if b_norm == 0.0 else b_norm

    if math.sqrt(rs_old) / b_norm < tol:
        return x, 0, math.sqrt(rs_old) / b_norm

    for k in range(1, max_iter + 1):
        ap = apply_operator(p, n, h)
        alpha = rs_old / float(np.dot(p, ap))
        x = x + alpha * p
        r = r - alpha * ap

        rs_new = float(np.dot(r, r))
        rel_res = math.sqrt(rs_new) / b_norm
        if rel_res < tol:
            return x, k, rel_res

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, math.sqrt(rs_old) / b_norm


def solve_once(n: int) -> Dict[str, float]:
    """Solve the PDE on an n x n interior grid and report errors."""
    if n < 2:
        raise ValueError("n must be at least 2.")

    h = 1.0 / (n + 1)
    b, x, y = build_rhs(n, h, rhs_function, boundary_function)
    u_inner, iterations, rel_residual = conjugate_gradient(n, h, b)

    xx_full, yy_full = np.meshgrid(x, y, indexing="ij")
    u_exact = exact_solution(xx_full, yy_full)

    u_num = np.zeros((n + 2, n + 2), dtype=float)
    u_num[0, :] = u_exact[0, :]
    u_num[-1, :] = u_exact[-1, :]
    u_num[:, 0] = u_exact[:, 0]
    u_num[:, -1] = u_exact[:, -1]
    u_num[1:-1, 1:-1] = u_inner.reshape(n, n)

    err_inner = u_num[1:-1, 1:-1] - u_exact[1:-1, 1:-1]
    l2_error = float(np.sqrt(np.mean(err_inner**2)))
    linf_error = float(np.max(np.abs(err_inner)))

    return {
        "n": float(n),
        "h": h,
        "unknowns": float(n * n),
        "cg_iters": float(iterations),
        "rel_residual": rel_residual,
        "l2_error": l2_error,
        "linf_error": linf_error,
    }


def convergence_study(grid_sizes: List[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    prev_l2 = None
    prev_linf = None

    for n in grid_sizes:
        row = solve_once(n)
        if prev_l2 is None:
            row["order_l2"] = float("nan")
            row["order_linf"] = float("nan")
        else:
            row["order_l2"] = math.log(prev_l2 / row["l2_error"], 2.0)
            row["order_linf"] = math.log(prev_linf / row["linf_error"], 2.0)
        prev_l2 = row["l2_error"]
        prev_linf = row["linf_error"]
        rows.append(row)
    return rows


def print_results(rows: List[Dict[str, float]]) -> None:
    if pd is not None:
        frame = pd.DataFrame(rows)
        for col in ["n", "unknowns", "cg_iters"]:
            frame[col] = frame[col].astype(int)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
        return

    header = (
        f"{'n':>5} {'unknowns':>10} {'cg_iters':>10} {'h':>12} "
        f"{'rel_res':>12} {'l2_error':>14} {'linf_error':>14} {'ord_l2':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['n']):5d} {int(row['unknowns']):10d} "
            f"{int(row['cg_iters']):10d} {row['h']:12.4e} {row['rel_residual']:12.4e} "
            f"{row['l2_error']:14.6e} {row['linf_error']:14.6e} {row['order_l2']:9.4f}"
        )


def main() -> None:
    grid_sizes = [8, 16, 32, 64]
    rows = convergence_study(grid_sizes)
    print("Finite Difference Method for 2D Elliptic PDE (-Delta u = f)")
    print("Exact solution: u*(x,y) = exp(x+y)")
    print("Linear solver: matrix-free Conjugate Gradient (NumPy)")
    print_results(rows)


if __name__ == "__main__":
    main()
