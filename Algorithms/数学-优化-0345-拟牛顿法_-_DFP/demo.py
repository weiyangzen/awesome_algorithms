"""Minimal runnable MVP for DFP quasi-Newton optimization.

The script optimizes the 2D Rosenbrock function from a fixed start point,
implements DFP + Armijo backtracking line search from scratch, and prints
compact convergence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class DFPResult:
    x: Array
    f: float
    grad_norm: float
    iterations: int
    converged: bool
    function_evals: int
    gradient_evals: int
    history: List[Tuple[int, float, float, float, float]]


def rosenbrock(x: Array) -> float:
    """Rosenbrock function in n-dimensions."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_grad(x: Array) -> Array:
    """Gradient of Rosenbrock function in n-dimensions."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    g[-1] = 200.0 * (x[-1] - x[-2] ** 2)
    if x.size > 2:
        g[1:-1] = (
            200.0 * (x[1:-1] - x[:-2] ** 2)
            - 400.0 * x[1:-1] * (x[2:] - x[1:-1] ** 2)
            - 2.0 * (1.0 - x[1:-1])
        )
    return g


def backtracking_line_search(
    f: Callable[[Array], float],
    x: Array,
    fx: float,
    g: Array,
    p: Array,
    initial_step: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    max_backtracks: int = 40,
) -> Tuple[float, Array, float, int]:
    """Armijo backtracking line search."""
    alpha = initial_step
    slope = float(np.dot(g, p))
    evaluations = 0

    candidate_x = x
    candidate_fx = fx

    for _ in range(max_backtracks):
        candidate_x = x + alpha * p
        candidate_fx = f(candidate_x)
        evaluations += 1

        if candidate_fx <= fx + c1 * alpha * slope:
            return alpha, candidate_x, candidate_fx, evaluations

        alpha *= shrink

    return alpha, candidate_x, candidate_fx, evaluations


def dfp_optimize(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x0: Array,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> DFPResult:
    """Run DFP optimization with basic numerical safeguards."""
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x0 must be a non-empty 1D vector")

    n = x.size
    I = np.eye(n)
    H = np.eye(n)

    fx = f(x)
    g = grad(x)
    function_evals = 1
    gradient_evals = 1

    history: List[Tuple[int, float, float, float, float]] = []

    converged = False
    for it in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g, ord=2))
        if grad_norm < tol:
            converged = True
            break

        p = -H @ g
        if float(np.dot(g, p)) >= -1e-14:
            p = -g
            H = I.copy()

        alpha, x_new, fx_new, fevals = backtracking_line_search(
            f=f,
            x=x,
            fx=fx,
            g=g,
            p=p,
        )
        function_evals += fevals

        g_new = grad(x_new)
        gradient_evals += 1

        s = x_new - x
        y = g_new - g
        ys = float(np.dot(y, s))
        Hy = H @ y
        yHy = float(np.dot(y, Hy))

        if ys > 1e-12 and yHy > 1e-12:
            H = H + np.outer(s, s) / ys - np.outer(Hy, Hy) / yHy
            H = 0.5 * (H + H.T)
        else:
            H = I.copy()

        x, fx, g = x_new, fx_new, g_new
        history.append((it, fx, float(np.linalg.norm(g, ord=2)), alpha, ys))

    final_grad_norm = float(np.linalg.norm(g, ord=2))
    if final_grad_norm < tol:
        converged = True

    return DFPResult(
        x=x,
        f=float(fx),
        grad_norm=final_grad_norm,
        iterations=len(history),
        converged=converged,
        function_evals=function_evals,
        gradient_evals=gradient_evals,
        history=history,
    )


def print_history_summary(history: List[Tuple[int, float, float, float, float]]) -> None:
    if not history:
        print("No iterations recorded (already converged at initial point).")
        return

    head = history[:6]
    tail = history[-3:] if len(history) > 10 else []

    print("Iter | f(x)         | ||grad||      | step      | y^T s")
    print("-----+--------------+---------------+-----------+-----------")
    for it, fx, gn, alpha, ys in head:
        print(f"{it:4d} | {fx: .6e} | {gn: .6e} | {alpha: .3e} | {ys: .3e}")

    if tail:
        print("  ...")
        for it, fx, gn, alpha, ys in tail:
            print(f"{it:4d} | {fx: .6e} | {gn: .6e} | {alpha: .3e} | {ys: .3e}")


def main() -> None:
    x0 = np.array([-1.2, 1.0], dtype=float)

    result = dfp_optimize(
        f=rosenbrock,
        grad=rosenbrock_grad,
        x0=x0,
        tol=1e-8,
        max_iter=500,
    )

    print("DFP demo on Rosenbrock function")
    print(f"Initial point: {x0.tolist()}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final x: {result.x.tolist()}")
    print(f"Final f(x): {result.f:.12e}")
    print(f"Final ||grad||: {result.grad_norm:.12e}")
    print(f"Function evaluations: {result.function_evals}")
    print(f"Gradient evaluations: {result.gradient_evals}")

    if not result.converged:
        raise RuntimeError("DFP did not converge within max_iter.")
    if not np.allclose(result.x, np.array([1.0, 1.0]), atol=5e-5):
        raise RuntimeError(f"Unexpected optimizer result: x={result.x}")

    print("\nIteration trace (head/tail):")
    print_history_summary(result.history)
    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()
