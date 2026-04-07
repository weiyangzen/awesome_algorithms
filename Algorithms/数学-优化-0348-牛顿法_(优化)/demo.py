"""Newton method (optimization) MVP demo.

This script implements a transparent damped Newton optimizer with Armijo
backtracking line search, then runs two fixed test problems:
1) Rosenbrock function (non-convex)
2) SPD quadratic objective (convex)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, float, float, float]


@dataclass
class NewtonResult:
    x: Array
    f: float
    grad_norm: float
    iterations: int
    converged: bool
    function_evals: int
    gradient_evals: int
    hessian_evals: int
    history: List[HistoryItem]
    message: str


def check_vector(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def rosenbrock(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_grad(x: Array) -> Array:
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


def rosenbrock_hess(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.zeros((n, n), dtype=float)

    h[0, 0] = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
    h[0, 1] = -400.0 * x[0]
    h[1, 0] = h[0, 1]

    for i in range(1, n - 1):
        h[i, i - 1] = -400.0 * x[i - 1]
        h[i, i] = 1200.0 * x[i] ** 2 - 400.0 * x[i + 1] + 202.0
        h[i, i + 1] = -400.0 * x[i]

    if n > 2:
        h[n - 1, n - 2] = -400.0 * x[n - 2]
        h[n - 1, n - 1] = 200.0
    else:
        h[1, 1] = 200.0

    return h


def make_quadratic_problem(a: Array, b: Array) -> Tuple[Callable[[Array], float], Callable[[Array], Array], Callable[[Array], Array]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be square matrix, got shape={a.shape}.")
    if b.ndim != 1 or b.shape[0] != a.shape[0]:
        raise ValueError(f"b shape mismatch: A={a.shape}, b={b.shape}.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("A/b contains non-finite values.")

    def f(x: Array) -> float:
        return float(0.5 * x.T @ a @ x - b.T @ x)

    def g(x: Array) -> Array:
        return a @ x - b

    def h(_: Array) -> Array:
        return a

    return f, g, h


def modified_newton_direction(
    h: Array,
    g: Array,
    base_damping: float,
    max_trials: int,
) -> Tuple[Array, float, bool]:
    n = g.size
    identity = np.eye(n, dtype=float)
    damping = base_damping

    for _ in range(max_trials):
        try:
            p = np.linalg.solve(h + damping * identity, -g)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        if np.all(np.isfinite(p)) and float(np.dot(g, p)) < -1e-14:
            return p, damping, True

        damping *= 10.0

    return -g.copy(), 0.0, False


def armijo_backtracking(
    f: Callable[[Array], float],
    x: Array,
    fx: float,
    g: Array,
    p: Array,
    c1: float,
    shrink: float,
    min_step: float,
    max_backtracks: int,
) -> Tuple[float, Array, float, int, bool]:
    slope = float(np.dot(g, p))
    if slope >= 0.0:
        raise ValueError("Search direction is not a descent direction.")

    alpha = 1.0
    evals = 0
    candidate_x = x
    candidate_fx = fx

    for _ in range(max_backtracks):
        candidate_x = x + alpha * p
        candidate_fx = f(candidate_x)
        evals += 1

        if np.isfinite(candidate_fx) and candidate_fx <= fx + c1 * alpha * slope:
            return alpha, candidate_x, float(candidate_fx), evals, True

        alpha *= shrink
        if alpha < min_step:
            break

    return alpha, candidate_x, float(candidate_fx), evals, False


def newton_optimize(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    hess: Callable[[Array], Array],
    x0: Array,
    tol: float = 1e-8,
    max_iter: int = 100,
    base_damping: float = 1e-8,
    max_damping_trials: int = 8,
    c1: float = 1e-4,
    line_search_shrink: float = 0.5,
    min_step: float = 1e-12,
    max_backtracks: int = 40,
) -> NewtonResult:
    x = check_vector("x0", x0).copy()

    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if base_damping < 0.0:
        raise ValueError("base_damping must be >= 0.")
    if max_damping_trials <= 0:
        raise ValueError("max_damping_trials must be > 0.")
    if not (0.0 < c1 < 1.0):
        raise ValueError("c1 must be in (0, 1).")
    if not (0.0 < line_search_shrink < 1.0):
        raise ValueError("line_search_shrink must be in (0, 1).")
    if min_step <= 0.0:
        raise ValueError("min_step must be > 0.")
    if max_backtracks <= 0:
        raise ValueError("max_backtracks must be > 0.")

    fx = float(f(x))
    g = np.asarray(grad(x), dtype=float)

    if not np.isfinite(fx) or not np.all(np.isfinite(g)):
        raise ValueError("Initial function value or gradient is non-finite.")

    function_evals = 1
    gradient_evals = 1
    hessian_evals = 0

    history: List[HistoryItem] = []
    converged = False
    message = "max_iter reached"

    for it in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        grad_tol = tol * (1.0 + float(np.linalg.norm(x)))
        if grad_norm <= grad_tol:
            converged = True
            message = "gradient tolerance reached"
            break

        h = np.asarray(hess(x), dtype=float)
        hessian_evals += 1
        if h.shape != (x.size, x.size) or not np.all(np.isfinite(h)):
            message = "invalid Hessian encountered"
            break

        p, damping_used, used_modified_newton = modified_newton_direction(
            h=h,
            g=g,
            base_damping=base_damping,
            max_trials=max_damping_trials,
        )

        if not used_modified_newton:
            damping_used = 0.0

        slope = float(np.dot(g, p))
        if slope >= -1e-14:
            p = -g
            slope = -float(np.dot(g, g))
            damping_used = 0.0

        alpha, x_new, fx_new, fevals, accepted = armijo_backtracking(
            f=f,
            x=x,
            fx=fx,
            g=g,
            p=p,
            c1=c1,
            shrink=line_search_shrink,
            min_step=min_step,
            max_backtracks=max_backtracks,
        )
        function_evals += fevals

        if not accepted:
            message = "line search failed"
            break

        g_new = np.asarray(grad(x_new), dtype=float)
        gradient_evals += 1
        if not np.all(np.isfinite(g_new)):
            message = "non-finite gradient encountered"
            break

        step_norm = float(np.linalg.norm(x_new - x))
        grad_new_norm = float(np.linalg.norm(g_new))
        history.append((it, float(fx_new), grad_new_norm, float(alpha), float(damping_used), step_norm))

        x, fx, g = x_new, float(fx_new), g_new

        step_tol = tol * (1.0 + float(np.linalg.norm(x)))
        if step_norm <= step_tol:
            converged = True
            message = "step tolerance reached"
            break

    return NewtonResult(
        x=x,
        f=float(fx),
        grad_norm=float(np.linalg.norm(g)),
        iterations=len(history),
        converged=converged,
        function_evals=function_evals,
        gradient_evals=gradient_evals,
        hessian_evals=hessian_evals,
        history=history,
        message=message,
    )


def relative_error(abs_err: float, ref_norm: float, eps: float = 1e-15) -> float:
    return float(abs(abs_err) / (abs(ref_norm) + eps))


def print_history(history: List[HistoryItem], max_lines: int = 12) -> None:
    print("iter | f(x_k)           | ||grad||         | alpha            | damping          | ||step||")
    print("-" * 98)
    for it, fx, gnorm, alpha, damping, step_norm in history[:max_lines]:
        print(
            f"{it:4d} | {fx:16.9e} | {gnorm:16.9e} | {alpha:16.9e} | {damping:16.9e} | {step_norm:16.9e}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_case(
    name: str,
    f: Callable[[Array], float],
    g: Callable[[Array], Array],
    h: Callable[[Array], Array],
    x0: Array,
    reference_x: Optional[Array],
    tol: float,
    max_iter: int,
) -> dict:
    print(f"\n=== Case: {name} ===")
    result = newton_optimize(
        f=f,
        grad=g,
        hess=h,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
    )

    print(f"Converged: {result.converged}")
    print(f"Stop reason: {result.message}")
    print(f"Iterations: {result.iterations}")
    print(f"Final x: {result.x}")
    print(f"Final f(x): {result.f:.12e}")
    print(f"Final ||grad||: {result.grad_norm:.12e}")
    print(f"Function evals: {result.function_evals}")
    print(f"Gradient evals: {result.gradient_evals}")
    print(f"Hessian evals: {result.hessian_evals}")

    print("Iteration trace:")
    print_history(result.history)

    rel_x_error = np.nan
    if reference_x is not None:
        ref = np.asarray(reference_x, dtype=float)
        abs_x_error = float(np.linalg.norm(result.x - ref))
        rel_x_error = relative_error(abs_x_error, float(np.linalg.norm(ref)))
        print(f"Reference x*: {ref}")
        print(f"Absolute x error: {abs_x_error:.12e}")
        print(f"Relative x error: {rel_x_error:.12e}")

    return {
        "converged": float(result.converged),
        "grad_norm": result.grad_norm,
        "rel_x_error": rel_x_error,
        "iterations": float(result.iterations),
    }


def main() -> None:
    tol = 1e-8
    max_iter = 120

    quad_a = np.array(
        [
            [8.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=float,
    )
    quad_b = np.array([2.0, -1.0, 1.0], dtype=float)
    quad_f, quad_g, quad_h = make_quadratic_problem(quad_a, quad_b)
    quad_ref = np.linalg.solve(quad_a, quad_b)

    cases = [
        {
            "name": "Rosenbrock-2D",
            "f": rosenbrock,
            "g": rosenbrock_grad,
            "h": rosenbrock_hess,
            "x0": np.array([-1.2, 1.0], dtype=float),
            "reference_x": np.array([1.0, 1.0], dtype=float),
        },
        {
            "name": "SPD-Quadratic-3D",
            "f": quad_f,
            "g": quad_g,
            "h": quad_h,
            "x0": np.array([3.0, -2.0, 1.0], dtype=float),
            "reference_x": quad_ref,
        },
    ]

    stats = []
    for case in cases:
        stats.append(
            run_case(
                name=case["name"],
                f=case["f"],
                g=case["g"],
                h=case["h"],
                x0=case["x0"],
                reference_x=case["reference_x"],
                tol=tol,
                max_iter=max_iter,
            )
        )

    max_grad_norm = max(item["grad_norm"] for item in stats)
    finite_rel_errors = [item["rel_x_error"] for item in stats if np.isfinite(item["rel_x_error"])]
    max_rel_error = max(finite_rel_errors) if finite_rel_errors else np.nan
    all_converged = all(bool(item["converged"]) for item in stats)

    print("\n=== Summary ===")
    print(f"All cases converged: {all_converged}")
    print(f"Max final gradient norm: {max_grad_norm:.12e}")
    print(f"Max relative x error: {max_rel_error:.12e}")

    if not all_converged:
        raise RuntimeError("At least one case did not converge.")
    if max_grad_norm > 1e-6:
        raise RuntimeError(f"Final gradient too large: {max_grad_norm}")
    if np.isfinite(max_rel_error) and max_rel_error > 1e-5:
        raise RuntimeError(f"Relative x error too large: {max_rel_error}")

    print("Validation checks passed.")


if __name__ == "__main__":
    main()
