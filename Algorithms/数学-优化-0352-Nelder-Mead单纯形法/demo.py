"""Minimal runnable MVP for Nelder-Mead simplex optimization.

This script implements the core Nelder-Mead update rules from scratch:
reflection, expansion, contraction, and shrink. It then runs two fixed
benchmark cases without any interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, float, str]


@dataclass
class NelderMeadResult:
    x: Array
    fx: float
    iterations: int
    function_evals: int
    converged: bool
    history: List[HistoryItem]


def check_vector(name: str, x: Array) -> None:
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def make_initial_simplex(x0: Array, step: float = 0.2) -> Array:
    """Construct a deterministic initial simplex around x0."""
    check_vector("x0", x0)
    if step <= 0:
        raise ValueError("step must be positive.")

    n = x0.size
    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0

    for i in range(n):
        point = x0.copy()
        # Slightly scale by coordinate magnitude to avoid tiny/huge mismatch.
        delta = step * max(1.0, abs(point[i]))
        point[i] += delta
        simplex[i + 1] = point

    return simplex


def nelder_mead(
    fun: Callable[[Array], float],
    x0: Array,
    initial_step: float = 0.2,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    tol_x: float = 1e-8,
    tol_f: float = 1e-10,
    max_iter: int = 600,
    max_eval: int = 5000,
) -> NelderMeadResult:
    """Run Nelder-Mead simplex method for unconstrained minimization."""
    x0 = np.asarray(x0, dtype=float)
    check_vector("x0", x0)
    if not (alpha > 0 and gamma > 1 and 0 < rho < 1 and 0 < sigma < 1):
        raise ValueError("Require alpha>0, gamma>1, 0<rho<1, 0<sigma<1.")
    if tol_x <= 0 or tol_f <= 0:
        raise ValueError("tol_x and tol_f must be positive.")
    if max_iter <= 0 or max_eval <= 0:
        raise ValueError("max_iter and max_eval must be positive.")

    simplex = make_initial_simplex(x0=x0, step=initial_step)
    n = x0.size

    fvals = np.array([float(fun(v)) for v in simplex], dtype=float)
    if not np.all(np.isfinite(fvals)):
        raise ValueError("Objective returns non-finite values on initial simplex.")

    function_evals = n + 1
    history: List[HistoryItem] = []
    converged = False

    for it in range(1, max_iter + 1):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        best = simplex[0]
        worst = simplex[-1]

        diameter = float(np.max(np.linalg.norm(simplex - best, axis=1)))
        f_spread = float(np.max(np.abs(fvals - fvals[0])))
        history.append((it, float(fvals[0]), diameter, f_spread, "sort"))

        if diameter < tol_x and f_spread < tol_f:
            converged = True
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = float(fun(xr))
        function_evals += 1

        action = "reflect"

        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = float(fun(xe))
            function_evals += 1
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
                action = "expand"
            else:
                simplex[-1] = xr
                fvals[-1] = fr
                action = "reflect-best"
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
            action = "reflect-middle"
        else:
            # Contraction branch
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = float(fun(xc))
                function_evals += 1
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    action = "contract-out"
                else:
                    action = "shrink"
            else:
                # Inside contraction
                xc = centroid + rho * (worst - centroid)
                fc = float(fun(xc))
                function_evals += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    action = "contract-in"
                else:
                    action = "shrink"

            if action == "shrink":
                best = simplex[0].copy()
                for i in range(1, n + 1):
                    simplex[i] = best + sigma * (simplex[i] - best)
                    fvals[i] = float(fun(simplex[i]))
                function_evals += n

        history[-1] = (it, float(fvals.min()), diameter, f_spread, action)

        if function_evals >= max_eval:
            break

    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]

    return NelderMeadResult(
        x=simplex[0].copy(),
        fx=float(fvals[0]),
        iterations=len(history),
        function_evals=function_evals,
        converged=converged,
        history=history,
    )


def rosenbrock_2d(x: Array) -> float:
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def shifted_quadratic(x: Array) -> float:
    return float((x[0] - 2.0) ** 2 + 10.0 * (x[1] + 1.0) ** 2)


def print_history(history: Sequence[HistoryItem], max_lines: int = 8) -> None:
    print("iter | best f(x)       | simplex_diam    | f_spread        | action")
    print("-----+-----------------+-----------------+-----------------+--------------")

    head = list(history[:max_lines])
    tail = list(history[-3:]) if len(history) > (max_lines + 3) else []

    for it, best_fx, diam, spread, action in head:
        print(f"{it:4d} | {best_fx: .8e} | {diam: .8e} | {spread: .8e} | {action}")

    if tail:
        print("  ...")
        for it, best_fx, diam, spread, action in tail:
            print(f"{it:4d} | {best_fx: .8e} | {diam: .8e} | {spread: .8e} | {action}")


def run_case(name: str, fun: Callable[[Array], float], x0: Array, reference: Array) -> Tuple[float, float, int]:
    print(f"\n=== Case: {name} ===")
    result = nelder_mead(
        fun=fun,
        x0=x0,
        initial_step=0.3,
        tol_x=1e-8,
        tol_f=1e-10,
        max_iter=800,
        max_eval=6000,
    )

    err = float(np.linalg.norm(result.x - reference))
    print(f"Initial x0: {x0.tolist()}")
    print(f"Converged by tolerance: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Function evaluations: {result.function_evals}")
    print(f"Estimated minimizer: {result.x.tolist()}")
    print(f"Reference minimizer: {reference.tolist()}")
    print(f"Final objective: {result.fx:.12e}")
    print(f"L2 error to reference: {err:.12e}")

    print("Iteration trace (head/tail):")
    print_history(result.history)

    return result.fx, err, result.iterations


def main() -> None:
    cases = [
        {
            "name": "Rosenbrock-2D",
            "fun": rosenbrock_2d,
            "x0": np.array([-1.2, 1.0], dtype=float),
            "reference": np.array([1.0, 1.0], dtype=float),
            "fx_tol": 1e-8,
            "err_tol": 3e-4,
        },
        {
            "name": "Shifted quadratic bowl",
            "fun": shifted_quadratic,
            "x0": np.array([5.0, -4.0], dtype=float),
            "reference": np.array([2.0, -1.0], dtype=float),
            "fx_tol": 1e-10,
            "err_tol": 1e-6,
        },
    ]

    max_fx = 0.0
    max_err = 0.0
    total_iters = 0

    for case in cases:
        fx, err, iters = run_case(
            name=case["name"],
            fun=case["fun"],
            x0=case["x0"],
            reference=case["reference"],
        )
        if fx > case["fx_tol"]:
            raise RuntimeError(
                f"Case {case['name']} failed objective tolerance: {fx:.3e} > {case['fx_tol']:.3e}"
            )
        if err > case["err_tol"]:
            raise RuntimeError(
                f"Case {case['name']} failed solution tolerance: {err:.3e} > {case['err_tol']:.3e}"
            )

        max_fx = max(max_fx, fx)
        max_err = max(max_err, err)
        total_iters += iters

    print("\n=== Summary ===")
    print(f"Cases: {len(cases)}")
    print(f"Max final objective among cases: {max_fx:.12e}")
    print(f"Max solution L2 error among cases: {max_err:.12e}")
    print(f"Average iterations: {total_iters / len(cases):.2f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
