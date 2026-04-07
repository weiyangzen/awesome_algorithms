"""Minimal runnable MVP for Sequential Quadratic Programming (SQP).

This demo solves a small nonlinear constrained optimization problem with
SciPy's SLSQP solver (an SQP implementation), while exposing objective,
gradients, constraints, iteration trace, and KKT-style diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize


Array = np.ndarray


@dataclass
class SQPDemoResult:
    x: Array
    f: float
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int
    constraint_values: Dict[str, float]
    approx_stationarity_residual: float
    history: pd.DataFrame


def objective(x: Array) -> float:
    """Quadratic objective with mild variable coupling."""
    x0, x1 = float(x[0]), float(x[1])
    return (x0 - 1.5) ** 2 + (x1 - 0.5) ** 2 + 0.2 * x0 * x1


def objective_grad(x: Array) -> Array:
    """Analytic gradient of the objective."""
    x0, x1 = float(x[0]), float(x[1])
    return np.array([2.0 * (x0 - 1.5) + 0.2 * x1, 2.0 * (x1 - 0.5) + 0.2 * x0], dtype=float)


def c_circle(x: Array) -> float:
    """Inequality: stay inside unit circle -> 1 - x0^2 - x1^2 >= 0."""
    x0, x1 = float(x[0]), float(x[1])
    return 1.0 - x0 * x0 - x1 * x1


def c_circle_jac(x: Array) -> Array:
    x0, x1 = float(x[0]), float(x[1])
    return np.array([-2.0 * x0, -2.0 * x1], dtype=float)


def c_sum(x: Array) -> float:
    """Inequality: x0 + x1 - 0.5 >= 0."""
    x0, x1 = float(x[0]), float(x[1])
    return x0 + x1 - 0.5


def c_sum_jac(_: Array) -> Array:
    return np.array([1.0, 1.0], dtype=float)


def c_x_nonnegative(x: Array) -> float:
    """Inequality: x0 >= 0."""
    return float(x[0])


def c_x_nonnegative_jac(_: Array) -> Array:
    return np.array([1.0, 0.0], dtype=float)


def evaluate_constraints(x: Array) -> Dict[str, float]:
    return {
        "c_circle(>=0)": c_circle(x),
        "c_sum(>=0)": c_sum(x),
        "c_x_nonnegative(>=0)": c_x_nonnegative(x),
    }


def estimate_stationarity_residual(x: Array, active_tol: float = 1e-6) -> float:
    """Estimate KKT stationarity residual on active inequalities.

    For active constraints (defined as g(x) >= 0), use KKT stationarity:
        grad f(x) - A * mu ~= 0,  mu >= 0
    where columns of A are active-constraint gradients.
    then report residual norm.
    """
    grad = objective_grad(x)
    values_and_jacs: List[Tuple[float, Array]] = [
        (c_circle(x), c_circle_jac(x)),
        (c_sum(x), c_sum_jac(x)),
        (c_x_nonnegative(x), c_x_nonnegative_jac(x)),
    ]

    active_grads = [jac for value, jac in values_and_jacs if value <= active_tol]
    if not active_grads:
        return float(np.linalg.norm(grad, ord=2))

    a_mat = np.column_stack(active_grads)
    multipliers, *_ = np.linalg.lstsq(a_mat, grad, rcond=None)
    multipliers = np.maximum(multipliers, 0.0)
    residual = grad - a_mat @ multipliers
    return float(np.linalg.norm(residual, ord=2))


def solve_sqp_demo() -> SQPDemoResult:
    x0 = np.array([0.2, 0.9], dtype=float)
    bounds = Bounds(lb=np.array([-1.5, -1.5]), ub=np.array([1.5, 1.5]))

    constraints = [
        {"type": "ineq", "fun": c_circle, "jac": c_circle_jac},
        {"type": "ineq", "fun": c_sum, "jac": c_sum_jac},
        {"type": "ineq", "fun": c_x_nonnegative, "jac": c_x_nonnegative_jac},
    ]

    trace_rows: List[Dict[str, float]] = []

    def callback(xk: Array) -> None:
        cvals = evaluate_constraints(xk)
        trace_rows.append(
            {
                "iter": float(len(trace_rows) + 1),
                "x0": float(xk[0]),
                "x1": float(xk[1]),
                "f(x)": objective(xk),
                "c_circle": cvals["c_circle(>=0)"],
                "c_sum": cvals["c_sum(>=0)"],
                "c_x_nonnegative": cvals["c_x_nonnegative(>=0)"],
            }
        )

    result = minimize(
        fun=objective,
        x0=x0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"ftol": 1e-9, "maxiter": 100, "disp": False},
    )

    x_star = np.asarray(result.x, dtype=float)
    cvals = evaluate_constraints(x_star)
    stationarity = estimate_stationarity_residual(x_star)
    history = pd.DataFrame(trace_rows)

    return SQPDemoResult(
        x=x_star,
        f=float(result.fun),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nit=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        constraint_values=cvals,
        approx_stationarity_residual=stationarity,
        history=history,
    )


def print_history(history: pd.DataFrame) -> None:
    if history.empty:
        print("No callback iterations recorded.")
        return

    if len(history) <= 8:
        view = history
    else:
        view = pd.concat([history.head(5), history.tail(3)], axis=0)

    print("Iteration trace (head/tail):")
    print(
        view.to_string(
            index=False,
            float_format=lambda x: f"{x: .6f}",
        )
    )


def main() -> None:
    out = solve_sqp_demo()

    print("SQP demo via SciPy SLSQP")
    print(f"success: {out.success}")
    print(f"status/message: {out.status} / {out.message}")
    print(f"iterations (nit): {out.nit}")
    print(f"function evals (nfev): {out.nfev}")
    print(f"gradient evals (njev): {out.njev}")
    print(f"x*: {out.x.tolist()}")
    print(f"f(x*): {out.f:.12e}")

    print("Constraint values at x* (must be >= 0):")
    for name, value in out.constraint_values.items():
        print(f"  {name}: {value:.12e}")

    print(f"Approx. stationarity residual: {out.approx_stationarity_residual:.6e}")
    print_history(out.history)

    # Minimal self-checks for unattended validation.
    if not out.success:
        raise RuntimeError("SLSQP did not converge.")
    min_constraint = min(out.constraint_values.values())
    if min_constraint < -1e-7:
        raise RuntimeError(f"Constraint violation detected: min={min_constraint:.3e}")
    if out.f >= objective(np.array([0.2, 0.9], dtype=float)):
        raise RuntimeError("Objective did not improve from initial point.")

    print("Validation checks passed.")


if __name__ == "__main__":
    main()
