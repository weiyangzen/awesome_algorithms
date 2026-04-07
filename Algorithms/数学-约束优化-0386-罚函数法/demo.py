"""Quadratic penalty method for constrained optimization (minimal runnable MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


Array = np.ndarray


@dataclass
class ConstrainedProblem:
    """Container for a smooth constrained optimization problem."""

    dimension: int
    description: str
    objective: Callable[[Array], float]
    objective_grad: Callable[[Array], Array]
    eq_constraints: Callable[[Array], Array]
    eq_jacobian: Callable[[Array], Array]
    ineq_constraints: Callable[[Array], Array]
    ineq_jacobian: Callable[[Array], Array]


@dataclass
class PenaltyIterRecord:
    """One outer-iteration record of the penalty method."""

    outer_iter: int
    rho: float
    x: Array
    f_raw: float
    eq_norm: float
    ineq_violation_norm: float
    penalty_term: float
    augmented_value: float
    inner_iterations: int
    inner_success: bool


def validate_problem(problem: ConstrainedProblem, x0: Array) -> None:
    """Validate shapes, finiteness, and consistency at initial point."""
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")
    if x0.shape[0] != problem.dimension:
        raise ValueError("x0 dimension does not match problem.dimension.")
    if not np.isfinite(x0).all():
        raise ValueError("x0 contains NaN or Inf.")

    f0 = float(problem.objective(x0))
    grad0 = np.asarray(problem.objective_grad(x0), dtype=float)
    h0 = np.asarray(problem.eq_constraints(x0), dtype=float)
    jh0 = np.asarray(problem.eq_jacobian(x0), dtype=float)
    g0 = np.asarray(problem.ineq_constraints(x0), dtype=float)
    jg0 = np.asarray(problem.ineq_jacobian(x0), dtype=float)

    if not np.isfinite(f0):
        raise ValueError("objective(x0) must be finite.")
    if grad0.shape != (problem.dimension,):
        raise ValueError("objective_grad(x0) has invalid shape.")

    if h0.ndim != 1:
        raise ValueError("eq_constraints(x) must return a 1D array.")
    if g0.ndim != 1:
        raise ValueError("ineq_constraints(x) must return a 1D array.")

    if jh0.shape != (h0.shape[0], problem.dimension):
        raise ValueError("eq_jacobian(x) shape must be (n_eq, dimension).")
    if jg0.shape != (g0.shape[0], problem.dimension):
        raise ValueError("ineq_jacobian(x) shape must be (n_ineq, dimension).")

    if not (
        np.isfinite(grad0).all()
        and np.isfinite(h0).all()
        and np.isfinite(jh0).all()
        and np.isfinite(g0).all()
        and np.isfinite(jg0).all()
    ):
        raise ValueError("Problem functions evaluated at x0 contain NaN or Inf.")


def penalty_components(problem: ConstrainedProblem, x: Array, rho: float) -> Tuple[float, float, float, float, float]:
    """Return f(x), ||h||2, ||max(g,0)||2, penalty term, augmented objective."""
    f_raw = float(problem.objective(x))
    h = np.asarray(problem.eq_constraints(x), dtype=float)
    g = np.asarray(problem.ineq_constraints(x), dtype=float)

    h_norm = float(np.linalg.norm(h, ord=2))
    g_violation = np.maximum(g, 0.0)
    g_violation_norm = float(np.linalg.norm(g_violation, ord=2))

    penalty_term = 0.5 * rho * (float(np.dot(h, h)) + float(np.dot(g_violation, g_violation)))
    augmented_value = f_raw + penalty_term
    return f_raw, h_norm, g_violation_norm, penalty_term, augmented_value


def penalty_objective(problem: ConstrainedProblem, x: Array, rho: float) -> float:
    """Quadratic penalty objective for equality + inequality constraints.

    Minimize:
      Q_rho(x) = f(x) + (rho/2) * ||h(x)||^2 + (rho/2) * ||max(g(x), 0)||^2
    where equality constraints are h(x)=0 and inequalities are g(x)<=0.
    """
    _, _, _, _, augmented = penalty_components(problem, x, rho)
    return float(augmented)


def penalty_gradient(problem: ConstrainedProblem, x: Array, rho: float) -> Array:
    """Gradient of Q_rho(x) using Jacobians of constraints."""
    grad = np.asarray(problem.objective_grad(x), dtype=float).copy()

    h = np.asarray(problem.eq_constraints(x), dtype=float)
    jh = np.asarray(problem.eq_jacobian(x), dtype=float)
    if h.size > 0:
        grad += rho * (jh.T @ h)

    g = np.asarray(problem.ineq_constraints(x), dtype=float)
    jg = np.asarray(problem.ineq_jacobian(x), dtype=float)
    if g.size > 0:
        g_plus = np.maximum(g, 0.0)
        grad += rho * (jg.T @ g_plus)

    return grad


def quadratic_penalty_method(
    problem: ConstrainedProblem,
    x0: Array,
    *,
    rho0: float = 1.0,
    penalty_growth: float = 10.0,
    feasibility_tol: float = 1e-8,
    max_outer_iters: int = 12,
    inner_gtol: float = 1e-10,
    inner_maxiter: int = 500,
) -> Tuple[Array, List[PenaltyIterRecord]]:
    """Solve constrained problem by outer penalty updates + inner unconstrained minimization."""
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive.")
    if penalty_growth <= 1.0:
        raise ValueError("penalty_growth must be > 1.")
    if feasibility_tol <= 0.0:
        raise ValueError("feasibility_tol must be positive.")
    if max_outer_iters <= 0:
        raise ValueError("max_outer_iters must be positive.")
    if inner_gtol <= 0.0:
        raise ValueError("inner_gtol must be positive.")
    if inner_maxiter <= 0:
        raise ValueError("inner_maxiter must be positive.")

    x = np.asarray(x0, dtype=float).copy()
    validate_problem(problem, x)

    rho = float(rho0)
    history: List[PenaltyIterRecord] = []

    for outer in range(1, max_outer_iters + 1):
        result = minimize(
            fun=lambda z: penalty_objective(problem, np.asarray(z, dtype=float), rho),
            x0=x,
            jac=lambda z: penalty_gradient(problem, np.asarray(z, dtype=float), rho),
            method="BFGS",
            options={"gtol": inner_gtol, "maxiter": inner_maxiter, "disp": False},
        )

        x = np.asarray(result.x, dtype=float)
        f_raw, eq_norm, ineq_violation_norm, penalty_term, augmented_value = penalty_components(problem, x, rho)

        history.append(
            PenaltyIterRecord(
                outer_iter=outer,
                rho=rho,
                x=x.copy(),
                f_raw=f_raw,
                eq_norm=eq_norm,
                ineq_violation_norm=ineq_violation_norm,
                penalty_term=penalty_term,
                augmented_value=augmented_value,
                inner_iterations=int(result.nit),
                inner_success=bool(result.success),
            )
        )

        if not np.isfinite(augmented_value):
            raise RuntimeError("Augmented objective became non-finite during optimization.")

        feasibility_measure = max(eq_norm, ineq_violation_norm)
        if feasibility_measure < feasibility_tol:
            return x, history

        rho *= penalty_growth

    final_feasibility = max(history[-1].eq_norm, history[-1].ineq_violation_norm)
    raise RuntimeError(
        "Penalty method did not reach feasibility tolerance. "
        f"final feasibility measure={final_feasibility:.3e}, target={feasibility_tol:.3e}."
    )


def build_demo_problem() -> Tuple[ConstrainedProblem, Array, Array]:
    """Build a tiny constrained problem with known optimum x*=[1,0].

    Problem:
      min f(x1,x2) = (x1-2)^2 + (x2-1)^2
      s.t. h(x)=x1+x2-1 = 0
           g1(x)=-x1 <= 0
           g2(x)=-x2 <= 0
    """

    def objective(x: Array) -> float:
        return float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2)

    def objective_grad(x: Array) -> Array:
        return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)], dtype=float)

    def eq_constraints(x: Array) -> Array:
        return np.array([x[0] + x[1] - 1.0], dtype=float)

    def eq_jacobian(x: Array) -> Array:
        _ = x
        return np.array([[1.0, 1.0]], dtype=float)

    def ineq_constraints(x: Array) -> Array:
        return np.array([-x[0], -x[1]], dtype=float)

    def ineq_jacobian(x: Array) -> Array:
        _ = x
        return np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=float)

    problem = ConstrainedProblem(
        dimension=2,
        description=(
            "min (x1-2)^2 + (x2-1)^2, s.t. x1+x2=1, x1>=0, x2>=0; "
            "true constrained optimum at x*=[1,0]."
        ),
        objective=objective,
        objective_grad=objective_grad,
        eq_constraints=eq_constraints,
        eq_jacobian=eq_jacobian,
        ineq_constraints=ineq_constraints,
        ineq_jacobian=ineq_jacobian,
    )

    x0 = np.array([2.5, -0.5], dtype=float)
    known_solution = np.array([1.0, 0.0], dtype=float)
    return problem, x0, known_solution


def solve_reference_slsqp(problem: ConstrainedProblem, x0: Array) -> Optional[Tuple[Array, float]]:
    """Reference constrained solve using SLSQP (for validation only)."""
    h0 = np.asarray(problem.eq_constraints(x0), dtype=float)
    g0 = np.asarray(problem.ineq_constraints(x0), dtype=float)

    constraints = []
    for idx in range(h0.shape[0]):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda z, i=idx: float(problem.eq_constraints(np.asarray(z, dtype=float))[i]),
                "jac": lambda z, i=idx: np.asarray(problem.eq_jacobian(np.asarray(z, dtype=float))[i], dtype=float),
            }
        )

    for idx in range(g0.shape[0]):
        constraints.append(
            {
                "type": "ineq",
                # SLSQP uses c(z) >= 0, while we define g(z) <= 0.
                "fun": lambda z, i=idx: float(-problem.ineq_constraints(np.asarray(z, dtype=float))[i]),
                "jac": lambda z, i=idx: np.asarray(-problem.ineq_jacobian(np.asarray(z, dtype=float))[i], dtype=float),
            }
        )

    result = minimize(
        fun=lambda z: float(problem.objective(np.asarray(z, dtype=float))),
        x0=x0,
        jac=lambda z: np.asarray(problem.objective_grad(np.asarray(z, dtype=float)), dtype=float),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        return None
    return np.asarray(result.x, dtype=float), float(result.fun)


def print_history(history: List[PenaltyIterRecord]) -> None:
    """Print outer iterations in a compact table."""
    print("outer | rho          | f(x)         | ||h||2       | ||g+||2      | penalty      | Q_rho        | inner | ok")
    print("-" * 118)
    for rec in history:
        print(
            f"{rec.outer_iter:>5d} | {rec.rho:>12.4e} | {rec.f_raw:>11.7f} | "
            f"{rec.eq_norm:>11.4e} | {rec.ineq_violation_norm:>11.4e} | "
            f"{rec.penalty_term:>11.4e} | {rec.augmented_value:>11.7f} | "
            f"{rec.inner_iterations:>5d} | {str(rec.inner_success):>2s}"
        )


def main() -> None:
    problem, x0, known_solution = build_demo_problem()

    print("=== Quadratic Penalty Method Demo ===")
    print(problem.description)
    print(f"initial x0: {np.array2string(x0, precision=6)}")

    x_sol, history = quadratic_penalty_method(
        problem,
        x0,
        rho0=1.0,
        penalty_growth=10.0,
        feasibility_tol=1e-8,
        max_outer_iters=12,
        inner_gtol=1e-10,
        inner_maxiter=500,
    )

    print()
    print_history(history)

    h = problem.eq_constraints(x_sol)
    g = problem.ineq_constraints(x_sol)
    g_plus = np.maximum(g, 0.0)

    print("\nFinal penalty-method solution:")
    print(f"x*: {np.array2string(x_sol, precision=10)}")
    print(f"f(x*): {problem.objective(x_sol):.10f}")
    print(f"||h(x*)||2: {np.linalg.norm(h):.3e}")
    print(f"||max(g(x*),0)||2: {np.linalg.norm(g_plus):.3e}")
    print(f"known solution: {np.array2string(known_solution, precision=10)}")
    print(f"||x* - known||2: {np.linalg.norm(x_sol - known_solution):.3e}")

    ref = solve_reference_slsqp(problem, x0)
    if ref is None:
        print("SLSQP reference: failed or unavailable.")
    else:
        ref_x, ref_f = ref
        print("\nSLSQP reference (constraint-native solve):")
        print(f"x_ref: {np.array2string(ref_x, precision=10)}")
        print(f"f(x_ref): {ref_f:.10f}")
        print(f"||x* - x_ref||2: {np.linalg.norm(x_sol - ref_x):.3e}")


if __name__ == "__main__":
    main()
