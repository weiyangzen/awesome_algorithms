"""Log-barrier method for inequality-constrained optimization (minimal runnable MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


Array = np.ndarray


@dataclass
class InequalityConstrainedProblem:
    """Container for a smooth optimization problem with g(x) <= 0 constraints."""

    dimension: int
    description: str
    objective: Callable[[Array], float]
    objective_grad: Callable[[Array], Array]
    objective_hessian: Callable[[Array], Array]
    ineq_constraints: Callable[[Array], Array]
    ineq_jacobian: Callable[[Array], Array]


@dataclass
class BarrierIterRecord:
    """One outer-iteration record of the barrier method."""

    outer_iter: int
    t: float
    x: Array
    f_raw: float
    max_g: float
    barrier_value: float
    surrogate_duality_gap: float
    inner_iterations: int
    inner_grad_norm: float


def validate_problem(problem: InequalityConstrainedProblem, x0: Array) -> None:
    """Validate shapes, finiteness, and strict feasibility of the initial point."""
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")
    if x0.shape[0] != problem.dimension:
        raise ValueError("x0 dimension does not match problem.dimension.")
    if not np.isfinite(x0).all():
        raise ValueError("x0 contains NaN or Inf.")

    f0 = float(problem.objective(x0))
    grad0 = np.asarray(problem.objective_grad(x0), dtype=float)
    hess0 = np.asarray(problem.objective_hessian(x0), dtype=float)
    g0 = np.asarray(problem.ineq_constraints(x0), dtype=float)
    jg0 = np.asarray(problem.ineq_jacobian(x0), dtype=float)

    if not np.isfinite(f0):
        raise ValueError("objective(x0) must be finite.")
    if grad0.shape != (problem.dimension,):
        raise ValueError("objective_grad(x0) has invalid shape.")
    if hess0.shape != (problem.dimension, problem.dimension):
        raise ValueError("objective_hessian(x0) has invalid shape.")
    if g0.ndim != 1:
        raise ValueError("ineq_constraints(x) must return a 1D array.")
    if jg0.shape != (g0.shape[0], problem.dimension):
        raise ValueError("ineq_jacobian(x) shape must be (n_ineq, dimension).")

    if not (
        np.isfinite(grad0).all()
        and np.isfinite(hess0).all()
        and np.isfinite(g0).all()
        and np.isfinite(jg0).all()
    ):
        raise ValueError("Problem functions evaluated at x0 contain NaN or Inf.")

    if np.any(g0 >= 0.0):
        raise ValueError(
            "x0 must be strictly feasible for log barrier (all g_i(x0) < 0)."
        )


def is_strictly_feasible(problem: InequalityConstrainedProblem, x: Array) -> bool:
    """Return whether x is in the strict interior of the inequality feasible set."""
    g = np.asarray(problem.ineq_constraints(x), dtype=float)
    return bool(np.all(g < 0.0))


def barrier_objective(problem: InequalityConstrainedProblem, x: Array, t: float) -> float:
    """Log-barrier objective phi_t(x) = t f(x) - sum_i log(-g_i(x))."""
    g = np.asarray(problem.ineq_constraints(x), dtype=float)
    if np.any(g >= 0.0):
        return float("inf")

    f_raw = float(problem.objective(x))
    barrier_term = -float(np.sum(np.log(-g)))
    return t * f_raw + barrier_term


def barrier_gradient(problem: InequalityConstrainedProblem, x: Array, t: float) -> Array:
    """Gradient of the log-barrier objective."""
    g = np.asarray(problem.ineq_constraints(x), dtype=float)
    if np.any(g >= 0.0):
        raise ValueError("Gradient is undefined outside the strict interior.")

    grad_f = np.asarray(problem.objective_grad(x), dtype=float)
    jac_g = np.asarray(problem.ineq_jacobian(x), dtype=float)

    # d/dx[-log(-g_i(x))] = -(1 / g_i(x)) * grad g_i(x)
    return t * grad_f - jac_g.T @ (1.0 / g)


def barrier_hessian(problem: InequalityConstrainedProblem, x: Array, t: float) -> Array:
    """Hessian of phi_t(x) for smooth objective and affine constraints.

    For affine g_i, Hessian is:
      t * Hessian(f) + J_g^T diag(1 / g_i(x)^2) J_g
    """
    g = np.asarray(problem.ineq_constraints(x), dtype=float)
    if np.any(g >= 0.0):
        raise ValueError("Hessian is undefined outside the strict interior.")

    hess_f = np.asarray(problem.objective_hessian(x), dtype=float)
    jac_g = np.asarray(problem.ineq_jacobian(x), dtype=float)
    inv_g_sq = 1.0 / np.square(g)
    hess_barrier = jac_g.T @ (inv_g_sq[:, None] * jac_g)
    hessian = t * hess_f + hess_barrier
    return 0.5 * (hessian + hessian.T)


def solve_barrier_subproblem(
    problem: InequalityConstrainedProblem,
    x_init: Array,
    t: float,
    *,
    inner_tol: float,
    max_inner_iters: int,
    line_search_alpha: float,
    line_search_beta: float,
) -> Tuple[Array, int, float]:
    """Solve min_x phi_t(x) with damped Newton + backtracking.

    The line-search guarantees strict feasibility and Armijo decrease.
    """
    x = np.asarray(x_init, dtype=float).copy()
    last_grad_norm = float("inf")

    for iteration in range(1, max_inner_iters + 1):
        grad = barrier_gradient(problem, x, t)
        grad_norm = float(np.linalg.norm(grad, ord=2))
        last_grad_norm = grad_norm
        if grad_norm < inner_tol:
            return x, iteration - 1, grad_norm

        hessian = barrier_hessian(problem, x, t)
        try:
            direction = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            direction = -grad

        phi_x = barrier_objective(problem, x, t)
        if not np.isfinite(phi_x):
            raise RuntimeError("Barrier objective became non-finite in inner solve.")

        slope = float(np.dot(grad, direction))
        if slope >= 0.0:
            direction = -grad
            slope = float(np.dot(grad, direction))

        step = 1.0
        accepted = False
        for _ in range(80):
            candidate = x + step * direction
            if not is_strictly_feasible(problem, candidate):
                step *= line_search_beta
                continue

            phi_candidate = barrier_objective(problem, candidate, t)
            if np.isfinite(phi_candidate) and phi_candidate <= phi_x + line_search_alpha * step * slope:
                accepted = True
                break
            step *= line_search_beta

        if not accepted:
            raise RuntimeError("Backtracking line-search failed to find a feasible descent step.")

        x = candidate

    return x, max_inner_iters, last_grad_norm


def log_barrier_method(
    problem: InequalityConstrainedProblem,
    x0: Array,
    *,
    t0: float = 1.0,
    mu: float = 8.0,
    duality_gap_tol: float = 1e-7,
    max_outer_iters: int = 12,
    inner_tol: float = 1e-9,
    max_inner_iters: int = 500,
    line_search_alpha: float = 1e-4,
    line_search_beta: float = 0.5,
) -> Tuple[Array, List[BarrierIterRecord]]:
    """Solve inequality-constrained optimization by the log-barrier method."""
    if t0 <= 0.0:
        raise ValueError("t0 must be positive.")
    if mu <= 1.0:
        raise ValueError("mu must be > 1.")
    if duality_gap_tol <= 0.0:
        raise ValueError("duality_gap_tol must be positive.")
    if max_outer_iters <= 0:
        raise ValueError("max_outer_iters must be positive.")
    if inner_tol <= 0.0:
        raise ValueError("inner_tol must be positive.")
    if max_inner_iters <= 0:
        raise ValueError("max_inner_iters must be positive.")
    if not (0.0 < line_search_alpha < 0.5):
        raise ValueError("line_search_alpha must be in (0, 0.5).")
    if not (0.0 < line_search_beta < 1.0):
        raise ValueError("line_search_beta must be in (0, 1).")

    x = np.asarray(x0, dtype=float).copy()
    validate_problem(problem, x)

    m = np.asarray(problem.ineq_constraints(x), dtype=float).shape[0]
    if m == 0:
        raise ValueError("Barrier method requires at least one inequality constraint.")

    t = float(t0)
    history: List[BarrierIterRecord] = []

    for outer in range(1, max_outer_iters + 1):
        x, inner_iters, inner_grad_norm = solve_barrier_subproblem(
            problem,
            x,
            t,
            inner_tol=inner_tol,
            max_inner_iters=max_inner_iters,
            line_search_alpha=line_search_alpha,
            line_search_beta=line_search_beta,
        )

        f_raw = float(problem.objective(x))
        g = np.asarray(problem.ineq_constraints(x), dtype=float)
        max_g = float(np.max(g))
        barrier_value = float(barrier_objective(problem, x, t))
        surrogate_gap = float(m / t)

        history.append(
            BarrierIterRecord(
                outer_iter=outer,
                t=t,
                x=x.copy(),
                f_raw=f_raw,
                max_g=max_g,
                barrier_value=barrier_value,
                surrogate_duality_gap=surrogate_gap,
                inner_iterations=inner_iters,
                inner_grad_norm=inner_grad_norm,
            )
        )

        if not np.isfinite(barrier_value):
            raise RuntimeError("Barrier objective became non-finite during outer iterations.")

        if surrogate_gap < duality_gap_tol:
            return x, history

        t *= mu

    raise RuntimeError(
        "Barrier method did not reach duality-gap tolerance. "
        f"final surrogate gap={history[-1].surrogate_duality_gap:.3e}, target={duality_gap_tol:.3e}."
    )


def build_demo_problem() -> Tuple[InequalityConstrainedProblem, Array, Array]:
    """Build a tiny convex inequality-constrained problem with known boundary optimum.

    Problem:
      min f(x1,x2) = (x1-1.5)^2 + (x2-0.5)^2
      s.t. g1(x)=x1+x2-1 <= 0
           g2(x)=-x1 <= 0
           g3(x)=-x2 <= 0

    Feasible set is a triangle; constrained optimum is x*=[1,0].
    """

    def objective(x: Array) -> float:
        return float((x[0] - 1.5) ** 2 + (x[1] - 0.5) ** 2)

    def objective_grad(x: Array) -> Array:
        return np.array([2.0 * (x[0] - 1.5), 2.0 * (x[1] - 0.5)], dtype=float)

    def objective_hessian(x: Array) -> Array:
        _ = x
        return np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float)

    def ineq_constraints(x: Array) -> Array:
        return np.array([x[0] + x[1] - 1.0, -x[0], -x[1]], dtype=float)

    def ineq_jacobian(x: Array) -> Array:
        _ = x
        return np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float)

    problem = InequalityConstrainedProblem(
        dimension=2,
        description=(
            "min (x1-1.5)^2 + (x2-0.5)^2, s.t. x1+x2<=1, x1>=0, x2>=0; "
            "true constrained optimum at x*=[1,0]."
        ),
        objective=objective,
        objective_grad=objective_grad,
        objective_hessian=objective_hessian,
        ineq_constraints=ineq_constraints,
        ineq_jacobian=ineq_jacobian,
    )

    x0 = np.array([0.4, 0.4], dtype=float)
    known_solution = np.array([1.0, 0.0], dtype=float)
    return problem, x0, known_solution


def solve_reference_slsqp(problem: InequalityConstrainedProblem, x0: Array) -> Optional[Tuple[Array, float]]:
    """Reference constrained solve with SLSQP (validation only)."""
    if minimize is None:
        return None

    g0 = np.asarray(problem.ineq_constraints(x0), dtype=float)
    constraints = []
    for idx in range(g0.shape[0]):
        constraints.append(
            {
                "type": "ineq",
                # SLSQP expects c(z) >= 0 while we define g(z) <= 0.
                "fun": lambda z, i=idx: float(-problem.ineq_constraints(np.asarray(z, dtype=float))[i]),
                "jac": lambda z, i=idx: np.asarray(
                    -problem.ineq_jacobian(np.asarray(z, dtype=float))[i],
                    dtype=float,
                ),
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


def print_history(history: List[BarrierIterRecord]) -> None:
    """Print a compact table for outer iterations."""
    print("outer | t            | f(x)         | max g(x)     | phi_t(x)     | m/t          | inner | ||grad||")
    print("-" * 106)
    for rec in history:
        print(
            f"{rec.outer_iter:>5d} | {rec.t:>12.4e} | {rec.f_raw:>11.7f} | "
            f"{rec.max_g:>11.4e} | {rec.barrier_value:>11.7f} | "
            f"{rec.surrogate_duality_gap:>11.4e} | {rec.inner_iterations:>5d} | {rec.inner_grad_norm:>8.2e}"
        )


def main() -> None:
    problem, x0, known_solution = build_demo_problem()

    print("=== Log-Barrier Method Demo ===")
    print(problem.description)
    print(f"initial x0: {np.array2string(x0, precision=6)}")

    x_sol, history = log_barrier_method(
        problem,
        x0,
        t0=1.0,
        mu=8.0,
        duality_gap_tol=1e-7,
        max_outer_iters=12,
        inner_tol=1e-9,
        max_inner_iters=500,
        line_search_alpha=1e-4,
        line_search_beta=0.5,
    )

    print()
    print_history(history)

    g = np.asarray(problem.ineq_constraints(x_sol), dtype=float)
    print("\nFinal barrier-method solution:")
    print(f"x*: {np.array2string(x_sol, precision=10)}")
    print(f"f(x*): {problem.objective(x_sol):.10f}")
    print(f"max g(x*): {np.max(g):.3e} (<= 0 means feasible)")
    print(f"known solution: {np.array2string(known_solution, precision=10)}")
    print(f"||x* - known||2: {np.linalg.norm(x_sol - known_solution):.3e}")

    ref = solve_reference_slsqp(problem, x0)
    if ref is None:
        print("SLSQP reference: unavailable or failed.")
    else:
        ref_x, ref_f = ref
        print("\nSLSQP reference (constraint-native solve):")
        print(f"x_ref: {np.array2string(ref_x, precision=10)}")
        print(f"f(x_ref): {ref_f:.10f}")
        print(f"||x* - x_ref||2: {np.linalg.norm(x_sol - ref_x):.3e}")


if __name__ == "__main__":
    main()
