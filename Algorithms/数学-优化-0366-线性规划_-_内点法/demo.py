"""Linear Programming via barrier interior-point path-following (minimal MVP)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


Array = np.ndarray


def validate_lp_inputs(A: Array, b: Array, c: Array, x0: Array, feas_tol: float = 1e-9) -> None:
    """Validate standard-form LP inputs and strict-feasible starting point."""
    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    if b.ndim != 1 or c.ndim != 1 or x0.ndim != 1:
        raise ValueError("b, c, x0 must be 1D arrays.")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("b dimension mismatch with A.")
    if c.shape[0] != n or x0.shape[0] != n:
        raise ValueError("c/x0 dimension mismatch with A.")
    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(c).all() and np.isfinite(x0).all()):
        raise ValueError("Input contains NaN or Inf.")
    if np.min(x0) <= 0.0:
        raise ValueError("x0 must be strictly positive for log-barrier interior-point.")
    if np.max(np.abs(A @ x0 - b)) > feas_tol:
        raise ValueError("x0 must satisfy Ax=b within tolerance.")


def kkt_residual(A: Array, b: Array, c: Array, x: Array, nu: Array, t: float) -> Tuple[Array, Array]:
    """Return dual and primal residuals for barrier KKT conditions."""
    r_dual = t * c - 1.0 / x + A.T @ nu
    r_primal = A @ x - b
    return r_dual, r_primal


def solve_kkt_step(A: Array, x: Array, r_dual: Array, r_primal: Array) -> Tuple[Array, Array]:
    """Solve Newton direction from augmented KKT system."""
    m, n = A.shape
    h_diag = 1.0 / (x * x)
    hessian = np.diag(h_diag)

    top = np.hstack([hessian, A.T])
    bottom = np.hstack([A, np.zeros((m, m), dtype=float)])
    kkt = np.vstack([top, bottom])
    rhs = -np.concatenate([r_dual, r_primal])

    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"KKT system solve failed: {exc}") from exc

    dx = sol[:n]
    dnu = sol[n:]
    return dx, dnu


def centering_newton(
    A: Array,
    b: Array,
    c: Array,
    x_init: Array,
    nu_init: Array,
    t: float,
    *,
    newton_tol: float,
    max_newton_iter: int,
    backtrack_beta: float,
    residual_c1: float,
) -> Tuple[Array, Array, int]:
    """Solve one barrier subproblem with damped Newton iterations."""
    m, _ = A.shape
    x = x_init.copy()
    if nu_init.shape != (m,):
        raise ValueError("nu_init has invalid shape.")
    nu = nu_init.copy()

    for it in range(1, max_newton_iter + 1):
        r_dual, r_primal = kkt_residual(A, b, c, x, nu, t)
        r_norm = float(np.sqrt(np.dot(r_dual, r_dual) + np.dot(r_primal, r_primal)))
        if r_norm < newton_tol * max(1.0, t):
            return x, nu, it - 1

        dx, dnu = solve_kkt_step(A, x, r_dual, r_primal)

        # Keep x strictly positive.
        alpha = 1.0
        neg_idx = dx < 0.0
        if np.any(neg_idx):
            alpha = min(alpha, 0.99 * float(np.min(-x[neg_idx] / dx[neg_idx])))

        # Residual backtracking.
        while True:
            x_next = x + alpha * dx
            nu_next = nu + alpha * dnu
            if np.min(x_next) <= 0.0:
                alpha *= backtrack_beta
                if alpha < 1e-16:
                    raise RuntimeError("Line search failed: cannot keep positivity.")
                continue

            r_dual_next, r_primal_next = kkt_residual(A, b, c, x_next, nu_next, t)
            r_norm_next = float(
                np.sqrt(np.dot(r_dual_next, r_dual_next) + np.dot(r_primal_next, r_primal_next))
            )
            if r_norm_next <= (1.0 - residual_c1 * alpha) * r_norm:
                break
            alpha *= backtrack_beta
            if alpha < 1e-16:
                raise RuntimeError("Line search failed: residual not decreasing.")

        x = x_next
        nu = nu_next

        if np.linalg.norm(alpha * dx) < 1e-14:
            return x, nu, it

    raise RuntimeError("Exceeded max_newton_iter without centering convergence.")


def barrier_path_following_lp(
    A: Array,
    b: Array,
    c: Array,
    x0: Array,
    *,
    mu: float = 12.0,
    outer_tol: float = 1e-9,
    newton_tol: float = 1e-10,
    max_outer: int = 60,
    max_newton_iter: int = 80,
    backtrack_beta: float = 0.5,
    residual_c1: float = 1e-2,
) -> Tuple[Array, List[Tuple[int, float, float, float, int, float]]]:
    """Solve LP by log-barrier path-following interior-point method."""
    if mu <= 1.0:
        raise ValueError("mu must be > 1.")
    if outer_tol <= 0.0 or newton_tol <= 0.0:
        raise ValueError("Tolerances must be positive.")
    if max_outer <= 0 or max_newton_iter <= 0:
        raise ValueError("Iteration limits must be positive integers.")
    if not (0.0 < backtrack_beta < 1.0):
        raise ValueError("backtrack_beta must be in (0, 1).")
    if not (0.0 < residual_c1 < 1.0):
        raise ValueError("residual_c1 must be in (0, 1).")

    validate_lp_inputs(A, b, c, x0)
    n = A.shape[1]
    m = A.shape[0]

    x = x0.copy()
    nu = np.zeros(m, dtype=float)
    t = 1.0
    history: List[Tuple[int, float, float, float, int, float]] = []

    for outer in range(1, max_outer + 1):
        x, nu, inner_iters = centering_newton(
            A,
            b,
            c,
            x,
            nu,
            t,
            newton_tol=newton_tol,
            max_newton_iter=max_newton_iter,
            backtrack_beta=backtrack_beta,
            residual_c1=residual_c1,
        )

        objective = float(np.dot(c, x))
        gap_est = float(n / t)
        min_x = float(np.min(x))
        history.append((outer, t, objective, gap_est, inner_iters, min_x))

        if gap_est < outer_tol:
            return x, history

        t *= mu

    raise RuntimeError("Exceeded max_outer without barrier convergence.")


def build_demo_lp_model() -> Dict[str, Array]:
    """Build a small LP and convert it to standard form with slacks.

    Original LP:
      maximize 3*x1 + 2*x2
      s.t. x1 + x2 <= 4
           x1 <= 2
           x2 <= 3
           x1, x2 >= 0
    """
    A = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],  # x1 + x2 + s1 = 4
            [1.0, 0.0, 0.0, 1.0, 0.0],  # x1 + s2 = 2
            [0.0, 1.0, 0.0, 0.0, 1.0],  # x2 + s3 = 3
        ],
        dtype=float,
    )
    b = np.array([4.0, 2.0, 3.0], dtype=float)
    c = np.array([-3.0, -2.0, 0.0, 0.0, 0.0], dtype=float)  # minimize -profit

    # Strictly feasible start: x1=1, x2=1, s1=2, s2=1, s3=2.
    x0 = np.array([1.0, 1.0, 2.0, 1.0, 2.0], dtype=float)
    return {"A": A, "b": b, "c": c, "x0": x0}


def try_scipy_reference() -> Optional[Tuple[Array, float]]:
    """Use SciPy HiGHS as optional reference (not the main solver)."""
    try:
        from scipy.optimize import linprog  # type: ignore
    except Exception:
        return None

    c = np.array([-3.0, -2.0], dtype=float)
    A_ub = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    b_ub = np.array([4.0, 2.0, 3.0], dtype=float)
    bounds = [(0.0, None), (0.0, None)]
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    return np.asarray(res.x, dtype=float), float(res.fun)


def print_outer_history(history: List[Tuple[int, float, float, float, int, float]]) -> None:
    """Pretty-print outer iterations."""
    print("outer | t            | c^T x        | gap=n/t      | inner | min(x)")
    print("-" * 72)
    for outer, t, objective, gap_est, inner_iters, min_x in history:
        print(
            f"{outer:>5d} | {t:>12.4e} | {objective:>11.6f} | "
            f"{gap_est:>11.4e} | {inner_iters:>5d} | {min_x:>8.3e}"
        )


def main() -> None:
    model = build_demo_lp_model()
    A = model["A"]
    b = model["b"]
    c = model["c"]
    x0 = model["x0"]

    x_sol, history = barrier_path_following_lp(
        A,
        b,
        c,
        x0,
        mu=12.0,
        outer_tol=1e-9,
        newton_tol=1e-10,
        max_outer=60,
        max_newton_iter=80,
    )

    print("=== Barrier Interior-Point LP Demo ===")
    print_outer_history(history)

    objective = float(np.dot(c, x_sol))
    primal_res = float(np.max(np.abs(A @ x_sol - b)))

    print("\nFinal solution (standard-form vars [x1, x2, s1, s2, s3]):")
    print(np.array2string(x_sol, precision=8, suppress_small=False))
    print(f"objective (min c^T x): {objective:.10f}")
    print(f"max|Ax-b|: {primal_res:.3e}")
    print(f"min(x): {np.min(x_sol):.3e}")

    # For original maximize problem: max 3*x1 + 2*x2 = -(min objective).
    x1, x2 = float(x_sol[0]), float(x_sol[1])
    max_profit = 3.0 * x1 + 2.0 * x2
    print(f"implied original max objective 3*x1+2*x2: {max_profit:.10f}")

    ref = try_scipy_reference()
    if ref is None:
        print("SciPy reference: unavailable (scipy not installed or solver failed).")
    else:
        ref_x, ref_fun = ref
        print("\nSciPy HiGHS reference (original 2-variable LP):")
        print(f"x_ref: {np.array2string(ref_x, precision=8)}")
        print(f"objective_ref (min -3*x1-2*x2): {ref_fun:.10f}")
        delta = abs(objective - ref_fun)
        print(f"|objective - objective_ref|: {delta:.3e}")


if __name__ == "__main__":
    main()
