"""Primal-dual interior-point method MVP for linear programming.

Problem form solved in this demo:
    minimize    c^T x
    subject to  A x = b
                x >= 0

Dual form:
    maximize    b^T y
    subject to  A^T y + s = c
                s >= 0

The implementation uses a Mehrotra-style predictor-corrector primal-dual IPM
with an infeasible start. Core linear algebra is written directly in NumPy
instead of calling a black-box LP solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class IterationRecord:
    iter_id: int
    primal_res: float
    dual_res: float
    mu: float
    alpha_pri: float
    alpha_dual: float
    sigma: float
    objective: float


@dataclass
class PDIPMResult:
    x: Array
    y: Array
    s: Array
    converged: bool
    iterations: int
    history: List[IterationRecord]


def validate_lp_data(a_mat: Array, b: Array, c: Array) -> Tuple[int, int]:
    if a_mat.ndim != 2:
        raise ValueError(f"A must be 2D, got shape={a_mat.shape}.")
    m, n = a_mat.shape
    if b.shape != (m,):
        raise ValueError(f"b must have shape {(m,)}, got {b.shape}.")
    if c.shape != (n,):
        raise ValueError(f"c must have shape {(n,)}, got {c.shape}.")
    if not (np.all(np.isfinite(a_mat)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        raise ValueError("A, b, c must be finite.")
    if m == 0 or n == 0:
        raise ValueError("A must be non-empty.")
    return m, n


def residual_metrics(a_mat: Array, b: Array, c: Array, x: Array, y: Array, s: Array) -> Tuple[Array, Array, Array, float, float, float]:
    r_p = a_mat @ x - b
    r_d = a_mat.T @ y + s - c
    r_c = x * s

    primal_res = float(np.linalg.norm(r_p, ord=np.inf) / (1.0 + np.linalg.norm(b, ord=np.inf)))
    dual_res = float(np.linalg.norm(r_d, ord=np.inf) / (1.0 + np.linalg.norm(c, ord=np.inf)))

    mu = float(np.dot(x, s) / x.size)
    gap = float(mu / (1.0 + abs(np.dot(c, x))))
    return r_p, r_d, r_c, primal_res, dual_res, gap


def step_to_boundary(v: Array, dv: Array, tau: float) -> float:
    negative = dv < 0.0
    if np.any(negative):
        alpha_max = float(np.min(-v[negative] / dv[negative]))
        return float(min(1.0, tau * alpha_max))
    return 1.0


def solve_newton_direction(
    a_mat: Array,
    x: Array,
    s: Array,
    r_p: Array,
    r_d: Array,
    rhs3: Array,
    regularization: float,
) -> Tuple[Array, Array, Array]:
    inv_s = 1.0 / s
    d = x * inv_s

    # Normal equation matrix: M = A * diag(x/s) * A^T.
    m_mat = (a_mat * d) @ a_mat.T
    if regularization > 0.0:
        m_mat = m_mat + regularization * np.eye(m_mat.shape[0], dtype=float)

    rhs = -r_p - a_mat @ ((rhs3 + x * r_d) * inv_s)

    try:
        dy = np.linalg.solve(m_mat, rhs)
    except np.linalg.LinAlgError:
        dy = np.linalg.lstsq(m_mat, rhs, rcond=None)[0]

    ds = -r_d - a_mat.T @ dy
    dx = (rhs3 + x * r_d + x * (a_mat.T @ dy)) * inv_s
    return dx, dy, ds


def primal_dual_interior_point(
    a_mat: Array,
    b: Array,
    c: Array,
    max_iter: int = 80,
    tol: float = 1e-8,
    tau: float = 0.995,
    regularization: float = 1e-10,
) -> PDIPMResult:
    m, n = validate_lp_data(a_mat, b, c)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1).")

    # Infeasible but strictly positive initial point.
    x = np.ones(n, dtype=float)
    s = np.ones(n, dtype=float)
    y = np.zeros(m, dtype=float)

    history: List[IterationRecord] = []

    for k in range(1, max_iter + 1):
        r_p, r_d, r_c, primal_res, dual_res, gap = residual_metrics(a_mat, b, c, x, y, s)

        if max(primal_res, dual_res, gap) <= tol:
            return PDIPMResult(x=x, y=y, s=s, converged=True, iterations=k - 1, history=history)

        mu = float(np.dot(x, s) / n)

        # Predictor (affine-scaling) direction: sigma = 0, no centering term.
        rhs3_aff = -r_c
        dx_aff, dy_aff, ds_aff = solve_newton_direction(
            a_mat=a_mat,
            x=x,
            s=s,
            r_p=r_p,
            r_d=r_d,
            rhs3=rhs3_aff,
            regularization=regularization,
        )

        alpha_aff_pri = step_to_boundary(x, dx_aff, tau=1.0)
        alpha_aff_dual = step_to_boundary(s, ds_aff, tau=1.0)

        x_aff = x + alpha_aff_pri * dx_aff
        s_aff = s + alpha_aff_dual * ds_aff
        mu_aff = float(np.dot(x_aff, s_aff) / n)

        sigma = float((mu_aff / mu) ** 3) if mu > 0.0 else 0.0
        sigma = float(np.clip(sigma, 0.0, 1.0))

        # Corrector RHS: -XSe - Dx_aff*Ds_aff*e + sigma*mu*e
        rhs3_corr = -r_c - dx_aff * ds_aff + sigma * mu * np.ones(n, dtype=float)
        dx, dy, ds = solve_newton_direction(
            a_mat=a_mat,
            x=x,
            s=s,
            r_p=r_p,
            r_d=r_d,
            rhs3=rhs3_corr,
            regularization=regularization,
        )

        alpha_pri = step_to_boundary(x, dx, tau=tau)
        alpha_dual = step_to_boundary(s, ds, tau=tau)

        x = x + alpha_pri * dx
        y = y + alpha_dual * dy
        s = s + alpha_dual * ds

        # Keep strict positivity under numerical noise.
        x = np.maximum(x, 1e-15)
        s = np.maximum(s, 1e-15)

        history.append(
            IterationRecord(
                iter_id=k,
                primal_res=primal_res,
                dual_res=dual_res,
                mu=mu,
                alpha_pri=alpha_pri,
                alpha_dual=alpha_dual,
                sigma=sigma,
                objective=float(np.dot(c, x)),
            )
        )

    return PDIPMResult(x=x, y=y, s=s, converged=False, iterations=max_iter, history=history)


def print_history(history: List[IterationRecord], max_lines: int = 15) -> None:
    print("iter | primal_res      | dual_res        | mu              | alpha_p         | alpha_d         | sigma           | c^T x")
    print("-" * 140)
    for item in history[:max_lines]:
        print(
            f"{item.iter_id:4d} | {item.primal_res:14.7e} | {item.dual_res:14.7e} | {item.mu:14.7e} | "
            f"{item.alpha_pri:14.7e} | {item.alpha_dual:14.7e} | {item.sigma:14.7e} | {item.objective: .9e}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_demo_case() -> Dict[str, float]:
    # LP in standard equality form with nonnegative variables (x3, x4 are slacks):
    #   minimize   -3 x1 - x2
    #   s.t.       x1 + x2 + x3      = 4
    #              2x1 + x2     + x4 = 5
    #              x >= 0
    a_mat = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    b = np.array([4.0, 5.0], dtype=float)
    c = np.array([-3.0, -1.0, 0.0, 0.0], dtype=float)

    # Analytic optimum for this LP.
    x_ref = np.array([2.5, 0.0, 1.5, 0.0], dtype=float)

    result = primal_dual_interior_point(a_mat=a_mat, b=b, c=c, max_iter=80, tol=1e-9)
    print_history(result.history)

    x_star = result.x
    y_star = result.y
    s_star = result.s
    obj = float(np.dot(c, x_star))
    ref_obj = float(np.dot(c, x_ref))

    rp = np.linalg.norm(a_mat @ x_star - b, ord=np.inf)
    rd = np.linalg.norm(a_mat.T @ y_star + s_star - c, ord=np.inf)
    comp = float(np.dot(x_star, s_star) / x_star.size)

    abs_err = float(np.linalg.norm(x_star - x_ref, ord=np.inf))

    print("\n=== Final Solution ===")
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"x*: {x_star}")
    print(f"y*: {y_star}")
    print(f"s*: {s_star}")
    print(f"objective c^T x: {obj:.12f}")
    print(f"reference objective: {ref_obj:.12f}")
    print(f"primal residual inf-norm: {rp:.3e}")
    print(f"dual residual inf-norm:   {rd:.3e}")
    print(f"average complementarity mu: {comp:.3e}")
    print(f"||x* - x_ref||_inf: {abs_err:.3e}")

    scipy_obj = np.nan
    try:
        from scipy.optimize import linprog  # type: ignore

        scipy_res = linprog(c=c, A_eq=a_mat, b_eq=b, bounds=[(0.0, None)] * c.size, method="highs")
        if scipy_res.success and scipy_res.x is not None:
            scipy_obj = float(c @ scipy_res.x)
            print(f"SciPy linprog objective (highs): {scipy_obj:.12f}")
            print(f"|obj - obj_scipy|: {abs(obj - scipy_obj):.3e}")
        else:
            print("SciPy linprog comparison skipped: solver did not report success.")
    except Exception:
        print("SciPy linprog comparison skipped: SciPy not available.")

    return {
        "converged": float(result.converged),
        "iterations": float(result.iterations),
        "objective": obj,
        "reference_objective": ref_obj,
        "primal_residual": float(rp),
        "dual_residual": float(rd),
        "mu": comp,
        "x_inf_error": abs_err,
        "scipy_objective": scipy_obj,
    }


def main() -> None:
    summary = run_demo_case()
    pass_flag = (
        summary["converged"] > 0.5
        and summary["primal_residual"] < 1e-7
        and summary["dual_residual"] < 1e-7
        and summary["mu"] < 1e-7
        and summary["x_inf_error"] < 1e-5
    )

    print("\n=== Summary ===")
    for key in [
        "converged",
        "iterations",
        "objective",
        "reference_objective",
        "primal_residual",
        "dual_residual",
        "mu",
        "x_inf_error",
    ]:
        print(f"{key}: {summary[key]}")
    print(f"pass: {pass_flag}")

    if not pass_flag:
        raise RuntimeError("PD-IPM demo did not meet validation thresholds.")


if __name__ == "__main__":
    main()
