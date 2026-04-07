"""Frank-Wolfe algorithm MVP on a simplex-constrained convex quadratic program.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize


@dataclass
class FWResult:
    x: np.ndarray
    history: Dict[str, List[float]]
    iterations: int
    converged: bool


def quadratic_objective(x: np.ndarray, q: np.ndarray, c: np.ndarray) -> float:
    """Compute f(x) = 0.5 * x^T Q x + c^T x."""
    return float(0.5 * x @ (q @ x) + c @ x)


def quadratic_gradient(x: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute grad f(x) = Qx + c."""
    return q @ x + c


def is_on_simplex(x: np.ndarray, atol: float = 1e-8) -> bool:
    """Check x >= 0 and sum(x) = 1 within tolerance."""
    return bool(np.min(x) >= -atol and abs(np.sum(x) - 1.0) <= atol)


def lmo_on_simplex(grad: np.ndarray) -> np.ndarray:
    """Linear minimization oracle on simplex: argmin_s <grad, s> is one-hot at argmin grad."""
    s = np.zeros_like(grad)
    s[int(np.argmin(grad))] = 1.0
    return s


def exact_line_search_for_quadratic(
    x: np.ndarray,
    s: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-14,
) -> float:
    """Closed-form line search on segment x + gamma*(s-x), gamma in [0,1]."""
    d = s - x
    grad_x = quadratic_gradient(x, q, c)
    denom = float(d @ (q @ d))
    numer = -float(d @ grad_x)

    if denom <= eps:
        # Nearly linear along direction: choose boundary minimizing first-order model.
        return 1.0 if numer > 0.0 else 0.0

    gamma = numer / denom
    return float(np.clip(gamma, 0.0, 1.0))


def frank_wolfe_simplex_qp(
    q: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 1500,
    tol_gap: float = 1e-8,
    use_exact_line_search: bool = True,
) -> FWResult:
    """Solve min 0.5 x^T Q x + c^T x s.t. x in simplex via Frank-Wolfe."""
    if q.shape[0] != q.shape[1]:
        raise ValueError("q must be square")
    if q.shape[0] != c.shape[0] or c.shape[0] != x0.shape[0]:
        raise ValueError("dimension mismatch among q, c, x0")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    x = x0.astype(float).copy()
    if not is_on_simplex(x):
        raise ValueError("x0 must lie on simplex")

    history: Dict[str, List[float]] = {
        "objective": [],
        "gap": [],
        "step_size": [],
    }

    converged = False

    for k in range(max_iter):
        grad = quadratic_gradient(x, q, c)
        s = lmo_on_simplex(grad)
        gap = float((x - s) @ grad)
        obj = quadratic_objective(x, q, c)

        history["objective"].append(obj)
        history["gap"].append(gap)

        if gap <= tol_gap:
            history["step_size"].append(0.0)
            converged = True
            return FWResult(x=x, history=history, iterations=k + 1, converged=converged)

        if use_exact_line_search:
            gamma = exact_line_search_for_quadratic(x, s, q, c)
        else:
            gamma = 2.0 / (k + 2.0)

        history["step_size"].append(float(gamma))
        x = x + gamma * (s - x)

    return FWResult(x=x, history=history, iterations=max_iter, converged=converged)


def solve_reference_with_slsqp(q: np.ndarray, c: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Reference solution using scipy SLSQP for validation."""

    def fun(z: np.ndarray) -> float:
        return quadratic_objective(z, q, c)

    def jac(z: np.ndarray) -> np.ndarray:
        return quadratic_gradient(z, q, c)

    n = x0.shape[0]
    constraints = [{"type": "eq", "fun": lambda z: np.sum(z) - 1.0, "jac": lambda z: np.ones_like(z)}]
    bounds = [(0.0, 1.0) for _ in range(n)]

    res = minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 5000, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"Reference solver failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def build_quadratic_problem(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reproducible strongly-convex quadratic problem on the simplex."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    q = (a.T @ a) / float(n)
    q = q + 0.15 * np.eye(n)  # Strong convexity margin.
    c = rng.normal(loc=0.0, scale=0.3, size=n)
    x0 = np.full(n, 1.0 / n)
    return q, c, x0


def main() -> None:
    n = 20
    q, c, x0 = build_quadratic_problem(n=n, seed=11)

    result = frank_wolfe_simplex_qp(
        q=q,
        c=c,
        x0=x0,
        max_iter=2000,
        tol_gap=1e-8,
        use_exact_line_search=True,
    )

    x_fw = result.x
    obj_fw = quadratic_objective(x_fw, q, c)
    gap_fw = result.history["gap"][-1]

    x_ref = solve_reference_with_slsqp(q=q, c=c, x0=x0)
    obj_ref = quadratic_objective(x_ref, q, c)

    feasibility_sum_err = abs(float(np.sum(x_fw) - 1.0))
    feasibility_min = float(np.min(x_fw))
    objective_drop = result.history["objective"][0] - obj_fw
    obj_diff_vs_ref = obj_fw - obj_ref

    print("=== Frank-Wolfe on Simplex-Constrained Quadratic Program ===")
    print(f"dimension n: {n}")
    print(f"iterations: {result.iterations}")
    print(f"converged by gap: {result.converged}")
    print(f"final objective (FW): {obj_fw:.12f}")
    print(f"reference objective (SLSQP): {obj_ref:.12f}")
    print(f"objective diff (FW - REF): {obj_diff_vs_ref:.3e}")
    print(f"final FW gap: {gap_fw:.3e}")
    print(f"sum(x)-1 error: {feasibility_sum_err:.3e}")
    print(f"min(x): {feasibility_min:.3e}")
    print(f"objective decrease from iter0: {objective_drop:.3e}")

    # MVP quality checks.
    if not is_on_simplex(x_fw, atol=1e-6):
        raise RuntimeError("Feasibility check failed: FW iterate is not on simplex.")
    if objective_drop <= 0.0:
        raise RuntimeError("Convergence check failed: objective did not decrease.")
    if obj_diff_vs_ref > 5e-6:
        raise RuntimeError(
            "Accuracy check failed: FW objective is not sufficiently close to SLSQP reference."
        )

    print("Quality checks passed.")


if __name__ == "__main__":
    main()
