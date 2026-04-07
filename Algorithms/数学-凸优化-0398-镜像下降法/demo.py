"""Mirror Descent MVP on simplex-constrained convex quadratic optimization.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize


@dataclass
class MirrorDescentResult:
    x_last: np.ndarray
    x_best: np.ndarray
    history: Dict[str, List[float]]
    iterations: int
    converged: bool


def quadratic_objective(x: np.ndarray, q: np.ndarray, c: np.ndarray) -> float:
    """Compute f(x) = 0.5 * x^T Q x + c^T x."""
    return float(0.5 * x @ (q @ x) + c @ x)


def quadratic_gradient(x: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute grad f(x) = Qx + c."""
    return q @ x + c


def is_on_simplex(x: np.ndarray, atol: float = 1e-9) -> bool:
    """Check x >= 0 and sum(x) = 1 within tolerance."""
    return bool(np.min(x) >= -atol and abs(float(np.sum(x)) - 1.0) <= atol)


def simplex_linear_oracle(grad: np.ndarray) -> np.ndarray:
    """Return s = argmin_{s in simplex} <grad, s>, i.e. one-hot at argmin grad."""
    s = np.zeros_like(grad)
    s[int(np.argmin(grad))] = 1.0
    return s


def mirror_descent_entropy_simplex(
    q: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 6000,
    eta0: float = 1.2,
    tol_gap: float = 1e-8,
    eps: float = 1e-15,
) -> MirrorDescentResult:
    """Mirror Descent with negative-entropy mirror map on simplex."""
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix")
    n = q.shape[0]
    if c.shape != (n,) or x0.shape != (n,):
        raise ValueError("c and x0 must match q dimension")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if eta0 <= 0.0:
        raise ValueError("eta0 must be positive")
    if tol_gap <= 0.0:
        raise ValueError("tol_gap must be positive")

    x = x0.astype(float).copy()
    if not is_on_simplex(x):
        raise ValueError("x0 must lie on simplex")

    # Keep strict positivity for log-domain update while preserving simplex feasibility.
    x = np.clip(x, eps, None)
    x = x / float(np.sum(x))

    history: Dict[str, List[float]] = {
        "objective": [],
        "gap": [],
        "step_size": [],
        "step_norm": [],
    }

    best_obj = quadratic_objective(x, q, c)
    x_best = x.copy()
    converged = False

    for k in range(max_iter):
        grad = quadratic_gradient(x, q, c)
        obj = quadratic_objective(x, q, c)
        s = simplex_linear_oracle(grad)
        gap = float((x - s) @ grad)

        if not (np.isfinite(obj) and np.isfinite(gap)):
            raise RuntimeError("non-finite objective or gap encountered")

        history["objective"].append(obj)
        history["gap"].append(gap)

        if obj < best_obj:
            best_obj = obj
            x_best = x.copy()

        if gap <= tol_gap:
            history["step_size"].append(0.0)
            history["step_norm"].append(0.0)
            converged = True
            return MirrorDescentResult(
                x_last=x,
                x_best=x_best,
                history=history,
                iterations=k + 1,
                converged=converged,
            )

        eta = eta0 / np.sqrt(k + 1.0)
        log_x = np.log(np.clip(x, eps, None))
        z = log_x - eta * grad
        z = z - float(np.max(z))  # Stabilize exp for large magnitudes.

        x_next = np.exp(z)
        x_next = x_next / float(np.sum(x_next))

        step_norm = float(np.linalg.norm(x_next - x))
        if not np.isfinite(step_norm):
            raise RuntimeError("non-finite step norm encountered")

        history["step_size"].append(float(eta))
        history["step_norm"].append(step_norm)

        x = x_next

    return MirrorDescentResult(
        x_last=x,
        x_best=x_best,
        history=history,
        iterations=max_iter,
        converged=converged,
    )


def solve_reference_with_slsqp(q: np.ndarray, c: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Reference simplex-constrained solver for validation only."""

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
        options={"ftol": 1e-12, "maxiter": 4000, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"reference solver failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def build_problem(n: int = 25, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible strongly-convex quadratic objective on simplex."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    q = (a.T @ a) / float(n)
    q = q + 0.2 * np.eye(n)
    c = rng.normal(loc=0.0, scale=0.25, size=n)
    x0 = np.full(n, 1.0 / n)
    return q, c, x0


def main() -> None:
    q, c, x0 = build_problem(n=25, seed=11)

    result = mirror_descent_entropy_simplex(
        q=q,
        c=c,
        x0=x0,
        max_iter=6000,
        eta0=1.2,
        tol_gap=1e-8,
    )

    x_md = result.x_best
    x_last = result.x_last
    x_ref = solve_reference_with_slsqp(q=q, c=c, x0=x0)

    initial_obj = result.history["objective"][0]
    obj_md = quadratic_objective(x_md, q, c)
    obj_ref = quadratic_objective(x_ref, q, c)
    obj_diff = obj_md - obj_ref

    final_gap = result.history["gap"][-1]
    min_gap = min(result.history["gap"])
    objective_drop = initial_obj - obj_md

    simplex_err = abs(float(np.sum(x_md) - 1.0))
    min_component = float(np.min(x_md))
    last_obj = quadratic_objective(x_last, q, c)

    print("=== Mirror Descent (Entropy) on Simplex-Constrained Quadratic ===")
    print(f"dimension n: {x0.shape[0]}")
    print(f"iterations: {result.iterations}")
    print(f"converged by gap: {result.converged}")
    print(f"initial objective: {initial_obj:.12f}")
    print(f"final objective (best MD): {obj_md:.12f}")
    print(f"final objective (last MD): {last_obj:.12f}")
    print(f"reference objective (SLSQP): {obj_ref:.12f}")
    print(f"objective diff (best MD - REF): {obj_diff:.3e}")
    print(f"final gap: {final_gap:.3e}")
    print(f"min gap in history: {min_gap:.3e}")
    print(f"sum(x)-1 error: {simplex_err:.3e}")
    print(f"min(x): {min_component:.3e}")
    print(f"objective decrease from iter0: {objective_drop:.3e}")

    if not is_on_simplex(x_md, atol=1e-7):
        raise RuntimeError("feasibility check failed: x_best is not on simplex")
    if objective_drop <= 0.0:
        raise RuntimeError("descent check failed: objective did not decrease")
    if obj_diff > 3e-4:
        raise RuntimeError("accuracy check failed: objective gap to reference is too large")
    if min_gap > 1e-3:
        raise RuntimeError("optimality check failed: mirror descent gap did not become small enough")

    print("Quality checks passed.")


if __name__ == "__main__":
    main()
