"""Nonlinear interior-point (log-barrier) minimal runnable MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class BarrierConfig:
    t0: float = 1.0
    mu: float = 8.0
    duality_gap_tol: float = 1e-8
    newton_tol: float = 1e-12
    max_outer_iter: int = 30
    max_newton_iter: int = 80
    armijo_c1: float = 1e-4
    backtrack_beta: float = 0.5


@dataclass
class CenteringReport:
    iters: int
    final_phi: float
    final_decrement2: float


# Demo problem:
#   minimize   f(x) = (x1 - 1)^2 + (x2 - 2)^2
#   subject to g1(x) = x1 + x2 - 2 <= 0
#              g2(x) = -x1 <= 0
#              g3(x) = -x2 <= 0
# True optimum is x* = [0.5, 1.5].

def objective(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)


def grad_objective(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)], dtype=float)


def hess_objective(_: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float)


def ineq_constraints(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] + x[1] - 2.0, -x[0], -x[1]], dtype=float)


def ineq_jacobian(_: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )


def barrier_value_grad_hess(x: np.ndarray, t: float) -> tuple[float, np.ndarray, np.ndarray]:
    c = ineq_constraints(x)
    if np.any(c >= 0.0):
        raise ValueError("Point is not strictly feasible for barrier objective.")

    j = ineq_jacobian(x)
    minus_c = -c

    f = objective(x)
    gf = grad_objective(x)
    hf = hess_objective(x)

    barrier_val = -np.sum(np.log(minus_c))
    barrier_grad = np.sum(j / minus_c[:, None], axis=0)

    barrier_hess = np.zeros((x.size, x.size), dtype=float)
    for i in range(c.size):
        ji = j[i]
        barrier_hess += np.outer(ji, ji) / (minus_c[i] ** 2)

    phi = t * f + barrier_val
    grad_phi = t * gf + barrier_grad
    hess_phi = t * hf + barrier_hess
    return float(phi), grad_phi, hess_phi


def solve_newton_direction(hess: np.ndarray, grad: np.ndarray) -> np.ndarray:
    rhs = -grad
    for power in range(8):
        reg = 1e-12 * (10.0**power)
        try:
            return np.linalg.solve(hess + reg * np.eye(hess.shape[0]), rhs)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.lstsq(hess, rhs, rcond=None)[0]


def centering_newton(
    x0: np.ndarray,
    t: float,
    cfg: BarrierConfig,
) -> tuple[np.ndarray, CenteringReport]:
    x = x0.copy()

    for k in range(1, cfg.max_newton_iter + 1):
        phi, grad_phi, hess_phi = barrier_value_grad_hess(x, t)
        direction = solve_newton_direction(hess_phi, grad_phi)
        decrement2 = float(-grad_phi @ direction)

        if decrement2 / 2.0 <= cfg.newton_tol:
            return x, CenteringReport(k, phi, decrement2)

        alpha = 1.0

        # Keep strict feasibility.
        while np.any(ineq_constraints(x + alpha * direction) >= 0.0):
            alpha *= cfg.backtrack_beta
            if alpha < 1e-16:
                raise RuntimeError("Line search failed to keep strict feasibility.")

        # Armijo decrease on barrier objective.
        while True:
            cand = x + alpha * direction
            cand_phi, _, _ = barrier_value_grad_hess(cand, t)
            if cand_phi <= phi + cfg.armijo_c1 * alpha * (grad_phi @ direction):
                break
            alpha *= cfg.backtrack_beta
            if alpha < 1e-16:
                raise RuntimeError("Line search failed to satisfy Armijo condition.")

        x = x + alpha * direction

    raise RuntimeError("Centering Newton did not converge within max_newton_iter.")


def interior_point(
    x0: np.ndarray,
    cfg: BarrierConfig,
) -> tuple[np.ndarray, list[tuple[int, float, float, int, float]]]:
    if np.any(ineq_constraints(x0) >= 0.0):
        raise ValueError("x0 must be strictly feasible for interior-point method.")

    x = x0.astype(float).copy()
    m = ineq_constraints(x).size
    t = cfg.t0
    logs: list[tuple[int, float, float, int, float]] = []

    for outer_iter in range(1, cfg.max_outer_iter + 1):
        x, report = centering_newton(x, t, cfg)
        gap_est = m / t
        logs.append((outer_iter, t, gap_est, report.iters, report.final_phi))
        if gap_est < cfg.duality_gap_tol:
            return x, logs
        t *= cfg.mu

    raise RuntimeError("Barrier loop did not satisfy duality-gap tolerance in time.")


def solve_with_scipy_reference(x0: np.ndarray) -> np.ndarray:
    constraints = [
        {"type": "ineq", "fun": lambda x: 2.0 - x[0] - x[1]},
        {"type": "ineq", "fun": lambda x: x[0]},
        {"type": "ineq", "fun": lambda x: x[1]},
    ]
    res = minimize(
        fun=objective,
        x0=x0,
        jac=grad_objective,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 200, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"SciPy reference solver failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    cfg = BarrierConfig()
    x0 = np.array([0.2, 0.2], dtype=float)  # strictly feasible

    x_ipm, logs = interior_point(x0=x0, cfg=cfg)
    x_ref = solve_with_scipy_reference(x0=x0)
    x_true = np.array([0.5, 1.5], dtype=float)

    print("== Nonlinear Interior-Point (Barrier + Newton) Demo ==")
    print(f"start x0: {x0}")
    print("\nouter_iter | t            | m/t          | newton_iters | centered_phi")
    for outer_iter, t, gap_est, newton_iters, phi in logs:
        print(
            f"{outer_iter:9d} | {t:12.4e} | {gap_est:12.4e} |"
            f" {newton_iters:12d} | {phi:12.6e}"
        )

    print("\nsolution (IPM):", x_ipm)
    print("solution (SciPy SLSQP ref):", x_ref)
    print("analytic solution:", x_true)

    print("\nobjective(IPM):", f"{objective(x_ipm):.10f}")
    print("max constraint value (<=0 feasible):", f"{np.max(ineq_constraints(x_ipm)):.3e}")
    print("distance to analytic solution:", f"{np.linalg.norm(x_ipm - x_true):.3e}")
    print("distance to SciPy reference:", f"{np.linalg.norm(x_ipm - x_ref):.3e}")


if __name__ == "__main__":
    main()
