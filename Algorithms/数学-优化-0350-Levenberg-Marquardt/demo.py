"""Levenberg-Marquardt algorithm MVP for nonlinear least-squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, float, float, float, int]


@dataclass
class LMResult:
    x: Array
    cost: float
    grad_inf_norm: float
    iterations: int
    accepted_steps: int
    converged: bool
    message: str
    function_evals: int
    jacobian_evals: int
    history: List[HistoryItem]


def ensure_vector(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def finite_difference_jacobian(
    residual_fun: Callable[[Array], Array],
    x: Array,
    r_at_x: Optional[Array] = None,
    eps: float = 1e-8,
) -> Array:
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    x = ensure_vector("x", x)
    r0 = ensure_vector("residual(x)", residual_fun(x) if r_at_x is None else r_at_x)
    m = r0.size
    n = x.size
    jac = np.zeros((m, n), dtype=float)

    for j in range(n):
        step = eps * (1.0 + abs(float(x[j])))
        x_step = x.copy()
        x_step[j] += step
        r_step = ensure_vector("residual(x + step)", residual_fun(x_step))
        if r_step.size != m:
            raise ValueError("Residual dimension changed during finite differences.")
        jac[:, j] = (r_step - r0) / step

    return jac


def lm_solve(
    residual_fun: Callable[[Array], Array],
    x0: Array,
    jacobian_fun: Optional[Callable[[Array], Array]] = None,
    max_iter: int = 120,
    tol_grad: float = 1e-8,
    tol_step: float = 1e-10,
    tol_cost: float = 1e-14,
    lambda0: float = 1e-3,
    max_lambda: float = 1e12,
) -> LMResult:
    x = ensure_vector("x0", x0).copy()
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol_grad <= 0.0 or tol_step <= 0.0 or tol_cost <= 0.0:
        raise ValueError("tol_grad/tol_step/tol_cost must be positive.")
    if lambda0 <= 0.0 or max_lambda <= lambda0:
        raise ValueError("Require 0 < lambda0 < max_lambda.")

    residual_evals = 0
    jacobian_evals = 0

    def eval_residual(z: Array) -> Array:
        nonlocal residual_evals
        residual_evals += 1
        return ensure_vector("residual(z)", residual_fun(z))

    def eval_jacobian(z: Array, r_at_z: Array) -> Array:
        nonlocal jacobian_evals
        jacobian_evals += 1
        if jacobian_fun is None:
            return finite_difference_jacobian(residual_fun=residual_fun, x=z, r_at_x=r_at_z)
        jac = np.asarray(jacobian_fun(z), dtype=float)
        if jac.ndim != 2 or jac.shape[0] != r_at_z.size or jac.shape[1] != z.size:
            raise ValueError(
                f"Invalid Jacobian shape {jac.shape}, expected ({r_at_z.size}, {z.size})."
            )
        if not np.all(np.isfinite(jac)):
            raise ValueError("Jacobian contains non-finite values.")
        return jac

    r = eval_residual(x)
    cost = 0.5 * float(np.dot(r, r))
    j_mat = eval_jacobian(x, r)
    a = j_mat.T @ j_mat
    g = j_mat.T @ r
    grad_inf = float(np.linalg.norm(g, ord=np.inf))

    history: List[HistoryItem] = []
    damping = float(lambda0)
    nu = 2.0
    accepted_steps = 0
    converged = False
    message = "max_iter reached"

    for k in range(1, max_iter + 1):
        grad_inf = float(np.linalg.norm(g, ord=np.inf))
        if grad_inf <= tol_grad:
            converged = True
            message = "gradient infinity-norm tolerance reached"
            break

        system = a + damping * np.eye(x.size, dtype=float)
        try:
            step = -np.linalg.solve(system, g)
        except np.linalg.LinAlgError:
            message = "linear solve failed"
            break

        step_norm = float(np.linalg.norm(step))
        if step_norm <= tol_step * (1.0 + float(np.linalg.norm(x))):
            converged = True
            message = "step tolerance reached"
            history.append((k, cost, grad_inf, step_norm, damping, float("inf"), 1))
            break

        x_trial = x + step
        r_trial = eval_residual(x_trial)
        cost_trial = 0.5 * float(np.dot(r_trial, r_trial))

        actual_reduction = cost - cost_trial
        predicted_reduction = 0.5 * float(step.T @ (damping * step - g))

        if predicted_reduction <= 0.0 or not np.isfinite(predicted_reduction):
            rho = -np.inf
        else:
            rho = actual_reduction / predicted_reduction

        accepted = int(bool(rho > 0.0 and np.isfinite(cost_trial)))
        if accepted:
            x = x_trial
            r = r_trial
            cost_prev = cost
            cost = cost_trial
            j_mat = eval_jacobian(x, r)
            a = j_mat.T @ j_mat
            g = j_mat.T @ r
            accepted_steps += 1

            damping_scale = max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            damping *= damping_scale
            damping = max(damping, 1e-15)
            nu = 2.0

            if abs(cost_prev - cost) <= tol_cost * (1.0 + abs(cost_prev)):
                converged = True
                message = "cost tolerance reached"

        else:
            damping *= nu
            nu *= 2.0
            if damping > max_lambda:
                message = "damping exceeded max_lambda"
                history.append((k, cost, grad_inf, step_norm, damping, rho, accepted))
                break

        history.append((k, cost, grad_inf, step_norm, damping, rho, accepted))
        if converged:
            break

    return LMResult(
        x=x,
        cost=cost,
        grad_inf_norm=float(np.linalg.norm(g, ord=np.inf)),
        iterations=len(history),
        accepted_steps=accepted_steps,
        converged=converged,
        message=message,
        function_evals=residual_evals,
        jacobian_evals=jacobian_evals,
        history=history,
    )


def make_exponential_case() -> Tuple[
    Callable[[Array], Array],
    Callable[[Array], Array],
    Array,
    Array,
]:
    t = np.linspace(0.0, 2.0, 30)
    true_params = np.array([2.5, -0.8, 0.5], dtype=float)

    y = true_params[0] * np.exp(true_params[1] * t) + true_params[2]

    def residual(theta: Array) -> Array:
        a, b, c = theta
        return a * np.exp(b * t) + c - y

    def jacobian(theta: Array) -> Array:
        a, b, _ = theta
        exp_bt = np.exp(b * t)
        return np.column_stack([exp_bt, a * t * exp_bt, np.ones_like(t)])

    x0 = np.array([2.2, -0.7, 0.4], dtype=float)
    return residual, jacobian, x0, true_params


def make_circle_case() -> Tuple[Callable[[Array], Array], Array, Array]:
    true_params = np.array([1.5, -0.8, 2.2], dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    points = np.column_stack(
        [
            true_params[0] + true_params[2] * np.cos(angles),
            true_params[1] + true_params[2] * np.sin(angles),
        ]
    )

    def residual(theta: Array) -> Array:
        cx, cy, r = theta
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - r

    x0 = np.array([0.0, 0.0, 1.0], dtype=float)
    return residual, x0, true_params


def relative_error(abs_error: float, ref_norm: float, eps: float = 1e-15) -> float:
    return abs(abs_error) / (abs(ref_norm) + eps)


def print_history(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("iter | cost             | ||J^T r||_inf     | ||step||         | lambda           | rho              | acc")
    print("-" * 110)
    for k, cost, grad_inf, step_norm, damping, rho, accepted in history[:max_lines]:
        print(
            f"{k:4d} | {cost:16.9e} | {grad_inf:16.9e} | {step_norm:16.9e} | "
            f"{damping:16.9e} | {rho:16.9e} | {accepted:d}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_case(
    name: str,
    residual_fun: Callable[[Array], Array],
    x0: Array,
    reference: Array,
    jacobian_fun: Optional[Callable[[Array], Array]] = None,
    max_iter: int = 120,
) -> dict:
    print(f"\n=== Case: {name} ===")
    result = lm_solve(
        residual_fun=residual_fun,
        x0=x0,
        jacobian_fun=jacobian_fun,
        max_iter=max_iter,
    )

    print(f"Converged: {result.converged}")
    print(f"Stop reason: {result.message}")
    print(f"Iterations: {result.iterations}")
    print(f"Accepted steps: {result.accepted_steps}")
    print(f"Final x: {result.x}")
    print(f"Final cost: {result.cost:.12e}")
    print(f"Final ||J^T r||_inf: {result.grad_inf_norm:.12e}")
    print(f"Residual evals: {result.function_evals}")
    print(f"Jacobian evals: {result.jacobian_evals}")
    print("Iteration trace:")
    print_history(result.history)

    abs_err = float(np.linalg.norm(result.x - reference))
    rel_err = float(relative_error(abs_err, float(np.linalg.norm(reference))))

    print(f"Reference x*: {reference}")
    print(f"Absolute x error: {abs_err:.12e}")
    print(f"Relative x error: {rel_err:.12e}")

    return {
        "converged": float(result.converged),
        "grad_inf": float(result.grad_inf_norm),
        "abs_error": abs_err,
        "rel_error": rel_err,
        "iterations": float(result.iterations),
        "cost": float(result.cost),
    }


def main() -> None:
    exp_residual, exp_jacobian, exp_x0, exp_ref = make_exponential_case()
    circle_residual, circle_x0, circle_ref = make_circle_case()

    stats = []
    stats.append(
        run_case(
            name="Exponential-Curve-Fit (analytic Jacobian)",
            residual_fun=exp_residual,
            jacobian_fun=exp_jacobian,
            x0=exp_x0,
            reference=exp_ref,
            max_iter=100,
        )
    )
    stats.append(
        run_case(
            name="Circle-Fit (finite-difference Jacobian)",
            residual_fun=circle_residual,
            jacobian_fun=None,
            x0=circle_x0,
            reference=circle_ref,
            max_iter=120,
        )
    )

    max_grad = max(item["grad_inf"] for item in stats)
    max_rel_error = max(item["rel_error"] for item in stats)
    mean_iterations = float(np.mean([item["iterations"] for item in stats]))
    all_converged = all(bool(item["converged"]) for item in stats)

    print("\n=== Summary ===")
    print(f"All cases converged: {all_converged}")
    print(f"Max final ||J^T r||_inf: {max_grad:.12e}")
    print(f"Max relative parameter error: {max_rel_error:.12e}")
    print(f"Average iterations: {mean_iterations:.2f}")

    if not all_converged:
        raise RuntimeError("At least one case did not converge.")
    if max_grad > 1e-6:
        raise RuntimeError(f"Gradient infinity norm too large: {max_grad}")
    if max_rel_error > 5e-5:
        raise RuntimeError(f"Relative parameter error too large: {max_rel_error}")

    print("Validation checks passed.")


if __name__ == "__main__":
    main()
