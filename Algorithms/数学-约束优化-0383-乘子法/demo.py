"""Method of multipliers (augmented Lagrangian) MVP demo.

The script solves equality-constrained optimization problems:
    minimize f(x) subject to c(x) = 0

using the classical method of multipliers:
1) approximately minimize augmented Lagrangian in x
2) update multipliers lambda <- lambda + rho * c(x)

No interactive input is required. Two fixed test cases are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, float, float, float, int]


@dataclass
class SubproblemResult:
    x: Array
    aug_grad_norm: float
    iterations: int
    converged: bool
    line_search_failed: bool


@dataclass
class MultiplierResult:
    x: Array
    lam: Array
    objective: float
    constraint_norm: float
    aug_grad_norm: float
    iterations: int
    converged: bool
    message: str
    history: List[HistoryItem]


def check_vector(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def ensure_constraint_vector(cval: Array) -> Array:
    arr = np.asarray(cval, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Constraint value must be 1D and non-empty, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Constraint value contains non-finite values.")
    return arr


def ensure_jacobian_shape(jac: Array, m: int, n: int) -> Array:
    mat = np.asarray(jac, dtype=float)
    if mat.shape != (m, n):
        raise ValueError(f"Constraint Jacobian shape mismatch, expected {(m, n)}, got {mat.shape}.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Constraint Jacobian contains non-finite values.")
    return mat


def augmented_lagrangian(
    objective: Callable[[Array], float],
    constraint: Callable[[Array], Array],
    x: Array,
    lam: Array,
    rho: float,
) -> float:
    cval = ensure_constraint_vector(constraint(x))
    val = float(objective(x)) + float(lam @ cval) + 0.5 * rho * float(cval @ cval)
    if not np.isfinite(val):
        raise ValueError("Augmented Lagrangian value is non-finite.")
    return val


def augmented_gradient(
    objective_grad: Callable[[Array], Array],
    constraint: Callable[[Array], Array],
    constraint_jacobian: Callable[[Array], Array],
    x: Array,
    lam: Array,
    rho: float,
) -> Array:
    g = check_vector("objective gradient", objective_grad(x))
    cval = ensure_constraint_vector(constraint(x))
    jac = ensure_jacobian_shape(constraint_jacobian(x), m=cval.size, n=x.size)
    return g + jac.T @ (lam + rho * cval)


def solve_augmented_subproblem(
    objective: Callable[[Array], float],
    objective_grad: Callable[[Array], Array],
    constraint: Callable[[Array], Array],
    constraint_jacobian: Callable[[Array], Array],
    x_init: Array,
    lam: Array,
    rho: float,
    grad_tol: float,
    max_iter: int,
    c1: float = 1e-4,
    shrink: float = 0.5,
    min_step: float = 1e-12,
    max_backtracks: int = 40,
) -> SubproblemResult:
    x = check_vector("x_init", x_init).copy()

    if grad_tol <= 0.0:
        raise ValueError("grad_tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if not (0.0 < c1 < 1.0):
        raise ValueError("c1 must be in (0, 1).")
    if not (0.0 < shrink < 1.0):
        raise ValueError("shrink must be in (0, 1).")
    if min_step <= 0.0:
        raise ValueError("min_step must be > 0.")
    if max_backtracks <= 0:
        raise ValueError("max_backtracks must be > 0.")

    for it in range(1, max_iter + 1):
        g_aug = augmented_gradient(objective_grad, constraint, constraint_jacobian, x, lam, rho)
        g_norm = float(np.linalg.norm(g_aug))
        if g_norm <= grad_tol * (1.0 + float(np.linalg.norm(x))):
            return SubproblemResult(
                x=x,
                aug_grad_norm=g_norm,
                iterations=it - 1,
                converged=True,
                line_search_failed=False,
            )

        direction = -g_aug
        slope = float(g_aug @ direction)
        if slope >= 0.0:
            return SubproblemResult(
                x=x,
                aug_grad_norm=g_norm,
                iterations=it - 1,
                converged=False,
                line_search_failed=True,
            )

        fx_aug = augmented_lagrangian(objective, constraint, x, lam, rho)
        alpha = 1.0
        accepted = False

        for _ in range(max_backtracks):
            trial = x + alpha * direction
            trial_aug = augmented_lagrangian(objective, constraint, trial, lam, rho)
            if trial_aug <= fx_aug + c1 * alpha * slope:
                x = trial
                accepted = True
                break
            alpha *= shrink
            if alpha < min_step:
                break

        if not accepted:
            return SubproblemResult(
                x=x,
                aug_grad_norm=g_norm,
                iterations=it,
                converged=False,
                line_search_failed=True,
            )

    final_grad = augmented_gradient(objective_grad, constraint, constraint_jacobian, x, lam, rho)
    return SubproblemResult(
        x=x,
        aug_grad_norm=float(np.linalg.norm(final_grad)),
        iterations=max_iter,
        converged=False,
        line_search_failed=False,
    )


def multiplier_method(
    objective: Callable[[Array], float],
    objective_grad: Callable[[Array], Array],
    constraint: Callable[[Array], Array],
    constraint_jacobian: Callable[[Array], Array],
    x0: Array,
    lam0: Optional[Array] = None,
    rho0: float = 1.0,
    rho_scale: float = 2.0,
    rho_max: float = 1e6,
    tol_constraint: float = 1e-8,
    tol_aug_grad: float = 1e-6,
    max_outer_iter: int = 50,
    max_inner_iter: int = 400,
) -> MultiplierResult:
    x = check_vector("x0", x0).copy()
    c0 = ensure_constraint_vector(constraint(x))

    if lam0 is None:
        lam = np.zeros_like(c0)
    else:
        lam = check_vector("lam0", lam0).copy()
        if lam.size != c0.size:
            raise ValueError(f"lam0 dimension mismatch, expected {c0.size}, got {lam.size}.")

    if rho0 <= 0.0:
        raise ValueError("rho0 must be > 0.")
    if rho_scale <= 1.0:
        raise ValueError("rho_scale must be > 1.")
    if rho_max <= 0.0:
        raise ValueError("rho_max must be > 0.")
    if tol_constraint <= 0.0:
        raise ValueError("tol_constraint must be > 0.")
    if tol_aug_grad <= 0.0:
        raise ValueError("tol_aug_grad must be > 0.")
    if max_outer_iter <= 0:
        raise ValueError("max_outer_iter must be > 0.")
    if max_inner_iter <= 0:
        raise ValueError("max_inner_iter must be > 0.")

    rho = float(rho0)
    history: List[HistoryItem] = []
    converged = False
    message = "max_outer_iter reached"
    prev_c_norm = float("inf")

    for k in range(1, max_outer_iter + 1):
        sub = solve_augmented_subproblem(
            objective=objective,
            objective_grad=objective_grad,
            constraint=constraint,
            constraint_jacobian=constraint_jacobian,
            x_init=x,
            lam=lam,
            rho=rho,
            grad_tol=tol_aug_grad,
            max_iter=max_inner_iter,
        )
        x = sub.x

        cval = ensure_constraint_vector(constraint(x))
        c_norm = float(np.linalg.norm(cval))
        aug_grad = augmented_gradient(objective_grad, constraint, constraint_jacobian, x, lam, rho)
        aug_grad_norm = float(np.linalg.norm(aug_grad))
        obj = float(objective(x))

        history.append(
            (
                k,
                obj,
                c_norm,
                aug_grad_norm,
                rho,
                float(np.linalg.norm(lam)),
                sub.iterations,
            )
        )

        cons_tol = tol_constraint * (1.0 + float(np.linalg.norm(x)))
        grad_tol = tol_aug_grad * (1.0 + float(np.linalg.norm(x)))
        if c_norm <= cons_tol and aug_grad_norm <= grad_tol:
            converged = True
            message = "constraint and augmented-gradient tolerances reached"
            break

        lam = lam + rho * cval

        if c_norm > 0.5 * prev_c_norm:
            rho = min(rho * rho_scale, rho_max)
        prev_c_norm = c_norm

        if sub.line_search_failed:
            message = "inner line search failed"
            break

    final_c = ensure_constraint_vector(constraint(x))
    final_aug_grad = augmented_gradient(objective_grad, constraint, constraint_jacobian, x, lam, rho)

    return MultiplierResult(
        x=x,
        lam=lam,
        objective=float(objective(x)),
        constraint_norm=float(np.linalg.norm(final_c)),
        aug_grad_norm=float(np.linalg.norm(final_aug_grad)),
        iterations=len(history),
        converged=converged,
        message=message,
        history=history,
    )


def print_history(history: List[HistoryItem], max_rows: int = 12) -> None:
    print("iter | objective         | ||c(x)||          | ||grad L_rho||    | rho              | ||lambda||        | inner_iters")
    print("-" * 124)
    for row in history[:max_rows]:
        it, obj, c_norm, grad_norm, rho, lam_norm, inner_iters = row
        print(
            f"{it:4d} | {obj:16.9e} | {c_norm:16.9e} | {grad_norm:16.9e} | {rho:16.9e} | {lam_norm:16.9e} | {inner_iters:11d}"
        )
    if len(history) > max_rows:
        print(f"... ({len(history) - max_rows} more outer iterations omitted)")


def run_case(
    name: str,
    objective: Callable[[Array], float],
    objective_grad: Callable[[Array], Array],
    constraint: Callable[[Array], Array],
    constraint_jacobian: Callable[[Array], Array],
    x0: Array,
    reference_x: Optional[Array],
    reference_lam: Optional[Array],
) -> dict:
    print(f"\n=== Case: {name} ===")
    result = multiplier_method(
        objective=objective,
        objective_grad=objective_grad,
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        x0=x0,
        rho0=1.0,
        rho_scale=2.0,
        rho_max=1e5,
        tol_constraint=1e-8,
        tol_aug_grad=1e-6,
        max_outer_iter=40,
        max_inner_iter=500,
    )

    print(f"Converged: {result.converged}")
    print(f"Stop reason: {result.message}")
    print(f"Outer iterations: {result.iterations}")
    print(f"Final x: {result.x}")
    print(f"Final lambda: {result.lam}")
    print(f"Final objective: {result.objective:.12e}")
    print(f"Final ||c(x)||: {result.constraint_norm:.12e}")
    print(f"Final ||grad L_rho||: {result.aug_grad_norm:.12e}")
    print("Iteration trace:")
    print_history(result.history)

    rel_x_error = np.nan
    rel_lam_error = np.nan

    if reference_x is not None:
        ref_x = check_vector("reference_x", reference_x)
        abs_x_error = float(np.linalg.norm(result.x - ref_x))
        rel_x_error = abs_x_error / (float(np.linalg.norm(ref_x)) + 1e-15)
        print(f"Reference x*: {ref_x}")
        print(f"Absolute x error: {abs_x_error:.12e}")
        print(f"Relative x error: {rel_x_error:.12e}")

    if reference_lam is not None:
        ref_lam = check_vector("reference_lam", reference_lam)
        abs_lam_error = float(np.linalg.norm(result.lam - ref_lam))
        rel_lam_error = abs_lam_error / (float(np.linalg.norm(ref_lam)) + 1e-15)
        print(f"Reference lambda*: {ref_lam}")
        print(f"Absolute lambda error: {abs_lam_error:.12e}")
        print(f"Relative lambda error: {rel_lam_error:.12e}")

    return {
        "converged": float(result.converged),
        "constraint_norm": result.constraint_norm,
        "aug_grad_norm": result.aug_grad_norm,
        "rel_x_error": rel_x_error,
        "rel_lam_error": rel_lam_error,
    }


def build_case_linear_equality() -> dict:
    # min (x1-1)^2 + (x2-2)^2 s.t. x1 + x2 - 2 = 0
    def objective(x: Array) -> float:
        return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)

    def objective_grad(x: Array) -> Array:
        return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)], dtype=float)

    def constraint(x: Array) -> Array:
        return np.array([x[0] + x[1] - 2.0], dtype=float)

    def constraint_jacobian(_: Array) -> Array:
        return np.array([[1.0, 1.0]], dtype=float)

    return {
        "name": "Linear-Equality-Quadratic",
        "objective": objective,
        "objective_grad": objective_grad,
        "constraint": constraint,
        "constraint_jacobian": constraint_jacobian,
        "x0": np.array([3.5, -1.0], dtype=float),
        "reference_x": np.array([0.5, 1.5], dtype=float),
        "reference_lam": np.array([1.0], dtype=float),
    }


def build_case_nonlinear_constraint() -> dict:
    # min x1^2 + x2^2 s.t. x1 * x2 - 1 = 0
    # Positive initial point targets the solution near [1, 1].
    def objective(x: Array) -> float:
        return float(x[0] ** 2 + x[1] ** 2)

    def objective_grad(x: Array) -> Array:
        return np.array([2.0 * x[0], 2.0 * x[1]], dtype=float)

    def constraint(x: Array) -> Array:
        return np.array([x[0] * x[1] - 1.0], dtype=float)

    def constraint_jacobian(x: Array) -> Array:
        return np.array([[x[1], x[0]]], dtype=float)

    return {
        "name": "Nonlinear-Constraint-Hyperbola",
        "objective": objective,
        "objective_grad": objective_grad,
        "constraint": constraint,
        "constraint_jacobian": constraint_jacobian,
        "x0": np.array([1.5, 0.8], dtype=float),
        "reference_x": np.array([1.0, 1.0], dtype=float),
        "reference_lam": np.array([-2.0], dtype=float),
    }


def main() -> None:
    cases = [
        build_case_linear_equality(),
        build_case_nonlinear_constraint(),
    ]

    stats = []
    for case in cases:
        stats.append(
            run_case(
                name=case["name"],
                objective=case["objective"],
                objective_grad=case["objective_grad"],
                constraint=case["constraint"],
                constraint_jacobian=case["constraint_jacobian"],
                x0=case["x0"],
                reference_x=case["reference_x"],
                reference_lam=case["reference_lam"],
            )
        )

    all_converged = all(bool(item["converged"]) for item in stats)
    max_constraint_norm = max(item["constraint_norm"] for item in stats)
    max_aug_grad_norm = max(item["aug_grad_norm"] for item in stats)

    rel_x_candidates = [item["rel_x_error"] for item in stats if np.isfinite(item["rel_x_error"])]
    rel_lam_candidates = [item["rel_lam_error"] for item in stats if np.isfinite(item["rel_lam_error"])]
    max_rel_x_error = max(rel_x_candidates) if rel_x_candidates else np.nan
    max_rel_lam_error = max(rel_lam_candidates) if rel_lam_candidates else np.nan

    print("\n=== Summary ===")
    print(f"All cases converged: {all_converged}")
    print(f"Max final ||c(x)||: {max_constraint_norm:.12e}")
    print(f"Max final ||grad L_rho||: {max_aug_grad_norm:.12e}")
    print(f"Max relative x error: {max_rel_x_error:.12e}")
    print(f"Max relative lambda error: {max_rel_lam_error:.12e}")

    if not all_converged:
        raise RuntimeError("At least one case did not converge.")
    if max_constraint_norm > 1e-6:
        raise RuntimeError(f"Constraint violation too large: {max_constraint_norm}")
    if max_aug_grad_norm > 1e-4:
        raise RuntimeError(f"Augmented gradient too large: {max_aug_grad_norm}")
    if np.isfinite(max_rel_x_error) and max_rel_x_error > 5e-4:
        raise RuntimeError(f"Relative x error too large: {max_rel_x_error}")
    if np.isfinite(max_rel_lam_error) and max_rel_lam_error > 2e-2:
        raise RuntimeError(f"Relative lambda error too large: {max_rel_lam_error}")

    print("Validation checks passed.")


if __name__ == "__main__":
    main()
