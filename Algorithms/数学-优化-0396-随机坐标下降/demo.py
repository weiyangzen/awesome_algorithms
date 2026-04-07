"""Minimal runnable MVP for Randomized Coordinate Descent (RCD).

This demo solves ridge regression with a randomized coordinate descent solver:

    min_x  (1/(2m)) * ||A x - b||_2^2 + (lambda/2) * ||x||_2^2

For this quadratic objective, each coordinate update uses an exact 1D minimizer:

    x_j <- x_j - grad_j / L_j,
    L_j = ||A[:, j]||_2^2 / m + lambda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RCDResult:
    """Container for RCD outputs and diagnostics."""

    x: np.ndarray
    updates_used: int
    epochs: float
    converged: bool
    objective: float
    grad_inf: float
    objective_star: float
    objective_gap: float
    relative_gap: float
    history: list[float]


def matmul_checked(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    """Compute matrix product while guarding against non-finite outputs.

    Some BLAS backends may surface benign floating-point warnings from matmul.
    We suppress warning emission here but still enforce finite-value outputs.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = left @ right

    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"Non-finite values produced in matmul: {label}")

    return out


def validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    max_updates: int,
    tol: float,
) -> None:
    """Validate shapes and numerical constraints."""
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    m, p = A.shape
    if m <= 0 or p <= 0:
        raise ValueError("A must have positive shape (m, p).")
    if b.ndim != 1 or b.shape[0] != m:
        raise ValueError("b must be a 1D vector of length m.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite values.")
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if max_updates <= 0:
        raise ValueError("max_updates must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def objective_value(residual: np.ndarray, x: np.ndarray, lam: float, m: int) -> float:
    """Compute f(x) using residual r = A x - b."""
    return 0.5 * float(residual @ residual) / float(m) + 0.5 * lam * float(x @ x)


def gradient_inf_norm(
    A: np.ndarray,
    residual: np.ndarray,
    x: np.ndarray,
    lam: float,
    m: int,
) -> float:
    """Compute ||grad f(x)||_inf for stopping and reporting."""
    grad = matmul_checked(A.T, residual, label="A.T @ residual") / float(m) + lam * x
    return float(np.linalg.norm(grad, ord=np.inf))


def solve_ridge_closed_form(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Reference closed-form solution: (A^T A / m + lam I) x = A^T b / m."""
    m, p = A.shape
    hessian = matmul_checked(A.T, A, label="A.T @ A") / float(m) + lam * np.eye(p)
    rhs = matmul_checked(A.T, b, label="A.T @ b") / float(m)
    return np.linalg.solve(hessian, rhs)


def randomized_coordinate_descent_ridge(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    max_updates: int,
    tol: float,
    seed: int,
) -> RCDResult:
    """Run randomized coordinate descent on ridge regression."""
    validate_inputs(A=A, b=b, lam=lam, max_updates=max_updates, tol=tol)

    A = A.astype(float, copy=False)
    b = b.astype(float, copy=False)
    m, p = A.shape

    # Coordinate-wise Lipschitz constants of partial gradients.
    lipschitz = np.sum(A * A, axis=0) / float(m) + lam
    if np.any(lipschitz <= 0.0):
        raise ValueError("Encountered non-positive coordinate Lipschitz constant.")

    rng = np.random.default_rng(seed)
    x = np.zeros(p, dtype=float)
    residual = -b.copy()  # because x = 0 -> A x - b = -b

    history: list[float] = [objective_value(residual=residual, x=x, lam=lam, m=m)]

    converged = False
    updates_used = 0

    for t in range(1, max_updates + 1):
        j = int(rng.integers(0, p))
        col_j = A[:, j]

        grad_j = float(col_j @ residual) / float(m) + lam * x[j]
        step = -grad_j / lipschitz[j]

        if step != 0.0:
            x[j] += step
            residual += step * col_j

        updates_used = t

        # Check stopping once per epoch (p coordinate updates).
        if t % p == 0 or t == max_updates:
            history.append(objective_value(residual=residual, x=x, lam=lam, m=m))
            grad_inf = gradient_inf_norm(A=A, residual=residual, x=x, lam=lam, m=m)
            if grad_inf <= tol:
                converged = True
                break

    objective = objective_value(residual=residual, x=x, lam=lam, m=m)
    grad_inf = gradient_inf_norm(A=A, residual=residual, x=x, lam=lam, m=m)

    x_star = solve_ridge_closed_form(A=A, b=b, lam=lam)
    residual_star = matmul_checked(A, x_star, label="A @ x_star") - b
    objective_star = objective_value(residual=residual_star, x=x_star, lam=lam, m=m)

    objective_gap = max(0.0, objective - objective_star)
    relative_gap = objective_gap / max(abs(objective_star), 1e-12)

    return RCDResult(
        x=x,
        updates_used=updates_used,
        epochs=updates_used / float(p),
        converged=converged,
        objective=objective,
        grad_inf=grad_inf,
        objective_star=objective_star,
        objective_gap=objective_gap,
        relative_gap=relative_gap,
        history=history,
    )


def build_problem(m: int = 220, p: int = 35, seed: int = 396) -> tuple[np.ndarray, np.ndarray]:
    """Build a reproducible synthetic regression problem."""
    if m <= 0 or p <= 0:
        raise ValueError("m and p must be positive.")

    rng = np.random.default_rng(seed)
    A = rng.normal(loc=0.0, scale=1.0, size=(m, p))
    x_true = rng.normal(loc=0.0, scale=1.0, size=p)
    noise = 0.03 * rng.normal(loc=0.0, scale=1.0, size=m)
    b = matmul_checked(A, x_true, label="A @ x_true") + noise
    return A, b


def run_checks(result: RCDResult, objective_initial: float) -> None:
    """Fail fast if behavior is unexpectedly poor."""
    history_array = np.asarray(result.history, dtype=float)
    non_monotone_jumps = np.sum(np.diff(history_array) > 1e-10)

    if not result.converged:
        raise AssertionError("RCD did not converge within max_updates.")
    if result.objective >= objective_initial:
        raise AssertionError("Objective did not decrease from initialization.")
    if result.grad_inf > 5e-8:
        raise AssertionError(f"Gradient infinity norm too large: {result.grad_inf:.3e}")
    if result.relative_gap > 1e-9:
        raise AssertionError(f"Relative objective gap too large: {result.relative_gap:.3e}")
    if non_monotone_jumps > 0:
        raise AssertionError(
            f"Objective history is not monotone at {int(non_monotone_jumps)} checkpoint(s)."
        )


def main() -> None:
    lam = 0.15
    max_updates = 40_000
    tol = 1e-8
    seed = 396

    A, b = build_problem(m=220, p=35, seed=seed)
    m, p = A.shape

    x0 = np.zeros(p, dtype=float)
    residual0 = matmul_checked(A, x0, label="A @ x0") - b
    objective_initial = objective_value(residual=residual0, x=x0, lam=lam, m=m)

    result = randomized_coordinate_descent_ridge(
        A=A,
        b=b,
        lam=lam,
        max_updates=max_updates,
        tol=tol,
        seed=seed,
    )

    run_checks(result=result, objective_initial=objective_initial)

    print("Randomized Coordinate Descent MVP report")
    print(f"shape(A)                       : ({m}, {p})")
    print(f"lambda                         : {lam:.3f}")
    print(f"max_updates                    : {max_updates}")
    print(f"tol                            : {tol:.1e}")
    print(f"updates_used                   : {result.updates_used}")
    print(f"epochs_used                    : {result.epochs:.2f}")
    print(f"converged                      : {result.converged}")
    print(f"objective_initial              : {objective_initial:.12f}")
    print(f"objective_final                : {result.objective:.12f}")
    print(f"objective_star(closed_form)    : {result.objective_star:.12f}")
    print(f"objective_gap                  : {result.objective_gap:.3e}")
    print(f"relative_gap                   : {result.relative_gap:.3e}")
    print(f"gradient_inf_norm              : {result.grad_inf:.3e}")
    print(f"history_points                 : {len(result.history)}")

    print("\nFirst 5 coordinates of solution x:")
    print(np.array2string(result.x[:5], precision=8, suppress_small=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
