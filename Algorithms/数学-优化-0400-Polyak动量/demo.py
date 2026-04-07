"""Minimal runnable MVP for Polyak momentum (Heavy-Ball method).

This script compares plain gradient descent with Polyak momentum on an
ill-conditioned strongly convex quadratic objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IterRecord:
    """Per-iteration diagnostics for optimization."""

    step: int
    objective: float
    suboptimality: float
    grad_norm: float
    dist_to_opt: float


@dataclass
class RunResult:
    """Container for one optimizer run."""

    w: np.ndarray
    history: List[IterRecord]
    converged: bool


def validate_problem_inputs(A: np.ndarray, b: np.ndarray, w0: np.ndarray) -> Tuple[float, float]:
    """Validate quadratic problem inputs and return (mu, L)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0]")
    if w0.ndim != 1 or w0.shape[0] != A.shape[0]:
        raise ValueError("w0 must be a 1D vector with length equal to A.shape[0]")

    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(w0).all()):
        raise ValueError("A, b, w0 must contain only finite values")

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("A must be symmetric")

    eigvals = np.linalg.eigvalsh(A)
    mu = float(eigvals[0])
    L = float(eigvals[-1])
    if mu <= 0.0:
        raise ValueError("A must be positive definite (smallest eigenvalue must be > 0)")
    if not np.isfinite(mu) or not np.isfinite(L):
        raise ValueError("Failed to derive finite spectral bounds for A")
    return mu, L


def make_ill_conditioned_quadratic(
    dim: int = 20,
    condition_number: float = 1e3,
    seed: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate SPD quadratic objective with controllable condition number.

    Objective: f(w) = 0.5 * w^T A w - b^T w
    """
    if dim <= 1:
        raise ValueError("dim must be > 1")
    if condition_number <= 1.0:
        raise ValueError("condition_number must be > 1")

    rng = np.random.default_rng(seed)

    # Random orthonormal basis.
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigvals = np.geomspace(1.0, condition_number, dim)

    # Use einsum instead of '@' to avoid backend-specific matmul warnings.
    A = np.einsum("ik,k,jk->ij", q, eigvals, q)
    A = 0.5 * (A + A.T)
    b = rng.normal(size=dim)

    w_star = np.linalg.solve(A, b)
    mu = float(eigvals[0])
    L = float(eigvals[-1])
    return A, b, w_star, mu, L


def objective_quadratic(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Compute f(w) = 0.5 * w^T A w - b^T w."""
    Aw = np.einsum("ij,j->i", A, w)
    return float(0.5 * np.dot(w, Aw) - np.dot(b, w))


def gradient_quadratic(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute gradient: grad f(w) = A w - b."""
    return np.einsum("ij,j->i", A, w) - b


def run_gradient_descent(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    w_star: np.ndarray,
    step_size: float,
    max_iters: int,
    tol: float,
) -> RunResult:
    """Run plain gradient descent."""
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")

    f_star = objective_quadratic(A, b, w_star)
    w = w0.copy()
    history: List[IterRecord] = []
    converged = False

    for step in range(max_iters + 1):
        grad = gradient_quadratic(A, b, w)
        grad_norm = float(np.linalg.norm(grad))
        obj = objective_quadratic(A, b, w)
        history.append(
            IterRecord(
                step=step,
                objective=obj,
                suboptimality=obj - f_star,
                grad_norm=grad_norm,
                dist_to_opt=float(np.linalg.norm(w - w_star)),
            )
        )

        if grad_norm <= tol:
            converged = True
            break

        w = w - step_size * grad

    return RunResult(w=w, history=history, converged=converged)


def run_polyak_momentum(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    w_star: np.ndarray,
    step_size: float,
    beta: float,
    max_iters: int,
    tol: float,
) -> RunResult:
    """Run Polyak momentum (Heavy-Ball) method.

    v_{k+1} = beta * v_k - step_size * grad f(w_k)
    w_{k+1} = w_k + v_{k+1}
    """
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1")

    f_star = objective_quadratic(A, b, w_star)
    w = w0.copy()
    velocity = np.zeros_like(w)
    history: List[IterRecord] = []
    converged = False

    for step in range(max_iters + 1):
        grad = gradient_quadratic(A, b, w)
        grad_norm = float(np.linalg.norm(grad))
        obj = objective_quadratic(A, b, w)
        history.append(
            IterRecord(
                step=step,
                objective=obj,
                suboptimality=obj - f_star,
                grad_norm=grad_norm,
                dist_to_opt=float(np.linalg.norm(w - w_star)),
            )
        )

        if grad_norm <= tol:
            converged = True
            break

        velocity = beta * velocity - step_size * grad
        w = w + velocity

    return RunResult(w=w, history=history, converged=converged)


def first_step_below_suboptimality(history: List[IterRecord], threshold: float) -> int | None:
    """Return first iteration step where suboptimality <= threshold."""
    for item in history:
        if item.suboptimality <= threshold:
            return item.step
    return None


def print_history(name: str, history: List[IterRecord], max_rows: int = 8) -> None:
    """Print a compact sampled trajectory from long optimization history."""
    print(f"\n[{name}] sampled trajectory")
    print("step | objective       | suboptimality  | grad_norm      | dist_to_opt")
    print("-----+-----------------+----------------+----------------+----------------")

    if len(history) <= max_rows:
        selected = history
    else:
        idxs = np.linspace(0, len(history) - 1, max_rows, dtype=int)
        selected = [history[i] for i in idxs]

    for item in selected:
        print(
            f"{item.step:4d} | {item.objective: .8e} | {item.suboptimality: .8e} "
            f"| {item.grad_norm: .8e} | {item.dist_to_opt: .8e}"
        )


def main() -> None:
    print("Polyak Momentum MVP (MATH-0400)")
    print("=" * 78)

    A, b, w_star, mu, L = make_ill_conditioned_quadratic(dim=20, condition_number=1e3, seed=400)

    rng = np.random.default_rng(123)
    w0 = rng.normal(size=A.shape[0])

    mu_checked, L_checked = validate_problem_inputs(A, b, w0)
    if abs(mu_checked - mu) > 1e-8 or abs(L_checked - L) > 1e-8:
        raise RuntimeError("spectral checks are inconsistent")

    max_iters = 220
    tol = 1e-8

    # Keep the same step size for fair comparison; only momentum term differs.
    step_size = 1.0 / L
    beta = 0.60

    gd_res = run_gradient_descent(
        A=A,
        b=b,
        w0=w0,
        w_star=w_star,
        step_size=step_size,
        max_iters=max_iters,
        tol=tol,
    )
    hb_res = run_polyak_momentum(
        A=A,
        b=b,
        w0=w0,
        w_star=w_star,
        step_size=step_size,
        beta=beta,
        max_iters=max_iters,
        tol=tol,
    )

    print(f"dimension: {A.shape[0]}")
    print(f"mu: {mu:.6f}, L: {L:.6f}, condition_number: {L / mu:.1f}")
    print(f"step_size: {step_size:.6e}, beta: {beta:.2f}, max_iters: {max_iters}, tol: {tol:.1e}")

    print_history("Gradient Descent", gd_res.history)
    print_history("Polyak Momentum", hb_res.history)

    gd_last = gd_res.history[-1]
    hb_last = hb_res.history[-1]

    threshold = 1.0
    gd_hit = first_step_below_suboptimality(gd_res.history, threshold)
    hb_hit = first_step_below_suboptimality(hb_res.history, threshold)

    print("\nFinal comparison")
    print("-" * 78)
    print(f"GD:      step={gd_last.step:3d}, suboptimality={gd_last.suboptimality:.8e}, grad_norm={gd_last.grad_norm:.8e}")
    print(f"Polyak:  step={hb_last.step:3d}, suboptimality={hb_last.suboptimality:.8e}, grad_norm={hb_last.grad_norm:.8e}")
    print(f"suboptimality ratio (Polyak/GD): {hb_last.suboptimality / gd_last.suboptimality:.6f}")
    print(f"first step with suboptimality <= {threshold:.1f}: GD={gd_hit}, Polyak={hb_hit}")

    if not (hb_last.suboptimality < gd_last.suboptimality):
        raise RuntimeError("Polyak momentum did not outperform GD in final suboptimality")
    if gd_hit is None or hb_hit is None or not (hb_hit < gd_hit):
        raise RuntimeError("Polyak momentum did not reach target suboptimality faster than GD")

    print("=" * 78)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
