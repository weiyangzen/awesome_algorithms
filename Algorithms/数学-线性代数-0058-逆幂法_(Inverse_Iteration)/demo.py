"""Minimal runnable MVP for Inverse Iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class InverseIterationResult:
    eigenvalue: float
    eigenvector: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    residual_history: List[float]


def validate_inputs(a: np.ndarray, sigma: float, max_iters: int, tol: float) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if not np.isfinite(a).all():
        raise ValueError("Input matrix A must contain only finite values.")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite.")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def inverse_iteration(
    a: np.ndarray,
    sigma: float = 0.0,
    max_iters: int = 200,
    tol: float = 1e-12,
    seed: int = 0,
) -> InverseIterationResult:
    """Compute an eigenpair near shift sigma via shifted inverse iteration."""
    validate_inputs(a=a, sigma=sigma, max_iters=max_iters, tol=tol)

    n = a.shape[0]
    shifted = a - sigma * np.eye(n, dtype=float)

    if np.linalg.matrix_rank(shifted) < n:
        raise ValueError("A - sigma I is singular; choose a different sigma.")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x_norm = float(np.linalg.norm(x))
    if x_norm == 0.0:
        raise RuntimeError("Random initialization produced a zero vector.")
    x = x / x_norm

    residual_history: List[float] = []
    lambda_hat = 0.0
    residual = float("inf")

    for it in range(1, max_iters + 1):
        # Solve (A - sigma I) y = x, then normalize.
        y = np.linalg.solve(shifted, x)
        y_norm = float(np.linalg.norm(y))
        if y_norm == 0.0 or not np.isfinite(y_norm):
            raise RuntimeError("Encountered an invalid iterate with zero/non-finite norm.")

        x_next = y / y_norm

        # Align sign to reduce direction oscillation in logs/comparisons.
        if float(np.dot(x_next, x)) < 0.0:
            x_next = -x_next

        ax_next = a @ x_next
        lambda_hat = float(np.dot(x_next, ax_next))
        residual = float(np.linalg.norm(ax_next - lambda_hat * x_next))
        residual_history.append(residual)

        x = x_next

        if residual < tol:
            return InverseIterationResult(
                eigenvalue=lambda_hat,
                eigenvector=x,
                residual_norm=residual,
                iterations=it,
                converged=True,
                residual_history=residual_history,
            )

    return InverseIterationResult(
        eigenvalue=lambda_hat,
        eigenvector=x,
        residual_norm=residual,
        iterations=max_iters,
        converged=False,
        residual_history=residual_history,
    )


def build_demo_matrix() -> np.ndarray:
    """Build a symmetric matrix with known spectrum but nontrivial eigenvectors."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((4, 4))
    q, _ = np.linalg.qr(m)
    lambdas = np.array([0.8, 2.0, 3.5, 5.0], dtype=float)
    a = q @ np.diag(lambdas) @ q.T
    return 0.5 * (a + a.T)


def reference_eigenpair_near_shift(a: np.ndarray, sigma: float) -> tuple[float, np.ndarray]:
    """Reference eigenpair nearest to sigma (used only for validation)."""
    eigvals, eigvecs = np.linalg.eigh(a)
    idx = int(np.argmin(np.abs(eigvals - sigma)))
    eigval = float(eigvals[idx])
    eigvec = eigvecs[:, idx]
    eigvec = eigvec / float(np.linalg.norm(eigvec))
    return eigval, eigvec


def run_checks(
    a: np.ndarray,
    sigma: float,
    result: InverseIterationResult,
    ref_lambda: float,
    ref_vec: np.ndarray,
) -> None:
    if not result.converged:
        raise AssertionError("Inverse iteration did not converge within max_iters.")

    if result.residual_norm > 1e-10:
        raise AssertionError(f"Residual too large: {result.residual_norm:.3e}")

    eigval_err = abs(result.eigenvalue - ref_lambda)
    if eigval_err > 1e-10:
        raise AssertionError(f"Target eigenvalue error too large: {eigval_err:.3e}")

    cosine = abs(float(np.dot(result.eigenvector, ref_vec)))
    if cosine < 0.999999:
        raise AssertionError(f"Eigenvector alignment too weak: {cosine:.9f}")

    target_gap = abs(ref_lambda - sigma)
    all_eigs = np.linalg.eigvalsh(a)
    nearest_gap = float(np.min(np.abs(all_eigs - sigma)))
    if abs(target_gap - nearest_gap) > 1e-12:
        raise AssertionError("Reference eigenpair is not the nearest-to-shift eigenpair.")

    if not np.isfinite(result.eigenvector).all():
        raise AssertionError("Estimated eigenvector contains non-finite values.")


def main() -> None:
    a = build_demo_matrix()
    sigma = 2.1

    result = inverse_iteration(a, sigma=sigma, max_iters=120, tol=1e-12, seed=42)
    ref_lambda, ref_vec = reference_eigenpair_near_shift(a, sigma=sigma)
    run_checks(a=a, sigma=sigma, result=result, ref_lambda=ref_lambda, ref_vec=ref_vec)

    eigvals = np.linalg.eigvalsh(a)
    eigval_err = abs(result.eigenvalue - ref_lambda)
    cosine = abs(float(np.dot(result.eigenvector, ref_vec)))

    print("Inverse Iteration demo")
    print(f"matrix_shape={a.shape}")
    print(f"shift_sigma={sigma:.6f}")
    print(f"eigenvalues={[float(v) for v in eigvals]}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"lambda_hat={result.eigenvalue:.12f}")
    print(f"lambda_ref={ref_lambda:.12f}")
    print(f"eigval_error={eigval_err:.3e}")
    print(f"residual_norm={result.residual_norm:.3e}")
    print(f"direction_cosine={cosine:.9f}")

    preview = result.residual_history[:5]
    tail = result.residual_history[-3:]
    print(f"residual_history_head={[f'{v:.3e}' for v in preview]}")
    print(f"residual_history_tail={[f'{v:.3e}' for v in tail]}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
