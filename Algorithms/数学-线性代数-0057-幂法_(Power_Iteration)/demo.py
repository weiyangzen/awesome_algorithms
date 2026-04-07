"""Minimal runnable MVP for Power Iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PowerIterationResult:
    eigenvalue: float
    eigenvector: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    residual_history: List[float]


def validate_inputs(a: np.ndarray, max_iters: int, tol: float) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if not np.isfinite(a).all():
        raise ValueError("Input matrix A must contain only finite values.")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def power_iteration(
    a: np.ndarray,
    max_iters: int = 300,
    tol: float = 1e-12,
    seed: int = 0,
) -> PowerIterationResult:
    """Compute dominant eigenpair approximation using power iteration."""
    validate_inputs(a, max_iters=max_iters, tol=tol)

    n = a.shape[0]
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
        y = a @ x
        y_norm = float(np.linalg.norm(y))
        if y_norm == 0.0 or not np.isfinite(y_norm):
            raise RuntimeError("Encountered an invalid iterate with zero/non-finite norm.")

        x_next = y / y_norm

        # Align sign to reduce direction oscillation when comparing vectors.
        if float(np.dot(x_next, x)) < 0.0:
            x_next = -x_next

        ax_next = a @ x_next
        lambda_hat = float(np.dot(x_next, ax_next))
        residual = float(np.linalg.norm(ax_next - lambda_hat * x_next))
        residual_history.append(residual)

        x = x_next

        if residual < tol:
            return PowerIterationResult(
                eigenvalue=lambda_hat,
                eigenvector=x,
                residual_norm=residual,
                iterations=it,
                converged=True,
                residual_history=residual_history,
            )

    return PowerIterationResult(
        eigenvalue=lambda_hat,
        eigenvector=x,
        residual_norm=residual,
        iterations=max_iters,
        converged=False,
        residual_history=residual_history,
    )


def dominant_eigenpair_numpy(a: np.ndarray) -> tuple[float, np.ndarray]:
    """Reference dominant eigenpair from NumPy (used only for final validation)."""
    eigvals, eigvecs = np.linalg.eig(a)
    idx = int(np.argmax(np.abs(eigvals)))

    eigval = np.real_if_close(eigvals[idx])
    eigvec = np.real_if_close(eigvecs[:, idx]).astype(float)
    eigvec /= float(np.linalg.norm(eigvec))

    return float(eigval), eigvec


def run_checks(result: PowerIterationResult, ref_lambda: float, ref_vec: np.ndarray) -> None:
    if not result.converged:
        raise AssertionError("Power iteration did not converge within max_iters.")

    eigval_err = abs(result.eigenvalue - ref_lambda)
    if eigval_err > 1e-9:
        raise AssertionError(f"Dominant eigenvalue error too large: {eigval_err:.3e}")

    if result.residual_norm > 1e-9:
        raise AssertionError(f"Residual too large: {result.residual_norm:.3e}")

    cosine = abs(float(np.dot(result.eigenvector, ref_vec)))
    if cosine < 0.999999:
        raise AssertionError(f"Eigenvector alignment too weak: {cosine:.9f}")

    if not np.isfinite(result.eigenvector).all():
        raise AssertionError("Estimated eigenvector contains non-finite values.")


def main() -> None:
    # Symmetric matrix with a unique dominant eigenvalue in magnitude.
    a = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.5, 1.5],
        ],
        dtype=float,
    )

    result = power_iteration(a, max_iters=300, tol=1e-12, seed=42)
    ref_lambda, ref_vec = dominant_eigenpair_numpy(a)
    run_checks(result, ref_lambda, ref_vec)

    eigval_err = abs(result.eigenvalue - ref_lambda)
    cosine = abs(float(np.dot(result.eigenvector, ref_vec)))

    print("Power Iteration demo")
    print(f"matrix_shape={a.shape}")
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
