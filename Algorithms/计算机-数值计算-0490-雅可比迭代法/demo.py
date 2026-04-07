"""Minimal runnable MVP for Jacobi iteration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JacobiResult:
    """Container for Jacobi iteration outputs and diagnostics."""

    x: np.ndarray
    iterations: int
    converged: bool
    step_error_inf: float
    residual_l2: float
    spectral_radius_B: float


def validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None,
    tol: float,
    max_iter: int,
) -> None:
    """Validate matrix/vector shapes and numerical sanity."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0].")
    if x0 is not None and x0.shape != (n,):
        raise ValueError("x0 must have shape (n,).")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite values.")
    if x0 is not None and not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain only finite values.")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("A has a zero (or near-zero) diagonal entry; Jacobi is undefined.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")


def jacobi_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 1000,
) -> JacobiResult:
    """Solve Ax=b by Jacobi iteration using step infinity norm as stop criterion."""
    validate_inputs(A=A, b=b, x0=x0, tol=tol, max_iter=max_iter)

    A = A.astype(float, copy=False)
    b = b.astype(float, copy=False)
    n = A.shape[0]

    diag = np.diag(A)
    R = A - np.diag(diag)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float, copy=True)

    converged = False
    step_error_inf = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / diag
        step_error_inf = float(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new
        iterations = k

        if step_error_inf < tol:
            converged = True
            break

    residual_l2 = float(np.linalg.norm(A @ x - b, ord=2))

    # Iteration matrix B = -D^{-1}(L+U) = -D^{-1}R.
    B = -(R / diag[:, None])
    spectral_radius_B = float(np.max(np.abs(np.linalg.eigvals(B))))

    return JacobiResult(
        x=x,
        iterations=iterations,
        converged=converged,
        step_error_inf=step_error_inf,
        residual_l2=residual_l2,
        spectral_radius_B=spectral_radius_B,
    )


def build_strictly_diagonally_dominant_system(
    n: int,
    seed: int = 490,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reproducible strictly diagonally dominant system Ax=b."""
    if n <= 1:
        raise ValueError("n must be >= 2.")

    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))

    # Enforce strict diagonal dominance, which guarantees Jacobi convergence.
    row_abs_sum = np.sum(np.abs(A), axis=1)
    A[np.arange(n), np.arange(n)] = row_abs_sum + 1.0

    x_true = np.linspace(1.0, float(n), num=n) / float(n)
    b = A @ x_true
    return A, b, x_true


def run_checks(result: JacobiResult, x_true: np.ndarray, x_ref: np.ndarray) -> None:
    """Fail fast if convergence or accuracy is unexpectedly poor."""
    err_true_inf = float(np.linalg.norm(result.x - x_true, ord=np.inf))
    err_ref_inf = float(np.linalg.norm(result.x - x_ref, ord=np.inf))

    if not result.converged:
        raise AssertionError("Jacobi did not converge within max_iter.")
    if result.spectral_radius_B >= 1.0:
        raise AssertionError(
            f"Spectral radius condition violated: rho(B)={result.spectral_radius_B:.6f}"
        )
    if result.residual_l2 > 1e-8:
        raise AssertionError(f"Residual too large: {result.residual_l2:.3e}")
    if err_ref_inf > 1e-8:
        raise AssertionError(f"Deviation from numpy solve too large: {err_ref_inf:.3e}")
    if err_true_inf > 1e-8:
        raise AssertionError(f"Deviation from ground truth too large: {err_true_inf:.3e}")


def main() -> None:
    n = 8
    tol = 1e-11
    max_iter = 1000

    A, b, x_true = build_strictly_diagonally_dominant_system(n=n, seed=490)
    result = jacobi_solve(A=A, b=b, x0=None, tol=tol, max_iter=max_iter)

    x_ref = np.linalg.solve(A, b)
    err_true_inf = float(np.linalg.norm(result.x - x_true, ord=np.inf))
    err_ref_inf = float(np.linalg.norm(result.x - x_ref, ord=np.inf))

    run_checks(result=result, x_true=x_true, x_ref=x_ref)

    print("Jacobi iteration MVP report")
    print(f"matrix_size                    : {n}")
    print(f"max_iter                       : {max_iter}")
    print(f"tol                            : {tol:.1e}")
    print(f"iterations_used                : {result.iterations}")
    print(f"converged                      : {result.converged}")
    print(f"spectral_radius_B              : {result.spectral_radius_B:.6f}")
    print(f"step_error_inf                 : {result.step_error_inf:.3e}")
    print(f"residual_l2                    : {result.residual_l2:.3e}")
    print(f"solution_error_inf_vs_x_true   : {err_true_inf:.3e}")
    print(f"solution_error_inf_vs_numpy    : {err_ref_inf:.3e}")

    print("\nApprox solution x:")
    print(np.array2string(result.x, precision=8, suppress_small=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
