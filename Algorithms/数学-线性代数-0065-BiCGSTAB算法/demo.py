"""Minimal runnable MVP for the BiCGSTAB algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Vector = np.ndarray
Preconditioner = Callable[[Vector], Vector]


@dataclass
class BiCGSTABResult:
    """Container for solver outputs and diagnostics."""

    x: Vector
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_history: list[float]
    status: str


def validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    maxiter: int,
    rtol: float,
    atol: float,
    breakdown_tol: float,
) -> None:
    """Validate matrix/vector shapes and numeric sanity."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0].")
    if x0.ndim != 1 or x0.shape[0] != n:
        raise ValueError("x0 must be a 1D vector with length equal to A.shape[0].")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative.")
    if breakdown_tol <= 0:
        raise ValueError("breakdown_tol must be positive.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite numbers.")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x0)):
        raise ValueError("b and x0 must contain only finite numbers.")


def bicgstab(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    maxiter: int | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    M_inv: Preconditioner | None = None,
    breakdown_tol: float = 1e-15,
) -> BiCGSTABResult:
    """Solve Ax=b using the BiCGSTAB iteration (left-preconditioned form)."""
    n = A.shape[0]
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    maxiter_use = 2 * n if maxiter is None else int(maxiter)
    M_inv_use = (lambda v: v) if M_inv is None else M_inv

    validate_inputs(A, b, x, maxiter_use, rtol, atol, breakdown_tol)

    b_norm = float(np.linalg.norm(b))
    tol = max(rtol * b_norm, atol)

    r = b - np.dot(A, x)
    r_hat = r.copy()

    residual0 = float(np.linalg.norm(r))
    residual_history = [residual0]
    if residual0 <= tol:
        rel0 = residual0 / (b_norm + 1e-30)
        return BiCGSTABResult(
            x=x,
            converged=True,
            iterations=0,
            residual_norm=residual0,
            relative_residual=rel0,
            residual_history=residual_history,
            status="initial_guess_satisfies_tolerance",
        )

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)

    for k in range(1, maxiter_use + 1):
        rho = float(np.dot(r_hat, r))
        if abs(rho) <= breakdown_tol:
            break

        if k == 1:
            p = r.copy()
        else:
            if abs(omega) <= breakdown_tol:
                break
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

        p_hat = M_inv_use(p)
        v = np.dot(A, p_hat)

        denom = float(np.dot(r_hat, v))
        if abs(denom) <= breakdown_tol:
            break

        alpha = rho / denom
        s = r - alpha * v

        s_norm = float(np.linalg.norm(s))
        if s_norm <= tol:
            x = x + alpha * p_hat
            residual_history.append(s_norm)
            rel = s_norm / (b_norm + 1e-30)
            return BiCGSTABResult(
                x=x,
                converged=True,
                iterations=k,
                residual_norm=s_norm,
                relative_residual=rel,
                residual_history=residual_history,
                status="converged_after_s_step",
            )

        s_hat = M_inv_use(s)
        t = np.dot(A, s_hat)

        tt = float(np.dot(t, t))
        if abs(tt) <= breakdown_tol:
            break

        omega = float(np.dot(t, s) / tt)
        if abs(omega) <= breakdown_tol:
            break

        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t

        r_norm = float(np.linalg.norm(r))
        residual_history.append(r_norm)

        if r_norm <= tol:
            rel = r_norm / (b_norm + 1e-30)
            return BiCGSTABResult(
                x=x,
                converged=True,
                iterations=k,
                residual_norm=r_norm,
                relative_residual=rel,
                residual_history=residual_history,
                status="converged_after_full_update",
            )

        rho_prev = rho

    last_norm = residual_history[-1]
    rel_last = last_norm / (b_norm + 1e-30)
    return BiCGSTABResult(
        x=x,
        converged=False,
        iterations=len(residual_history) - 1,
        residual_norm=last_norm,
        relative_residual=rel_last,
        residual_history=residual_history,
        status="breakdown_or_maxiter_reached",
    )


def jacobi_preconditioner(A: np.ndarray, eps: float = 1e-15) -> Preconditioner:
    """Return a simple diagonal inverse preconditioner."""
    diag = np.diag(A).astype(float).copy()
    diag[np.abs(diag) < eps] = 1.0
    inv_diag = 1.0 / diag

    def apply(v: np.ndarray) -> np.ndarray:
        return inv_diag * v

    return apply


def build_test_system(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic non-symmetric, diagonally-dominant system."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A += n * np.eye(n)
    x_true = rng.normal(size=n)
    b = np.dot(A, x_true)
    return A, b, x_true


def run_scipy_reference(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    rtol: float,
    maxiter: int,
) -> dict[str, float | int] | None:
    """Optionally run SciPy's bicgstab for cross-checking."""
    try:
        from scipy.sparse.linalg import bicgstab as scipy_bicgstab
    except Exception:
        return None

    try:
        x_ref, info = scipy_bicgstab(A, b, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter)
    except TypeError:
        # Older SciPy versions use `tol` instead of `rtol`.
        x_ref, info = scipy_bicgstab(A, b, x0=x0, tol=rtol, maxiter=maxiter)

    residual = float(np.linalg.norm(b - np.dot(A, x_ref)))
    return {
        "info": int(info),
        "residual_norm": residual,
        "relative_residual": residual / (float(np.linalg.norm(b)) + 1e-30),
    }


def main() -> None:
    n = 80
    maxiter = 300
    rtol = 1e-10

    A, b, x_true = build_test_system(n=n, seed=7)
    x0 = np.zeros(n, dtype=float)

    M_inv = jacobi_preconditioner(A)
    result = bicgstab(A=A, b=b, x0=x0, maxiter=maxiter, rtol=rtol, M_inv=M_inv)

    abs_error = float(np.linalg.norm(result.x - x_true))
    rel_error = abs_error / (float(np.linalg.norm(x_true)) + 1e-30)

    if not result.converged:
        raise AssertionError(
            "BiCGSTAB did not converge in MVP demo. "
            f"status={result.status}, rel_res={result.relative_residual:.3e}"
        )
    if result.relative_residual > 1e-8:
        raise AssertionError(f"Relative residual too large: {result.relative_residual:.3e}")
    if rel_error > 1e-8:
        raise AssertionError(f"Relative solution error too large: {rel_error:.3e}")

    print("BiCGSTAB MVP report")
    print(f"matrix_size                 : {n}")
    print(f"maxiter                     : {maxiter}")
    print(f"converged                   : {result.converged}")
    print(f"status                      : {result.status}")
    print(f"iterations                  : {result.iterations}")
    print(f"final_residual_norm         : {result.residual_norm:.3e}")
    print(f"final_relative_residual     : {result.relative_residual:.3e}")
    print(f"relative_solution_error     : {rel_error:.3e}")
    print(f"residual_history_head       : {[f'{v:.2e}' for v in result.residual_history[:5]]}")
    print(f"residual_history_tail       : {[f'{v:.2e}' for v in result.residual_history[-5:]]}")

    scipy_ref = run_scipy_reference(A=A, b=b, x0=x0, rtol=rtol, maxiter=maxiter)
    if scipy_ref is None:
        print("scipy_reference             : skipped (SciPy unavailable)")
    else:
        print("scipy_reference             : available")
        print(f"scipy_info                  : {scipy_ref['info']}")
        print(f"scipy_relative_residual     : {scipy_ref['relative_residual']:.3e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
