"""Minimal runnable MVP for preconditioning via PCG + Jacobi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCGResult:
    """Container for one PCG run."""

    x: np.ndarray
    iterations: int
    converged: bool
    residual_history: np.ndarray
    relative_residual: float
    relative_solution_error: float


def build_ill_conditioned_tridiagonal_spd(
    n: int,
    condition_scale: float = 1e6,
    coupling: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a strictly diagonally-dominant SPD tridiagonal matrix.

    Matrix form:
        A = tridiag(off, diag, off)
    where
        diag_i ~ logspace(1, condition_scale),
        off_i = -coupling * sqrt(diag_i * diag_{i+1}).

    This construction creates strong diagonal scale disparity, so Jacobi
    preconditioning can visibly improve CG convergence.
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if condition_scale <= 1.0:
        raise ValueError("condition_scale must be > 1.")
    if not (0.0 < coupling < 0.5):
        raise ValueError("coupling must be in (0, 0.5) to preserve SPD margin.")

    diag = np.logspace(0.0, np.log10(condition_scale), n, dtype=float)
    off = -coupling * np.sqrt(diag[:-1] * diag[1:])

    radius = np.zeros(n, dtype=float)
    radius[0] = abs(off[0])
    radius[-1] = abs(off[-1])
    radius[1:-1] = np.abs(off[:-1]) + np.abs(off[1:])

    if np.min(diag - radius) <= 0.0:
        raise ValueError("Constructed matrix is not strictly diagonally dominant.")

    return diag, off


def tridiag_matvec(diag: np.ndarray, off: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute y = A x for tridiagonal A represented by (diag, off)."""
    if diag.ndim != 1 or off.ndim != 1 or x.ndim != 1:
        raise ValueError("diag, off, x must be 1D arrays.")
    n = diag.size
    if off.size != n - 1 or x.size != n:
        raise ValueError("Shape mismatch in tridiagonal matvec.")

    y = diag * x
    y[:-1] += off * x[1:]
    y[1:] += off * x[:-1]
    return y


def make_reference_solution(n: int) -> np.ndarray:
    """Deterministic smooth ground-truth vector x_true."""
    grid = np.arange(1, n + 1, dtype=float)
    x_true = np.sin(np.pi * grid / (n + 1)) + 0.1 * np.cos(3.0 * np.pi * grid / (n + 1))
    return x_true


def pcg_solve(
    diag: np.ndarray,
    off: np.ndarray,
    b: np.ndarray,
    apply_preconditioner,
    max_iters: int = 5000,
    tol: float = 1e-8,
) -> PCGResult:
    """Solve Ax=b by (left) preconditioned conjugate gradient.

    The preconditioner callback computes z = M^{-1} r.
    """
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    n = diag.size
    if b.shape != (n,):
        raise ValueError("b shape mismatch.")

    x = np.zeros_like(b)
    r = b - tridiag_matvec(diag, off, x)
    z = apply_preconditioner(r)

    if z.shape != r.shape or not np.all(np.isfinite(z)):
        raise ValueError("Preconditioner produced invalid output.")

    p = z.copy()
    rz_old = float(np.dot(r, z))

    if rz_old <= 0.0:
        raise ValueError("Preconditioner must be SPD-compatible (r^T M^{-1} r > 0).")

    r0_norm = float(np.linalg.norm(r))
    history = [r0_norm]

    if r0_norm == 0.0:
        return PCGResult(
            x=x,
            iterations=0,
            converged=True,
            residual_history=np.array(history, dtype=float),
            relative_residual=0.0,
            relative_solution_error=float("nan"),
        )

    converged = False
    iters = 0

    for k in range(1, max_iters + 1):
        Ap = tridiag_matvec(diag, off, p)
        denom = float(np.dot(p, Ap))

        if denom <= 1e-30:
            break

        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = float(np.linalg.norm(r))
        history.append(r_norm)

        if r_norm <= tol * r0_norm:
            converged = True
            iters = k
            break

        z = apply_preconditioner(r)
        rz_new = float(np.dot(r, z))

        if rz_new <= 1e-30:
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        iters = k

    relative_residual = history[-1] / history[0]

    return PCGResult(
        x=x,
        iterations=iters,
        converged=converged,
        residual_history=np.array(history, dtype=float),
        relative_residual=float(relative_residual),
        relative_solution_error=float("nan"),
    )


def to_dense_tridiagonal(diag: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Create dense matrix only for diagnostics/condition-number reporting."""
    n = diag.size
    A = np.diag(diag)
    A += np.diag(off, k=1)
    A += np.diag(off, k=-1)
    return A


def estimate_condition_numbers(diag: np.ndarray, off: np.ndarray) -> tuple[float, float]:
    """Estimate cond(A) and cond(D^{-1/2} A D^{-1/2}) in 2-norm."""
    A = to_dense_tridiagonal(diag, off)
    d_inv_sqrt = 1.0 / np.sqrt(diag)
    A_pre = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    cond_a = float(np.linalg.cond(A))
    cond_pre = float(np.linalg.cond(A_pre))
    return cond_a, cond_pre


def run_checks(
    plain: PCGResult,
    jacobi: PCGResult,
    cond_plain: float,
    cond_pre: float,
) -> None:
    """Validate expected preconditioning behavior."""
    if not plain.converged:
        raise AssertionError("Unpreconditioned CG did not converge on this test case.")
    if not jacobi.converged:
        raise AssertionError("Jacobi-preconditioned CG did not converge.")

    if jacobi.iterations >= plain.iterations:
        raise AssertionError(
            f"Jacobi preconditioning did not reduce iterations: plain={plain.iterations}, "
            f"jacobi={jacobi.iterations}."
        )

    if cond_pre >= cond_plain:
        raise AssertionError(
            f"Preconditioned condition number not improved: cond(A)={cond_plain:.3e}, "
            f"cond(pre)={cond_pre:.3e}."
        )

    if jacobi.relative_solution_error > 1e-8:
        raise AssertionError(
            f"Jacobi-PCG solution error too large: {jacobi.relative_solution_error:.3e}"
        )


def main() -> None:
    n = 300
    condition_scale = 1e6
    coupling = 0.02
    tol = 1e-8
    max_iters = 5000

    diag, off = build_ill_conditioned_tridiagonal_spd(
        n=n,
        condition_scale=condition_scale,
        coupling=coupling,
    )

    x_true = make_reference_solution(n)
    b = tridiag_matvec(diag, off, x_true)

    inv_diag = 1.0 / diag

    plain = pcg_solve(
        diag=diag,
        off=off,
        b=b,
        apply_preconditioner=lambda r: r,
        max_iters=max_iters,
        tol=tol,
    )

    jacobi = pcg_solve(
        diag=diag,
        off=off,
        b=b,
        apply_preconditioner=lambda r: inv_diag * r,
        max_iters=max_iters,
        tol=tol,
    )

    plain.relative_solution_error = float(
        np.linalg.norm(plain.x - x_true) / np.linalg.norm(x_true)
    )
    jacobi.relative_solution_error = float(
        np.linalg.norm(jacobi.x - x_true) / np.linalg.norm(x_true)
    )

    cond_plain, cond_pre = estimate_condition_numbers(diag, off)

    run_checks(plain=plain, jacobi=jacobi, cond_plain=cond_plain, cond_pre=cond_pre)

    speedup = plain.iterations / max(jacobi.iterations, 1)

    print("Preconditioning MVP report (PCG + Jacobi)")
    print(f"problem_size_n                 : {n}")
    print(f"condition_scale_target         : {condition_scale:.1e}")
    print(f"coupling                       : {coupling:.3f}")
    print(f"tolerance                      : {tol:.1e}")
    print(f"max_iters                      : {max_iters}")
    print(f"cond2(A)                       : {cond_plain:.3e}")
    print(f"cond2(D^(-1/2) A D^(-1/2))     : {cond_pre:.3e}")
    print("")
    print(f"CG(no preconditioner) iterations : {plain.iterations}")
    print(f"CG(no preconditioner) rel_res    : {plain.relative_residual:.3e}")
    print(f"CG(no preconditioner) rel_x_err  : {plain.relative_solution_error:.3e}")
    print("")
    print(f"PCG(Jacobi) iterations           : {jacobi.iterations}")
    print(f"PCG(Jacobi) rel_res              : {jacobi.relative_residual:.3e}")
    print(f"PCG(Jacobi) rel_x_err            : {jacobi.relative_solution_error:.3e}")
    print("")
    print(f"iteration_speedup (plain/jacobi) : {speedup:.2f}x")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
