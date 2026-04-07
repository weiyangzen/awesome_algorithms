"""Minimal runnable MVP for GMRES (Generalized Minimal Residual)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMRESResult:
    """Container for GMRES outputs and diagnostics."""

    x: np.ndarray
    converged: bool
    iterations: int
    relative_residual: float
    residual_history: list[float]
    krylov_dim: int
    arnoldi_relation_error: float


def validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
) -> None:
    """Validate dimensions and basic numerical conditions."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must be a 1D vector with length A.shape[0].")
    if x0.ndim != 1 or x0.shape[0] != n:
        raise ValueError("x0 must be a 1D vector with length A.shape[0].")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(x0)):
        raise ValueError("A, b, x0 must contain only finite numbers.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def compute_givens(a: float, b: float) -> tuple[float, float]:
    """Return (c, s) such that [[c, s],[-s, c]] @ [a, b]^T = [r, 0]^T."""
    r = float(np.hypot(a, b))
    if r == 0.0:
        return 1.0, 0.0
    return a / r, b / r


def gmres_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int | None = None,
    tol: float = 1e-10,
    reorthogonalize: bool = True,
) -> GMRESResult:
    """Solve Ax=b with unrestarted GMRES up to max_iter steps."""
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=float)
    else:
        x0 = x0.astype(float, copy=True)

    if max_iter is None:
        max_iter = n
    max_iter = min(max_iter, n)

    validate_inputs(A=A, b=b, x0=x0, max_iter=max_iter, tol=tol)

    norm_b = float(np.linalg.norm(b))
    rhs_scale = max(norm_b, 1.0)

    r0 = b - A.dot(x0)
    beta = float(np.linalg.norm(r0))

    if beta <= tol * rhs_scale:
        return GMRESResult(
            x=x0,
            converged=True,
            iterations=0,
            relative_residual=beta / rhs_scale,
            residual_history=[beta],
            krylov_dim=0,
            arnoldi_relation_error=0.0,
        )

    V = np.zeros((n, max_iter + 1), dtype=float)
    H_arnoldi = np.zeros((max_iter + 1, max_iter), dtype=float)
    R = np.zeros((max_iter + 1, max_iter), dtype=float)
    c = np.zeros(max_iter, dtype=float)
    s = np.zeros(max_iter, dtype=float)
    g = np.zeros(max_iter + 1, dtype=float)

    V[:, 0] = r0 / beta
    g[0] = beta

    residual_history = [beta]
    krylov_dim = 0
    converged = False

    eps = float(np.finfo(float).eps)

    for j in range(max_iter):
        w = A.dot(V[:, j])

        # Modified Gram-Schmidt orthogonalization.
        for i in range(j + 1):
            hij = float(np.dot(V[:, i], w))
            H_arnoldi[i, j] = hij
            w = w - hij * V[:, i]

        # Optional second pass improves orthogonality.
        if reorthogonalize:
            for i in range(j + 1):
                corr = float(np.dot(V[:, i], w))
                H_arnoldi[i, j] += corr
                w = w - corr * V[:, i]

        H_arnoldi[j + 1, j] = float(np.linalg.norm(w))

        if H_arnoldi[j + 1, j] > eps:
            V[:, j + 1] = w / H_arnoldi[j + 1, j]

        # Copy the fresh Arnoldi column into the least-squares working matrix.
        R[: j + 2, j] = H_arnoldi[: j + 2, j]

        # Apply existing Givens rotations to the new working column.
        for i in range(j):
            h_i_j = R[i, j]
            h_ip1_j = R[i + 1, j]
            R[i, j] = c[i] * h_i_j + s[i] * h_ip1_j
            R[i + 1, j] = -s[i] * h_i_j + c[i] * h_ip1_j

        # Build and apply a new Givens rotation at row pair (j, j+1).
        c[j], s[j] = compute_givens(R[j, j], R[j + 1, j])
        R[j, j] = c[j] * R[j, j] + s[j] * R[j + 1, j]
        R[j + 1, j] = 0.0

        g_j = g[j]
        g_j1 = g[j + 1]
        g[j] = c[j] * g_j + s[j] * g_j1
        g[j + 1] = -s[j] * g_j + c[j] * g_j1

        krylov_dim = j + 1
        residual_history.append(abs(g[j + 1]))

        relative_residual_est = abs(g[j + 1]) / rhs_scale
        if relative_residual_est <= tol:
            converged = True
            break

    if krylov_dim == 0:
        x = x0.copy()
    else:
        upper = R[:krylov_dim, :krylov_dim]
        rhs = g[:krylov_dim]
        try:
            y = np.linalg.solve(upper, rhs)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(upper, rhs, rcond=None)[0]
        x = x0 + V[:, :krylov_dim].dot(y)

    true_relative_residual = float(np.linalg.norm(b - A.dot(x)) / rhs_scale)
    converged = converged or (true_relative_residual <= tol)

    if krylov_dim == 0:
        arnoldi_relation_error = 0.0
    else:
        AV = A.dot(V[:, :krylov_dim])
        VH = V[:, : krylov_dim + 1].dot(H_arnoldi[: krylov_dim + 1, :krylov_dim])
        denom = float(np.linalg.norm(AV)) + eps
        arnoldi_relation_error = float(np.linalg.norm(AV - VH) / denom)

    return GMRESResult(
        x=x,
        converged=converged,
        iterations=krylov_dim,
        relative_residual=true_relative_residual,
        residual_history=residual_history,
        krylov_dim=krylov_dim,
        arnoldi_relation_error=arnoldi_relation_error,
    )


def build_nonsymmetric_system(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic non-symmetric linear system A x_true = b."""
    if n < 4:
        raise ValueError("n must be at least 4.")

    main = 4.0 * np.ones(n)
    upper = -1.2 * np.ones(n - 1)
    lower = 0.8 * np.ones(n - 1)

    A = np.diag(main) + np.diag(upper, k=1) + np.diag(lower, k=-1)

    # Add two corner couplings so A is not purely tridiagonal and is clearly non-symmetric.
    A[0, -1] = 0.2
    A[-1, 0] = -0.1

    grid = np.linspace(0.0, 1.0, n)
    x_true = np.sin(2.0 * np.pi * grid) + 0.5 * np.cos(5.0 * np.pi * grid)
    b = A.dot(x_true)
    return A, b, x_true


def run_checks(result: GMRESResult, x_true: np.ndarray, tol: float) -> None:
    """Fail fast when core guarantees are not met."""
    if result.iterations <= 0:
        raise AssertionError("GMRES did not perform any iteration.")
    if not result.converged:
        raise AssertionError("GMRES did not converge under the configured tolerance/budget.")
    if len(result.residual_history) != result.iterations + 1:
        raise AssertionError("Residual history length is inconsistent with iteration count.")

    # Residual should be non-increasing in exact arithmetic; tolerate tiny roundoff wiggles.
    hist = np.array(result.residual_history, dtype=float)
    diffs = np.diff(hist)
    if np.max(diffs, initial=0.0) > 1e-10:
        raise AssertionError("Residual history is not monotone within tolerance.")

    solution_rel_error = float(np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true))
    if solution_rel_error > 1e-8:
        raise AssertionError(f"Solution relative error too large: {solution_rel_error:.3e}")
    if result.relative_residual > max(10.0 * tol, 1e-10):
        raise AssertionError(f"Relative residual too large: {result.relative_residual:.3e}")
    if result.arnoldi_relation_error > 1e-10:
        raise AssertionError(
            f"Arnoldi relation error too large: {result.arnoldi_relation_error:.3e}"
        )


def main() -> None:
    n = 60
    tol = 1e-10
    max_iter = 35

    A, b, x_true = build_nonsymmetric_system(n=n)

    result = gmres_solve(A=A, b=b, x0=None, max_iter=max_iter, tol=tol, reorthogonalize=True)

    solution_rel_error = float(np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true))

    run_checks(result=result, x_true=x_true, tol=tol)

    print("GMRES MVP report")
    print(f"matrix_size                 : {n}")
    print(f"max_iter_budget             : {max_iter}")
    print(f"effective_iterations        : {result.iterations}")
    print(f"converged                   : {result.converged}")
    print(f"relative_residual           : {result.relative_residual:.3e}")
    print(f"solution_relative_error     : {solution_rel_error:.3e}")
    print(f"arnoldi_relation_error      : {result.arnoldi_relation_error:.3e}")
    print(f"residual_history_head       : {[f'{v:.2e}' for v in result.residual_history[:5]]}")
    print(f"residual_history_tail       : {[f'{v:.2e}' for v in result.residual_history[-5:]]}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
