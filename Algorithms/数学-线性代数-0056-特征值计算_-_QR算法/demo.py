"""Minimal runnable MVP for eigenvalue computation via shifted QR iteration.

This demo focuses on real symmetric matrices and implements:
- implicit shifted QR updates using a Wilkinson-style shift
- deflation on converged subdiagonal entries
- validation against NumPy reference eigenvalues
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class QREigenResult:
    eigenvalues: np.ndarray
    schur_form: np.ndarray
    iterations: int
    converged: bool
    final_offdiag_norm: float
    offdiag_history: List[float]
    remaining_block_size: int


def validate_inputs(a: np.ndarray, max_iters: int, tol: float) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if not np.isfinite(a).all():
        raise ValueError("Input matrix A must contain only finite values.")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def wilkinson_shift_symmetric(block: np.ndarray) -> float:
    """Return Wilkinson shift from the bottom-right 2x2 symmetric block."""
    if block.shape[0] < 2:
        return float(block[-1, -1])

    a = float(block[-2, -2])
    b = float(block[-2, -1])
    c = float(block[-1, -1])

    if abs(b) < 1e-18:
        return c

    delta = (a - c) / 2.0
    sign = 1.0 if delta >= 0.0 else -1.0
    denom = abs(delta) + np.sqrt(delta * delta + b * b)
    if denom == 0.0:
        return c
    return c - sign * (b * b) / denom


def shifted_qr_eigenvalues_symmetric(
    a: np.ndarray,
    max_iters: int = 4000,
    tol: float = 1e-12,
) -> QREigenResult:
    """Compute eigenvalues of a real symmetric matrix using shifted QR iteration."""
    validate_inputs(a, max_iters=max_iters, tol=tol)

    if not np.allclose(a, a.T, atol=1e-12, rtol=0.0):
        raise ValueError("This MVP expects a symmetric matrix.")

    n = a.shape[0]
    ak = a.astype(float).copy()
    eigvals = np.zeros(n, dtype=float)
    offdiag_history: List[float] = []
    iterations = 0

    if n == 1:
        eigvals[0] = float(ak[0, 0])
        return QREigenResult(
            eigenvalues=eigvals,
            schur_form=ak,
            iterations=0,
            converged=True,
            final_offdiag_norm=0.0,
            offdiag_history=[],
            remaining_block_size=0,
        )

    m = n
    while m > 1:
        while True:
            sub = abs(float(ak[m - 1, m - 2]))
            scale = abs(float(ak[m - 1, m - 1])) + abs(float(ak[m - 2, m - 2])) + 1.0
            if sub <= tol * scale:
                ak[m - 1, m - 2] = 0.0
                ak[m - 2, m - 1] = 0.0
                break

            if iterations >= max_iters:
                remaining_diag = np.diag(ak)[:m]
                eigvals[:m] = remaining_diag
                final_offdiag = float(np.linalg.norm(np.tril(ak, k=-1), ord="fro"))
                return QREigenResult(
                    eigenvalues=eigvals,
                    schur_form=ak,
                    iterations=iterations,
                    converged=False,
                    final_offdiag_norm=final_offdiag,
                    offdiag_history=offdiag_history,
                    remaining_block_size=m,
                )

            active = ak[:m, :m]
            mu = wilkinson_shift_symmetric(active)
            q, r = np.linalg.qr(active - mu * np.eye(m))
            ak[:m, :m] = r @ q + mu * np.eye(m)

            # Keep symmetry tight against floating-point drift.
            ak[:m, :m] = 0.5 * (ak[:m, :m] + ak[:m, :m].T)

            offdiag = float(np.linalg.norm(np.tril(ak[:m, :m], k=-1), ord="fro"))
            offdiag_history.append(offdiag)
            iterations += 1

        eigvals[m - 1] = float(ak[m - 1, m - 1])
        m -= 1

    eigvals[0] = float(ak[0, 0])
    final_offdiag = float(np.linalg.norm(np.tril(ak, k=-1), ord="fro"))
    return QREigenResult(
        eigenvalues=eigvals,
        schur_form=ak,
        iterations=iterations,
        converged=True,
        final_offdiag_norm=final_offdiag,
        offdiag_history=offdiag_history,
        remaining_block_size=0,
    )


def run_checks(a: np.ndarray, result: QREigenResult, ref_eigs: np.ndarray) -> None:
    if not result.converged:
        raise AssertionError(
            f"QR iteration did not converge, remaining_block_size={result.remaining_block_size}"
        )
    if not np.isfinite(result.eigenvalues).all():
        raise AssertionError("Estimated eigenvalues contain non-finite values.")

    eig_err = float(np.max(np.abs(np.sort(result.eigenvalues) - np.sort(ref_eigs))))
    if eig_err > 1e-8:
        raise AssertionError(f"Eigenvalue error too large: {eig_err:.3e}")

    trace_err = float(abs(np.sum(result.eigenvalues) - np.trace(a)))
    if trace_err > 1e-8:
        raise AssertionError(f"Trace consistency failed: {trace_err:.3e}")

    if result.final_offdiag_norm > 1e-8:
        raise AssertionError(f"Final off-diagonal norm too large: {result.final_offdiag_norm:.3e}")


def build_demo_matrix() -> np.ndarray:
    """Build a deterministic symmetric matrix with known eigenvalues."""
    rng = np.random.default_rng(2026)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    target_eigs = np.array([9.0, 5.0, 3.0, 1.5, -2.0, -4.0], dtype=float)
    a = q @ np.diag(target_eigs) @ q.T
    return 0.5 * (a + a.T)


def main() -> None:
    a = build_demo_matrix()
    result = shifted_qr_eigenvalues_symmetric(a, max_iters=4000, tol=1e-12)
    ref_eigs = np.linalg.eigvalsh(a)
    run_checks(a, result, ref_eigs)

    eig_err = float(np.max(np.abs(np.sort(result.eigenvalues) - np.sort(ref_eigs))))
    print("Shifted QR eigenvalue demo (symmetric matrix)")
    print(f"matrix_shape={a.shape}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"final_offdiag_norm={result.final_offdiag_norm:.3e}")
    print(f"max_abs_eig_error={eig_err:.3e}")
    print(f"trace_error={abs(np.sum(result.eigenvalues) - np.trace(a)):.3e}")
    print("estimated_eigs_sorted=", np.sort(result.eigenvalues))
    print("reference_eigs_sorted=", np.sort(ref_eigs))

    if result.offdiag_history:
        head = result.offdiag_history[:5]
        tail = result.offdiag_history[-3:]
        print(f"offdiag_history_head={[f'{v:.3e}' for v in head]}")
        print(f"offdiag_history_tail={[f'{v:.3e}' for v in tail]}")
    else:
        print("offdiag_history=[]")

    print("All checks passed.")


if __name__ == "__main__":
    main()
