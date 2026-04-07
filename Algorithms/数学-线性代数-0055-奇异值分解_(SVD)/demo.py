"""Minimal runnable MVP for Singular Value Decomposition (SVD).

This demo implements compact SVD through the eigen-decomposition of A^T A:
1) build Gram matrix G = A^T A
2) solve eigenpairs of G to obtain right singular vectors and singular values
3) recover left singular vectors via u_i = A v_i / sigma_i
4) validate against NumPy SVD and pseudo-inverse identities
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVDResult:
    u: np.ndarray
    singular_values: np.ndarray
    vt: np.ndarray
    rank: int
    reconstruction_error: float
    u_orthogonality_error: float
    v_orthogonality_error: float


def validate_inputs(a: np.ndarray, tol: float) -> None:
    if a.ndim != 2:
        raise ValueError("Input matrix A must be 2-D.")
    if not np.isfinite(a).all():
        raise ValueError("Input matrix A must contain only finite values.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def compact_svd_from_gram(a: np.ndarray, tol: float = 1e-12) -> SVDResult:
    """Compute compact SVD via eigendecomposition of G = A^T A.

    Returns U_r, s_r, V_r^T where r is numerical rank under the given tolerance.
    """
    validate_inputs(a, tol=tol)

    m, n = a.shape
    gram = a.T @ a

    # G is symmetric PSD; eigh is numerically suitable here.
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0.0)
    singular_values_all = np.sqrt(eigvals)

    if singular_values_all.size == 0:
        return SVDResult(
            u=np.zeros((m, 0), dtype=float),
            singular_values=np.zeros((0,), dtype=float),
            vt=np.zeros((0, n), dtype=float),
            rank=0,
            reconstruction_error=float(np.linalg.norm(a, ord="fro")),
            u_orthogonality_error=0.0,
            v_orthogonality_error=0.0,
        )

    scale = max(float(singular_values_all[0]), 1.0)
    mask = singular_values_all > tol * scale
    singular_values = singular_values_all[mask]
    v = eigvecs[:, mask]
    rank = int(singular_values.size)

    if rank == 0:
        u = np.zeros((m, 0), dtype=float)
        vt = np.zeros((0, n), dtype=float)
        reconstruction = np.zeros_like(a)
        u_orth_err = 0.0
        v_orth_err = 0.0
    else:
        u = (a @ v) / singular_values[np.newaxis, :]
        vt = v.T
        reconstruction = u @ np.diag(singular_values) @ vt
        u_orth_err = float(np.linalg.norm(u.T @ u - np.eye(rank), ord="fro"))
        v_orth_err = float(np.linalg.norm(vt @ vt.T - np.eye(rank), ord="fro"))

    reconstruction_error = float(np.linalg.norm(a - reconstruction, ord="fro"))
    return SVDResult(
        u=u,
        singular_values=singular_values,
        vt=vt,
        rank=rank,
        reconstruction_error=reconstruction_error,
        u_orthogonality_error=u_orth_err,
        v_orthogonality_error=v_orth_err,
    )


def pseudo_inverse_from_svd(result: SVDResult) -> np.ndarray:
    """Build Moore-Penrose pseudo-inverse from compact SVD factors."""
    if result.rank == 0:
        n = result.vt.shape[1]
        m = result.u.shape[0]
        return np.zeros((n, m), dtype=float)

    inv_sigma = np.diag(1.0 / result.singular_values)
    return result.vt.T @ inv_sigma @ result.u.T


def best_rank_k_approx(result: SVDResult, k: int) -> np.ndarray:
    """Return rank-k truncated approximation under compact SVD factors."""
    if k < 0:
        raise ValueError("k must be non-negative.")

    m = result.u.shape[0]
    n = result.vt.shape[1]
    kk = min(k, result.rank)
    if kk == 0:
        return np.zeros((m, n), dtype=float)
    return result.u[:, :kk] @ np.diag(result.singular_values[:kk]) @ result.vt[:kk, :]


def run_checks(a: np.ndarray, result: SVDResult) -> dict[str, float]:
    s_ref = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    s_hat_full = np.zeros_like(s_ref)
    s_hat_full[: result.rank] = result.singular_values
    sigma_error = float(np.max(np.abs(s_hat_full - s_ref)))
    if sigma_error > 1e-6:
        raise AssertionError(f"Singular value error too large: {sigma_error:.3e}")

    if result.reconstruction_error > 1e-9:
        raise AssertionError(
            f"Reconstruction error too large: {result.reconstruction_error:.3e}"
        )

    if result.u_orthogonality_error > 1e-9:
        raise AssertionError(
            f"U orthogonality error too large: {result.u_orthogonality_error:.3e}"
        )
    if result.v_orthogonality_error > 1e-9:
        raise AssertionError(
            f"V orthogonality error too large: {result.v_orthogonality_error:.3e}"
        )

    pinv_hat = pseudo_inverse_from_svd(result)
    pinv_ref = np.linalg.pinv(a)
    pinv_rel_error = float(
        np.linalg.norm(pinv_hat - pinv_ref, ord="fro")
        / max(np.linalg.norm(pinv_ref, ord="fro"), 1e-12)
    )
    if pinv_rel_error > 1e-8:
        raise AssertionError(f"Pseudo-inverse relative error too large: {pinv_rel_error:.3e}")

    mp_identity_error = float(np.linalg.norm(a @ pinv_hat @ a - a, ord="fro"))
    if mp_identity_error > 1e-8:
        raise AssertionError(
            f"Moore-Penrose identity error too large: {mp_identity_error:.3e}"
        )

    approx_rank1 = best_rank_k_approx(result, 1)
    approx_rank2 = best_rank_k_approx(result, 2)
    trunc_err_rank1 = float(np.linalg.norm(a - approx_rank1, ord="fro"))
    trunc_err_rank2 = float(np.linalg.norm(a - approx_rank2, ord="fro"))
    if trunc_err_rank2 > trunc_err_rank1 + 1e-12:
        raise AssertionError(
            "Rank-2 approximation should not be worse than rank-1 approximation."
        )

    return {
        "sigma_error": sigma_error,
        "pinv_rel_error": pinv_rel_error,
        "mp_identity_error": mp_identity_error,
        "trunc_err_rank1": trunc_err_rank1,
        "trunc_err_rank2": trunc_err_rank2,
    }


def build_demo_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic 6x4 matrix with known singular values."""
    rng = np.random.default_rng(2026)
    q_u, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    q_v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    sigma_true = np.array([7.0, 3.0, 1.2, 0.0], dtype=float)
    a = q_u[:, :4] @ np.diag(sigma_true) @ q_v.T
    return a, sigma_true


def main() -> None:
    a, sigma_true = build_demo_matrix()
    tol = 1e-6
    result = compact_svd_from_gram(a, tol=tol)
    metrics = run_checks(a, result)
    sigma_ref = np.linalg.svd(a, full_matrices=False, compute_uv=False)

    print("SVD demo via eigendecomposition of A^T A")
    print(f"matrix_shape={a.shape}")
    print(f"rank_tol={tol}")
    print(f"rank={result.rank}")
    print(f"singular_values_true={sigma_true}")
    print(f"singular_values_est={result.singular_values}")
    print(f"singular_values_ref={sigma_ref}")
    print(f"reconstruction_error={result.reconstruction_error:.3e}")
    print(f"u_orthogonality_error={result.u_orthogonality_error:.3e}")
    print(f"v_orthogonality_error={result.v_orthogonality_error:.3e}")
    print(f"sigma_error={metrics['sigma_error']:.3e}")
    print(f"pinv_rel_error={metrics['pinv_rel_error']:.3e}")
    print(f"mp_identity_error={metrics['mp_identity_error']:.3e}")
    print(f"trunc_err_rank1={metrics['trunc_err_rank1']:.3e}")
    print(f"trunc_err_rank2={metrics['trunc_err_rank2']:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
