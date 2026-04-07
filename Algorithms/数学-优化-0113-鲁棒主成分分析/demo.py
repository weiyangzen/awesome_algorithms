"""Minimal runnable MVP for Robust PCA (Principal Component Pursuit via IALM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RPCAResult:
    """Container for Robust PCA decomposition outputs and diagnostics."""

    low_rank: np.ndarray
    sparse: np.ndarray
    iterations: int
    converged: bool
    relative_residual: float
    estimated_rank: int
    estimated_sparsity: float


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft-thresholding operator: S_tau(x)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svd_threshold(x: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular-value thresholding operator: D_tau(X)."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s_shrunk = np.maximum(s - tau, 0.0)
    rank = int(np.sum(s_shrunk > 0.0))

    if rank == 0:
        return np.zeros_like(x), 0

    # Use einsum to avoid spurious matmul runtime warnings observed in this environment.
    low_rank = np.einsum("ir,r,rj->ij", u[:, :rank], s_shrunk[:rank], vt[:rank, :])
    return low_rank, rank


def validate_inputs(m: np.ndarray, lam: float, tol: float, max_iter: int, rho: float) -> None:
    """Validate matrix shape and core optimizer hyperparameters."""
    if m.ndim != 2:
        raise ValueError("Input matrix must be 2D.")
    if not np.all(np.isfinite(m)):
        raise ValueError("Input matrix contains non-finite values.")
    if lam <= 0.0:
        raise ValueError("lam must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if rho <= 1.0:
        raise ValueError("rho must be greater than 1.0.")


def robust_pca_ialm(
    m: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 2000,
    rho: float = 1.5,
    mu: float | None = None,
) -> RPCAResult:
    """Decompose M into low-rank + sparse parts via inexact ALM (IALM)."""
    m = np.asarray(m, dtype=float)
    rows, cols = m.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(rows, cols))
    validate_inputs(m=m, lam=lam, tol=tol, max_iter=max_iter, rho=rho)

    norm_two = float(np.linalg.norm(m, 2))
    norm_inf = float(np.max(np.abs(m)))
    dual_norm = max(norm_two, norm_inf / lam)

    y = m / (dual_norm + 1e-12)

    if mu is None:
        mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7

    low_rank = np.zeros_like(m)
    sparse = np.zeros_like(m)
    m_fro = float(np.linalg.norm(m, ord="fro")) + 1e-12

    converged = False
    relative_residual = float("inf")
    estimated_rank = 0
    iterations = 0

    for k in range(1, max_iter + 1):
        low_rank, estimated_rank = svd_threshold(m - sparse + y / mu, 1.0 / mu)
        sparse = soft_threshold(m - low_rank + y / mu, lam / mu)

        residual = m - low_rank - sparse
        relative_residual = float(np.linalg.norm(residual, ord="fro") / m_fro)

        y = y + mu * residual
        mu = min(mu * rho, mu_bar)

        iterations = k
        if relative_residual < tol:
            converged = True
            break

    estimated_sparsity = float(np.mean(np.abs(sparse) > 1e-9))

    return RPCAResult(
        low_rank=low_rank,
        sparse=sparse,
        iterations=iterations,
        converged=converged,
        relative_residual=relative_residual,
        estimated_rank=estimated_rank,
        estimated_sparsity=estimated_sparsity,
    )


def build_synthetic_rpca_case(
    rows: int = 70,
    cols: int = 60,
    rank: int = 4,
    sparse_ratio: float = 0.06,
    seed: int = 113,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a reproducible low-rank + sparse matrix for RPCA validation."""
    if rows <= 1 or cols <= 1:
        raise ValueError("rows and cols must both be >= 2.")
    if rank <= 0 or rank >= min(rows, cols):
        raise ValueError("rank must be in (0, min(rows, cols)).")
    if not (0.0 < sparse_ratio < 1.0):
        raise ValueError("sparse_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)

    u = rng.normal(0.0, 1.0, size=(rows, rank))
    v = rng.normal(0.0, 1.0, size=(cols, rank))
    low_rank_true = np.einsum("ik,jk->ij", u, v) / np.sqrt(float(rank))
    low_rank_true = low_rank_true / (np.std(low_rank_true) + 1e-12)

    sparse_true = np.zeros((rows, cols), dtype=float)
    total = rows * cols
    nnz = max(1, int(total * sparse_ratio))
    picked = rng.choice(total, size=nnz, replace=False)

    magnitudes = rng.uniform(6.0, 12.0, size=nnz)
    signs = rng.choice(np.array([-1.0, 1.0]), size=nnz)
    sparse_true.flat[picked] = signs * magnitudes

    observed = low_rank_true + sparse_true
    return observed, low_rank_true, sparse_true


def support_metrics(
    sparse_pred: np.ndarray,
    sparse_true: np.ndarray,
    threshold: float = 1e-3,
) -> tuple[float, float, float]:
    """Compute precision/recall/F1 for recovered sparse support."""
    pred_mask = np.abs(sparse_pred) > threshold
    true_mask = np.abs(sparse_true) > 0.0

    tp = int(np.sum(pred_mask & true_mask))
    fp = int(np.sum(pred_mask & ~true_mask))
    fn = int(np.sum(~pred_mask & true_mask))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def run_checks(
    result: RPCAResult,
    observed: np.ndarray,
    low_rank_true: np.ndarray,
    sparse_true: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Fail fast when decomposition quality is far from expected."""
    recon = result.low_rank + result.sparse

    low_rank_rel_err = float(
        np.linalg.norm(result.low_rank - low_rank_true, ord="fro")
        / (np.linalg.norm(low_rank_true, ord="fro") + 1e-12)
    )
    sparse_rel_err = float(
        np.linalg.norm(result.sparse - sparse_true, ord="fro")
        / (np.linalg.norm(sparse_true, ord="fro") + 1e-12)
    )
    recon_rel_err = float(
        np.linalg.norm(recon - observed, ord="fro")
        / (np.linalg.norm(observed, ord="fro") + 1e-12)
    )
    precision, recall, f1 = support_metrics(result.sparse, sparse_true)

    if not result.converged:
        raise AssertionError("RPCA did not converge within max_iter.")
    if result.relative_residual > 1e-6:
        raise AssertionError(f"Residual is too large: {result.relative_residual:.3e}")
    if low_rank_rel_err > 0.20:
        raise AssertionError(f"Low-rank recovery error too large: {low_rank_rel_err:.3e}")
    if sparse_rel_err > 0.20:
        raise AssertionError(f"Sparse recovery error too large: {sparse_rel_err:.3e}")
    if recon_rel_err > 1e-6:
        raise AssertionError(f"Reconstruction error too large: {recon_rel_err:.3e}")
    if f1 < 0.95:
        raise AssertionError(f"Sparse support F1 too low: {f1:.3f}")

    return low_rank_rel_err, sparse_rel_err, recon_rel_err, precision, recall


def main() -> None:
    observed, low_rank_true, sparse_true = build_synthetic_rpca_case(
        rows=70,
        cols=60,
        rank=4,
        sparse_ratio=0.06,
        seed=113,
    )

    lam = 1.0 / np.sqrt(max(observed.shape))
    result = robust_pca_ialm(
        m=observed,
        lam=lam,
        tol=1e-7,
        max_iter=2000,
        rho=1.5,
    )

    low_rank_rel_err, sparse_rel_err, recon_rel_err, precision, recall = run_checks(
        result=result,
        observed=observed,
        low_rank_true=low_rank_true,
        sparse_true=sparse_true,
    )

    print("Robust PCA (IALM) MVP report")
    print(f"matrix_shape                     : {observed.shape}")
    print(f"lambda                           : {lam:.6f}")
    print(f"iterations_used                  : {result.iterations}")
    print(f"converged                        : {result.converged}")
    print(f"relative_residual                : {result.relative_residual:.3e}")
    print(f"estimated_rank                   : {result.estimated_rank}")
    print(f"estimated_sparsity               : {result.estimated_sparsity:.4f}")
    print(f"low_rank_relative_error          : {low_rank_rel_err:.3e}")
    print(f"sparse_relative_error            : {sparse_rel_err:.3e}")
    print(f"reconstruction_relative_error    : {recon_rel_err:.3e}")
    print(f"sparse_support_precision         : {precision:.3f}")
    print(f"sparse_support_recall            : {recall:.3f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
