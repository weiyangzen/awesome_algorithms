"""Minimal runnable MVP for randomized SVD (from-scratch implementation)."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)


@dataclass
class RandomizedSVDResult:
    """Container for truncated randomized SVD outputs."""

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    m: int
    n: int
    rank: int
    target_dim: int
    n_iter: int


def _as_finite_2d_array(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("matrix must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("matrix contains non-finite values")
    return arr


def _ensure_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def randomized_svd(
    matrix: np.ndarray,
    rank: int,
    oversample: int = 8,
    n_iter: int = 1,
    seed: int = 0,
) -> RandomizedSVDResult:
    """Compute a truncated SVD using randomized range finding.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix A with shape (m, n).
    rank : int
        Target truncated rank k.
    oversample : int
        Extra sketch dimension p; l = min(k + p, min(m, n)).
    n_iter : int
        Number of power iterations to improve spectral separation.
    seed : int
        Random seed for reproducibility.
    """

    A = _as_finite_2d_array(matrix)
    m, n = A.shape
    max_rank = min(m, n)

    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError("rank must be an integer")
    if not isinstance(oversample, int) or isinstance(oversample, bool):
        raise TypeError("oversample must be an integer")
    if not isinstance(n_iter, int) or isinstance(n_iter, bool):
        raise TypeError("n_iter must be an integer")

    if rank <= 0 or rank > max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    if oversample < 0:
        raise ValueError("oversample must be >= 0")
    if n_iter < 0:
        raise ValueError("n_iter must be >= 0")

    target_dim = min(rank + oversample, max_rank)
    rng = np.random.default_rng(seed)

    # 1) Random projection: Y = A @ Omega
    omega = rng.standard_normal(size=(n, target_dim))
    Y = A @ omega
    _ensure_finite("Y (initial sketch)", Y)

    # 2) Optional power iterations: Y <- A(A^T Y)
    for i in range(n_iter):
        Y = A @ (A.T @ Y)
        _ensure_finite(f"Y (power iter {i + 1})", Y)

    # 3) Orthonormal basis for the sampled range
    Q, _ = np.linalg.qr(Y, mode="reduced")
    _ensure_finite("Q", Q)

    # 4) Project to a small matrix and solve exact SVD there
    B = Q.T @ A
    _ensure_finite("B", B)
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    _ensure_finite("S", S)

    # 5) Lift left singular vectors back to original space
    U = Q @ Ub
    _ensure_finite("U", U)

    return RandomizedSVDResult(
        U=U[:, :rank],
        S=S[:rank],
        Vt=Vt[:rank, :],
        m=m,
        n=n,
        rank=rank,
        target_dim=target_dim,
        n_iter=n_iter,
    )


def reconstruct(result: RandomizedSVDResult) -> np.ndarray:
    A_hat = result.U @ np.diag(result.S) @ result.Vt
    _ensure_finite("A_hat", A_hat)
    return A_hat


def relative_fro_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = np.linalg.norm(reference, ord="fro")
    if denom == 0.0:
        return float(np.linalg.norm(estimate - reference, ord="fro"))
    return float(np.linalg.norm(estimate - reference, ord="fro") / denom)


def build_demo_matrix(
    m: int = 400,
    n: int = 250,
    latent_rank: int = 30,
    noise_std: float = 1e-2,
    seed: int = 7,
) -> np.ndarray:
    """Generate a near-low-rank matrix with controllable spectrum."""

    if latent_rank <= 0 or latent_rank > min(m, n):
        raise ValueError("latent_rank must be in [1, min(m, n)]")

    rng = np.random.default_rng(seed)
    U0, _ = np.linalg.qr(rng.standard_normal((m, latent_rank)), mode="reduced")
    V0, _ = np.linalg.qr(rng.standard_normal((n, latent_rank)), mode="reduced")

    singular_values = np.geomspace(50.0, 0.5, latent_rank)
    low_rank_part = U0 @ np.diag(singular_values) @ V0.T
    _ensure_finite("low_rank_part", low_rank_part)
    noise = noise_std * rng.standard_normal((m, n))
    A = low_rank_part + noise
    _ensure_finite("A", A)
    return A


def main() -> None:
    A = build_demo_matrix()

    rank = 20
    oversample = 10
    n_iter = 2
    seed = 1234

    rand = randomized_svd(A, rank=rank, oversample=oversample, n_iter=n_iter, seed=seed)
    A_rand = reconstruct(rand)

    # Exact truncated SVD baseline (best rank-k approximation in Frobenius norm)
    Ue, Se, Vte = np.linalg.svd(A, full_matrices=False)
    A_best = Ue[:, :rank] @ np.diag(Se[:rank]) @ Vte[:rank, :]

    rand_err = relative_fro_error(A, A_rand)
    best_err = relative_fro_error(A, A_best)
    sigma_rel_err = float(
        np.linalg.norm(rand.S - Se[:rank]) / (np.linalg.norm(Se[:rank]) + 1e-15)
    )
    captured_energy = float(np.sum(rand.S**2) / np.sum(Se**2))

    print("Randomized SVD MVP")
    print("-" * 60)
    print(f"matrix shape              : {A.shape}")
    print(f"target rank k             : {rank}")
    print(f"oversample p              : {oversample}")
    print(f"sketch dim l              : {rand.target_dim}")
    print(f"power iterations q        : {n_iter}")
    print(f"relative Fro error (rand) : {rand_err:.6e}")
    print(f"relative Fro error (best) : {best_err:.6e}")
    print(f"error ratio rand/best     : {rand_err / (best_err + 1e-15):.4f}")
    print(f"top-k singular rel error  : {sigma_rel_err:.6e}")
    print(f"captured energy ratio     : {captured_energy:.6f}")
    print("top-8 singular values (rand):", np.array2string(rand.S[:8], precision=4))
    print("top-8 singular values (exact):", np.array2string(Se[:8], precision=4))


if __name__ == "__main__":
    main()
