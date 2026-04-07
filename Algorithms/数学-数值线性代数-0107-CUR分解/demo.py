"""CUR decomposition MVP.

This script demonstrates a transparent CUR pipeline:
1) build a synthetic low-rank matrix,
2) sample rows/columns (leverage-score or uniform),
3) construct U via pseudoinverses,
4) compare reconstruction errors with truncated SVD baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CURConfig:
    """Configuration for deterministic demo execution."""

    m: int = 140
    n: int = 100
    true_rank: int = 8
    target_rank: int = 8
    n_cols: int = 20
    n_rows: int = 20
    noise_std: float = 0.02
    seed: int = 7


@dataclass
class CURResult:
    """Container for CUR decomposition artifacts."""

    c_idx: np.ndarray
    r_idx: np.ndarray
    c_mat: np.ndarray
    u_mat: np.ndarray
    r_mat: np.ndarray
    a_hat: np.ndarray


def make_low_rank_matrix(config: CURConfig) -> np.ndarray:
    """Generate A = L @ R + Gaussian noise."""
    rng = np.random.default_rng(config.seed)
    left = rng.normal(size=(config.m, config.true_rank))
    right = rng.normal(size=(config.true_rank, config.n))
    # Some NumPy/OpenBLAS builds can emit spurious floating warnings on matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        signal = left @ right
    noise = config.noise_std * rng.normal(size=(config.m, config.n))
    return signal + noise


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return ||a-b||_F / ||a||_F."""
    num = np.linalg.norm(a - b, ord="fro")
    den = np.linalg.norm(a, ord="fro")
    return float(num / den)


def top_leverage_indices(
    singular_vectors: np.ndarray,
    count: int,
    axis_role: str,
) -> np.ndarray:
    """Pick top leverage-score indices.

    For rows: singular_vectors should be U_k with shape (m, k).
    For cols: singular_vectors should be V_k^T with shape (k, n), and we use
              column-wise squared norms.
    """
    if count < 1:
        raise ValueError("count must be >= 1")

    if axis_role == "row":
        # U_k: leverage_i = ||U_k[i, :]||^2
        scores = np.sum(singular_vectors * singular_vectors, axis=1)
        dim = singular_vectors.shape[0]
    elif axis_role == "col":
        # V_k^T: leverage_j = ||V_k[:, j]||^2
        scores = np.sum(singular_vectors * singular_vectors, axis=0)
        dim = singular_vectors.shape[1]
    else:
        raise ValueError("axis_role must be 'row' or 'col'")

    if count > dim:
        raise ValueError(f"count={count} exceeds dimension={dim}")

    picked = np.argpartition(scores, -count)[-count:]
    picked.sort()
    return picked.astype(int)


def random_indices(dim: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample unique sorted indices."""
    if count < 1 or count > dim:
        raise ValueError("count must satisfy 1 <= count <= dim")
    idx = rng.choice(dim, size=count, replace=False)
    idx.sort()
    return idx.astype(int)


def cur_decomposition(
    a: np.ndarray,
    k: int,
    n_cols: int,
    n_rows: int,
    method: str,
    rng: np.random.Generator,
) -> CURResult:
    """Build CUR decomposition of A."""
    m, n = a.shape
    if not (1 <= k <= min(m, n)):
        raise ValueError("k must satisfy 1 <= k <= min(m, n)")
    if not (1 <= n_cols <= n):
        raise ValueError("n_cols out of range")
    if not (1 <= n_rows <= m):
        raise ValueError("n_rows out of range")

    u, _, vt = np.linalg.svd(a, full_matrices=False)
    uk = u[:, :k]
    vtk = vt[:k, :]

    if method == "leverage":
        c_idx = top_leverage_indices(vtk, n_cols, axis_role="col")
        r_idx = top_leverage_indices(uk, n_rows, axis_role="row")
    elif method == "uniform":
        c_idx = random_indices(n, n_cols, rng)
        r_idx = random_indices(m, n_rows, rng)
    else:
        raise ValueError("method must be 'leverage' or 'uniform'")

    c_mat = a[:, c_idx]
    r_mat = a[r_idx, :]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u_mat = np.linalg.pinv(c_mat) @ a @ np.linalg.pinv(r_mat)
        a_hat = c_mat @ u_mat @ r_mat
    return CURResult(
        c_idx=c_idx,
        r_idx=r_idx,
        c_mat=c_mat,
        u_mat=u_mat,
        r_mat=r_mat,
        a_hat=a_hat,
    )


def truncated_svd_error(a: np.ndarray, k: int) -> float:
    """Compute relative error of best rank-k truncated SVD."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        a_k = (u[:, :k] * s[:k]) @ vt[:k, :]
    return relative_fro_error(a, a_k)


def storage_ratio(m: int, n: int, c: int, r: int) -> float:
    """(storage of C,U,R) / (storage of A) in scalar count."""
    original = m * n
    cur_store = m * c + c * r + r * n
    return float(cur_store / original)


def main() -> None:
    config = CURConfig()
    rng = np.random.default_rng(config.seed + 1234)

    a = make_low_rank_matrix(config)

    cur_lev = cur_decomposition(
        a=a,
        k=config.target_rank,
        n_cols=config.n_cols,
        n_rows=config.n_rows,
        method="leverage",
        rng=rng,
    )
    cur_uni = cur_decomposition(
        a=a,
        k=config.target_rank,
        n_cols=config.n_cols,
        n_rows=config.n_rows,
        method="uniform",
        rng=rng,
    )

    err_lev = relative_fro_error(a, cur_lev.a_hat)
    err_uni = relative_fro_error(a, cur_uni.a_hat)
    err_svd = truncated_svd_error(a, config.target_rank)
    sr = storage_ratio(config.m, config.n, config.n_cols, config.n_rows)

    print("CUR分解 MVP（NumPy）")
    print(
        f"shape=({config.m}, {config.n}), true_rank={config.true_rank}, "
        f"target_rank={config.target_rank}, cols={config.n_cols}, rows={config.n_rows}"
    )
    print(f"noise_std={config.noise_std:.3f}, seed={config.seed}")
    print(f"storage_ratio(CUR/A) = {sr:.4f}")
    print()
    print("Relative Frobenius Error:")
    print(f"  CUR(top-leverage)  : {err_lev:.6e}")
    print(f"  CUR(uniform-rand)  : {err_uni:.6e}")
    print(f"  Truncated SVD rank-{config.target_rank}: {err_svd:.6e}")
    print()
    print(
        "Picked column indices (top-leverage): "
        + np.array2string(cur_lev.c_idx, separator=", ")
    )
    print(
        "Picked row indices (top-leverage): "
        + np.array2string(cur_lev.r_idx, separator=", ")
    )


if __name__ == "__main__":
    main()
