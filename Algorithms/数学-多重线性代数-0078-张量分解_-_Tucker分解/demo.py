"""Minimal runnable MVP for Tucker decomposition using HOSVD + HOOI.

This script is self-contained and requires only numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class TuckerResult:
    core: Array
    factors: List[Array]
    errors: List[float]


def unfold(tensor: Array, mode: int) -> Array:
    """Mode-n unfolding: shape (I_mode, prod(other_dims))."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(unfolded: Array, mode: int, shape: Sequence[int]) -> Array:
    """Inverse operation of mode-n unfolding."""
    full_shape = [shape[mode], *[shape[i] for i in range(len(shape)) if i != mode]]
    tensor = np.reshape(unfolded, full_shape)
    return np.moveaxis(tensor, 0, mode)


def n_mode_product(tensor: Array, matrix: Array, mode: int) -> Array:
    """Compute tensor x_mode matrix, with matrix shape (J, I_mode)."""
    unfolded = unfold(tensor, mode)
    # Some BLAS backends can emit spurious floating warnings for valid matmul calls.
    # We suppress those warnings but still enforce a strict finite-value check.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        product = matrix @ unfolded
    if not np.isfinite(product).all():
        raise FloatingPointError("n_mode_product produced non-finite values")
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(product, mode, new_shape)


def top_left_singular_vectors(mat: Array, rank: int) -> Array:
    """Return first 'rank' columns of U from SVD(mat)."""
    u, _, _ = np.linalg.svd(mat, full_matrices=False)
    return u[:, :rank]


def hosvd_init(x: Array, ranks: Sequence[int]) -> List[Array]:
    """HOSVD initialization for Tucker factors."""
    factors: List[Array] = []
    for mode, rank in enumerate(ranks):
        x_mode = unfold(x, mode)
        factors.append(top_left_singular_vectors(x_mode, rank))
    return factors


def project_to_core(x: Array, factors: Sequence[Array]) -> Array:
    """Compute core = x x1 U1^T x2 U2^T ..."""
    core = x
    for mode, factor in enumerate(factors):
        core = n_mode_product(core, factor.T, mode)
    return core


def reconstruct(core: Array, factors: Sequence[Array]) -> Array:
    """Reconstruct tensor from Tucker factors and core."""
    x_hat = core
    for mode, factor in enumerate(factors):
        x_hat = n_mode_product(x_hat, factor, mode)
    return x_hat


def relative_fro_error(x: Array, x_hat: Array) -> float:
    """Relative Frobenius norm error."""
    return float(np.linalg.norm(x - x_hat) / np.linalg.norm(x))


def hooi(
    x: Array,
    ranks: Sequence[int],
    max_iters: int = 50,
    tol: float = 1e-8,
) -> TuckerResult:
    """Higher-Order Orthogonal Iteration for Tucker decomposition."""
    if len(ranks) != x.ndim:
        raise ValueError("ranks length must equal tensor order")
    if any(r <= 0 for r in ranks):
        raise ValueError("all ranks must be positive")
    if any(r > d for r, d in zip(ranks, x.shape)):
        raise ValueError("each rank must be <= corresponding dimension")

    factors = hosvd_init(x, ranks)
    errors: List[float] = []

    for _ in range(max_iters):
        for mode in range(x.ndim):
            projected = x
            for other_mode in range(x.ndim):
                if other_mode == mode:
                    continue
                projected = n_mode_product(projected, factors[other_mode].T, other_mode)

            mode_matrix = unfold(projected, mode)
            factors[mode] = top_left_singular_vectors(mode_matrix, ranks[mode])

        core = project_to_core(x, factors)
        x_hat = reconstruct(core, factors)
        err = relative_fro_error(x, x_hat)
        errors.append(err)

        if len(errors) >= 2 and abs(errors[-2] - errors[-1]) < tol:
            break

    return TuckerResult(core=core, factors=factors, errors=errors)


def random_orthonormal(rows: int, cols: int, rng: np.random.Generator) -> Array:
    """Generate a random column-orthonormal matrix with shape (rows, cols)."""
    q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q[:, :cols]


def build_synthetic_tensor(
    shape: Tuple[int, int, int],
    ranks: Tuple[int, int, int],
    noise_std: float,
    seed: int = 42,
) -> Tuple[Array, Array, List[Array]]:
    """Generate a low-rank Tucker tensor and a noisy observation."""
    rng = np.random.default_rng(seed)
    true_factors = [random_orthonormal(shape[i], ranks[i], rng) for i in range(3)]
    true_core = rng.standard_normal(ranks)

    clean = reconstruct(true_core, true_factors)
    noise = noise_std * rng.standard_normal(shape)
    observed = clean + noise
    return observed, clean, true_factors


def main() -> None:
    shape = (30, 25, 20)
    ranks = (3, 4, 2)
    observed, clean, _ = build_synthetic_tensor(shape, ranks, noise_std=0.03, seed=7)

    init_factors = hosvd_init(observed, ranks)
    init_core = project_to_core(observed, init_factors)
    init_recon = reconstruct(init_core, init_factors)
    init_err_observed = relative_fro_error(observed, init_recon)
    init_err_clean = relative_fro_error(clean, init_recon)

    result = hooi(observed, ranks, max_iters=60, tol=1e-9)
    final_recon = reconstruct(result.core, result.factors)

    final_err_observed = relative_fro_error(observed, final_recon)
    final_err_clean = relative_fro_error(clean, final_recon)

    print("=== Tucker Decomposition MVP (HOSVD + HOOI) ===")
    print(f"Tensor shape: {shape}")
    print(f"Target ranks: {ranks}")
    print(f"HOOI iterations: {len(result.errors)}")
    print()

    print("[Reconstruction error w.r.t. observed tensor]")
    print(f"HOSVD init error : {init_err_observed:.6f}")
    print(f"HOOI final error : {final_err_observed:.6f}")
    print()

    print("[Reconstruction error w.r.t. clean low-rank tensor]")
    print(f"HOSVD init error : {init_err_clean:.6f}")
    print(f"HOOI final error : {final_err_clean:.6f}")
    print()

    print("[HOOI error trace on observed tensor]")
    for i, err in enumerate(result.errors, start=1):
        print(f"iter {i:02d}: {err:.6f}")


if __name__ == "__main__":
    main()
