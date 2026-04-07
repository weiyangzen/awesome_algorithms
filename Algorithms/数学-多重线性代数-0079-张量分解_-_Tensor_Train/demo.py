"""Tensor Train (TT) decomposition MVP via explicit TT-SVD.

This demo keeps the algorithm transparent: no black-box tensor library calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class TTConfig:
    """Configuration for the Tensor Train demonstration."""

    shape: Tuple[int, ...] = (10, 9, 8, 7)
    true_ranks: Tuple[int, ...] = (1, 3, 4, 3, 1)
    max_rank: int = 4
    rel_tol: float = 1e-8
    noise_std: float = 0.002
    seed: int = 17

    def __post_init__(self) -> None:
        if len(self.shape) < 3:
            raise ValueError("Tensor order must be >= 3 for this MVP.")
        if min(self.shape) <= 1:
            raise ValueError("Each mode size must be > 1.")
        if len(self.true_ranks) != len(self.shape) + 1:
            raise ValueError("true_ranks length must be len(shape)+1.")
        if self.true_ranks[0] != 1 or self.true_ranks[-1] != 1:
            raise ValueError("Tensor Train boundary ranks must satisfy r0=rd=1.")
        if any(r <= 0 for r in self.true_ranks):
            raise ValueError("All TT ranks must be positive.")
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive.")
        if self.rel_tol < 0:
            raise ValueError("rel_tol must be non-negative.")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")


@dataclass
class TTResult:
    """Result of TT-SVD decomposition."""

    cores: List[Array]
    ranks: Tuple[int, ...]
    local_discarded_energy: List[float]


def tt_parameter_count(cores: Sequence[Array]) -> int:
    """Count free parameters in TT cores."""

    return int(sum(core.size for core in cores))


def relative_fro_error(reference: Array, estimate: Array) -> float:
    """Relative Frobenius error ||reference-estimate|| / ||reference||."""

    norm_ref = float(np.linalg.norm(reference))
    if not np.isfinite(norm_ref) or norm_ref <= 0:
        raise ValueError("Reference tensor norm must be positive and finite.")
    return float(np.linalg.norm(reference - estimate) / norm_ref)


def reconstruct_tt(cores: Sequence[Array]) -> Array:
    """Reconstruct a dense tensor from TT cores."""

    if not cores:
        raise ValueError("cores must be non-empty")
    if cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
        raise ValueError("Boundary TT ranks must be 1.")

    contracted = cores[0]
    for core in cores[1:]:
        if contracted.shape[-1] != core.shape[0]:
            raise ValueError("Adjacent TT cores have incompatible ranks.")
        contracted = np.tensordot(contracted, core, axes=([-1], [0]))
    return np.squeeze(contracted, axis=(0, -1))


def build_random_tt_cores(
    shape: Sequence[int],
    ranks: Sequence[int],
    rng: np.random.Generator,
) -> List[Array]:
    """Generate random TT cores with mild scaling for numerical stability."""

    if len(ranks) != len(shape) + 1:
        raise ValueError("ranks length must equal len(shape)+1")

    cores: List[Array] = []
    for i, n_i in enumerate(shape):
        r_left = ranks[i]
        r_right = ranks[i + 1]
        scale = 1.0 / np.sqrt(max(1, r_left * n_i))
        cores.append(rng.normal(scale=scale, size=(r_left, n_i, r_right)))
    return cores


def choose_rank_from_singular_values(
    singular_values: Array,
    max_rank: int,
    local_delta_sq: float,
) -> int:
    """Choose TT rank using tolerance budget and max-rank cap."""

    m = singular_values.size
    if m == 0:
        raise ValueError("No singular values returned by SVD.")

    # Tail energy after keeping first r singular values.
    sv_sq = singular_values**2
    tail_sq_reversed = np.cumsum(sv_sq[::-1])
    tail_sq = tail_sq_reversed[::-1]

    rank_by_tol = m
    for r in range(1, m + 1):
        discarded_sq = float(tail_sq[r]) if r < m else 0.0
        if discarded_sq <= local_delta_sq:
            rank_by_tol = r
            break

    chosen = min(rank_by_tol, max_rank, m)
    return max(1, int(chosen))


def tt_svd(x: Array, max_rank: int, rel_tol: float) -> TTResult:
    """Compute Tensor Train decomposition using explicit TT-SVD."""

    order = x.ndim
    if order < 2:
        raise ValueError("Tensor order must be at least 2.")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive.")
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative.")

    norm_x = float(np.linalg.norm(x))
    if not np.isfinite(norm_x) or norm_x <= 0:
        raise ValueError("Input tensor norm must be positive and finite.")

    # Standard TT-SVD tolerance split across (order-1) truncation stages.
    local_delta_sq = (rel_tol * norm_x) ** 2 / max(order - 1, 1)

    working = np.array(x, dtype=float, copy=True)
    cores: List[Array] = []
    local_discarded_energy: List[float] = []
    ranks: List[int] = [1]
    r_prev = 1

    for mode in range(order - 1):
        n_mode = x.shape[mode]
        matrix = working.reshape(r_prev * n_mode, -1)

        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        r_next = choose_rank_from_singular_values(s, max_rank=max_rank, local_delta_sq=local_delta_sq)

        discarded_energy = float(np.sum(s[r_next:] ** 2))
        local_discarded_energy.append(discarded_energy)

        u_trunc = u[:, :r_next]
        s_trunc = s[:r_next]
        vh_trunc = vh[:r_next, :]

        core = u_trunc.reshape(r_prev, n_mode, r_next)
        cores.append(core)

        working = s_trunc[:, None] * vh_trunc
        ranks.append(r_next)
        r_prev = r_next

    last_core = working.reshape(r_prev, x.shape[-1], 1)
    cores.append(last_core)
    ranks.append(1)

    return TTResult(
        cores=cores,
        ranks=tuple(ranks),
        local_discarded_energy=local_discarded_energy,
    )


def make_synthetic_tensor(config: TTConfig) -> Tuple[Array, Array, List[Array]]:
    """Create a noisy tensor from known TT cores."""

    rng = np.random.default_rng(config.seed)
    true_cores = build_random_tt_cores(config.shape, config.true_ranks, rng)
    clean = reconstruct_tt(true_cores)
    noise = rng.normal(scale=config.noise_std, size=config.shape)
    observed = clean + noise
    return observed, clean, true_cores


def run_checks(
    config: TTConfig,
    result: TTResult,
    rel_err_observed: float,
    rel_err_clean: float,
    dense_params: int,
) -> None:
    """Acceptance checks for this MVP."""

    if len(result.cores) != len(config.shape):
        raise AssertionError("Number of TT cores must equal tensor order.")
    if result.ranks[0] != 1 or result.ranks[-1] != 1:
        raise AssertionError("Boundary TT ranks must be 1.")
    if not all(np.isfinite(core).all() for core in result.cores):
        raise AssertionError("TT cores contain non-finite values.")
    if not np.isfinite(rel_err_observed) or not np.isfinite(rel_err_clean):
        raise AssertionError("Reconstruction errors must be finite.")
    if rel_err_observed >= 0.25:
        raise AssertionError(f"Observed-tensor relative error too high: {rel_err_observed:.6f}")
    if rel_err_clean >= 0.25:
        raise AssertionError(f"Clean-tensor relative error too high: {rel_err_clean:.6f}")
    if tt_parameter_count(result.cores) >= dense_params:
        raise AssertionError("TT representation should be more compact than dense tensor.")
    if not all(e >= -1e-12 for e in result.local_discarded_energy):
        raise AssertionError("Discarded energies must be non-negative.")


def main() -> None:
    config = TTConfig()

    x_observed, x_clean, true_cores = make_synthetic_tensor(config)

    result = tt_svd(x_observed, max_rank=config.max_rank, rel_tol=config.rel_tol)
    x_hat = reconstruct_tt(result.cores)

    rel_err_observed = relative_fro_error(x_observed, x_hat)
    rel_err_clean = relative_fro_error(x_clean, x_hat)

    dense_params = int(np.prod(config.shape))
    est_params = tt_parameter_count(result.cores)
    true_params = tt_parameter_count(true_cores)
    compression_ratio = dense_params / est_params

    print("=== Tensor Train Decomposition MVP (TT-SVD) ===")
    print(f"shape={config.shape}, order={len(config.shape)}")
    print(f"true_ranks={config.true_ranks}")
    print(f"estimated_ranks={result.ranks}")
    print(f"max_rank={config.max_rank}, rel_tol={config.rel_tol}, noise_std={config.noise_std}")
    print()

    print("[Reconstruction Error]")
    print(f"relative_error_vs_observed={rel_err_observed:.6f}")
    print(f"relative_error_vs_clean={rel_err_clean:.6f}")
    print()

    print("[Compression]")
    print(f"dense_parameters={dense_params}")
    print(f"true_tt_parameters={true_params}")
    print(f"estimated_tt_parameters={est_params}")
    print(f"compression_ratio_dense_over_tt={compression_ratio:.3f}x")
    print()

    print("[Local TT-SVD Truncation Energy]")
    for i, e in enumerate(result.local_discarded_energy, start=1):
        print(f"step_{i}_discarded_energy={e:.6e}")

    run_checks(
        config=config,
        result=result,
        rel_err_observed=rel_err_observed,
        rel_err_clean=rel_err_clean,
        dense_params=dense_params,
    )
    print("All TT-SVD checks passed.")


if __name__ == "__main__":
    main()
