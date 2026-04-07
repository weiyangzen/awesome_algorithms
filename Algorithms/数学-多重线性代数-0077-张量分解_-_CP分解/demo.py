"""CP decomposition MVP using explicit ALS updates for a 3-way tensor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class CPALSConfig:
    """Configuration for the synthetic CP-ALS demonstration."""

    shape: Tuple[int, int, int] = (18, 16, 14)
    rank: int = 3
    max_iter: int = 250
    tol: float = 1e-8
    ridge: float = 1e-6
    noise_std: float = 0.03
    seed: int = 7

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError("Only 3-way tensors are supported in this MVP.")
        if min(self.shape) <= 1:
            raise ValueError("Each tensor mode size must be > 1.")
        if self.rank <= 0 or self.rank > min(self.shape):
            raise ValueError("rank must satisfy 1 <= rank <= min(shape).")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")


def normalize_columns(matrix: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize each column to unit norm and return (normalized, norms)."""

    norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(norms < eps, 1.0, norms)
    return matrix / safe_norms, safe_norms


def normalize_factors(factors: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Normalize all factor matrices and absorb scales into CP weights."""

    rank = factors[0].shape[1]
    weights = np.ones(rank, dtype=float)
    normalized: List[np.ndarray] = []
    for factor in factors:
        normed, norms = normalize_columns(factor)
        normalized.append(normed)
        weights *= norms
    return weights, normalized


def reconstruct_cp(weights: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct a 3-way tensor from CP weights and factors."""

    a, b, c = factors
    return np.einsum("ir,jr,kr,r->ijk", a, b, c, weights, optimize=True)


def hadamard_gram(factors: Sequence[np.ndarray], skip_mode: int) -> np.ndarray:
    """Compute Hadamard product of Gram matrices except one mode."""

    rank = factors[0].shape[1]
    gram = np.ones((rank, rank), dtype=float)
    for mode, factor in enumerate(factors):
        if mode == skip_mode:
            continue
        gram *= factor.T @ factor
    return gram


def mttkrp_mode0(x: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    """MTTKRP for mode-0."""

    b, c = factors[1], factors[2]
    return np.einsum("ijk,jr,kr->ir", x, b, c, optimize=True)


def mttkrp_mode1(x: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    """MTTKRP for mode-1."""

    a, c = factors[0], factors[2]
    return np.einsum("ijk,ir,kr->jr", x, a, c, optimize=True)


def mttkrp_mode2(x: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    """MTTKRP for mode-2."""

    a, b = factors[0], factors[1]
    return np.einsum("ijk,ir,jr->kr", x, a, b, optimize=True)


def solve_factor_update(rhs: np.ndarray, gram: np.ndarray, ridge: float) -> np.ndarray:
    """Solve X * gram = rhs for X in a numerically stable way."""

    rank = gram.shape[0]
    lhs = gram + ridge * np.eye(rank)
    return np.linalg.solve(lhs, rhs.T).T


def cp_als(
    x: np.ndarray,
    rank: int,
    max_iter: int,
    tol: float,
    ridge: float,
    seed: int,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Run CP-ALS and return (weights, factors, relative_error_history)."""

    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(size=(x.shape[0], rank)),
        rng.normal(size=(x.shape[1], rank)),
        rng.normal(size=(x.shape[2], rank)),
    ]
    _, factors = normalize_factors(factors)

    norm_x = np.linalg.norm(x)
    if not np.isfinite(norm_x) or norm_x <= 0:
        raise ValueError("Input tensor norm must be positive and finite.")

    history: List[float] = []

    for _ in range(max_iter):
        gram0 = hadamard_gram(factors, skip_mode=0)
        rhs0 = mttkrp_mode0(x, factors)
        factors[0] = solve_factor_update(rhs0, gram0, ridge)

        gram1 = hadamard_gram(factors, skip_mode=1)
        rhs1 = mttkrp_mode1(x, factors)
        factors[1] = solve_factor_update(rhs1, gram1, ridge)

        gram2 = hadamard_gram(factors, skip_mode=2)
        rhs2 = mttkrp_mode2(x, factors)
        factors[2] = solve_factor_update(rhs2, gram2, ridge)

        weights, factors = normalize_factors(factors)
        x_hat = reconstruct_cp(weights, factors)
        rel_err = np.linalg.norm(x - x_hat) / norm_x
        history.append(float(rel_err))

        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol:
            break

    return weights, factors, np.asarray(history, dtype=float)


def best_alignment_score(
    est_factors: Sequence[np.ndarray],
    true_factors: Sequence[np.ndarray],
) -> float:
    """Return best average column-cosine score over all component permutations."""

    rank = est_factors[0].shape[1]
    est_norm = [normalize_columns(f)[0] for f in est_factors]
    true_norm = [normalize_columns(f)[0] for f in true_factors]

    best = -1.0
    for perm in permutations(range(rank)):
        per_component_scores = []
        for r in range(rank):
            mode_scores = []
            for mode in range(3):
                dot = float(np.dot(est_norm[mode][:, r], true_norm[mode][:, perm[r]]))
                mode_scores.append(abs(dot))
            per_component_scores.append(float(np.mean(mode_scores)))
        score = float(np.mean(per_component_scores))
        if score > best:
            best = score
    return best


def make_synthetic_tensor(config: CPALSConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Create a noisy low-rank tensor with known CP factors."""

    rng = np.random.default_rng(config.seed)
    true_factors = [
        rng.normal(size=(config.shape[0], config.rank)),
        rng.normal(size=(config.shape[1], config.rank)),
        rng.normal(size=(config.shape[2], config.rank)),
    ]
    true_weights = rng.uniform(0.8, 1.6, size=config.rank)

    clean = reconstruct_cp(true_weights, true_factors)
    noise = rng.normal(scale=config.noise_std, size=config.shape)
    observed = clean + noise
    return observed, clean, true_weights, true_factors


def run_checks(history: np.ndarray, rel_err_obs: float, rel_err_clean: float, align: float) -> None:
    """Basic acceptance checks for this MVP."""

    if history.size == 0:
        raise AssertionError("No ALS iterations were executed.")
    if not np.all(np.isfinite(history)):
        raise AssertionError("Error history contains non-finite values.")
    if rel_err_obs >= 0.35:
        raise AssertionError(f"Reconstruction error vs observed tensor too high: {rel_err_obs:.6f}")
    if rel_err_clean >= 0.30:
        raise AssertionError(f"Reconstruction error vs clean tensor too high: {rel_err_clean:.6f}")
    # Allow tiny numerical jitter but require clear overall improvement.
    if history[-1] > history[0] - 1e-4:
        raise AssertionError("ALS did not improve reconstruction error.")
    if align <= 0.60:
        raise AssertionError(f"Recovered components are weakly aligned with truth: {align:.6f}")


def main() -> None:
    config = CPALSConfig()

    x_obs, x_clean, true_weights, true_factors = make_synthetic_tensor(config)

    est_weights, est_factors, history = cp_als(
        x=x_obs,
        rank=config.rank,
        max_iter=config.max_iter,
        tol=config.tol,
        ridge=config.ridge,
        seed=config.seed + 123,
    )

    x_hat = reconstruct_cp(est_weights, est_factors)

    rel_err_obs = np.linalg.norm(x_obs - x_hat) / np.linalg.norm(x_obs)
    rel_err_clean = np.linalg.norm(x_clean - x_hat) / np.linalg.norm(x_clean)
    alignment = best_alignment_score(est_factors, true_factors)

    print("=== CP-ALS Demo (3-way tensor) ===")
    print(f"shape={config.shape}, rank={config.rank}, max_iter={config.max_iter}, tol={config.tol}")
    print(f"iterations_run={history.size}")
    print(f"initial_rel_error={history[0]:.6f}")
    print(f"final_rel_error={history[-1]:.6f}")
    print(f"rel_error_vs_observed={rel_err_obs:.6f}")
    print(f"rel_error_vs_clean={rel_err_clean:.6f}")
    print(f"best_component_alignment={alignment:.6f}")
    print(f"true_weight_range=({true_weights.min():.4f}, {true_weights.max():.4f})")
    print(f"estimated_weight_range=({est_weights.min():.4f}, {est_weights.max():.4f})")

    run_checks(history, rel_err_obs, rel_err_clean, alignment)
    print("All CP-ALS checks passed.")


if __name__ == "__main__":
    main()
