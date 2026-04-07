"""Minimal runnable MVP for Non-negative Matrix Factorization (MATH-0105)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class NMFRunStats:
    """Lightweight diagnostics for one NMF run."""

    iterations: int
    initial_loss: float
    final_loss: float
    relative_reconstruction_error: float


def validate_nonnegative_matrix(v: np.ndarray) -> None:
    """Validate matrix shape and non-negativity constraints for NMF input."""
    if v.ndim != 2:
        raise ValueError("Input V must be a 2D matrix.")
    if v.size == 0:
        raise ValueError("Input V must be non-empty.")
    if np.any(v < 0.0):
        min_val = float(np.min(v))
        raise ValueError(f"Input V must be non-negative, but minimum value is {min_val}.")


def frobenius_loss(v: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    """Return objective value 0.5 * ||V - WH||_F^2."""
    residual = v - w @ h
    return 0.5 * float(np.linalg.norm(residual, ord="fro") ** 2)


def nmf_multiplicative_update(
    v: Sequence[Sequence[float]],
    rank: int,
    max_iter: int = 2000,
    tol: float = 1e-7,
    epsilon: float = 1e-10,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[float], NMFRunStats]:
    """Compute NMF with multiplicative updates for Frobenius objective.

    Args:
        v: Non-negative matrix with shape (m, n).
        rank: Factorization rank r.
        max_iter: Maximum number of iterations.
        tol: Relative objective improvement threshold for early stop.
        epsilon: Small value to avoid division by zero.
        random_state: Deterministic RNG seed.

    Returns:
        w: Basis matrix with shape (m, r), non-negative.
        h: Coefficient matrix with shape (r, n), non-negative.
        history: Objective values per iteration (including initial point).
        stats: Run diagnostics.
    """
    v_mat = np.array(v, dtype=float, copy=True)
    validate_nonnegative_matrix(v_mat)

    m, n = v_mat.shape
    if not (1 <= rank <= min(m, n)):
        raise ValueError(f"rank must be in [1, min(m,n)] = [1, {min(m, n)}], got {rank}.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    rng = np.random.default_rng(random_state)
    w = rng.random((m, rank)) + 1e-3
    h = rng.random((rank, n)) + 1e-3

    history: List[float] = []
    current_loss = frobenius_loss(v_mat, w, h)
    history.append(current_loss)

    iterations_done = 0

    for it in range(1, max_iter + 1):
        # Update H.
        h_numerator = w.T @ v_mat
        h_denominator = (w.T @ w @ h) + epsilon
        h *= h_numerator / h_denominator

        # Update W.
        w_numerator = v_mat @ h.T
        w_denominator = (w @ (h @ h.T)) + epsilon
        w *= w_numerator / w_denominator

        # Normalize each column of W and absorb scale into H.
        col_norms = np.linalg.norm(w, axis=0) + epsilon
        w /= col_norms
        h *= col_norms[:, np.newaxis]

        new_loss = frobenius_loss(v_mat, w, h)
        history.append(new_loss)
        iterations_done = it

        rel_improve = abs(current_loss - new_loss) / max(abs(current_loss), epsilon)
        current_loss = new_loss

        if rel_improve < tol:
            break

    rel_recon_error = float(np.linalg.norm(v_mat - w @ h, ord="fro") / (np.linalg.norm(v_mat, ord="fro") + epsilon))

    stats = NMFRunStats(
        iterations=iterations_done,
        initial_loss=history[0],
        final_loss=history[-1],
        relative_reconstruction_error=rel_recon_error,
    )
    return w, h, history, stats


def run_synthetic_case() -> None:
    """Exact low-rank synthetic case to verify convergence behavior."""
    print("[Case 1] synthetic low-rank matrix")

    rng = np.random.default_rng(2026)
    m, n, r = 30, 24, 4

    w_true = rng.uniform(0.2, 2.0, size=(m, r))
    h_true = rng.uniform(0.2, 2.0, size=(r, n))
    v = w_true @ h_true

    w, h, history, stats = nmf_multiplicative_update(
        v,
        rank=r,
        max_iter=2500,
        tol=1e-8,
        random_state=7,
    )

    print(f"shape(V) = {v.shape}, rank = {r}")
    print(f"iterations = {stats.iterations}")
    print(f"initial_loss = {stats.initial_loss:.6e}")
    print(f"final_loss = {stats.final_loss:.6e}")
    print(f"relative_reconstruction_error = {stats.relative_reconstruction_error:.6e}")

    assert np.all(w >= -1e-12), "W must stay non-negative."
    assert np.all(h >= -1e-12), "H must stay non-negative."
    assert history[-1] < history[0], "Objective should decrease from initialization."
    assert stats.relative_reconstruction_error < 8e-2, "Reconstruction error is unexpectedly high."

    print("checks: non-negativity and loss decrease passed")
    print("-" * 72)


def run_topic_case() -> None:
    """Small interpretable term-document demo."""
    print("[Case 2] tiny term-document topic demo")

    terms = [
        "算法",
        "模型",
        "训练",
        "数据",
        "比赛",
        "球员",
        "进球",
        "联赛",
    ]
    docs = ["tech_A", "tech_B", "tech_C", "sports_A", "sports_B", "sports_C"]

    # rows: terms, cols: documents
    v = np.array(
        [
            [9, 8, 7, 0, 1, 0],
            [8, 9, 7, 1, 0, 0],
            [7, 8, 9, 0, 1, 0],
            [9, 7, 8, 1, 0, 1],
            [0, 1, 0, 8, 9, 7],
            [1, 0, 1, 9, 8, 8],
            [0, 1, 0, 8, 9, 9],
            [0, 0, 1, 7, 8, 9],
        ],
        dtype=float,
    )

    w, h, history, stats = nmf_multiplicative_update(
        v,
        rank=2,
        max_iter=3000,
        tol=1e-8,
        random_state=11,
    )

    print(f"shape(V) = {v.shape}, rank = 2")
    print(f"iterations = {stats.iterations}")
    print(f"initial_loss = {history[0]:.6e}")
    print(f"final_loss = {history[-1]:.6e}")
    print(f"relative_reconstruction_error = {stats.relative_reconstruction_error:.6e}")

    for topic_idx in range(2):
        top_term_idx = np.argsort(w[:, topic_idx])[::-1][:3]
        top_doc_idx = np.argsort(h[topic_idx, :])[::-1][:2]
        top_terms = [terms[i] for i in top_term_idx]
        top_docs = [docs[j] for j in top_doc_idx]
        print(f"topic {topic_idx}: top_terms={top_terms}, top_docs={top_docs}")

    assert np.all(w >= -1e-12), "W must stay non-negative."
    assert np.all(h >= -1e-12), "H must stay non-negative."
    assert history[-1] < history[0], "Objective should decrease from initialization."

    print("checks: interpretability demo finished")
    print("-" * 72)


def main() -> None:
    print("NMF MVP (MATH-0105)")
    print("=" * 72)

    run_synthetic_case()
    run_topic_case()

    print("All NMF checks passed.")


if __name__ == "__main__":
    main()
