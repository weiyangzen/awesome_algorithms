"""Minimal runnable MVP for Hidden Markov Model (HMM), MATH-0320.

This script implements a discrete HMM from scratch with NumPy:
- sequence sampling,
- scaled forward-backward inference,
- Baum-Welch (EM) parameter learning,
- Viterbi decoding.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple

import numpy as np


@dataclass
class FBResult:
    """Forward-backward outputs for a single observation sequence."""

    alpha_hat: np.ndarray  # (T, N)
    beta_hat: np.ndarray  # (T, N)
    gamma: np.ndarray  # (T, N)
    xi: np.ndarray  # (T-1, N, N)
    log_likelihood: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    s = vec.sum()
    if s <= 0.0:
        raise ValueError("Cannot normalize: non-positive sum.")
    return vec / s


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    if np.any(s <= 0.0):
        raise ValueError("Cannot normalize rows: at least one row has non-positive sum.")
    return mat / s


def _assert_stochastic(pi: np.ndarray, a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> None:
    if not np.isclose(pi.sum(), 1.0, atol=atol):
        raise ValueError("pi must sum to 1.")
    if np.any(pi < -atol):
        raise ValueError("pi has negative entries.")

    if np.any(a < -atol) or not np.allclose(a.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Transition matrix A must be row-stochastic.")

    if np.any(b < -atol) or not np.allclose(b.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Emission matrix B must be row-stochastic.")


def sample_hmm(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    length: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample latent states and observations from a discrete HMM."""
    rng = np.random.default_rng(seed)
    n_states = pi.shape[0]
    n_obs = b.shape[1]

    states = np.zeros(length, dtype=np.int64)
    obs = np.zeros(length, dtype=np.int64)

    states[0] = rng.choice(n_states, p=pi)
    obs[0] = rng.choice(n_obs, p=b[states[0]])

    for t in range(1, length):
        states[t] = rng.choice(n_states, p=a[states[t - 1]])
        obs[t] = rng.choice(n_obs, p=b[states[t]])

    return states, obs


def forward_backward_scaled(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    obs: np.ndarray,
    eps: float = 1e-300,
) -> FBResult:
    """Scaled forward-backward for discrete HMM."""
    n_states = pi.shape[0]
    t_len = obs.shape[0]

    alpha_hat = np.zeros((t_len, n_states), dtype=np.float64)
    beta_hat = np.zeros((t_len, n_states), dtype=np.float64)
    c = np.zeros(t_len, dtype=np.float64)

    alpha0 = pi * b[:, obs[0]]
    c0 = alpha0.sum()
    if c0 <= eps:
        raise ValueError("Initial scaling factor is too small.")
    c[0] = c0
    alpha_hat[0] = alpha0 / c0

    for t in range(1, t_len):
        alpha_t = (alpha_hat[t - 1] @ a) * b[:, obs[t]]
        ct = alpha_t.sum()
        if ct <= eps:
            raise ValueError(f"Scaling factor c[{t}] is too small.")
        c[t] = ct
        alpha_hat[t] = alpha_t / ct

    # With this scaling convention, log P(o_1:T) = sum_t log c_t.
    log_likelihood = float(np.sum(np.log(c)))

    beta_hat[t_len - 1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta_t = a @ (b[:, obs[t + 1]] * beta_hat[t + 1])
        beta_hat[t] = beta_t / c[t + 1]

    gamma = alpha_hat * beta_hat
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((t_len - 1, n_states, n_states), dtype=np.float64)
    for t in range(t_len - 1):
        right = b[:, obs[t + 1]] * beta_hat[t + 1]
        numer = alpha_hat[t][:, None] * a * right[None, :]
        denom = numer.sum()
        if denom <= eps:
            raise ValueError(f"xi normalization failed at t={t}.")
        xi[t] = numer / denom

    return FBResult(
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        gamma=gamma,
        xi=xi,
        log_likelihood=log_likelihood,
    )


def baum_welch_train(
    obs: np.ndarray,
    n_states: int,
    n_obs: int,
    n_iter: int,
    seed: int,
    smoothing: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Train a discrete HMM with Baum-Welch (single sequence EM)."""
    rng = np.random.default_rng(seed)

    pi = _normalize(rng.random(n_states))
    a = _normalize_rows(rng.random((n_states, n_states)))
    b = _normalize_rows(rng.random((n_states, n_obs)))

    log_history: List[float] = []

    for _ in range(n_iter):
        fb = forward_backward_scaled(pi=pi, a=a, b=b, obs=obs)
        log_history.append(fb.log_likelihood)

        gamma = fb.gamma
        xi = fb.xi

        # M-step: initial distribution.
        pi = gamma[0]

        # M-step: transition matrix.
        trans_denom = gamma[:-1].sum(axis=0) + smoothing * n_states
        trans_numer = xi.sum(axis=0) + smoothing
        a = trans_numer / trans_denom[:, None]

        # M-step: emission matrix.
        emit_denom = gamma.sum(axis=0) + smoothing * n_obs
        emit_numer = np.full((n_states, n_obs), smoothing, dtype=np.float64)
        for symbol in range(n_obs):
            mask = obs == symbol
            if np.any(mask):
                emit_numer[:, symbol] += gamma[mask].sum(axis=0)
        b = emit_numer / emit_denom[:, None]

        # Keep stochastic constraints exact under floating-point errors.
        pi = _normalize(pi)
        a = _normalize_rows(a)
        b = _normalize_rows(b)

    return pi, a, b, log_history


def viterbi_decode(pi: np.ndarray, a: np.ndarray, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Most probable hidden-state path with Viterbi algorithm."""
    t_len = obs.shape[0]
    n_states = pi.shape[0]
    eps = 1e-300

    log_pi = np.log(np.maximum(pi, eps))
    log_a = np.log(np.maximum(a, eps))
    log_b = np.log(np.maximum(b, eps))

    delta = np.zeros((t_len, n_states), dtype=np.float64)
    psi = np.zeros((t_len, n_states), dtype=np.int64)

    delta[0] = log_pi + log_b[:, obs[0]]

    for t in range(1, t_len):
        for j in range(n_states):
            scores = delta[t - 1] + log_a[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_b[j, obs[t]]

    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def best_label_permutation(
    pred_states: np.ndarray,
    true_states: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, float]:
    """Resolve HMM label switching by best permutation on predicted labels."""
    best_acc = -1.0
    best_perm = None

    for perm in permutations(range(n_states)):
        mapped = np.array([perm[s] for s in pred_states], dtype=np.int64)
        acc = float(np.mean(mapped == true_states))
        if acc > best_acc:
            best_acc = acc
            best_perm = np.array(perm, dtype=np.int64)

    if best_perm is None:
        raise RuntimeError("Failed to find label permutation.")

    return best_perm, best_acc


def align_params_to_true_order(
    pi_est: np.ndarray,
    a_est: np.ndarray,
    b_est: np.ndarray,
    pred_to_true_perm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder estimated parameters into true-label order for readable comparison."""
    n_states = pi_est.shape[0]
    true_to_pred = np.zeros(n_states, dtype=np.int64)
    for pred_label, true_label in enumerate(pred_to_true_perm):
        true_to_pred[true_label] = pred_label

    pi_aligned = pi_est[true_to_pred]
    a_aligned = a_est[true_to_pred][:, true_to_pred]
    b_aligned = b_est[true_to_pred]
    return pi_aligned, a_aligned, b_aligned


def main() -> None:
    print("Hidden Markov Model (HMM) MVP - MATH-0320")
    print("=" * 80)

    # Ground-truth model: 3 hidden states, 4 observation symbols.
    pi_true = np.array([0.50, 0.30, 0.20], dtype=np.float64)
    a_true = np.array(
        [
            [0.78, 0.16, 0.06],
            [0.14, 0.72, 0.14],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float64,
    )
    b_true = np.array(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.15, 0.50, 0.25, 0.10],
            [0.05, 0.20, 0.30, 0.45],
        ],
        dtype=np.float64,
    )
    _assert_stochastic(pi_true, a_true, b_true)

    seq_len = 220
    true_states, obs = sample_hmm(pi=pi_true, a=a_true, b=b_true, length=seq_len, seed=20260320)

    n_states = pi_true.shape[0]
    n_obs = b_true.shape[1]

    pi_est, a_est, b_est, log_hist = baum_welch_train(
        obs=obs,
        n_states=n_states,
        n_obs=n_obs,
        n_iter=30,
        seed=7,
        smoothing=1e-6,
    )
    _assert_stochastic(pi_est, a_est, b_est)

    # EM should not decrease log-likelihood (up to tiny numeric tolerance).
    diffs = np.diff(np.array(log_hist, dtype=np.float64))
    min_improvement = float(diffs.min()) if diffs.size else 0.0
    if np.any(diffs < -1e-8):
        raise AssertionError(f"EM log-likelihood decreased: min delta = {min_improvement:.3e}")

    pred_states = viterbi_decode(pi=pi_est, a=a_est, b=b_est, obs=obs)
    perm, mapped_acc = best_label_permutation(pred_states, true_states, n_states)

    pi_aligned, a_aligned, b_aligned = align_params_to_true_order(pi_est, a_est, b_est, perm)

    pi_mae = float(np.mean(np.abs(pi_aligned - pi_true)))
    a_mae = float(np.mean(np.abs(a_aligned - a_true)))
    b_mae = float(np.mean(np.abs(b_aligned - b_true)))

    print(f"sequence length: {seq_len}")
    print(f"EM iterations: {len(log_hist)}")
    print(f"log-likelihood: iter0={log_hist[0]:.6f}, iter_last={log_hist[-1]:.6f}")
    print(f"minimum per-iteration delta: {min_improvement:.3e}")
    print("-" * 80)
    print("True pi      :", np.round(pi_true, 4))
    print("Aligned pi^  :", np.round(pi_aligned, 4))
    print(f"MAE(pi): {pi_mae:.4f}")
    print("-" * 80)
    print("True A:")
    print(np.round(a_true, 4))
    print("Aligned A^:")
    print(np.round(a_aligned, 4))
    print(f"MAE(A): {a_mae:.4f}")
    print("-" * 80)
    print("True B:")
    print(np.round(b_true, 4))
    print("Aligned B^:")
    print(np.round(b_aligned, 4))
    print(f"MAE(B): {b_mae:.4f}")
    print("-" * 80)
    print(f"Viterbi state accuracy (after best label permutation): {mapped_acc:.4f}")

    # Lightweight quality guards for a deterministic MVP run.
    assert log_hist[-1] > log_hist[0]
    assert mapped_acc > 0.45

    print("All checks passed.")


if __name__ == "__main__":
    main()
