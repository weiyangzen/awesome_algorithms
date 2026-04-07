"""Minimal runnable MVP for Forward-Backward Algorithm (MATH-0321).

This script implements the forward-backward algorithm for a discrete Hidden
Markov Model (HMM) from scratch (no HMM black-box library).

Given:
- initial state probabilities pi,
- transition matrix A,
- emission matrix B,
- observation sequence o_1...o_T,

it computes:
- scaled forward messages alpha_hat,
- scaled backward messages beta_hat,
- smoothed state posteriors gamma_t(i)=P(z_t=i|o_1:T),
- pairwise posteriors xi_t(i,j)=P(z_t=i,z_{t+1}=j|o_1:T),
- log-likelihood log P(o_1:T).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FBResult:
    """Container for scaled forward-backward outputs."""

    alpha_hat: np.ndarray  # shape (T, N)
    beta_hat: np.ndarray  # shape (T, N)
    gamma: np.ndarray  # shape (T, N)
    xi: np.ndarray  # shape (T-1, N, N)
    c: np.ndarray  # scaling factors, shape (T,)
    log_likelihood: float


def _assert_row_stochastic(mat: np.ndarray, name: str, atol: float = 1e-12) -> None:
    """Check nonnegativity and row sums = 1 for probability matrices."""
    if np.any(mat < -atol):
        raise ValueError(f"{name} has negative entries.")
    row_sums = mat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"{name} rows must sum to 1. Got {row_sums}.")


def sample_hmm_sequence(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    length: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample latent states and observations from a categorical HMM."""
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
    """Run scaled forward-backward for a discrete HMM.

    Scaling is applied at every time step to avoid underflow for long sequences.
    """
    n_states = pi.shape[0]
    t_len = obs.shape[0]

    alpha_hat = np.zeros((t_len, n_states), dtype=np.float64)
    beta_hat = np.zeros((t_len, n_states), dtype=np.float64)
    c = np.zeros(t_len, dtype=np.float64)

    # Forward init.
    alpha0 = pi * b[:, obs[0]]
    c0 = alpha0.sum()
    if c0 <= eps:
        raise ValueError("Initial scaling factor is too small; check model/observations.")
    c[0] = c0
    alpha_hat[0] = alpha0 / c0

    # Forward recursion.
    for t in range(1, t_len):
        alpha_t = (alpha_hat[t - 1] @ a) * b[:, obs[t]]
        ct = alpha_t.sum()
        if ct <= eps:
            raise ValueError(f"Scaling factor c[{t}] is too small; model underflow/zero path.")
        c[t] = ct
        alpha_hat[t] = alpha_t / ct

    log_likelihood = float(np.sum(np.log(c)))

    # Backward init.
    beta_hat[t_len - 1] = 1.0

    # Backward recursion.
    for t in range(t_len - 2, -1, -1):
        beta_t = a @ (b[:, obs[t + 1]] * beta_hat[t + 1])
        beta_hat[t] = beta_t / c[t + 1]

    # Smoothed state posterior.
    gamma = alpha_hat * beta_hat
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Pairwise posterior xi_t(i,j) for t=0..T-2.
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
        c=c,
        log_likelihood=log_likelihood,
    )


def forward_raw(pi: np.ndarray, a: np.ndarray, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Unscaled forward pass for numerical cross-check on short sequences."""
    n_states = pi.shape[0]
    t_len = obs.shape[0]
    alpha = np.zeros((t_len, n_states), dtype=np.float64)

    alpha[0] = pi * b[:, obs[0]]
    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ a) * b[:, obs[t]]
    return alpha


def backward_raw(a: np.ndarray, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Unscaled backward pass for numerical cross-check on short sequences."""
    n_states = a.shape[0]
    t_len = obs.shape[0]
    beta = np.zeros((t_len, n_states), dtype=np.float64)

    beta[t_len - 1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = a @ (b[:, obs[t + 1]] * beta[t + 1])
    return beta


def main() -> None:
    print("Forward-Backward Algorithm MVP (MATH-0321)")
    print("=" * 72)

    # Model setup: 3 hidden states, 4 observation symbols.
    pi = np.array([0.55, 0.30, 0.15], dtype=np.float64)
    a = np.array(
        [
            [0.72, 0.18, 0.10],
            [0.22, 0.63, 0.15],
            [0.25, 0.25, 0.50],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.15, 0.35, 0.35, 0.15],
            [0.05, 0.20, 0.30, 0.45],
        ],
        dtype=np.float64,
    )

    _assert_row_stochastic(a, "A")
    _assert_row_stochastic(b, "B")
    if not np.isclose(pi.sum(), 1.0):
        raise ValueError("pi must sum to 1.")

    seq_len = 20
    latent_states, obs = sample_hmm_sequence(pi=pi, a=a, b=b, length=seq_len, seed=321)

    fb = forward_backward_scaled(pi=pi, a=a, b=b, obs=obs)

    # Cross-check scaled results with raw forward/backward on this short sequence.
    alpha_raw = forward_raw(pi=pi, a=a, b=b, obs=obs)
    beta_raw = backward_raw(a=a, b=b, obs=obs)
    likelihood_raw = float(alpha_raw[-1].sum())
    loglik_raw = float(np.log(likelihood_raw))

    gamma_raw = alpha_raw * beta_raw
    gamma_raw /= gamma_raw.sum(axis=1, keepdims=True)

    print(f"sequence length: {seq_len}")
    print(f"scaled log-likelihood: {fb.log_likelihood:.10f}")
    print(f"raw    log-likelihood: {loglik_raw:.10f}")
    print(f"abs diff: {abs(fb.log_likelihood - loglik_raw):.3e}")
    print("-" * 72)

    # Show a compact posterior table.
    print("t  obs  true_state   gamma(state=0)  gamma(state=1)  gamma(state=2)")
    max_rows = min(seq_len, 10)
    for t in range(max_rows):
        print(
            f"{t:>2d}  {obs[t]:>3d}      {latent_states[t]:>3d}"
            f"         {fb.gamma[t, 0]:>10.6f}      {fb.gamma[t, 1]:>10.6f}      {fb.gamma[t, 2]:>10.6f}"
        )
    print("-" * 72)

    # Core consistency checks.
    assert np.allclose(fb.alpha_hat.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(fb.gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(fb.xi.sum(axis=(1, 2)), 1.0, atol=1e-10)

    # Marginalization identities between gamma and xi.
    # sum_j xi_t(i,j) = gamma_t(i)
    assert np.allclose(fb.xi.sum(axis=2), fb.gamma[:-1], atol=2e-9)
    # sum_i xi_t(i,j) = gamma_{t+1}(j)
    assert np.allclose(fb.xi.sum(axis=1), fb.gamma[1:], atol=2e-9)

    # Scaled-vs-raw posterior/log-likelihood agreement.
    assert abs(fb.log_likelihood - loglik_raw) < 1e-10
    assert np.allclose(fb.gamma, gamma_raw, atol=2e-10)

    # Example summary statistic: expected state occupancy counts.
    expected_counts = fb.gamma.sum(axis=0)
    most_probable_states = np.argmax(fb.gamma, axis=1)
    print("expected state occupancy counts:", np.round(expected_counts, 4))
    print("first 10 MAP smoothed states:", most_probable_states[:10])
    print("All checks passed.")


if __name__ == "__main__":
    main()
