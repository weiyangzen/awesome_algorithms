"""Minimal runnable MVP for Baum-Welch algorithm (MATH-0323).

This script trains a discrete HMM from observations only, using
Baum-Welch (EM for HMM) implemented from scratch with NumPy.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FBResult:
    """Scaled forward-backward outputs for one observation sequence."""

    alpha_hat: np.ndarray  # shape (T, N)
    beta_hat: np.ndarray  # shape (T, N)
    gamma: np.ndarray  # shape (T, N)
    xi: np.ndarray  # shape (T-1, N, N)
    c: np.ndarray  # shape (T,)
    log_likelihood: float


@dataclass
class BaumWelchResult:
    """Training summary for Baum-Welch."""

    initial_pi: np.ndarray
    initial_a: np.ndarray
    initial_b: np.ndarray
    learned_pi: np.ndarray
    learned_a: np.ndarray
    learned_b: np.ndarray
    log_likelihood_trace: np.ndarray
    iterations_run: int


def _assert_prob_vector(vec: np.ndarray, name: str, atol: float = 1e-12) -> None:
    if vec.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if np.any(vec < -atol):
        raise ValueError(f"{name} has negative entries.")
    if not np.isclose(vec.sum(), 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1, got {vec.sum()}.")


def _assert_row_stochastic(mat: np.ndarray, name: str, atol: float = 1e-12) -> None:
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if np.any(mat < -atol):
        raise ValueError(f"{name} has negative entries.")
    row_sums = mat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"{name} rows must sum to 1. Got {row_sums}.")


def _normalize_vector(vec: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    total = float(vec.sum())
    if total <= eps:
        return np.full_like(vec, 1.0 / vec.shape[0], dtype=np.float64)
    return vec / total


def _normalize_rows(mat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    out = np.array(mat, dtype=np.float64, copy=True)
    row_sums = out.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= eps
    if np.any(bad):
        out[bad] = 1.0
        row_sums = out.sum(axis=1, keepdims=True)
    out /= row_sums
    return out


def _random_hmm_params(n_states: int, n_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = _normalize_vector(rng.random(n_states) + 1e-3)
    a = _normalize_rows(rng.random((n_states, n_states)) + 1e-3)
    b = _normalize_rows(rng.random((n_states, n_obs)) + 1e-3)
    return pi, a, b


def sample_hmm_sequence(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample hidden states and observations from a discrete HMM."""
    if length <= 0:
        raise ValueError("length must be positive.")

    _assert_prob_vector(pi, "pi")
    _assert_row_stochastic(a, "A")
    _assert_row_stochastic(b, "B")

    n_states = pi.shape[0]
    n_obs = b.shape[1]
    if a.shape != (n_states, n_states):
        raise ValueError("A must have shape (N, N).")

    rng = np.random.default_rng(seed)
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
    """Run scaled forward-backward for one sequence."""
    _assert_prob_vector(pi, "pi")
    _assert_row_stochastic(a, "A")
    _assert_row_stochastic(b, "B")

    if obs.ndim != 1:
        raise ValueError("obs must be a 1D array of symbol ids.")
    if obs.size < 2:
        raise ValueError("obs length must be at least 2 for xi computation.")

    n_states = pi.shape[0]
    t_len = obs.shape[0]
    n_obs = b.shape[1]

    if a.shape != (n_states, n_states):
        raise ValueError("A shape mismatch.")
    if np.any((obs < 0) | (obs >= n_obs)):
        raise ValueError("obs contains out-of-range symbol index.")

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
        c=c,
        log_likelihood=log_likelihood,
    )


def baum_welch_train(
    obs: np.ndarray,
    n_states: int,
    n_obs: int,
    max_iters: int = 40,
    tol: float = 1e-6,
    seed: int = 323,
) -> BaumWelchResult:
    """Train a discrete HMM with Baum-Welch (single sequence EM)."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    rng = np.random.default_rng(seed)
    pi, a, b = _random_hmm_params(n_states=n_states, n_obs=n_obs, rng=rng)

    initial_pi = pi.copy()
    initial_a = a.copy()
    initial_b = b.copy()

    ll_trace = []
    init_fb = forward_backward_scaled(pi=pi, a=a, b=b, obs=obs)
    ll_trace.append(init_fb.log_likelihood)

    iterations_run = 0
    for _ in range(max_iters):
        fb = forward_backward_scaled(pi=pi, a=a, b=b, obs=obs)
        old_ll = fb.log_likelihood

        gamma = fb.gamma
        xi = fb.xi

        new_pi = _normalize_vector(gamma[0])

        a_num = xi.sum(axis=0)
        a_den = gamma[:-1].sum(axis=0)
        new_a = np.zeros((n_states, n_states), dtype=np.float64)
        for i in range(n_states):
            if a_den[i] <= 1e-15:
                new_a[i] = 1.0 / n_states
            else:
                new_a[i] = a_num[i] / a_den[i]
        new_a = _normalize_rows(new_a)

        b_num = np.zeros((n_states, n_obs), dtype=np.float64)
        for k in range(n_obs):
            mask = obs == k
            if np.any(mask):
                b_num[:, k] = gamma[mask].sum(axis=0)

        b_den = gamma.sum(axis=0)
        new_b = np.zeros((n_states, n_obs), dtype=np.float64)
        for i in range(n_states):
            if b_den[i] <= 1e-15:
                new_b[i] = 1.0 / n_obs
            else:
                new_b[i] = b_num[i] / b_den[i]
        new_b = _normalize_rows(new_b)

        new_fb = forward_backward_scaled(pi=new_pi, a=new_a, b=new_b, obs=obs)
        new_ll = new_fb.log_likelihood

        if new_ll + 1e-9 < old_ll:
            raise RuntimeError(
                f"Log-likelihood decreased unexpectedly: old={old_ll}, new={new_ll}."
            )

        pi, a, b = new_pi, new_a, new_b
        ll_trace.append(new_ll)
        iterations_run += 1

        if new_ll - old_ll < tol:
            break

    return BaumWelchResult(
        initial_pi=initial_pi,
        initial_a=initial_a,
        initial_b=initial_b,
        learned_pi=pi,
        learned_a=a,
        learned_b=b,
        log_likelihood_trace=np.array(ll_trace, dtype=np.float64),
        iterations_run=iterations_run,
    )


def _fmt_matrix(mat: np.ndarray, precision: int = 4) -> str:
    return np.array2string(mat, precision=precision, suppress_small=False)


def main() -> None:
    print("Baum-Welch Algorithm MVP (MATH-0323)")
    print("=" * 72)

    # Ground-truth HMM used only for synthetic data generation.
    pi_true = np.array([0.60, 0.30, 0.10], dtype=np.float64)
    a_true = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.20, 0.25, 0.55],
        ],
        dtype=np.float64,
    )
    b_true = np.array(
        [
            [0.55, 0.30, 0.10, 0.05],
            [0.10, 0.20, 0.45, 0.25],
            [0.05, 0.10, 0.25, 0.60],
        ],
        dtype=np.float64,
    )

    _assert_prob_vector(pi_true, "pi_true")
    _assert_row_stochastic(a_true, "a_true")
    _assert_row_stochastic(b_true, "b_true")

    seq_len = 300
    states, obs = sample_hmm_sequence(
        pi=pi_true,
        a=a_true,
        b=b_true,
        length=seq_len,
        seed=20260407,
    )

    result = baum_welch_train(
        obs=obs,
        n_states=3,
        n_obs=4,
        max_iters=50,
        tol=1e-6,
        seed=9323,
    )

    ll_trace = result.log_likelihood_trace
    deltas = np.diff(ll_trace)

    # Core sanity checks.
    assert np.all(deltas >= -1e-8), "EM log-likelihood must be non-decreasing."
    _assert_prob_vector(result.learned_pi, "learned_pi")
    _assert_row_stochastic(result.learned_a, "learned_a")
    _assert_row_stochastic(result.learned_b, "learned_b")

    fb_final = forward_backward_scaled(
        pi=result.learned_pi,
        a=result.learned_a,
        b=result.learned_b,
        obs=obs,
    )
    assert np.allclose(fb_final.gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(fb_final.xi.sum(axis=(1, 2)), 1.0, atol=1e-10)
    assert np.allclose(fb_final.xi.sum(axis=2), fb_final.gamma[:-1], atol=2e-9)
    assert np.allclose(fb_final.xi.sum(axis=1), fb_final.gamma[1:], atol=2e-9)

    print(f"sequence length: {seq_len}")
    print(f"first 20 observations: {obs[:20].tolist()}")
    print(f"first 20 hidden states (ground truth): {states[:20].tolist()}")
    print("-" * 72)

    print(f"iterations run: {result.iterations_run}")
    print(f"initial log-likelihood: {ll_trace[0]:.6f}")
    print(f"final   log-likelihood: {ll_trace[-1]:.6f}")
    print(f"total improvement:      {ll_trace[-1] - ll_trace[0]:.6f}")
    print(f"min delta per iter:     {deltas.min():.6e}")
    print("-" * 72)

    head = ll_trace[: min(6, ll_trace.size)]
    tail = ll_trace[-min(6, ll_trace.size) :]
    print("log-likelihood trace (head):", np.round(head, 6).tolist())
    print("log-likelihood trace (tail):", np.round(tail, 6).tolist())
    print("-" * 72)

    print("True pi:")
    print(_fmt_matrix(pi_true))
    print("Learned pi:")
    print(_fmt_matrix(result.learned_pi))
    print("-" * 72)

    print("True A:")
    print(_fmt_matrix(a_true))
    print("Learned A:")
    print(_fmt_matrix(result.learned_a))
    print("-" * 72)

    print("True B:")
    print(_fmt_matrix(b_true))
    print("Learned B:")
    print(_fmt_matrix(result.learned_b))
    print("-" * 72)

    map_states = np.argmax(fb_final.gamma, axis=1)
    print("first 20 MAP smoothed states from learned model:")
    print(map_states[:20].tolist())
    print("All checks passed.")


if __name__ == "__main__":
    main()
