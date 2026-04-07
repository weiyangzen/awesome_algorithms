"""Minimal runnable MVP for Viterbi algorithm (MATH-0322).

The script implements Viterbi decoding for a discrete HMM in log-domain,
and validates correctness against exhaustive path search on small cases.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np


EPS = 1e-300


@dataclass
class ViterbiResult:
    """Container for Viterbi decoding outputs."""

    path: List[int]
    log_prob: float
    delta: np.ndarray
    psi: np.ndarray


def assert_hmm_params(pi: np.ndarray, a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> None:
    """Validate shape and stochastic constraints for a discrete HMM."""
    if pi.ndim != 1:
        raise ValueError(f"pi must be 1D, got shape={pi.shape}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D matrices.")

    n_states = pi.shape[0]
    if a.shape != (n_states, n_states):
        raise ValueError(f"A must be ({n_states}, {n_states}), got {a.shape}")
    if b.shape[0] != n_states:
        raise ValueError(f"B must have {n_states} rows, got {b.shape[0]}")

    if np.any(pi < -atol) or np.any(a < -atol) or np.any(b < -atol):
        raise ValueError("pi/A/B contain negative probabilities.")

    if not np.isclose(float(pi.sum()), 1.0, atol=atol):
        raise ValueError("pi must sum to 1.")
    if not np.allclose(a.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Each row of A must sum to 1.")
    if not np.allclose(b.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Each row of B must sum to 1.")


def safe_log(x: np.ndarray) -> np.ndarray:
    """Numerically safe log to avoid log(0)."""
    return np.log(np.maximum(x, EPS))


def encode_observations(tokens: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    """Map observation tokens to integer ids."""
    ids = []
    for tok in tokens:
        if tok not in vocab:
            raise ValueError(f"Unknown observation token: {tok}")
        ids.append(vocab[tok])
    return np.asarray(ids, dtype=np.int64)


def viterbi_decode(obs: Sequence[int], pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> ViterbiResult:
    """Decode the most probable hidden-state path with Viterbi DP."""
    assert_hmm_params(pi, a, b)

    obs_arr = np.asarray(obs, dtype=np.int64)
    if obs_arr.ndim != 1:
        raise ValueError(f"obs must be 1D, got shape={obs_arr.shape}")

    n_states = pi.shape[0]
    n_obs = b.shape[1]
    t_len = int(obs_arr.size)

    if t_len == 0:
        return ViterbiResult(path=[], log_prob=0.0, delta=np.zeros((0, n_states)), psi=np.zeros((0, n_states), dtype=np.int64))

    if np.any((obs_arr < 0) | (obs_arr >= n_obs)):
        raise ValueError("obs contains out-of-range symbol ids.")

    log_pi = safe_log(pi)
    log_a = safe_log(a)
    log_b = safe_log(b)

    delta = np.full((t_len, n_states), -np.inf, dtype=np.float64)
    psi = np.zeros((t_len, n_states), dtype=np.int64)

    # Initialization.
    delta[0, :] = log_pi + log_b[:, obs_arr[0]]

    # Recurrence.
    for t in range(1, t_len):
        ot = int(obs_arr[t])
        for j in range(n_states):
            scores = delta[t - 1, :] + log_a[:, j]
            best_prev = int(np.argmax(scores))
            psi[t, j] = best_prev
            delta[t, j] = float(scores[best_prev] + log_b[j, ot])

    # Termination + backtrace.
    last_state = int(np.argmax(delta[-1, :]))
    best_log_prob = float(delta[-1, last_state])

    path = [0] * t_len
    path[-1] = last_state
    for t in range(t_len - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])

    return ViterbiResult(path=path, log_prob=best_log_prob, delta=delta, psi=psi)


def path_log_prob(path: Sequence[int], obs: Sequence[int], pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Compute log P(path, obs) for a fixed state path."""
    path_arr = np.asarray(path, dtype=np.int64)
    obs_arr = np.asarray(obs, dtype=np.int64)

    if path_arr.shape != obs_arr.shape:
        raise ValueError("path and obs must have the same length.")
    if path_arr.size == 0:
        return 0.0

    log_pi = safe_log(pi)
    log_a = safe_log(a)
    log_b = safe_log(b)

    total = float(log_pi[path_arr[0]] + log_b[path_arr[0], obs_arr[0]])
    for t in range(1, path_arr.size):
        total += float(log_a[path_arr[t - 1], path_arr[t]] + log_b[path_arr[t], obs_arr[t]])
    return total


def exhaustive_decode(obs: Sequence[int], pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[List[int], float]:
    """Brute-force best path search, used only for small-case verification."""
    obs_arr = np.asarray(obs, dtype=np.int64)
    t_len = int(obs_arr.size)
    n_states = int(pi.shape[0])

    if t_len == 0:
        return [], 0.0

    best_path: List[int] | None = None
    best_score = -np.inf

    for path_tuple in product(range(n_states), repeat=t_len):
        score = path_log_prob(path_tuple, obs_arr, pi, a, b)
        if score > best_score:
            best_score = score
            best_path = list(path_tuple)

    if best_path is None:
        raise RuntimeError("Brute-force decoding failed unexpectedly.")
    return best_path, float(best_score)


def random_stochastic_vector(rng: np.random.Generator, n: int) -> np.ndarray:
    vec = rng.random(n)
    return vec / vec.sum()


def random_stochastic_matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    mat = rng.random((rows, cols))
    return mat / mat.sum(axis=1, keepdims=True)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    state_names = ["Rainy", "Sunny", "Cloudy"]
    obs_names = ["walk", "shop", "clean"]
    vocab = {name: idx for idx, name in enumerate(obs_names)}

    pi = np.array([0.40, 0.35, 0.25], dtype=np.float64)
    a = np.array(
        [
            [0.60, 0.25, 0.15],
            [0.20, 0.65, 0.15],
            [0.25, 0.25, 0.50],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            [0.20, 0.30, 0.50],
            [0.55, 0.35, 0.10],
            [0.20, 0.50, 0.30],
        ],
        dtype=np.float64,
    )

    obs_tokens = ["walk", "shop", "clean", "clean", "walk"]
    obs = encode_observations(obs_tokens, vocab)

    print("Viterbi Algorithm MVP - MATH-0322")
    print("=" * 80)
    print(f"Observation tokens: {obs_tokens}")
    print(f"Observation ids:    {obs.tolist()}")

    result = viterbi_decode(obs, pi, a, b)
    best_path_names = [state_names[s] for s in result.path]

    brute_path, brute_score = exhaustive_decode(obs, pi, a, b)
    brute_path_names = [state_names[s] for s in brute_path]

    recomputed = path_log_prob(result.path, obs, pi, a, b)

    score_match = bool(np.isclose(result.log_prob, brute_score, atol=1e-10))
    self_match = bool(np.isclose(result.log_prob, recomputed, atol=1e-10))

    print(f"Best path ids (Viterbi):   {result.path}")
    print(f"Best path names (Viterbi): {best_path_names}")
    print(f"Best log prob (Viterbi):   {result.log_prob:.12f}")
    print(f"Best path ids (Bruteforce):   {brute_path}")
    print(f"Best path names (Bruteforce): {brute_path_names}")
    print(f"Best log prob (Bruteforce):   {brute_score:.12f}")
    print(f"Checks: score_match={score_match}, self_match={self_match}")

    if not score_match:
        raise AssertionError("Viterbi score does not match brute-force optimum.")
    if not self_match:
        raise AssertionError("Viterbi path score is inconsistent with returned log_prob.")

    # Randomized small-case regression: compare DP vs exhaustive decode.
    rng = np.random.default_rng(2026)
    num_random_cases = 4
    for cid in range(1, num_random_cases + 1):
        n_states = 3
        n_obs = 4
        t_len = 6

        pi_r = random_stochastic_vector(rng, n_states)
        a_r = random_stochastic_matrix(rng, n_states, n_states)
        b_r = random_stochastic_matrix(rng, n_states, n_obs)
        obs_r = rng.integers(low=0, high=n_obs, size=t_len, dtype=np.int64)

        v_res = viterbi_decode(obs_r, pi_r, a_r, b_r)
        _, b_score = exhaustive_decode(obs_r, pi_r, a_r, b_r)

        if not np.isclose(v_res.log_prob, b_score, atol=1e-10):
            raise AssertionError(f"Random case {cid} failed: Viterbi != brute-force.")

    print(f"Random regression checks passed: {num_random_cases}/{num_random_cases}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
