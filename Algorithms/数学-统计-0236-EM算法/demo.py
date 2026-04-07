"""Expectation-Maximization (EM) MVP on a binomial mixture model.

This script demonstrates EM from scratch on a latent-variable setting:
- Latent component z_i selects one of K coins.
- Observation x_i is the number of heads in m tosses.
- Parameters are mixture weights and per-component head probabilities.

Run:
    python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import List, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class EMResult:
    weights: np.ndarray
    probs: np.ndarray
    responsibilities: np.ndarray
    log_likelihood_trace: List[float]
    converged: bool
    n_iter: int


def validate_inputs(
    head_counts: np.ndarray,
    n_tosses: int,
    n_components: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    head_counts = np.asarray(head_counts)
    if head_counts.ndim != 1:
        raise ValueError(f"head_counts must be 1D, got shape={head_counts.shape}")
    if head_counts.size < 2:
        raise ValueError("head_counts must contain at least 2 observations")
    if not np.issubdtype(head_counts.dtype, np.integer):
        raise ValueError("head_counts must contain integer counts")
    if n_tosses < 1:
        raise ValueError("n_tosses must be >= 1")
    if np.any(head_counts < 0) or np.any(head_counts > n_tosses):
        raise ValueError("head_counts must satisfy 0 <= x_i <= n_tosses")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")
    return head_counts.astype(float)


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    stable = np.exp(a - a_max)
    summed = np.sum(stable, axis=axis, keepdims=True)
    out = np.log(summed) + a_max
    return np.squeeze(out, axis=axis)


def log_binomial_coeff_vec(head_counts: np.ndarray, n_tosses: int) -> np.ndarray:
    n = float(n_tosses)
    return np.array(
        [
            math.lgamma(n + 1.0)
            - math.lgamma(float(h) + 1.0)
            - math.lgamma(n - float(h) + 1.0)
            for h in head_counts
        ],
        dtype=float,
    )


def e_step(
    head_counts: np.ndarray,
    n_tosses: int,
    weights: np.ndarray,
    probs: np.ndarray,
    log_coeff: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n_samples = head_counts.shape[0]
    n_components = weights.shape[0]
    weighted_log_prob = np.empty((n_samples, n_components), dtype=float)

    tails = float(n_tosses) - head_counts
    for k in range(n_components):
        p = float(np.clip(probs[k], EPS, 1.0 - EPS))
        wk = float(np.clip(weights[k], EPS, 1.0))
        weighted_log_prob[:, k] = (
            np.log(wk)
            + log_coeff
            + head_counts * np.log(p)
            + tails * np.log(1.0 - p)
        )

    log_norm = logsumexp(weighted_log_prob, axis=1)
    log_resp = weighted_log_prob - log_norm[:, None]
    responsibilities = np.exp(log_resp)
    total_log_likelihood = float(np.sum(log_norm))
    return responsibilities, total_log_likelihood


def m_step(
    head_counts: np.ndarray,
    n_tosses: int,
    responsibilities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = head_counts.shape[0]
    nk = np.sum(responsibilities, axis=0) + EPS

    weights = nk / float(n_samples)
    probs = (responsibilities.T @ head_counts) / (nk * float(n_tosses))
    probs = np.clip(probs, EPS, 1.0 - EPS)

    # Normalize again to guard against tiny numerical drift.
    weights = weights / np.sum(weights)
    return weights, probs


def fit_em_binomial_mixture(
    head_counts: np.ndarray,
    n_tosses: int,
    n_components: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 2026,
) -> EMResult:
    head_counts = validate_inputs(
        head_counts=head_counts,
        n_tosses=n_tosses,
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
    )

    rng = np.random.default_rng(seed)
    weights = np.full(n_components, 1.0 / float(n_components), dtype=float)

    # Spread initial probabilities to reduce symmetric dead starts.
    probs = np.linspace(0.2, 0.8, num=n_components, dtype=float)
    probs += rng.normal(0.0, 0.03, size=n_components)
    probs = np.clip(probs, 0.05, 0.95)

    log_coeff = log_binomial_coeff_vec(head_counts=head_counts, n_tosses=n_tosses)

    ll_trace: List[float] = []
    converged = False
    final_resp = np.empty((head_counts.shape[0], n_components), dtype=float)

    for iteration in range(1, max_iter + 1):
        resp, ll = e_step(
            head_counts=head_counts,
            n_tosses=n_tosses,
            weights=weights,
            probs=probs,
            log_coeff=log_coeff,
        )
        ll_trace.append(ll)
        final_resp = resp

        if iteration > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            converged = True
            return EMResult(
                weights=weights,
                probs=probs,
                responsibilities=final_resp,
                log_likelihood_trace=ll_trace,
                converged=converged,
                n_iter=iteration,
            )

        weights, probs = m_step(
            head_counts=head_counts,
            n_tosses=n_tosses,
            responsibilities=resp,
        )

    final_resp, final_ll = e_step(
        head_counts=head_counts,
        n_tosses=n_tosses,
        weights=weights,
        probs=probs,
        log_coeff=log_coeff,
    )
    if abs(final_ll - ll_trace[-1]) > 1e-12:
        ll_trace.append(final_ll)

    return EMResult(
        weights=weights,
        probs=probs,
        responsibilities=final_resp,
        log_likelihood_trace=ll_trace,
        converged=converged,
        n_iter=max_iter,
    )


def generate_synthetic_counts(
    n_samples: int,
    n_tosses: int,
    true_weights: np.ndarray,
    true_probs: np.ndarray,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_components = true_weights.shape[0]
    labels = rng.choice(n_components, size=n_samples, p=true_weights)
    head_counts = rng.binomial(n=n_tosses, p=true_probs[labels])
    return head_counts.astype(int), labels.astype(int)


def best_permutation_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_components: int,
) -> float:
    if n_components > 8:
        raise ValueError("permutation accuracy is limited to n_components <= 8")

    best = 0.0
    for perm in permutations(range(n_components)):
        mapped = np.array([perm[int(c)] for c in y_pred], dtype=int)
        acc = float(np.mean(mapped == y_true))
        if acc > best:
            best = acc
    return best


def print_trace_tail(trace: List[float], tail: int = 8) -> None:
    print("\n=== Log-Likelihood Trace (tail) ===")
    start = max(0, len(trace) - tail)
    for i in range(start, len(trace)):
        delta = float("nan") if i == 0 else trace[i] - trace[i - 1]
        print(f"iter={i + 1:3d}  ll={trace[i]:10.4f}  delta={delta:10.6f}")


def main() -> None:
    n_components = 2
    n_samples = 1200
    n_tosses = 12

    true_weights = np.array([0.62, 0.38], dtype=float)
    true_probs = np.array([0.84, 0.23], dtype=float)

    head_counts, true_labels = generate_synthetic_counts(
        n_samples=n_samples,
        n_tosses=n_tosses,
        true_weights=true_weights,
        true_probs=true_probs,
        seed=2026,
    )

    result = fit_em_binomial_mixture(
        head_counts=head_counts,
        n_tosses=n_tosses,
        n_components=n_components,
        max_iter=200,
        tol=1e-7,
        seed=2026,
    )

    pred_labels = np.argmax(result.responsibilities, axis=1)
    cluster_acc = best_permutation_accuracy(
        y_true=true_labels,
        y_pred=pred_labels,
        n_components=n_components,
    )

    diffs = np.diff(np.array(result.log_likelihood_trace, dtype=float))
    negative_steps = int(np.sum(diffs < -1e-8))

    sort_true = np.argsort(true_probs)
    sort_est = np.argsort(result.probs)
    aligned_true_probs = true_probs[sort_true]
    aligned_est_probs = result.probs[sort_est]
    aligned_true_weights = true_weights[sort_true]
    aligned_est_weights = result.weights[sort_est]

    print("=== EM for Binomial Mixture ===")
    print(f"samples={n_samples}, tosses_per_sample={n_tosses}, components={n_components}")
    print(f"converged={result.converged}, iterations_used={result.n_iter}")
    print(f"final_log_likelihood={result.log_likelihood_trace[-1]:.4f}")
    print(f"negative_ll_steps={negative_steps}")
    print(f"best_permutation_accuracy={cluster_acc:.4f}")

    print("\nTrue weights       :", np.round(true_weights, 4))
    print("Estimated weights  :", np.round(result.weights, 4))
    print("True probs (heads) :", np.round(true_probs, 4))
    print("Estimated probs    :", np.round(result.probs, 4))

    print_trace_tail(result.log_likelihood_trace, tail=8)

    prob_mae = float(np.mean(np.abs(aligned_est_probs - aligned_true_probs)))
    weight_mae = float(np.mean(np.abs(aligned_est_weights - aligned_true_weights)))

    # Minimal quality gates for reproducible MVP behavior.
    assert np.all(np.isfinite(result.weights)) and np.all(np.isfinite(result.probs))
    assert abs(float(np.sum(result.weights)) - 1.0) < 1e-8
    assert np.all(result.weights > 0.0)
    assert np.all((result.probs > 0.0) & (result.probs < 1.0))
    assert cluster_acc >= 0.78, f"cluster accuracy too low: {cluster_acc:.4f}"
    assert negative_steps == 0, "log-likelihood should be non-decreasing"
    assert prob_mae < 0.08, f"probability estimation error too high: {prob_mae:.4f}"
    assert weight_mae < 0.08, f"weight estimation error too high: {weight_mae:.4f}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
