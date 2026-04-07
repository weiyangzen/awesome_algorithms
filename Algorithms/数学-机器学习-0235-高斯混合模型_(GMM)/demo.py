"""Gaussian Mixture Model (GMM) MVP via EM from scratch.

This script:
- Generates synthetic data from a known Gaussian mixture.
- Fits a full-covariance GMM using the EM algorithm (numpy-only core).
- Prints convergence diagnostics and clustering quality.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class GMMParams:
    weights: np.ndarray  # shape: (K,)
    means: np.ndarray  # shape: (K, D)
    covariances: np.ndarray  # shape: (K, D, D)


@dataclass
class EMResult:
    params: GMMParams
    log_likelihood_trace: List[float]
    converged: bool
    n_iter: int
    responsibilities: np.ndarray


def validate_inputs(
    x: np.ndarray,
    n_components: int,
    reg_covar: float,
    max_iter: int,
    tol: float,
) -> Tuple[int, int]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("x must contain at least 2 samples")
    if n_features < 1:
        raise ValueError("x must contain at least 1 feature")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if n_components > n_samples:
        raise ValueError("n_components cannot exceed number of samples")
    if reg_covar <= 0.0:
        raise ValueError("reg_covar must be > 0")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")
    return n_samples, n_features


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    stabilized = np.exp(a - a_max)
    out = np.log(np.sum(stabilized, axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis)


def init_means_kmeanspp(x: np.ndarray, n_components: int, rng: np.random.Generator) -> np.ndarray:
    n_samples, n_features = x.shape
    means = np.empty((n_components, n_features), dtype=float)

    first = int(rng.integers(0, n_samples))
    means[0] = x[first]
    closest_dist_sq = np.sum((x - means[0]) ** 2, axis=1)

    for k in range(1, n_components):
        probs = closest_dist_sq / np.sum(closest_dist_sq)
        idx = int(rng.choice(n_samples, p=probs))
        means[k] = x[idx]
        dist_sq = np.sum((x - means[k]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

    return means


def initialize_parameters(
    x: np.ndarray,
    n_components: int,
    reg_covar: float,
    rng: np.random.Generator,
) -> GMMParams:
    n_samples, n_features = x.shape
    weights = np.full(n_components, 1.0 / n_components, dtype=float)
    means = init_means_kmeanspp(x, n_components=n_components, rng=rng)

    base_cov = np.cov(x, rowvar=False, bias=True)
    if base_cov.ndim == 0:
        base_cov = np.array([[float(base_cov)]], dtype=float)
    base_cov = np.asarray(base_cov, dtype=float)
    base_cov = 0.5 * (base_cov + base_cov.T)
    base_cov += reg_covar * np.eye(n_features, dtype=float)

    covariances = np.repeat(base_cov[None, :, :], n_components, axis=0)
    _ = n_samples  # keep signature parallel with later extensions
    return GMMParams(weights=weights, means=means, covariances=covariances)


def gaussian_log_pdf_full_cov(
    x: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    n_features = x.shape[1]
    centered = x - mean

    # Cholesky gives stable inverse-quadratic + log-det computation.
    chol = np.linalg.cholesky(covariance)
    solve = np.linalg.solve(chol, centered.T)
    maha = np.sum(solve * solve, axis=0)

    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    const = n_features * np.log(2.0 * np.pi)
    return -0.5 * (const + log_det + maha)


def e_step(x: np.ndarray, params: GMMParams) -> Tuple[np.ndarray, float]:
    n_samples = x.shape[0]
    n_components = params.weights.shape[0]

    weighted_log_prob = np.empty((n_samples, n_components), dtype=float)
    for k in range(n_components):
        weighted_log_prob[:, k] = np.log(params.weights[k] + EPS) + gaussian_log_pdf_full_cov(
            x=x,
            mean=params.means[k],
            covariance=params.covariances[k],
        )

    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    log_resp = weighted_log_prob - log_prob_norm[:, None]
    responsibilities = np.exp(log_resp)
    total_log_likelihood = float(np.sum(log_prob_norm))
    return responsibilities, total_log_likelihood


def m_step(x: np.ndarray, responsibilities: np.ndarray, reg_covar: float) -> GMMParams:
    n_samples, n_features = x.shape
    n_components = responsibilities.shape[1]

    nk = np.sum(responsibilities, axis=0) + EPS
    weights = nk / n_samples
    means = (responsibilities.T @ x) / nk[:, None]

    covariances = np.empty((n_components, n_features, n_features), dtype=float)
    for k in range(n_components):
        diff = x - means[k]
        weighted = responsibilities[:, k][:, None] * diff
        cov = (weighted.T @ diff) / nk[k]
        cov = 0.5 * (cov + cov.T)
        cov += reg_covar * np.eye(n_features, dtype=float)
        covariances[k] = cov

    return GMMParams(weights=weights, means=means, covariances=covariances)


def fit_gmm_em(
    x: np.ndarray,
    n_components: int,
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 2026,
) -> EMResult:
    validate_inputs(
        x=x,
        n_components=n_components,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
    )

    rng = np.random.default_rng(seed)
    params = initialize_parameters(
        x=x,
        n_components=n_components,
        reg_covar=reg_covar,
        rng=rng,
    )

    ll_trace: List[float] = []
    converged = False
    prev_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        resp, ll = e_step(x, params)
        ll_trace.append(ll)

        if iteration > 1 and abs(ll - prev_ll) < tol:
            converged = True
            return EMResult(
                params=params,
                log_likelihood_trace=ll_trace,
                converged=converged,
                n_iter=iteration,
                responsibilities=resp,
            )

        params = m_step(x=x, responsibilities=resp, reg_covar=reg_covar)
        prev_ll = ll

    final_resp, final_ll = e_step(x, params)
    if not ll_trace or abs(final_ll - ll_trace[-1]) > 1e-12:
        ll_trace.append(final_ll)

    return EMResult(
        params=params,
        log_likelihood_trace=ll_trace,
        converged=converged,
        n_iter=max_iter,
        responsibilities=final_resp,
    )


def sample_from_true_gmm(
    n_samples: int = 900,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, GMMParams]:
    rng = np.random.default_rng(seed)

    weights = np.array([0.50, 0.30, 0.20], dtype=float)
    means = np.array(
        [
            [-2.0, -1.0],
            [3.0, 2.0],
            [0.0, 5.0],
        ],
        dtype=float,
    )
    covariances = np.array(
        [
            [[0.70, 0.20], [0.20, 0.45]],
            [[0.55, -0.18], [-0.18, 0.85]],
            [[0.65, 0.10], [0.10, 0.50]],
        ],
        dtype=float,
    )

    components = rng.choice(weights.shape[0], size=n_samples, p=weights)
    x = np.empty((n_samples, means.shape[1]), dtype=float)

    for k in range(weights.shape[0]):
        idx = components == k
        count = int(np.sum(idx))
        if count == 0:
            continue
        x[idx] = rng.multivariate_normal(mean=means[k], cov=covariances[k], size=count)

    true_params = GMMParams(weights=weights, means=means, covariances=covariances)
    return x, components, true_params


def hard_labels(responsibilities: np.ndarray) -> np.ndarray:
    return np.argmax(responsibilities, axis=1)


def best_label_permutation_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_components: int) -> float:
    if n_components > 8:
        raise ValueError("Brute-force permutation accuracy is limited to n_components <= 8")

    best_acc = 0.0
    for perm in permutations(range(n_components)):
        mapped = np.array([perm[int(c)] for c in y_pred], dtype=int)
        acc = float(np.mean(mapped == y_true))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def count_free_parameters(n_components: int, n_features: int) -> int:
    weight_params = n_components - 1
    mean_params = n_components * n_features
    cov_params = n_components * (n_features * (n_features + 1) // 2)
    return weight_params + mean_params + cov_params


def aic_bic(total_log_likelihood: float, n_samples: int, n_params: int) -> Tuple[float, float]:
    aic = 2.0 * n_params - 2.0 * total_log_likelihood
    bic = np.log(n_samples) * n_params - 2.0 * total_log_likelihood
    return float(aic), float(bic)


def print_trace_summary(log_likelihood_trace: List[float], last_n: int = 6) -> None:
    print("\n=== Log-Likelihood Trace (tail) ===")
    start = max(0, len(log_likelihood_trace) - last_n)
    for i in range(start, len(log_likelihood_trace)):
        delta = float("nan") if i == 0 else log_likelihood_trace[i] - log_likelihood_trace[i - 1]
        print(
            f"iter={i + 1:3d}  ll={log_likelihood_trace[i]:10.4f}  "
            f"delta={delta:10.6f}"
        )


def main() -> None:
    n_components = 3
    x, true_labels, true_params = sample_from_true_gmm(n_samples=900, seed=2026)

    result = fit_gmm_em(
        x=x,
        n_components=n_components,
        reg_covar=1e-6,
        max_iter=200,
        tol=1e-4,
        seed=42,
    )

    pred_labels = hard_labels(result.responsibilities)
    acc = best_label_permutation_accuracy(
        y_true=true_labels,
        y_pred=pred_labels,
        n_components=n_components,
    )

    final_ll = result.log_likelihood_trace[-1]
    n_samples, n_features = x.shape
    n_params = count_free_parameters(n_components=n_components, n_features=n_features)
    aic, bic = aic_bic(total_log_likelihood=final_ll, n_samples=n_samples, n_params=n_params)

    print("=== GMM (EM) MVP Demo ===")
    print(f"data shape: {x.shape}")
    print(f"n_components: {n_components}")
    print(f"converged: {result.converged}")
    print(f"iterations used: {result.n_iter}")
    print(f"final log-likelihood: {final_ll:.4f}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")
    print(f"best-permutation clustering accuracy: {acc:.4f}")

    print("\n=== True vs Estimated Weights ===")
    print(f"true weights:      {np.round(true_params.weights, 4)}")
    print(f"estimated weights: {np.round(result.params.weights, 4)}")

    print("\n=== Estimated Means ===")
    for k, mu in enumerate(result.params.means):
        print(f"component {k}: mean={np.round(mu, 4)}")

    print_trace_summary(result.log_likelihood_trace, last_n=8)

    if len(result.log_likelihood_trace) >= 2:
        diffs = np.diff(np.array(result.log_likelihood_trace, dtype=float))
        monotone_violations = int(np.sum(diffs < -1e-6))
        print(f"\nmonotonicity check: negative ll steps = {monotone_violations}")

    # Basic quality assertions for this synthetic setting.
    assert acc > 0.85, "Clustering accuracy is unexpectedly low for this synthetic dataset."
    assert np.all(result.params.weights > 0.0), "All estimated mixture weights must be positive."


if __name__ == "__main__":
    main()
