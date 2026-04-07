"""Minimal runnable MVP for Gibbs sampling.

This script implements source-level Gibbs updates for a multivariate Gaussian:
- derive conditional Normal from precision matrix row
- update each coordinate sequentially
- collect post-warmup samples and verify moment recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

Array = np.ndarray


@dataclass
class GibbsConfig:
    """Configuration for one Gibbs chain."""

    num_warmup: int = 1200
    num_samples: int = 3200
    thinning: int = 1
    seed: int = 20260407


def gibbs_update_gaussian(
    x: Array,
    mean: Array,
    precision: Array,
    rng: np.random.Generator,
) -> None:
    """Run one full Gibbs sweep over all coordinates.

    For Gaussian target with precision Q:
    x_i | x_-i ~ Normal(mu_i - (Q_i,-i (x_-i - mu_-i)) / Q_ii, 1 / Q_ii)
    """

    centered = x - mean
    dim = x.shape[0]

    for i in range(dim):
        q_ii = precision[i, i]
        if q_ii <= 0.0:
            raise ValueError("Precision matrix diagonal must be positive.")

        # Sum_j Q_ij (x_j - mu_j) excluding j=i
        coupled = float(np.dot(precision[i], centered) - q_ii * centered[i])
        cond_mean = mean[i] - coupled / q_ii
        cond_std = float(np.sqrt(1.0 / q_ii))

        x[i] = rng.normal(loc=cond_mean, scale=cond_std)
        centered[i] = x[i] - mean[i]


def run_gibbs_gaussian(
    mean: Array,
    cov: Array,
    initial_state: Array,
    config: GibbsConfig,
) -> Dict[str, Array | float]:
    """Sample from multivariate Gaussian using coordinate Gibbs."""

    if mean.ndim != 1:
        raise ValueError("mean must be a 1D vector.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")
    if cov.shape[0] != mean.shape[0]:
        raise ValueError("Dimension mismatch between mean and cov.")

    precision = np.linalg.inv(cov)
    x = initial_state.astype(float).copy()
    rng = np.random.default_rng(config.seed)

    dim = mean.shape[0]
    samples = np.zeros((config.num_samples, dim), dtype=float)

    total_iters = config.num_warmup + config.num_samples * config.thinning
    collected = 0

    for itr in range(total_iters):
        gibbs_update_gaussian(x=x, mean=mean, precision=precision, rng=rng)

        if itr >= config.num_warmup:
            post_idx = itr - config.num_warmup
            if post_idx % config.thinning == 0:
                samples[collected] = x
                collected += 1

    return {
        "samples": samples,
        "precision": precision,
    }


def empirical_covariance(samples: Array) -> Array:
    """Compute sample covariance with Bessel correction."""

    centered = samples - np.mean(samples, axis=0, keepdims=True)
    return (centered.T @ centered) / (samples.shape[0] - 1)


def estimate_ess_per_dim(samples: Array, max_lag: int = 600) -> Array:
    """Estimate per-dimension ESS via positive autocorrelation truncation."""

    n, dim = samples.shape
    ess = np.zeros(dim, dtype=float)

    for d in range(dim):
        x = samples[:, d] - np.mean(samples[:, d])
        var = float(np.dot(x, x) / n)
        if var <= 1e-15:
            ess[d] = 1.0
            continue

        rho_sum = 0.0
        lag_upper = min(max_lag, n - 1)
        for lag in range(1, lag_upper + 1):
            auto_cov = float(np.dot(x[:-lag], x[lag:]) / (n - lag))
            rho = auto_cov / var
            if rho <= 0.0:
                break
            rho_sum += rho

        ess[d] = n / (1.0 + 2.0 * rho_sum)
        ess[d] = float(np.clip(ess[d], 1.0, float(n)))

    return ess


def lag1_autocorr(samples: Array) -> Array:
    """Lag-1 autocorrelation for each dimension."""

    n, dim = samples.shape
    if n < 2:
        return np.zeros(dim, dtype=float)

    out = np.zeros(dim, dtype=float)
    for d in range(dim):
        x = samples[:, d] - np.mean(samples[:, d])
        denom = float(np.dot(x, x))
        if denom <= 1e-15:
            out[d] = 0.0
            continue
        out[d] = float(np.dot(x[:-1], x[1:]) / denom)
    return out


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    true_mean = np.array([1.2, -0.8, 0.5], dtype=float)
    true_cov = np.array(
        [
            [1.00, 0.65, 0.30],
            [0.65, 1.60, 0.45],
            [0.30, 0.45, 1.20],
        ],
        dtype=float,
    )
    initial_state = np.array([3.0, -3.0, 2.5], dtype=float)

    config = GibbsConfig()
    result = run_gibbs_gaussian(
        mean=true_mean,
        cov=true_cov,
        initial_state=initial_state,
        config=config,
    )
    samples = result["samples"]

    est_mean = np.mean(samples, axis=0)
    est_cov = empirical_covariance(samples)
    ess = estimate_ess_per_dim(samples)
    acf1 = lag1_autocorr(samples)

    mean_l2_error = float(np.linalg.norm(est_mean - true_mean))
    cov_fro_error = float(np.linalg.norm(est_cov - true_cov, ord="fro"))

    print("=== Gibbs Sampling MVP (Multivariate Gaussian) ===")
    print(f"seed: {config.seed}")
    print(
        f"warmup_iters: {config.num_warmup}, "
        f"sample_iters: {config.num_samples}, thinning: {config.thinning}"
    )
    print(f"target_dim: {true_mean.shape[0]}")
    print(f"true_mean: {true_mean}")
    print(f"estimated_mean: {est_mean}")
    print(f"mean_l2_error: {mean_l2_error:.6f}")
    print(f"true_cov:\n{true_cov}")
    print(f"estimated_cov:\n{est_cov}")
    print(f"cov_fro_error: {cov_fro_error:.6f}")
    print(f"lag1_autocorr: {acf1}")
    print(f"ESS per dim: {ess}")

    assert mean_l2_error < 0.14, "Mean estimation error is too large."
    assert cov_fro_error < 0.26, "Covariance estimation error is too large."
    assert float(np.max(acf1)) < 0.90, "Lag-1 autocorrelation is unexpectedly high."
    assert float(np.min(ess)) > 380.0, "ESS is too low for this demo setup."

    print("All checks passed.")


if __name__ == "__main__":
    main()
