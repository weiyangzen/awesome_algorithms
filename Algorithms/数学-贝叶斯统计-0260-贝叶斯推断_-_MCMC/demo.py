"""Minimal runnable MVP for Bayesian inference via MCMC.

Model:
    Bayesian logistic regression with Gaussian prior.
Sampler:
    Random-Walk Metropolis (multi-chain).

This script is deterministic, requires no interactive input,
and prints posterior summaries plus basic MCMC diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class MCMCConfig:
    n_chains: int = 4
    n_steps: int = 8000
    burn_in: int = 3000
    thin: int = 5
    prior_std: float = 2.5
    proposal_scale: float = 0.25
    adapt_interval: int = 100
    target_accept_low: float = 0.20
    target_accept_high: float = 0.45


@dataclass
class ChainResult:
    samples: np.ndarray
    acceptance_rate: float
    final_proposal_scale: float


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0.0
    negative = ~positive
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    return out


def validate_dataset(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X rows={x.shape[0]}, y len={y.shape[0]}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must be non-empty in both dimensions.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, np.array([0, 1])) and not np.array_equal(unique_values, np.array([0])) and not np.array_equal(unique_values, np.array([1])):
        raise ValueError(f"y must only contain 0/1, got unique={unique_values}.")
    if unique_values.size < 2:
        raise ValueError("y has only one class; need both classes for logistic inference demo.")


def validate_config(config: MCMCConfig) -> None:
    if config.n_chains <= 1:
        raise ValueError("n_chains must be >= 2 for multi-chain diagnostics.")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if config.burn_in < 0:
        raise ValueError("burn_in must be >= 0.")
    if config.n_steps <= config.burn_in:
        raise ValueError("n_steps must be greater than burn_in.")
    if config.thin <= 0:
        raise ValueError("thin must be > 0.")
    if config.prior_std <= 0.0:
        raise ValueError("prior_std must be > 0.")
    if config.proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be > 0.")
    if config.adapt_interval <= 0:
        raise ValueError("adapt_interval must be > 0.")


def log_posterior(beta: np.ndarray, x: np.ndarray, y: np.ndarray, prior_std: float) -> float:
    eta = x @ beta
    log_likelihood = float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    prior_var = prior_std * prior_std
    p = beta.size
    log_prior = -0.5 * float(np.dot(beta, beta)) / prior_var - p * np.log(prior_std * np.sqrt(2.0 * np.pi))
    return log_likelihood + log_prior


def run_rw_metropolis_chain(
    x: np.ndarray,
    y: np.ndarray,
    initial_beta: np.ndarray,
    seed: int,
    config: MCMCConfig,
) -> ChainResult:
    rng = np.random.default_rng(seed)

    beta = initial_beta.astype(float, copy=True)
    dim = beta.size
    current_log_post = log_posterior(beta=beta, x=x, y=y, prior_std=config.prior_std)

    proposal_scale = float(config.proposal_scale)
    accepted_total = 0
    accepted_window = 0

    kept_samples: List[np.ndarray] = []

    for step in range(1, config.n_steps + 1):
        proposal = beta + rng.normal(loc=0.0, scale=proposal_scale, size=dim)
        proposal_log_post = log_posterior(beta=proposal, x=x, y=y, prior_std=config.prior_std)

        log_accept_ratio = proposal_log_post - current_log_post
        if np.log(rng.uniform()) < log_accept_ratio:
            beta = proposal
            current_log_post = proposal_log_post
            accepted_total += 1
            accepted_window += 1

        if step <= config.burn_in and step % config.adapt_interval == 0:
            window_rate = accepted_window / config.adapt_interval
            if window_rate < config.target_accept_low:
                proposal_scale *= 0.85
            elif window_rate > config.target_accept_high:
                proposal_scale *= 1.15
            accepted_window = 0

        if step > config.burn_in and (step - config.burn_in) % config.thin == 0:
            kept_samples.append(beta.copy())

    if not kept_samples:
        raise RuntimeError("No posterior samples kept. Increase n_steps or reduce burn_in/thin.")

    samples = np.vstack(kept_samples)
    acceptance_rate = accepted_total / config.n_steps

    return ChainResult(
        samples=samples,
        acceptance_rate=float(acceptance_rate),
        final_proposal_scale=float(proposal_scale),
    )


def rhat_per_dimension(chains: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat per parameter.

    chains shape: (n_chains, n_samples_per_chain, n_params)
    """
    m, n, p = chains.shape
    chain_means = np.mean(chains, axis=1)  # (m, p)
    grand_mean = np.mean(chain_means, axis=0)  # (p,)

    b = n * np.sum((chain_means - grand_mean) ** 2, axis=0) / (m - 1)
    w = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    var_hat = ((n - 1) / n) * w + (1.0 / n) * b
    rhat = np.sqrt(var_hat / w)
    return rhat


def ess_per_dimension(chains: np.ndarray, max_lag: int = 400) -> np.ndarray:
    """Rough ESS estimate using averaged autocorrelation truncation at first non-positive lag."""
    m, n, p = chains.shape
    max_lag = min(max_lag, n - 1)
    ess = np.empty(p, dtype=float)

    for d in range(p):
        chain_d = chains[:, :, d]
        means = np.mean(chain_d, axis=1, keepdims=True)
        centered = chain_d - means
        var0 = np.mean(np.mean(centered * centered, axis=1))

        if var0 <= 1e-15:
            ess[d] = float(m * n)
            continue

        rho_sum = 0.0
        for lag in range(1, max_lag + 1):
            acov_lag = np.mean(np.mean(centered[:, :-lag] * centered[:, lag:], axis=1))
            rho_lag = acov_lag / var0
            if rho_lag <= 0.0:
                break
            rho_sum += rho_lag

        tau = 1.0 + 2.0 * rho_sum
        raw_ess = (m * n) / tau
        ess[d] = float(min(max(raw_ess, 1.0), m * n))

    return ess


def summarize_posterior(samples: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    q025 = np.quantile(samples, 0.025, axis=0)
    q975 = np.quantile(samples, 0.975, axis=0)
    return {
        "mean": mean,
        "std": std,
        "q025": q025,
        "q975": q975,
    }


def make_synthetic_logistic_data(
    seed: int = 2026,
    n_samples: int = 450,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    x_raw = rng.normal(size=(n_samples, 3))
    x_raw -= np.mean(x_raw, axis=0)
    x_std = np.std(x_raw, axis=0)
    if np.any(x_std < 1e-12):
        raise RuntimeError("Unexpected near-constant synthetic feature.")
    x_raw /= x_std

    x = np.column_stack([np.ones(n_samples), x_raw])

    true_beta = np.array([-0.7, 1.3, -1.0, 0.8], dtype=float)
    prob = sigmoid(x @ true_beta)
    y = rng.binomial(1, prob).astype(float)

    validate_dataset(x=x, y=y)
    return x, y, true_beta


def main() -> None:
    x, y, true_beta = make_synthetic_logistic_data(seed=2026, n_samples=450)

    config = MCMCConfig(
        n_chains=4,
        n_steps=8000,
        burn_in=3000,
        thin=5,
        prior_std=2.5,
        proposal_scale=0.25,
        adapt_interval=100,
        target_accept_low=0.20,
        target_accept_high=0.45,
    )
    validate_config(config)

    rng_init = np.random.default_rng(12345)

    chain_results: List[ChainResult] = []
    for chain_id in range(config.n_chains):
        init = rng_init.normal(loc=0.0, scale=0.4, size=x.shape[1])
        chain_result = run_rw_metropolis_chain(
            x=x,
            y=y,
            initial_beta=init,
            seed=1000 + chain_id,
            config=config,
        )
        chain_results.append(chain_result)

    samples_by_chain = np.stack([c.samples for c in chain_results], axis=0)
    combined_samples = np.vstack([c.samples for c in chain_results])

    summary = summarize_posterior(combined_samples)
    rhat = rhat_per_dimension(samples_by_chain)
    ess = ess_per_dimension(samples_by_chain, max_lag=300)

    posterior_mean_prob = sigmoid(x @ summary["mean"])
    posterior_pred = (posterior_mean_prob >= 0.5).astype(float)
    accuracy = float(np.mean(posterior_pred == y))

    param_names = ["intercept", "beta_1", "beta_2", "beta_3"]

    print("=== Bayesian Inference via MCMC (Random-Walk Metropolis) ===")
    print(f"dataset: n={x.shape[0]}, p={x.shape[1]} (including intercept)")
    print(
        "config: "
        f"chains={config.n_chains}, steps={config.n_steps}, burn_in={config.burn_in}, thin={config.thin}"
    )
    print()

    print("Per-chain diagnostics:")
    for i, result in enumerate(chain_results, start=1):
        print(
            f"  chain {i}: acceptance_rate={result.acceptance_rate:.4f}, "
            f"final_proposal_scale={result.final_proposal_scale:.4f}, "
            f"kept_samples={result.samples.shape[0]}"
        )

    print("\nPosterior summary by parameter:")
    print("name       | true     | mean     | std      | q2.5     | q97.5    | R-hat   | ESS")
    print("------------------------------------------------------------------------------------------")

    for i, name in enumerate(param_names):
        print(
            f"{name:10s} | "
            f"{true_beta[i]:8.4f} | "
            f"{summary['mean'][i]:8.4f} | "
            f"{summary['std'][i]:8.4f} | "
            f"{summary['q025'][i]:8.4f} | "
            f"{summary['q975'][i]:8.4f} | "
            f"{rhat[i]:7.4f} | "
            f"{ess[i]:6.1f}"
        )

    print("\nPosterior predictive accuracy (using posterior mean beta): " f"{accuracy:.4f}")


if __name__ == "__main__":
    main()
