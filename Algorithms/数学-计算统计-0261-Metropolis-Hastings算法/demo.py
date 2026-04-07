"""Minimal runnable MVP for Metropolis-Hastings (MH) algorithm.

Task-specific demo:
- Target distribution: Gamma posterior of a Poisson rate lambda.
- Sampler: Metropolis-Hastings with log-normal random-walk proposal
  (asymmetric, so Hastings correction is explicit).
- Validation: compare sample moments/CDF with analytic Gamma posterior.

The script is deterministic, needs no interactive input, and prints concise diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.metrics import mean_squared_error


@dataclass
class MHConfig:
    n_steps: int = 16000
    burn_in: int = 3000
    thin: int = 4
    proposal_sigma: float = 0.35
    init_lambda: float = 1.0
    seed: int = 2026


@dataclass
class MHResult:
    samples: np.ndarray
    acceptance_rate: float
    full_chain: np.ndarray


def validate_config(config: MHConfig) -> None:
    if config.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if config.burn_in < 0:
        raise ValueError("burn_in must be >= 0.")
    if config.n_steps <= config.burn_in:
        raise ValueError("n_steps must be greater than burn_in.")
    if config.thin <= 0:
        raise ValueError("thin must be > 0.")
    if config.proposal_sigma <= 0.0:
        raise ValueError("proposal_sigma must be > 0.")
    if config.init_lambda <= 0.0:
        raise ValueError("init_lambda must be > 0.")


def generate_poisson_data(n_obs: int, lambda_true: float, seed: int) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be > 0.")
    if lambda_true <= 0.0:
        raise ValueError("lambda_true must be > 0.")

    rng = np.random.default_rng(seed)
    y = rng.poisson(lam=lambda_true, size=n_obs)
    return y.astype(int)


def posterior_gamma_params(
    counts: np.ndarray,
    alpha_prior: float,
    beta_prior: float,
) -> Tuple[float, float]:
    if counts.ndim != 1:
        raise ValueError("counts must be 1D.")
    if counts.size == 0:
        raise ValueError("counts must be non-empty.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    if alpha_prior <= 0.0 or beta_prior <= 0.0:
        raise ValueError("alpha_prior and beta_prior must be > 0.")

    alpha_post = alpha_prior + float(np.sum(counts))
    beta_post = beta_prior + float(counts.size)
    return alpha_post, beta_post


def log_gamma_target_unnormalized(x: float, alpha: float, beta: float) -> float:
    """Unnormalized log density for Gamma(alpha, rate=beta), x>0."""
    if x <= 0.0:
        return -np.inf
    return (alpha - 1.0) * np.log(x) - beta * x


def log_lognormal_q(x_to: float, x_from: float, sigma: float) -> float:
    """Log-density of q(x_to | x_from) for log-normal random walk.

    Proposal:
        log(x_to) ~ Normal(log(x_from), sigma^2)
    """
    if x_to <= 0.0 or x_from <= 0.0:
        return -np.inf
    z = (np.log(x_to) - np.log(x_from)) / sigma
    return -np.log(x_to * sigma * np.sqrt(2.0 * np.pi)) - 0.5 * z * z


def metropolis_hastings_gamma(alpha: float, beta: float, config: MHConfig) -> MHResult:
    validate_config(config)

    rng = np.random.default_rng(config.seed)
    current = float(config.init_lambda)
    current_log_target = log_gamma_target_unnormalized(current, alpha, beta)

    chain = np.empty(config.n_steps, dtype=float)
    accepted = 0

    for t in range(config.n_steps):
        proposal = current * np.exp(config.proposal_sigma * rng.normal())
        proposal_log_target = log_gamma_target_unnormalized(proposal, alpha, beta)

        # MH acceptance with explicit Hastings correction.
        log_accept_ratio = (
            proposal_log_target
            - current_log_target
            + log_lognormal_q(current, proposal, config.proposal_sigma)
            - log_lognormal_q(proposal, current, config.proposal_sigma)
        )

        if np.log(rng.uniform()) < log_accept_ratio:
            current = proposal
            current_log_target = proposal_log_target
            accepted += 1

        chain[t] = current

    kept = chain[config.burn_in :: config.thin]
    if kept.size == 0:
        raise RuntimeError("No samples kept after burn-in/thin. Adjust config.")

    return MHResult(
        samples=kept,
        acceptance_rate=float(accepted / config.n_steps),
        full_chain=chain,
    )


def estimate_ess(samples: np.ndarray, max_lag: int = 500) -> float:
    """Simple ESS estimator via autocorrelation truncation at first non-positive lag."""
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 3:
        return float(n)

    centered = x - np.mean(x)
    var0 = np.dot(centered, centered) / n
    if var0 <= 1e-15:
        return float(n)

    max_lag = min(max_lag, n - 1)
    rho_sum = 0.0

    for lag in range(1, max_lag + 1):
        acov = np.dot(centered[:-lag], centered[lag:]) / (n - lag)
        rho = acov / var0
        if rho <= 0.0:
            break
        rho_sum += rho

    tau = 1.0 + 2.0 * rho_sum
    ess = n / tau
    return float(np.clip(ess, 1.0, n))


def empirical_cdf(sorted_samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ranks = np.searchsorted(sorted_samples, grid, side="right")
    return ranks / sorted_samples.size


def main() -> None:
    # Synthetic Poisson observations for Bayesian inference of lambda.
    n_obs = 80
    lambda_true = 4.2
    alpha_prior = 2.0
    beta_prior = 1.5

    counts = generate_poisson_data(n_obs=n_obs, lambda_true=lambda_true, seed=7)
    alpha_post, beta_post = posterior_gamma_params(
        counts=counts,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
    )

    config = MHConfig(
        n_steps=16000,
        burn_in=3000,
        thin=4,
        proposal_sigma=0.35,
        init_lambda=1.0,
        seed=2026,
    )

    result = metropolis_hastings_gamma(alpha=alpha_post, beta=beta_post, config=config)

    samples = result.samples
    sorted_samples = np.sort(samples)

    dist = st.gamma(a=alpha_post, scale=1.0 / beta_post)
    theory_mean = float(dist.mean())
    theory_std = float(dist.std())
    theory_q025, theory_q50, theory_q975 = [float(v) for v in dist.ppf([0.025, 0.5, 0.975])]

    sample_mean = float(np.mean(samples))
    sample_std = float(np.std(samples, ddof=1))
    sample_q025, sample_q50, sample_q975 = [float(v) for v in np.quantile(samples, [0.025, 0.5, 0.975])]

    # Use torch for an independent numeric cross-check on sample statistics.
    ts = torch.from_numpy(samples)
    torch_mean = float(ts.mean().item())
    torch_std = float(ts.std(unbiased=True).item())

    # Use sklearn metric for a distribution-level goodness proxy.
    grid = np.linspace(theory_q025, theory_q975, 120)
    cdf_emp = empirical_cdf(sorted_samples, grid)
    cdf_true = dist.cdf(grid)
    cdf_mse = float(mean_squared_error(cdf_true, cdf_emp))

    ess = estimate_ess(samples, max_lag=500)

    table = pd.DataFrame(
        {
            "metric": ["mean", "std", "q2.5", "q50", "q97.5"],
            "sample": [sample_mean, sample_std, sample_q025, sample_q50, sample_q975],
            "theory": [theory_mean, theory_std, theory_q025, theory_q50, theory_q975],
        }
    )
    table["abs_error"] = (table["sample"] - table["theory"]).abs()

    mean_close = abs(sample_mean - theory_mean) < 0.08
    cdf_close = cdf_mse < 2e-3
    accept_ok = 0.15 <= result.acceptance_rate <= 0.70
    pass_check = bool(mean_close and cdf_close and accept_ok)

    print("=== Metropolis-Hastings MVP: Poisson-Gamma Posterior Sampling ===")
    print(
        f"data: n_obs={n_obs}, observed_sum={int(np.sum(counts))}, "
        f"observed_mean={float(np.mean(counts)):.4f}"
    )
    print(
        "posterior (analytic): "
        f"Gamma(alpha={alpha_post:.3f}, rate={beta_post:.3f})"
    )
    print(
        f"sampler: steps={config.n_steps}, burn_in={config.burn_in}, thin={config.thin}, "
        f"proposal_sigma={config.proposal_sigma:.3f}"
    )
    print(f"acceptance_rate={result.acceptance_rate:.4f}, kept_samples={samples.size}, ESS≈{ess:.1f}")

    print("\nPosterior summary (sample vs analytic):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nExtra checks:")
    print(f"cdf_mse={cdf_mse:.6e}")
    print(f"torch_mean_diff={abs(torch_mean - sample_mean):.6e}")
    print(f"torch_std_diff={abs(torch_std - sample_std):.6e}")
    print(f"pass_loose_check={pass_check}")


if __name__ == "__main__":
    main()
