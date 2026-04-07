"""Metropolis Algorithm MVP (1D random-walk sampler).

This script samples from a standard normal target distribution using only
its unnormalized log-density, demonstrating the core Metropolis acceptance
mechanism used in computational physics and MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for a reproducible Metropolis run."""

    seed: int = 20260407
    initial_x: float = 8.0
    proposal_std: float = 1.0
    burn_in: int = 5_000
    n_samples: int = 20_000
    thin: int = 2


def log_target_standard_normal(x: float) -> float:
    """Unnormalized log-density of N(0, 1): -x^2 / 2 + constant."""
    return -0.5 * x * x


def metropolis_random_walk_1d(
    log_target,
    initial_x: float,
    proposal_std: float,
    burn_in: int,
    n_samples: int,
    thin: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate samples with a symmetric random-walk Metropolis kernel."""
    if proposal_std <= 0:
        raise ValueError("proposal_std must be > 0")
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if thin <= 0:
        raise ValueError("thin must be > 0")

    total_steps = burn_in + n_samples * thin
    samples = np.empty(n_samples, dtype=float)

    x = float(initial_x)
    log_p_x = float(log_target(x))
    accepted = 0
    write_idx = 0

    for step in range(total_steps):
        x_proposal = x + rng.normal(loc=0.0, scale=proposal_std)
        log_p_proposal = float(log_target(x_proposal))
        log_alpha = log_p_proposal - log_p_x

        if np.log(rng.random()) < log_alpha:
            x = x_proposal
            log_p_x = log_p_proposal
            accepted += 1

        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples[write_idx] = x
            write_idx += 1

    acceptance_rate = accepted / total_steps
    return samples, acceptance_rate


def lag1_autocorrelation(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation as a quick chain-quality diagnostic."""
    if x.size < 2:
        return float("nan")
    centered = x - np.mean(x)
    denom = np.dot(centered[:-1], centered[:-1])
    if denom <= 0:
        return float("nan")
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def ar1_effective_sample_size(x: np.ndarray) -> float:
    """AR(1)-style ESS approximation from lag-1 autocorrelation."""
    rho1 = lag1_autocorrelation(x)
    if not np.isfinite(rho1):
        return float("nan")
    rho1 = float(np.clip(rho1, -0.999, 0.999))
    n = float(x.size)
    return n * (1.0 - rho1) / (1.0 + rho1)


def main() -> None:
    cfg = SamplerConfig()
    rng = np.random.default_rng(cfg.seed)

    samples, acceptance_rate = metropolis_random_walk_1d(
        log_target=log_target_standard_normal,
        initial_x=cfg.initial_x,
        proposal_std=cfg.proposal_std,
        burn_in=cfg.burn_in,
        n_samples=cfg.n_samples,
        thin=cfg.thin,
        rng=rng,
    )

    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1))
    q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95])
    rho1 = lag1_autocorrelation(samples)
    ess = ar1_effective_sample_size(samples)

    print("Metropolis Algorithm MVP (1D Standard Normal)")
    print(f"seed                : {cfg.seed}")
    print(
        f"config              : initial_x={cfg.initial_x}, "
        f"proposal_std={cfg.proposal_std}, burn_in={cfg.burn_in}, "
        f"n_samples={cfg.n_samples}, thin={cfg.thin}"
    )
    print(f"acceptance_rate     : {acceptance_rate:.4f}")
    print(f"sample_mean         : {mean:.6f} (target: 0.0)")
    print(f"sample_variance     : {var:.6f} (target: 1.0)")
    print(f"quantiles(5/50/95)  : {q05:.6f}, {q50:.6f}, {q95:.6f}")
    print(f"lag1_autocorrelation: {rho1:.6f}")
    print(f"ESS_AR1_approx      : {ess:.1f} / {samples.size}")


if __name__ == "__main__":
    main()
