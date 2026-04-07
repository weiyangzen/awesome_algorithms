"""Minimal runnable MVP for Markov Chain Monte Carlo (MCMC).

This demo samples the Boltzmann distribution of a 1D anharmonic oscillator
using Metropolis-Hastings and compares moment estimates against a high-resolution
numerical integration reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MCMCConfig:
    """Configuration for a single Metropolis-Hastings run."""

    beta: float = 1.6
    n_steps: int = 80_000
    burn_in: int = 8_000
    proposal_sigma: float = 1.2
    x0: float = 0.0
    seed: int = 2026


def potential_energy(x: np.ndarray | float) -> np.ndarray | float:
    """Double-well potential U(x) = x^4/4 - x^2/2."""

    return 0.25 * np.asarray(x) ** 4 - 0.5 * np.asarray(x) ** 2


def log_target_density(x: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Unnormalized log target density: log pi(x) = -beta * U(x)."""

    return -beta * potential_energy(x)


def metropolis_hastings_1d(config: MCMCConfig) -> tuple[np.ndarray, float]:
    """Generate samples from the target density with a symmetric proposal."""

    if config.burn_in >= config.n_steps:
        raise ValueError("burn_in must be smaller than n_steps")

    rng = np.random.default_rng(config.seed)
    chain = np.empty(config.n_steps, dtype=float)

    x = float(config.x0)
    log_px = float(log_target_density(x, config.beta))
    accepted = 0

    for i in range(config.n_steps):
        proposal = x + rng.normal(loc=0.0, scale=config.proposal_sigma)
        log_pp = float(log_target_density(proposal, config.beta))

        # Symmetric random-walk proposal -> q(x'|x)=q(x|x'), so acceptance
        # only depends on target-density ratio.
        log_alpha = min(0.0, log_pp - log_px)
        if np.log(rng.random()) < log_alpha:
            x = proposal
            log_px = log_pp
            accepted += 1

        chain[i] = x

    samples = chain[config.burn_in :]
    acceptance_rate = accepted / config.n_steps
    return samples, acceptance_rate


def integrated_autocorrelation_time(samples: np.ndarray, max_lag: int = 2_000) -> float:
    """Estimate integrated autocorrelation time (IACT) by positive-sequence truncation."""

    centered = samples - np.mean(samples)
    var = float(np.var(centered))
    if var <= 0.0:
        return 1.0

    n = centered.size
    max_lag = min(max_lag, n - 1)
    tau = 1.0

    for lag in range(1, max_lag + 1):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
        rho = acov / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho

    return max(tau, 1.0)


def effective_sample_size(samples: np.ndarray, max_lag: int = 2_000) -> float:
    """Estimate effective sample size from IACT."""

    tau = integrated_autocorrelation_time(samples=samples, max_lag=max_lag)
    return samples.size / tau


def estimate_moments(samples: np.ndarray) -> Dict[str, float]:
    """Monte Carlo estimates for selected moments."""

    return {
        "mean_x": float(np.mean(samples)),
        "mean_x2": float(np.mean(samples**2)),
        "mean_x4": float(np.mean(samples**4)),
    }


def reference_moments(beta: float, x_max: float = 4.5, n_grid: int = 200_001) -> Dict[str, float]:
    """Compute near-exact moments via high-resolution numerical quadrature on a grid."""

    grid = np.linspace(-x_max, x_max, n_grid, dtype=float)
    logw = log_target_density(grid, beta)
    logw -= np.max(logw)
    w = np.exp(logw)

    z = np.trapezoid(w, grid)
    mean_x = np.trapezoid(grid * w, grid) / z
    mean_x2 = np.trapezoid((grid**2) * w, grid) / z
    mean_x4 = np.trapezoid((grid**4) * w, grid) / z

    return {
        "mean_x": float(mean_x),
        "mean_x2": float(mean_x2),
        "mean_x4": float(mean_x4),
    }


def build_report(config: MCMCConfig) -> tuple[pd.DataFrame, dict[str, float], bool]:
    """Run simulation, compare with reference, and return a result table."""

    samples, acceptance_rate = metropolis_hastings_1d(config)
    mc = estimate_moments(samples)
    ref = reference_moments(beta=config.beta)

    iact = integrated_autocorrelation_time(samples)
    ess = effective_sample_size(samples)

    rows = []
    for key in ["mean_x", "mean_x2", "mean_x4"]:
        abs_err = abs(mc[key] - ref[key])
        rel_err = abs_err / abs(ref[key]) if abs(ref[key]) > 1e-10 else np.nan
        rows.append(
            {
                "metric": key,
                "mcmc_estimate": mc[key],
                "reference": ref[key],
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )

    table = pd.DataFrame(rows)

    checks = {
        "acceptance_rate": acceptance_rate,
        "iact": iact,
        "ess": ess,
        "pass_acceptance": 0.20 <= acceptance_rate <= 0.80,
        "pass_mean_symmetry": abs(mc["mean_x"]) < 0.08,
        "pass_x2_accuracy": float(table.loc[table["metric"] == "mean_x2", "rel_error"].iloc[0]) < 0.05,
        "pass_x4_accuracy": float(table.loc[table["metric"] == "mean_x4", "rel_error"].iloc[0]) < 0.08,
        "pass_ess": ess > 1_000.0,
    }

    passed = bool(
        checks["pass_acceptance"]
        and checks["pass_mean_symmetry"]
        and checks["pass_x2_accuracy"]
        and checks["pass_x4_accuracy"]
        and checks["pass_ess"]
    )
    return table, checks, passed


def main() -> None:
    config = MCMCConfig()
    table, checks, passed = build_report(config)

    pd.set_option("display.float_format", lambda v: f"{v: .6f}")

    print("=== Markov Chain Monte Carlo (Metropolis-Hastings) MVP ===")
    print(f"beta={config.beta}, n_steps={config.n_steps}, burn_in={config.burn_in}")
    print(f"proposal_sigma={config.proposal_sigma}, seed={config.seed}")
    print()

    print("Moment estimation summary:")
    print(table.to_string(index=False))
    print()

    print("Diagnostics:")
    print(f"acceptance_rate={checks['acceptance_rate']:.4f}")
    print(f"IACT={checks['iact']:.2f}")
    print(f"ESS={checks['ess']:.1f}")
    print()

    print("Validation checks:")
    print(f"- acceptance in [0.20, 0.80]: {checks['pass_acceptance']}")
    print(f"- symmetry |E[x]| < 0.08: {checks['pass_mean_symmetry']}")
    print(f"- relative error E[x^2] < 5%: {checks['pass_x2_accuracy']}")
    print(f"- relative error E[x^4] < 8%: {checks['pass_x4_accuracy']}")
    print(f"- ESS > 1000: {checks['pass_ess']}")
    print()

    print(f"Validation: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
