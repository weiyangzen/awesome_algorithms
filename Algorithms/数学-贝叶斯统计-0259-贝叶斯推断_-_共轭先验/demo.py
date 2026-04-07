"""Bayesian inference MVP with conjugate prior (MATH-0259).

This script demonstrates Beta-Binomial conjugacy for Bernoulli data:
1) Closed-form posterior update.
2) Sequential (online) posterior update.
3) Posterior summary + posterior predictive probabilities.
4) Grid-based posterior approximation check against closed-form Beta posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln
from scipy.stats import beta as beta_dist


@dataclass
class PosteriorState:
    """Posterior parameters and sufficient statistics for Beta-Binomial."""

    alpha: float
    beta: float
    n: int
    k: int


def validate_prior(alpha: float, beta: float) -> None:
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        raise ValueError("alpha and beta must be finite.")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive.")


def validate_binary_data(x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not np.all(np.isin(x, [0, 1])):
        raise ValueError("x must contain only 0/1 values.")


def generate_bernoulli_data(
    p_true: float = 0.68,
    n: int = 120,
    seed: int = 259,
) -> np.ndarray:
    """Generate Bernoulli observations with a fixed random seed."""
    if not np.isfinite(p_true) or p_true <= 0.0 or p_true >= 1.0:
        raise ValueError("p_true must be in (0, 1).")
    if n <= 0:
        raise ValueError("n must be positive.")

    rng = np.random.default_rng(seed)
    x = rng.binomial(n=1, p=p_true, size=n).astype(np.int64)
    return x


def batch_beta_posterior_update(
    x: np.ndarray,
    alpha_prior: float,
    beta_prior: float,
) -> PosteriorState:
    """Closed-form batch update for Beta prior + Bernoulli observations."""
    validate_prior(alpha_prior, beta_prior)
    validate_binary_data(x)

    n = int(x.size)
    k = int(np.sum(x))
    alpha_post = float(alpha_prior + k)
    beta_post = float(beta_prior + (n - k))
    return PosteriorState(alpha=alpha_post, beta=beta_post, n=n, k=k)


def sequential_beta_posterior_update(
    x: np.ndarray,
    alpha_prior: float,
    beta_prior: float,
) -> Tuple[PosteriorState, pd.DataFrame]:
    """Sequential (online) Beta posterior updates, one sample at a time."""
    validate_prior(alpha_prior, beta_prior)
    validate_binary_data(x)

    alpha = float(alpha_prior)
    beta = float(beta_prior)
    k_running = 0
    records = []

    for step, obs in enumerate(x, start=1):
        if int(obs) == 1:
            alpha += 1.0
            k_running += 1
        else:
            beta += 1.0

        records.append(
            {
                "step": step,
                "x_t": int(obs),
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": alpha / (alpha + beta),
            }
        )

    state = PosteriorState(alpha=alpha, beta=beta, n=int(x.size), k=int(k_running))
    trace = pd.DataFrame.from_records(records)
    return state, trace


def beta_posterior_summary(
    alpha_post: float,
    beta_post: float,
    credible_mass: float = 0.95,
) -> Dict[str, float]:
    """Compute posterior moments and credible interval for Beta(alpha_post, beta_post)."""
    validate_prior(alpha_post, beta_post)
    if credible_mass <= 0.0 or credible_mass >= 1.0:
        raise ValueError("credible_mass must be in (0,1).")

    lower_q = (1.0 - credible_mass) / 2.0
    upper_q = 1.0 - lower_q
    ci_low, ci_high = beta_dist.ppf([lower_q, upper_q], alpha_post, beta_post)

    mean = alpha_post / (alpha_post + beta_post)
    var = (alpha_post * beta_post) / (
        (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1.0)
    )
    return {
        "mean": float(mean),
        "variance": float(var),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def beta_binomial_predictive_pmf(
    m_future: int,
    alpha_post: float,
    beta_post: float,
) -> np.ndarray:
    """Posterior predictive PMF for future success count using Beta-Binomial."""
    validate_prior(alpha_post, beta_post)
    if m_future <= 0:
        raise ValueError("m_future must be positive.")

    ks = np.arange(m_future + 1, dtype=np.float64)
    log_comb = gammaln(m_future + 1.0) - gammaln(ks + 1.0) - gammaln(m_future - ks + 1.0)
    log_beta_ratio = betaln(ks + alpha_post, m_future - ks + beta_post) - betaln(
        alpha_post,
        beta_post,
    )
    pmf = np.exp(log_comb + log_beta_ratio)
    pmf /= np.sum(pmf)
    return pmf


def grid_vs_closed_form_l1_error(
    x: np.ndarray,
    alpha_prior: float,
    beta_prior: float,
    grid_size: int = 4000,
) -> float:
    """Numerically approximate posterior on a grid and compare with Beta closed-form."""
    validate_prior(alpha_prior, beta_prior)
    validate_binary_data(x)
    if grid_size < 200:
        raise ValueError("grid_size is too small for a stable check.")

    n = int(x.size)
    k = int(np.sum(x))
    eps = 1e-6
    p_grid = np.linspace(eps, 1.0 - eps, grid_size)

    log_prior = (alpha_prior - 1.0) * np.log(p_grid) + (beta_prior - 1.0) * np.log(1.0 - p_grid)
    log_lik = k * np.log(p_grid) + (n - k) * np.log(1.0 - p_grid)
    log_unnorm = log_prior + log_lik

    log_unnorm -= np.max(log_unnorm)
    posterior_grid = np.exp(log_unnorm)
    posterior_grid /= np.trapezoid(posterior_grid, p_grid)

    alpha_post = alpha_prior + k
    beta_post = beta_prior + (n - k)
    posterior_closed = beta_dist.pdf(p_grid, alpha_post, beta_post)
    posterior_closed /= np.trapezoid(posterior_closed, p_grid)

    l1_err = np.trapezoid(np.abs(posterior_grid - posterior_closed), p_grid)
    return float(l1_err)


def main() -> None:
    print("Bayesian Inference MVP: Conjugate Prior (MATH-0259)")
    print("=" * 78)

    alpha_prior = 2.0
    beta_prior = 2.0
    p_true = 0.68
    n_obs = 120
    future_horizon = 8

    x = generate_bernoulli_data(p_true=p_true, n=n_obs, seed=259)
    batch_state = batch_beta_posterior_update(x, alpha_prior, beta_prior)
    seq_state, trace = sequential_beta_posterior_update(x, alpha_prior, beta_prior)
    summary = beta_posterior_summary(batch_state.alpha, batch_state.beta, credible_mass=0.95)

    predictive_pmf = beta_binomial_predictive_pmf(
        m_future=future_horizon,
        alpha_post=batch_state.alpha,
        beta_post=batch_state.beta,
    )
    l1_error = grid_vs_closed_form_l1_error(
        x=x,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        grid_size=4000,
    )

    print(f"true p                             : {p_true:.6f}")
    print(f"observations n / successes k       : {batch_state.n} / {batch_state.k}")
    print(
        "posterior (batch) alpha, beta      : "
        f"{batch_state.alpha:.3f}, {batch_state.beta:.3f}"
    )
    print(
        "posterior (sequential) alpha, beta : "
        f"{seq_state.alpha:.3f}, {seq_state.beta:.3f}"
    )
    print(f"posterior mean                     : {summary['mean']:.6f}")
    print(f"posterior variance                 : {summary['variance']:.6f}")
    print(
        "95% credible interval              : "
        f"[{summary['ci_low']:.6f}, {summary['ci_high']:.6f}]"
    )
    print(f"P(next success | data)             : {summary['mean']:.6f}")
    print(f"grid-vs-closed posterior L1 error  : {l1_error:.6e}")

    pred_df = pd.DataFrame(
        {
            "k_future": np.arange(future_horizon + 1, dtype=int),
            "predictive_prob": predictive_pmf,
        }
    )
    print()
    print(f"Posterior predictive PMF for m={future_horizon} future trials:")
    print(pred_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print()
    print("Last 8 sequential updates:")
    print(trace.tail(8).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    batch_seq_match = (
        abs(batch_state.alpha - seq_state.alpha) < 1e-12
        and abs(batch_state.beta - seq_state.beta) < 1e-12
        and batch_state.k == seq_state.k
        and batch_state.n == seq_state.n
    )
    predictive_sum_ok = abs(float(np.sum(predictive_pmf)) - 1.0) < 1e-10
    next_prob_match = abs(summary["mean"] - batch_state.alpha / (batch_state.alpha + batch_state.beta)) < 1e-12

    if not batch_seq_match:
        raise RuntimeError("batch update and sequential update are inconsistent.")
    if not predictive_sum_ok:
        raise RuntimeError("predictive PMF does not sum to 1.")
    if not next_prob_match:
        raise RuntimeError("posterior mean and one-step predictive probability mismatch.")
    if not (l1_error < 3e-3):
        raise RuntimeError("grid posterior deviates too much from closed-form posterior.")

    print("=" * 78)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
