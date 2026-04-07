"""Minimal runnable MVP for Maximum Likelihood Estimation (MATH-0257).

This demo estimates parameters of a 1D Gaussian distribution with MLE via:
1) Closed-form solution.
2) Gradient ascent on average log-likelihood using a log-sigma reparameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FitResult:
    """Container for gradient-ascent MLE results."""

    mu_hat: float
    sigma_hat: float
    log_likelihood_history: List[float]
    n_iter: int
    converged: bool


def generate_normal_samples(
    true_mu: float = 2.5,
    true_sigma: float = 1.7,
    n: int = 800,
    seed: int = 257,
) -> np.ndarray:
    """Generate i.i.d. Gaussian samples."""
    if true_sigma <= 0:
        raise ValueError("true_sigma must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_mu, scale=true_sigma, size=n).astype(np.float64)


def closed_form_gaussian_mle(x: np.ndarray) -> Tuple[float, float]:
    """Compute closed-form MLE for 1D Gaussian parameters.

    Returns:
        mu_hat: sample mean
        sigma_hat: sqrt((1/n) * sum((x-mu_hat)^2))
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if x.size == 0:
        raise ValueError("x cannot be empty")

    mu_hat = float(np.mean(x))
    sigma_sq_hat = float(np.mean((x - mu_hat) ** 2))
    sigma_hat = float(np.sqrt(max(sigma_sq_hat, 1e-16)))
    return mu_hat, sigma_hat


def avg_log_likelihood_and_grad(
    mu: float,
    log_sigma: float,
    x: np.ndarray,
) -> Tuple[float, float, float]:
    """Average log-likelihood and gradients for Gaussian MLE.

    Parameterization:
        sigma = exp(log_sigma) > 0

    Average log-likelihood:
        l = -log_sigma - 0.5 * exp(-2*log_sigma) * mean((x-mu)^2) - 0.5*log(2*pi)

    Gradients:
        dl/dmu = exp(-2*log_sigma) * (mean(x) - mu)
        dl/dlog_sigma = -1 + exp(-2*log_sigma) * mean((x-mu)^2)
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    centered = x - mu
    mean_sq = float(np.mean(centered * centered))
    inv_sigma_sq = float(np.exp(-2.0 * log_sigma))

    avg_ll = -log_sigma - 0.5 * inv_sigma_sq * mean_sq - 0.5 * np.log(2.0 * np.pi)
    grad_mu = inv_sigma_sq * (float(np.mean(x)) - mu)
    grad_log_sigma = -1.0 + inv_sigma_sq * mean_sq
    return float(avg_ll), float(grad_mu), float(grad_log_sigma)


def gradient_ascent_mle(
    x: np.ndarray,
    mu_init: float,
    log_sigma_init: float,
    lr: float = 0.08,
    max_iter: int = 6000,
    tol: float = 1e-10,
) -> FitResult:
    """Fit Gaussian MLE by gradient ascent on average log-likelihood."""
    if lr <= 0:
        raise ValueError("lr must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    mu = float(mu_init)
    log_sigma = float(log_sigma_init)
    history: List[float] = []

    converged = False
    n_iter = 0

    for step in range(1, max_iter + 1):
        avg_ll, g_mu, g_log_sigma = avg_log_likelihood_and_grad(mu, log_sigma, x)
        history.append(avg_ll)

        mu_new = mu + lr * g_mu
        log_sigma_new = log_sigma + lr * g_log_sigma

        update_norm = max(abs(mu_new - mu), abs(log_sigma_new - log_sigma))
        mu, log_sigma = mu_new, log_sigma_new
        n_iter = step

        if update_norm < tol:
            converged = True
            break

    sigma_hat = float(np.exp(log_sigma))
    return FitResult(
        mu_hat=mu,
        sigma_hat=sigma_hat,
        log_likelihood_history=history,
        n_iter=n_iter,
        converged=converged,
    )


def finite_difference_grad_check(
    mu: float,
    log_sigma: float,
    x: np.ndarray,
    delta: float = 1e-6,
) -> Tuple[float, float]:
    """Return absolute errors for (d/dmu, d/dlog_sigma) gradient checks."""
    if delta <= 0:
        raise ValueError("delta must be positive")

    _, g_mu, g_log_sigma = avg_log_likelihood_and_grad(mu, log_sigma, x)

    ll_mu_pos, _, _ = avg_log_likelihood_and_grad(mu + delta, log_sigma, x)
    ll_mu_neg, _, _ = avg_log_likelihood_and_grad(mu - delta, log_sigma, x)
    num_mu = (ll_mu_pos - ll_mu_neg) / (2.0 * delta)

    ll_ls_pos, _, _ = avg_log_likelihood_and_grad(mu, log_sigma + delta, x)
    ll_ls_neg, _, _ = avg_log_likelihood_and_grad(mu, log_sigma - delta, x)
    num_log_sigma = (ll_ls_pos - ll_ls_neg) / (2.0 * delta)

    return abs(num_mu - g_mu), abs(num_log_sigma - g_log_sigma)


def main() -> None:
    print("Maximum Likelihood Estimation MVP (MATH-0257)")
    print("=" * 72)

    true_mu = 2.5
    true_sigma = 1.7
    x = generate_normal_samples(true_mu=true_mu, true_sigma=true_sigma, n=800, seed=257)

    mu_closed, sigma_closed = closed_form_gaussian_mle(x)

    mu0 = mu_closed + 1.2
    log_sigma0 = np.log(sigma_closed * 1.8)

    grad_err_mu, grad_err_log_sigma = finite_difference_grad_check(
        mu=mu0,
        log_sigma=float(log_sigma0),
        x=x,
        delta=1e-6,
    )

    fit = gradient_ascent_mle(
        x=x,
        mu_init=mu0,
        log_sigma_init=float(log_sigma0),
        lr=0.08,
        max_iter=6000,
        tol=1e-10,
    )

    ll_start = fit.log_likelihood_history[0]
    ll_end = fit.log_likelihood_history[-1]

    print(f"true mu={true_mu:.6f}, true sigma={true_sigma:.6f}")
    print(f"closed-form MLE  : mu={mu_closed:.6f}, sigma={sigma_closed:.6f}")
    print(f"gradient-ascent  : mu={fit.mu_hat:.6f}, sigma={fit.sigma_hat:.6f}")
    print(f"iters={fit.n_iter}, converged={fit.converged}")
    print(f"avg log-likelihood: start={ll_start:.6f}, end={ll_end:.6f}")
    print(
        "gradient-check abs error: "
        f"dmu={grad_err_mu:.3e}, dlog_sigma={grad_err_log_sigma:.3e}"
    )

    mu_gap = abs(fit.mu_hat - mu_closed)
    sigma_gap = abs(fit.sigma_hat - sigma_closed)

    print(f"|mu_grad - mu_closed|={mu_gap:.3e}")
    print(f"|sigma_grad - sigma_closed|={sigma_gap:.3e}")

    if not (grad_err_mu < 1e-6 and grad_err_log_sigma < 1e-6):
        raise RuntimeError("gradient check failed")
    if not (ll_end > ll_start):
        raise RuntimeError("log-likelihood did not improve")
    if not (mu_gap < 5e-5 and sigma_gap < 5e-5):
        raise RuntimeError("gradient-ascent MLE does not match closed-form MLE")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
