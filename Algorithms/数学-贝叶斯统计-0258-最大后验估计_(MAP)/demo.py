"""Minimal runnable MVP for Maximum A Posteriori estimation (MATH-0258).

This demo uses a Beta-Bernoulli model and provides:
1) Closed-form MLE / posterior mean / MAP.
2) Gradient-ascent MAP in logit space from scratch.
3) Finite-difference gradient check and automatic assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FitResult:
    """Container for gradient-ascent MAP results."""

    theta_map: float
    phi_hat: float
    log_posterior_history: List[float]
    n_iter: int
    converged: bool


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid for scalar input."""
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def generate_bernoulli_samples(
    true_theta: float = 0.62,
    n: int = 40,
    seed: int = 258,
) -> np.ndarray:
    """Generate i.i.d. Bernoulli samples as float array of 0/1."""
    if not (0.0 < true_theta < 1.0):
        raise ValueError("true_theta must be in (0, 1)")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    return rng.binomial(1, true_theta, size=n).astype(np.float64)


def summarize_posterior(
    x: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[int, int, float, float, float]:
    """Return (k, n, mle, posterior_mean, map_closed_form)."""
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if x.size == 0:
        raise ValueError("x cannot be empty")
    if not np.all((x == 0.0) | (x == 1.0)):
        raise ValueError("x must contain only 0/1 values")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")

    n = int(x.size)
    k = int(np.sum(x))

    mle = k / n

    alpha_post = alpha + k
    beta_post = beta + (n - k)

    posterior_mean = alpha_post / (alpha_post + beta_post)

    if alpha_post <= 1.0 and beta_post <= 1.0:
        raise ValueError(
            "Posterior mode is not unique (both boundaries). "
            "Use stronger prior or different data."
        )
    if alpha_post <= 1.0:
        map_closed = 0.0
    elif beta_post <= 1.0:
        map_closed = 1.0
    else:
        map_closed = (alpha_post - 1.0) / (alpha_post + beta_post - 2.0)

    return k, n, float(mle), float(posterior_mean), float(map_closed)


def log_posterior_and_grad_phi(
    phi: float,
    k: int,
    n: int,
    alpha: float,
    beta: float,
) -> Tuple[float, float, float]:
    """Compute log posterior (up to constant), grad wrt phi, and theta.

    Define:
        theta = sigmoid(phi)
        a = k + alpha - 1
        b = (n-k) + beta - 1
    Then:
        log p(theta|D) = a*log(theta) + b*log(1-theta) + const
        d/dphi log p(theta|D) = a - (a+b)*theta
    """
    if n <= 0:
        raise ValueError("n must be positive")

    a = float(k) + alpha - 1.0
    b = float(n - k) + beta - 1.0

    theta = sigmoid(phi)
    theta_safe = float(np.clip(theta, 1e-12, 1.0 - 1e-12))

    log_post = a * np.log(theta_safe) + b * np.log(1.0 - theta_safe)
    grad_phi = a - (a + b) * theta

    return float(log_post), float(grad_phi), float(theta)


def finite_difference_grad_check(
    phi: float,
    k: int,
    n: int,
    alpha: float,
    beta: float,
    delta: float = 1e-6,
) -> float:
    """Return absolute gradient error between analytic and numerical derivative."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    _, g_analytic, _ = log_posterior_and_grad_phi(phi, k, n, alpha, beta)

    lp_pos, _, _ = log_posterior_and_grad_phi(phi + delta, k, n, alpha, beta)
    lp_neg, _, _ = log_posterior_and_grad_phi(phi - delta, k, n, alpha, beta)
    g_numeric = (lp_pos - lp_neg) / (2.0 * delta)

    return abs(g_numeric - g_analytic)


def gradient_ascent_map(
    k: int,
    n: int,
    alpha: float,
    beta: float,
    phi_init: float,
    lr: float = 0.02,
    max_iter: int = 6000,
    tol: float = 1e-12,
) -> FitResult:
    """Maximize log posterior by gradient ascent in phi=logit(theta) space."""
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    phi = float(phi_init)
    history: List[float] = []

    converged = False
    n_iter = 0

    for step in range(1, max_iter + 1):
        log_post, grad_phi, _ = log_posterior_and_grad_phi(phi, k, n, alpha, beta)
        history.append(log_post)

        phi_new = phi + lr * grad_phi

        n_iter = step
        if abs(phi_new - phi) < tol:
            phi = phi_new
            converged = True
            break

        phi = phi_new

    theta_map = sigmoid(phi)
    return FitResult(
        theta_map=float(theta_map),
        phi_hat=float(phi),
        log_posterior_history=history,
        n_iter=n_iter,
        converged=converged,
    )


def main() -> None:
    print("Maximum A Posteriori MVP (MATH-0258)")
    print("=" * 72)

    true_theta = 0.62
    alpha = 8.0
    beta = 4.0

    x_small = generate_bernoulli_samples(true_theta=true_theta, n=40, seed=258)
    k_small, n_small, mle_small, post_mean_small, map_closed_small = summarize_posterior(
        x_small,
        alpha,
        beta,
    )

    phi0 = float(np.log(0.15 / 0.85))
    grad_err = finite_difference_grad_check(phi0, k_small, n_small, alpha, beta)

    fit = gradient_ascent_map(
        k=k_small,
        n=n_small,
        alpha=alpha,
        beta=beta,
        phi_init=phi0,
        lr=0.02,
        max_iter=6000,
        tol=1e-12,
    )

    lp_start = fit.log_posterior_history[0]
    lp_end = fit.log_posterior_history[-1]
    map_gap = abs(fit.theta_map - map_closed_small)

    x_large = generate_bernoulli_samples(true_theta=true_theta, n=4000, seed=1258)
    _, _, mle_large, _, map_closed_large = summarize_posterior(x_large, alpha, beta)

    shrink_small = abs(map_closed_small - mle_small)
    shrink_large = abs(map_closed_large - mle_large)

    print(f"prior Beta(alpha={alpha:.1f}, beta={beta:.1f}), true theta={true_theta:.4f}")
    print(f"small sample: n={n_small}, k={k_small}")
    print(
        "  small-sample estimates: "
        f"MLE={mle_small:.6f}, PosteriorMean={post_mean_small:.6f}, "
        f"MAP(closed)={map_closed_small:.6f}, MAP(grad)={fit.theta_map:.6f}"
    )
    print(f"gradient ascent iters={fit.n_iter}, converged={fit.converged}")
    print(f"log-posterior: start={lp_start:.6f}, end={lp_end:.6f}")
    print(f"gradient-check abs error={grad_err:.3e}")
    print(f"|MAP_grad - MAP_closed|={map_gap:.3e}")
    print(
        "prior influence |MAP-MLE|: "
        f"small_n={shrink_small:.6f}, large_n={shrink_large:.6f}"
    )

    if not (grad_err < 1e-7):
        raise RuntimeError("gradient check failed")
    if not (lp_end > lp_start):
        raise RuntimeError("log posterior did not improve")
    if not (map_gap < 1e-9):
        raise RuntimeError("gradient MAP does not match closed-form MAP")
    if not (shrink_large < shrink_small):
        raise RuntimeError("prior influence did not decrease for large sample")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
