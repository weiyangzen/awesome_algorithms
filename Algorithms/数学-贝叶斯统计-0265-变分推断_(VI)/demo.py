"""Variational Inference (VI) MVP: mean-field CAVI for a Gaussian-Gamma model.

Model:
    x_n | mu, tau ~ Normal(mu, tau^{-1})
    mu | tau      ~ Normal(mu0, (lambda0 * tau)^{-1})
    tau           ~ Gamma(a0, b0)   (shape-rate)

Variational family:
    q(mu) q(tau)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln


@dataclass
class VIResult:
    m_n: float
    lambda_n: float
    a_n: float
    b_n: float
    e_tau: float
    var_mu: float
    iterations: int
    converged: bool
    trace: pd.DataFrame


def validate_inputs(
    x: np.ndarray,
    mu0: float,
    lambda0: float,
    a0: float,
    b0: float,
    tol: float,
    max_iter: int,
) -> None:
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if x.size == 0:
        raise ValueError("x must contain at least one observation.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    if not np.isfinite([mu0, lambda0, a0, b0, tol]).all():
        raise ValueError("Hyperparameters must be finite numbers.")
    if lambda0 <= 0:
        raise ValueError("lambda0 must be > 0.")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("a0 and b0 must be > 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")


def compute_elbo(
    x: np.ndarray,
    mu0: float,
    lambda0: float,
    a0: float,
    b0: float,
    m_n: float,
    lambda_n: float,
    a_n: float,
    b_n: float,
) -> float:
    n = x.size
    e_tau = a_n / b_n
    e_log_tau = digamma(a_n) - np.log(b_n)
    var_mu = 1.0 / (lambda_n * e_tau)

    sq_data = np.sum((x - m_n) ** 2) + n * var_mu
    sq_prior = (m_n - mu0) ** 2 + var_mu

    e_log_p_x = 0.5 * n * (e_log_tau - np.log(2.0 * np.pi)) - 0.5 * e_tau * sq_data
    e_log_p_mu = 0.5 * (np.log(lambda0) + e_log_tau - np.log(2.0 * np.pi)) - 0.5 * lambda0 * e_tau * sq_prior
    e_log_p_tau = a0 * np.log(b0) - gammaln(a0) + (a0 - 1.0) * e_log_tau - b0 * e_tau

    e_log_q_mu = -0.5 * (np.log(2.0 * np.pi * var_mu) + 1.0)
    e_log_q_tau = a_n * np.log(b_n) - gammaln(a_n) + (a_n - 1.0) * e_log_tau - b_n * e_tau

    return float(e_log_p_x + e_log_p_mu + e_log_p_tau - e_log_q_mu - e_log_q_tau)


def cavi_normal_gamma(
    x: np.ndarray,
    mu0: float,
    lambda0: float,
    a0: float,
    b0: float,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> VIResult:
    validate_inputs(x, mu0, lambda0, a0, b0, tol, max_iter)

    n = x.size
    x_bar = float(np.mean(x))

    lambda_n = lambda0 + n
    m_n = (lambda0 * mu0 + n * x_bar) / lambda_n
    a_n = a0 + 0.5 * (n + 1.0)

    e_tau = a0 / b0
    records: list[dict[str, float]] = []
    converged = False

    for it in range(1, max_iter + 1):
        var_mu = 1.0 / (lambda_n * e_tau)

        data_term = np.sum((x - m_n) ** 2) + n * var_mu
        prior_term = lambda0 * ((m_n - mu0) ** 2 + var_mu)
        b_n = b0 + 0.5 * (data_term + prior_term)

        e_tau_new = a_n / b_n
        var_mu_new = 1.0 / (lambda_n * e_tau_new)
        elbo = compute_elbo(x, mu0, lambda0, a0, b0, m_n, lambda_n, a_n, b_n)

        records.append(
            {
                "iter": float(it),
                "e_tau": float(e_tau_new),
                "var_mu": float(var_mu_new),
                "b_n": float(b_n),
                "elbo": float(elbo),
            }
        )

        if abs(e_tau_new - e_tau) < tol:
            converged = True
            e_tau = e_tau_new
            break

        e_tau = e_tau_new

    trace = pd.DataFrame.from_records(records)
    final_b_n = float(trace.iloc[-1]["b_n"])
    final_var_mu = float(trace.iloc[-1]["var_mu"])

    return VIResult(
        m_n=float(m_n),
        lambda_n=float(lambda_n),
        a_n=float(a_n),
        b_n=final_b_n,
        e_tau=float(e_tau),
        var_mu=final_var_mu,
        iterations=len(trace),
        converged=converged,
        trace=trace,
    )


def exact_posterior_params(
    x: np.ndarray,
    mu0: float,
    lambda0: float,
    a0: float,
    b0: float,
) -> dict[str, float]:
    n = x.size
    x_bar = float(np.mean(x))

    lambda_post = lambda0 + n
    mu_post = (lambda0 * mu0 + n * x_bar) / lambda_post
    a_post = a0 + 0.5 * n
    b_post = b0 + 0.5 * (
        np.sum((x - x_bar) ** 2) + (lambda0 * n / lambda_post) * (x_bar - mu0) ** 2
    )

    return {
        "mu_post": float(mu_post),
        "lambda_post": float(lambda_post),
        "a_post": float(a_post),
        "b_post": float(b_post),
        "e_tau_post": float(a_post / b_post),
    }


def is_almost_monotone_non_decreasing(values: np.ndarray, atol: float = 1e-8) -> bool:
    if values.size <= 1:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -atol))


def make_synthetic_data(
    n: int = 200,
    true_mu: float = 2.5,
    true_tau: float = 4.0,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(1.0 / true_tau)
    return rng.normal(loc=true_mu, scale=sigma, size=n)


def main() -> None:
    x = make_synthetic_data()

    mu0 = 0.0
    lambda0 = 1.0
    a0 = 2.0
    b0 = 2.0

    vi = cavi_normal_gamma(
        x=x,
        mu0=mu0,
        lambda0=lambda0,
        a0=a0,
        b0=b0,
        tol=1e-12,
        max_iter=1000,
    )
    exact = exact_posterior_params(x, mu0, lambda0, a0, b0)

    elbo_values = vi.trace["elbo"].to_numpy(dtype=float)
    monotone = is_almost_monotone_non_decreasing(elbo_values)

    print("=== Variational Inference (Mean-Field CAVI) Demo ===")
    print(f"n_samples                 : {x.size}")
    print(f"sample_mean               : {np.mean(x):.6f}")
    print(f"sample_variance           : {np.var(x):.6f}")
    print(f"iterations                : {vi.iterations}")
    print(f"converged                 : {vi.converged}")
    print(f"ELBO monotone nondecrease : {monotone}")
    print()

    print("--- VI posterior moments ---")
    print(f"E_q[mu]                   : {vi.m_n:.6f}")
    print(f"Var_q(mu)                 : {vi.var_mu:.6f}")
    print(f"E_q[tau]                  : {vi.e_tau:.6f}")
    print(f"a_n, b_n                  : ({vi.a_n:.6f}, {vi.b_n:.6f})")
    print()

    print("--- Exact conjugate posterior summary ---")
    print(f"E[mu | x]                 : {exact['mu_post']:.6f}")
    print(f"E[tau | x]                : {exact['e_tau_post']:.6f}")
    print(f"a_post, b_post            : ({exact['a_post']:.6f}, {exact['b_post']:.6f})")
    print()

    print("--- Approximation gap ---")
    print(f"|E_q[mu] - E[mu|x]|       : {abs(vi.m_n - exact['mu_post']):.6e}")
    print(f"|E_q[tau]- E[tau|x]|      : {abs(vi.e_tau - exact['e_tau_post']):.6e}")
    print()

    print("--- Last 5 iterations ---")
    show_df = vi.trace.tail(5).copy()
    show_df["iter"] = show_df["iter"].astype(int)
    print(show_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
