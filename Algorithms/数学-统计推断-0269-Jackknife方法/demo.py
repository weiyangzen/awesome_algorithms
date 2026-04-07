"""Minimal runnable MVP for Jackknife (MATH-0269).

This script demonstrates:
1) Generic leave-one-out jackknife for scalar statistics.
2) Bias correction on biased variance (ddof=0).
3) Standard-error estimation for sample mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class JackknifeResult:
    """Container for jackknife outputs."""

    full_estimate: float
    loo_estimates: np.ndarray
    loo_mean: float
    bias_estimate: float
    bias_corrected_estimate: float
    standard_error: float


def biased_variance(x: np.ndarray) -> float:
    """Biased variance estimator with denominator n (ddof=0)."""
    mu = float(np.mean(x))
    return float(np.mean((x - mu) ** 2))


def jackknife(x: np.ndarray, statistic: Callable[[np.ndarray], float]) -> JackknifeResult:
    """Compute jackknife bias and standard error for a scalar statistic.

    Args:
        x: 1D sample array with length >= 2
        statistic: function that maps a sample to a scalar estimate
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    n = x.size
    if n < 2:
        raise ValueError("x must contain at least 2 samples")

    full_est = float(statistic(x))
    loo_est = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Explicitly form leave-one-out sample (transparent for educational MVP).
        x_loo = np.concatenate((x[:i], x[i + 1 :]))
        loo_est[i] = float(statistic(x_loo))

    loo_mean = float(np.mean(loo_est))
    bias_est = float((n - 1) * (loo_mean - full_est))
    bias_corrected = float(full_est - bias_est)

    se_sq = (n - 1) / n * float(np.sum((loo_est - loo_mean) ** 2))
    se = float(np.sqrt(max(se_sq, 0.0)))

    return JackknifeResult(
        full_estimate=full_est,
        loo_estimates=loo_est,
        loo_mean=loo_mean,
        bias_estimate=bias_est,
        bias_corrected_estimate=bias_corrected,
        standard_error=se,
    )


def monte_carlo_variance_experiment(
    n: int = 12,
    trials: int = 2500,
    true_sigma: float = 2.0,
    seed: int = 20260407,
) -> Dict[str, float]:
    """Compare bias of biased variance vs jackknife-corrected variance."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if true_sigma <= 0:
        raise ValueError("true_sigma must be positive")

    rng = np.random.default_rng(seed)
    true_var = true_sigma**2

    raw_vals = np.empty(trials, dtype=np.float64)
    jack_vals = np.empty(trials, dtype=np.float64)

    for t in range(trials):
        x = rng.normal(loc=0.0, scale=true_sigma, size=n).astype(np.float64)
        raw_vals[t] = biased_variance(x)
        jack_vals[t] = jackknife(x, biased_variance).bias_corrected_estimate

    raw_mean = float(np.mean(raw_vals))
    jack_mean = float(np.mean(jack_vals))
    raw_bias = raw_mean - true_var
    jack_bias = jack_mean - true_var

    return {
        "true_var": float(true_var),
        "raw_mean": raw_mean,
        "jack_mean": jack_mean,
        "raw_bias": float(raw_bias),
        "jack_bias": float(jack_bias),
        "abs_bias_reduction_ratio": float(
            abs(jack_bias) / (abs(raw_bias) + 1e-15)
        ),
    }


def main() -> None:
    print("Jackknife MVP (MATH-0269)")
    print("=" * 72)

    rng = np.random.default_rng(269)
    x_single = rng.normal(loc=1.5, scale=2.0, size=12).astype(np.float64)
    n = x_single.size

    var_jack = jackknife(x_single, biased_variance)
    var_unbiased = float(np.var(x_single, ddof=1))

    mean_jack = jackknife(x_single, lambda arr: float(np.mean(arr)))
    mean_se_analytic = float(np.std(x_single, ddof=1) / np.sqrt(n))

    print(f"single-sample size n = {n}")
    print(
        "variance (ddof=0) full estimate       = "
        f"{var_jack.full_estimate:.6f}"
    )
    print(
        "variance jackknife bias-corrected     = "
        f"{var_jack.bias_corrected_estimate:.6f}"
    )
    print(f"variance np.var(ddof=1)               = {var_unbiased:.6f}")
    print(f"variance jackknife SE                 = {var_jack.standard_error:.6f}")
    print(
        "mean jackknife SE                     = "
        f"{mean_jack.standard_error:.6f}"
    )
    print(f"mean analytic SE (std/sqrt(n))        = {mean_se_analytic:.6f}")

    mc = monte_carlo_variance_experiment(n=12, trials=2500, true_sigma=2.0, seed=269)
    print("-" * 72)
    print(f"MC true variance                      = {mc['true_var']:.6f}")
    print(f"MC raw(ddof=0) mean estimate          = {mc['raw_mean']:.6f}")
    print(f"MC jackknife-corrected mean estimate  = {mc['jack_mean']:.6f}")
    print(f"MC raw bias                           = {mc['raw_bias']:.6f}")
    print(f"MC jackknife bias                     = {mc['jack_bias']:.6f}")
    print(
        "MC |jack_bias| / |raw_bias|           = "
        f"{mc['abs_bias_reduction_ratio']:.4f}"
    )

    # Correctness checks for this MVP.
    if abs(var_jack.bias_corrected_estimate - var_unbiased) > 1e-10:
        raise RuntimeError(
            "jackknife bias-corrected variance does not match ddof=1 variance"
        )
    if abs(mean_jack.standard_error - mean_se_analytic) > 1e-10:
        raise RuntimeError("jackknife SE for mean does not match analytic SE")
    if not (abs(mc["jack_bias"]) < abs(mc["raw_bias"])):
        raise RuntimeError("jackknife failed to reduce variance-estimator bias")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
