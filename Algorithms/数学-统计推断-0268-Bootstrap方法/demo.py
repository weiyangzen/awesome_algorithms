"""Bootstrap method MVP.

This script implements core bootstrap logic from scratch:
- one-sample bootstrap confidence intervals
- two-sample bootstrap confidence intervals for mean difference
- optional comparison with scipy.stats.bootstrap
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - depends on local environment
    scipy_stats = None


Array1D = np.ndarray
OneSampleStatistic = Callable[[np.ndarray, int], np.ndarray | float]
TwoSampleStatistic = Callable[[np.ndarray, np.ndarray, int], np.ndarray | float]


def validate_1d_finite(values: np.ndarray, name: str) -> Array1D:
    """Validate that values is a non-empty finite 1D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def bootstrap_resample_indices(n: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw bootstrap indices with replacement."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    return rng.integers(0, n, size=(n_resamples, n), endpoint=False)


def bootstrap_statistic_1sample(
    sample: Array1D,
    statistic: OneSampleStatistic,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap distribution for a one-sample statistic."""
    x = validate_1d_finite(sample, "sample")
    idx = bootstrap_resample_indices(x.size, n_resamples=n_resamples, rng=rng)
    resamples = x[idx]  # shape: (B, n)
    stats = np.asarray(statistic(resamples, axis=1), dtype=float)
    if stats.shape != (n_resamples,):
        raise ValueError("statistic must return shape (n_resamples,) when axis=1")
    return stats


def bootstrap_statistic_2sample(
    x: Array1D,
    y: Array1D,
    statistic: TwoSampleStatistic,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap distribution for a two-sample statistic."""
    x_clean = validate_1d_finite(x, "x")
    y_clean = validate_1d_finite(y, "y")

    idx_x = bootstrap_resample_indices(x_clean.size, n_resamples=n_resamples, rng=rng)
    idx_y = bootstrap_resample_indices(y_clean.size, n_resamples=n_resamples, rng=rng)
    x_boot = x_clean[idx_x]  # shape: (B, n_x)
    y_boot = y_clean[idx_y]  # shape: (B, n_y)

    stats = np.asarray(statistic(x_boot, y_boot, axis=1), dtype=float)
    if stats.shape != (n_resamples,):
        raise ValueError("statistic must return shape (n_resamples,) when axis=1")
    return stats


def percentile_interval(bootstrap_stats: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)
    low, high = np.percentile(bootstrap_stats, [q_low, q_high])
    return float(low), float(high)


def basic_interval(
    theta_hat: float,
    bootstrap_stats: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Basic bootstrap confidence interval: [2*theta_hat-q_high, 2*theta_hat-q_low]."""
    q_low, q_high = percentile_interval(bootstrap_stats, alpha=alpha)
    return float(2.0 * theta_hat - q_high), float(2.0 * theta_hat - q_low)


def normal_interval(theta_hat: float, bootstrap_stats: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approximation interval based on bootstrap standard error."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    se = float(np.std(bootstrap_stats, ddof=1))
    z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0)) if scipy_stats is not None else 1.959963984540054
    return float(theta_hat - z * se), float(theta_hat + z * se)


def stat_mean(values: np.ndarray, axis: int) -> np.ndarray | float:
    return np.mean(values, axis=axis)


def stat_median(values: np.ndarray, axis: int) -> np.ndarray | float:
    return np.median(values, axis=axis)


def stat_mean_diff(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray | float:
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def scipy_bootstrap_percentile_1sample(
    sample: Array1D,
    statistic: OneSampleStatistic,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Compute SciPy percentile CI for one-sample statistic when available."""
    if scipy_stats is None:
        return float("nan"), float("nan")
    result = scipy_stats.bootstrap(
        (sample,),
        statistic,
        confidence_level=1.0 - alpha,
        n_resamples=n_resamples,
        method="percentile",
        vectorized=True,
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def scipy_bootstrap_percentile_2sample(
    x: Array1D,
    y: Array1D,
    statistic: TwoSampleStatistic,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Compute SciPy percentile CI for two-sample statistic when available."""
    if scipy_stats is None:
        return float("nan"), float("nan")
    result = scipy_stats.bootstrap(
        (x, y),
        statistic,
        confidence_level=1.0 - alpha,
        n_resamples=n_resamples,
        method="percentile",
        vectorized=True,
        paired=False,
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def run_one_sample_experiment(
    scenario: str,
    sample: Array1D,
    statistic: OneSampleStatistic,
    statistic_name: str,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """Run one-sample bootstrap experiment and return a report row."""
    clean = validate_1d_finite(sample, "sample")
    theta_hat = float(statistic(clean, axis=0))
    boot_stats = bootstrap_statistic_1sample(clean, statistic, n_resamples=n_resamples, rng=rng)

    ci_pct_low, ci_pct_high = percentile_interval(boot_stats, alpha=alpha)
    ci_basic_low, ci_basic_high = basic_interval(theta_hat, boot_stats, alpha=alpha)
    ci_norm_low, ci_norm_high = normal_interval(theta_hat, boot_stats, alpha=alpha)

    scipy_low, scipy_high = scipy_bootstrap_percentile_1sample(
        clean,
        statistic=statistic,
        alpha=alpha,
        n_resamples=n_resamples,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    return {
        "scenario": scenario,
        "statistic": statistic_name,
        "n": int(clean.size),
        "estimate": theta_hat,
        "bootstrap_bias": float(np.mean(boot_stats) - theta_hat),
        "bootstrap_se": float(np.std(boot_stats, ddof=1)),
        "pct_ci_low": ci_pct_low,
        "pct_ci_high": ci_pct_high,
        "basic_ci_low": ci_basic_low,
        "basic_ci_high": ci_basic_high,
        "normal_ci_low": ci_norm_low,
        "normal_ci_high": ci_norm_high,
        "scipy_pct_ci_low": scipy_low,
        "scipy_pct_ci_high": scipy_high,
    }


def run_two_sample_experiment(
    scenario: str,
    x: Array1D,
    y: Array1D,
    statistic: TwoSampleStatistic,
    statistic_name: str,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """Run two-sample bootstrap experiment and return a report row."""
    x_clean = validate_1d_finite(x, "x")
    y_clean = validate_1d_finite(y, "y")
    theta_hat = float(statistic(x_clean, y_clean, axis=0))
    boot_stats = bootstrap_statistic_2sample(
        x_clean, y_clean, statistic=statistic, n_resamples=n_resamples, rng=rng
    )

    ci_pct_low, ci_pct_high = percentile_interval(boot_stats, alpha=alpha)
    ci_basic_low, ci_basic_high = basic_interval(theta_hat, boot_stats, alpha=alpha)
    ci_norm_low, ci_norm_high = normal_interval(theta_hat, boot_stats, alpha=alpha)

    scipy_low, scipy_high = scipy_bootstrap_percentile_2sample(
        x_clean,
        y_clean,
        statistic=statistic,
        alpha=alpha,
        n_resamples=n_resamples,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    return {
        "scenario": scenario,
        "statistic": statistic_name,
        "n_x": int(x_clean.size),
        "n_y": int(y_clean.size),
        "estimate": theta_hat,
        "bootstrap_bias": float(np.mean(boot_stats) - theta_hat),
        "bootstrap_se": float(np.std(boot_stats, ddof=1)),
        "pct_ci_low": ci_pct_low,
        "pct_ci_high": ci_pct_high,
        "basic_ci_low": ci_basic_low,
        "basic_ci_high": ci_basic_high,
        "normal_ci_low": ci_norm_low,
        "normal_ci_high": ci_norm_high,
        "scipy_pct_ci_low": scipy_low,
        "scipy_pct_ci_high": scipy_high,
    }


def main() -> None:
    alpha = 0.05
    n_resamples = 4_000
    seed = 20260407
    rng = np.random.default_rng(seed)

    one_sample_records = [
        run_one_sample_experiment(
            scenario="One-sample (normal): estimate mean of N(2,1.5^2)",
            sample=rng.normal(loc=2.0, scale=1.5, size=140),
            statistic=stat_mean,
            statistic_name="mean",
            n_resamples=n_resamples,
            alpha=alpha,
            rng=rng,
        ),
        run_one_sample_experiment(
            scenario="One-sample (skewed): estimate median of Exponential(1)",
            sample=rng.exponential(scale=1.0, size=160),
            statistic=stat_median,
            statistic_name="median",
            n_resamples=n_resamples,
            alpha=alpha,
            rng=rng,
        ),
    ]

    two_sample_records = [
        run_two_sample_experiment(
            scenario="Two-sample: mean(X)-mean(Y), X~N(0.5,1), Y~N(0,1)",
            x=rng.normal(loc=0.5, scale=1.0, size=120),
            y=rng.normal(loc=0.0, scale=1.0, size=110),
            statistic=stat_mean_diff,
            statistic_name="mean_difference",
            n_resamples=n_resamples,
            alpha=alpha,
            rng=rng,
        ),
        run_two_sample_experiment(
            scenario="Two-sample: mean(X)-mean(Y), X~N(0,1), Y~N(0,1)",
            x=rng.normal(loc=0.0, scale=1.0, size=120),
            y=rng.normal(loc=0.0, scale=1.0, size=110),
            statistic=stat_mean_diff,
            statistic_name="mean_difference",
            n_resamples=n_resamples,
            alpha=alpha,
            rng=rng,
        ),
    ]

    one_df = pd.DataFrame(one_sample_records)
    two_df = pd.DataFrame(two_sample_records)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 72)

    print("=" * 96)
    print("Bootstrap Method MVP")
    print(f"random_seed={seed}, n_resamples={n_resamples}, alpha={alpha}")
    print(f"SciPy comparison available={scipy_stats is not None}")
    print("=" * 96)
    print("\n[One-sample bootstrap]")
    print(one_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n[Two-sample bootstrap]")
    print(two_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    overlap_summary = []
    for row in two_sample_records:
        contains_zero = bool(row["pct_ci_low"] <= 0.0 <= row["pct_ci_high"])
        overlap_summary.append(
            {
                "scenario": row["scenario"],
                "percentile_CI_contains_0": contains_zero,
            }
        )

    print("\n[Inference hint]")
    print(pd.DataFrame(overlap_summary).to_string(index=False))


if __name__ == "__main__":
    main()
