"""MVP for confidence interval computation.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    """Simple container for confidence interval outputs."""

    method: str
    confidence: float
    lower: float
    upper: float
    center: float
    margin: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _check_confidence(confidence: float) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return 1.0 - confidence


def z_interval_mean_known_sigma(
    samples: np.ndarray,
    sigma: float,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """CI for mean when population sigma is known.

    Formula:
        x_bar +/- z_(1-alpha/2) * sigma / sqrt(n)
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("samples must be non-empty")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    alpha = _check_confidence(confidence)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    center = float(np.mean(x))
    margin = float(z * sigma / np.sqrt(x.size))
    return ConfidenceInterval(
        method="mean_z_known_sigma",
        confidence=confidence,
        lower=center - margin,
        upper=center + margin,
        center=center,
        margin=margin,
    )


def t_interval_mean_unknown_sigma(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """CI for mean when population sigma is unknown.

    Formula:
        x_bar +/- t_(1-alpha/2, n-1) * s / sqrt(n)
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two samples are required for t interval")

    alpha = _check_confidence(confidence)
    n = x.size
    center = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1))
    t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = float(t_critical * sample_std / np.sqrt(n))
    return ConfidenceInterval(
        method="mean_t_unknown_sigma",
        confidence=confidence,
        lower=center - margin,
        upper=center + margin,
        center=center,
        margin=margin,
    )


def wilson_interval_proportion(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score interval for binomial proportion.

    Formula:
        p_tilde = (p_hat + z^2/(2n)) / (1 + z^2/n)
        half = z/(1+z^2/n) * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not (0 <= successes <= trials):
        raise ValueError(f"successes must be in [0, trials], got {successes}")

    alpha = _check_confidence(confidence)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    n = float(trials)
    p_hat = successes / n

    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * n**2)))
        / denom
    )

    lower = float(max(0.0, center - half))
    upper = float(min(1.0, center + half))
    center = float(center)
    margin = float(half)

    return ConfidenceInterval(
        method="proportion_wilson",
        confidence=confidence,
        lower=lower,
        upper=upper,
        center=center,
        margin=margin,
    )


def bootstrap_percentile_interval(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 4000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for a scalar statistic."""
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("samples must be non-empty")
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")

    alpha = _check_confidence(confidence)
    n = x.size
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled = x[indices]

    if statistic is np.mean:
        stat_values = resampled.mean(axis=1)
        stat_name = "mean"
    elif statistic is np.median:
        stat_values = np.median(resampled, axis=1)
        stat_name = "median"
    else:
        stat_values = np.array([float(statistic(row)) for row in resampled], dtype=float)
        stat_name = getattr(statistic, "__name__", "custom_stat")

    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    lower, upper = np.quantile(stat_values, [q_low, q_high])

    point = float(statistic(x))
    margin = float(max(point - lower, upper - point))

    return ConfidenceInterval(
        method=f"bootstrap_percentile_{stat_name}",
        confidence=confidence,
        lower=float(lower),
        upper=float(upper),
        center=point,
        margin=margin,
    )


def estimate_mean_interval_coverage(
    true_mean: float,
    true_sigma: float,
    n: int,
    confidence: float,
    n_trials: int = 1200,
    seed: int = 1,
) -> tuple[float, float]:
    """Monte Carlo estimate for mean-interval coverage (z and t)."""
    rng = np.random.default_rng(seed)
    z_hits = 0
    t_hits = 0

    for _ in range(n_trials):
        x = rng.normal(loc=true_mean, scale=true_sigma, size=n)
        z_ci = z_interval_mean_known_sigma(x, sigma=true_sigma, confidence=confidence)
        t_ci = t_interval_mean_unknown_sigma(x, confidence=confidence)
        z_hits += int(z_ci.contains(true_mean))
        t_hits += int(t_ci.contains(true_mean))

    return z_hits / n_trials, t_hits / n_trials


def _pretty(ci: ConfidenceInterval) -> str:
    return (
        f"{ci.method:28s} "
        f"[{ci.lower:8.4f}, {ci.upper:8.4f}] "
        f"center={ci.center:8.4f} margin={ci.margin:7.4f}"
    )


def main() -> None:
    confidence = 0.95
    rng = np.random.default_rng(20260407)

    true_mean = 10.0
    true_sigma = 2.5
    sample = rng.normal(loc=true_mean, scale=true_sigma, size=40)

    z_ci = z_interval_mean_known_sigma(sample, sigma=true_sigma, confidence=confidence)
    t_ci = t_interval_mean_unknown_sigma(sample, confidence=confidence)
    boot_mean_ci = bootstrap_percentile_interval(
        sample,
        statistic=np.mean,
        confidence=confidence,
        n_resamples=5000,
        seed=2026,
    )

    true_p = 0.62
    n_trials = 120
    successes = int(rng.binomial(n=n_trials, p=true_p))
    wilson_ci = wilson_interval_proportion(successes, n_trials, confidence=confidence)

    print("=== Confidence Interval MVP ===")
    print(f"sample size={sample.size}, confidence={confidence:.2f}")
    print(f"sample mean={sample.mean():.4f}, sample std(ddof=1)={sample.std(ddof=1):.4f}")
    print(_pretty(z_ci))
    print(_pretty(t_ci))
    print(_pretty(boot_mean_ci))
    print()
    print(f"binomial proportion sample: successes={successes}, trials={n_trials}")
    print(_pretty(wilson_ci))
    print()

    z_cov, t_cov = estimate_mean_interval_coverage(
        true_mean=true_mean,
        true_sigma=true_sigma,
        n=40,
        confidence=confidence,
        n_trials=1200,
        seed=99,
    )
    print("Empirical coverage on Gaussian data (Monte Carlo, 1200 runs):")
    print(f"z interval coverage ≈ {z_cov:.4f}")
    print(f"t interval coverage ≈ {t_cov:.4f}")

    # Basic sanity checks for non-interactive validation.
    for ci in (z_ci, t_ci, boot_mean_ci, wilson_ci):
        assert ci.lower <= ci.upper
        assert ci.margin >= 0.0

    assert z_ci.contains(float(sample.mean()))
    assert t_ci.contains(float(sample.mean()))
    assert 0.0 <= wilson_ci.lower <= wilson_ci.upper <= 1.0
    assert abs(t_ci.center - z_ci.center) < 1e-12
    assert 0.90 <= z_cov <= 0.99
    assert 0.90 <= t_cov <= 0.99


if __name__ == "__main__":
    main()
