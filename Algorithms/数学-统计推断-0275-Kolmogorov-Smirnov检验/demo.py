"""Kolmogorov-Smirnov test MVP.

This script implements manual KS statistics for one-sample and two-sample
settings, then compares them with SciPy's implementations.
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


ArrayLike1D = np.ndarray


def validate_1d_finite(values: ArrayLike1D, name: str) -> ArrayLike1D:
    """Return a safe 1D float array and validate basic assumptions."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def ks_statistic_one_sample(
    sample: ArrayLike1D, cdf: Callable[[ArrayLike1D], ArrayLike1D]
) -> tuple[float, float, float]:
    """Compute one-sample KS statistics (D+, D-, D) manually."""
    x = np.sort(validate_1d_finite(sample, "sample"))
    n = x.size

    cdf_values = np.asarray(cdf(x), dtype=float)
    if cdf_values.shape != x.shape:
        raise ValueError("cdf(sample) must return an array with the same shape")
    if np.any((cdf_values < 0.0) | (cdf_values > 1.0)):
        raise ValueError("cdf values must be in [0, 1]")

    i = np.arange(1.0, n + 1.0)
    d_plus = float(np.max(i / n - cdf_values))
    d_minus = float(np.max(cdf_values - (i - 1.0) / n))
    d = max(d_plus, d_minus)
    return d_plus, d_minus, d


def ks_statistic_two_sample(x: ArrayLike1D, y: ArrayLike1D) -> tuple[float, float]:
    """Compute two-sample KS statistic D manually on a merged support grid."""
    xs = np.sort(validate_1d_finite(x, "x"))
    ys = np.sort(validate_1d_finite(y, "y"))

    grid = np.sort(np.unique(np.concatenate([xs, ys])))
    fx = np.searchsorted(xs, grid, side="right") / xs.size
    gy = np.searchsorted(ys, grid, side="right") / ys.size
    abs_diff = np.abs(fx - gy)

    idx = int(np.argmax(abs_diff))
    d = float(abs_diff[idx])
    location = float(grid[idx])
    return d, location


def asymptotic_pvalue(d_stat: float, effective_n: float) -> float:
    """Approximate two-sided KS p-value using the Kolmogorov asymptotic law."""
    if effective_n <= 0:
        raise ValueError("effective_n must be positive")

    root_ne = math.sqrt(effective_n)
    lam = (root_ne + 0.12 + 0.11 / root_ne) * d_stat
    if scipy_stats is not None:
        p_value = float(scipy_stats.kstwobign.sf(lam))
    else:
        p_value = kolmogorov_sf(lam)
    return min(max(p_value, 0.0), 1.0)


def kolmogorov_sf(lam: float, tol: float = 1e-12, max_terms: int = 100_000) -> float:
    """Survival function Q_KS(lambda) via convergent alternating series."""
    if lam <= 0:
        return 1.0

    total = 0.0
    for k in range(1, max_terms + 1):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k**2) * (lam**2))
        total += term
        if abs(term) < tol:
            break
    return min(max(total, 0.0), 1.0)


def standard_normal_cdf(x: ArrayLike1D) -> ArrayLike1D:
    """Standard normal CDF based on erf; used when SciPy is unavailable."""
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def run_one_sample_experiment(
    scenario: str,
    sample: ArrayLike1D,
    cdf: Callable[[ArrayLike1D], ArrayLike1D],
    alpha: float,
) -> dict[str, float | int | str | bool]:
    """Run manual vs SciPy one-sample KS and return a record."""
    clean = validate_1d_finite(sample, "sample")
    d_plus, d_minus, d_manual = ks_statistic_one_sample(clean, cdf)
    p_manual = asymptotic_pvalue(d_manual, effective_n=clean.size)

    if scipy_stats is not None:
        scipy_res = scipy_stats.kstest(clean, cdf, alternative="two-sided", method="auto")
        d_scipy = float(scipy_res.statistic)
        p_scipy = float(scipy_res.pvalue)
        reject = bool(p_scipy < alpha)
    else:
        d_scipy = float("nan")
        p_scipy = float("nan")
        reject = bool(p_manual < alpha)

    return {
        "scenario": scenario,
        "n": int(clean.size),
        "D_plus": d_plus,
        "D_minus": d_minus,
        "manual_D": d_manual,
        "scipy_D": d_scipy,
        "|manual_D-scipy_D|": abs(d_manual - d_scipy) if not math.isnan(d_scipy) else float("nan"),
        "manual_p_asymptotic": p_manual,
        "scipy_p": p_scipy,
        "reject_H0_at_0.05": reject,
    }


def run_two_sample_experiment(
    scenario: str,
    x: ArrayLike1D,
    y: ArrayLike1D,
    alpha: float,
) -> dict[str, float | int | str | bool]:
    """Run manual vs SciPy two-sample KS and return a record."""
    x_clean = validate_1d_finite(x, "x")
    y_clean = validate_1d_finite(y, "y")

    d_manual, location_manual = ks_statistic_two_sample(x_clean, y_clean)
    effective_n = x_clean.size * y_clean.size / (x_clean.size + y_clean.size)
    p_manual = asymptotic_pvalue(d_manual, effective_n=effective_n)

    if scipy_stats is not None:
        scipy_res = scipy_stats.ks_2samp(x_clean, y_clean, alternative="two-sided", method="auto")
        d_scipy = float(scipy_res.statistic)
        p_scipy = float(scipy_res.pvalue)
        reject = bool(p_scipy < alpha)
    else:
        d_scipy = float("nan")
        p_scipy = float("nan")
        reject = bool(p_manual < alpha)

    return {
        "scenario": scenario,
        "n": int(x_clean.size),
        "m": int(y_clean.size),
        "manual_D": d_manual,
        "scipy_D": d_scipy,
        "|manual_D-scipy_D|": abs(d_manual - d_scipy) if not math.isnan(d_scipy) else float("nan"),
        "manual_D_location": location_manual,
        "manual_p_asymptotic": p_manual,
        "scipy_p": p_scipy,
        "reject_H0_at_0.05": reject,
    }


def main() -> None:
    alpha = 0.05
    rng = np.random.default_rng(20260407)
    n = 400

    one_sample_records = [
        run_one_sample_experiment(
            scenario="One-sample: N(0,1) sample vs N(0,1) CDF",
            sample=rng.normal(0.0, 1.0, size=n),
            cdf=standard_normal_cdf,
            alpha=alpha,
        ),
        run_one_sample_experiment(
            scenario="One-sample: Uniform(-2.5,2.5) sample vs N(0,1) CDF",
            sample=rng.uniform(-2.5, 2.5, size=n),
            cdf=standard_normal_cdf,
            alpha=alpha,
        ),
    ]

    two_sample_records = [
        run_two_sample_experiment(
            scenario="Two-sample: N(0,1) vs N(0,1)",
            x=rng.normal(0.0, 1.0, size=n),
            y=rng.normal(0.0, 1.0, size=n),
            alpha=alpha,
        ),
        run_two_sample_experiment(
            scenario="Two-sample: N(0,1) vs N(0.7,1)",
            x=rng.normal(0.0, 1.0, size=n),
            y=rng.normal(0.7, 1.0, size=n),
            alpha=alpha,
        ),
    ]

    one_df = pd.DataFrame(one_sample_records)
    two_df = pd.DataFrame(two_sample_records)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 80)

    print("=" * 80)
    print("Kolmogorov-Smirnov Test MVP")
    print(f"alpha = {alpha}, random_seed = 20260407")
    print(f"SciPy compare available = {scipy_stats is not None}")
    print("=" * 80)
    print("\n[One-sample KS]")
    print(one_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n[Two-sample KS]")
    print(two_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
