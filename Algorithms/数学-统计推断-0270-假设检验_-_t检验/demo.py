"""Minimal runnable MVP for t-test (MATH-0270).

This script implements two t-tests from formulas (non-black-box):
1) One-sample t-test.
2) Welch two-sample t-test.

SciPy is used only to verify numerical consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Alternative = Literal["two-sided", "greater", "less"]


@dataclass
class TTestResult:
    """Container for t-test outputs."""

    test_name: str
    statistic: float
    p_value: float
    df: float
    alpha: float
    reject_h0: bool
    effect_size: float
    ci_low: float
    ci_high: float


def _validate_alternative(alternative: Alternative) -> None:
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of: two-sided, greater, less")


def _p_value_from_t(t_stat: float, df: float, alternative: Alternative) -> float:
    """Convert t statistic to p-value under a chosen alternative."""
    _validate_alternative(alternative)
    if df <= 0:
        raise ValueError("df must be positive")

    if alternative == "two-sided":
        p = 2.0 * stats.t.sf(abs(t_stat), df=df)
    elif alternative == "greater":
        p = stats.t.sf(t_stat, df=df)
    else:
        p = stats.t.cdf(t_stat, df=df)

    return float(p)


def one_sample_t_test_manual(
    x: np.ndarray,
    mu0: float,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """One-sample t-test implemented from formulas."""
    _validate_alternative(alternative)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    n = x.size
    if n < 2:
        raise ValueError("one-sample t-test requires n >= 2")

    x_bar = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0.0:
        raise ValueError("sample standard deviation must be positive")

    se = s / np.sqrt(n)
    t_stat = (x_bar - mu0) / se
    df = float(n - 1)
    p_value = _p_value_from_t(t_stat=t_stat, df=df, alternative=alternative)

    mean_diff = x_bar - mu0
    if alternative == "two-sided":
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        margin = t_crit * se
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
    elif alternative == "greater":
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_low = mean_diff - t_crit * se
        ci_high = np.inf
    else:
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_low = -np.inf
        ci_high = mean_diff + t_crit * se

    effect_size = mean_diff / s  # Cohen's d for one-sample mean difference

    return TTestResult(
        test_name="one-sample t-test",
        statistic=float(t_stat),
        p_value=float(p_value),
        df=float(df),
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        effect_size=float(effect_size),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def welch_t_test_manual(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """Welch two-sample t-test implemented from formulas."""
    _validate_alternative(alternative)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("x1 and x2 must be 1D")
    if x1.size < 2 or x2.size < 2:
        raise ValueError("Welch t-test requires both sample sizes >= 2")

    n1, n2 = x1.size, x2.size
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    s1_sq = float(np.var(x1, ddof=1))
    s2_sq = float(np.var(x2, ddof=1))

    if s1_sq <= 0.0 or s2_sq <= 0.0:
        raise ValueError("both sample variances must be positive")

    v1 = s1_sq / n1
    v2 = s2_sq / n2
    se = float(np.sqrt(v1 + v2))
    t_stat = (m1 - m2) / se

    numerator = (v1 + v2) ** 2
    denominator = (v1**2) / (n1 - 1) + (v2**2) / (n2 - 1)
    df = float(numerator / denominator)

    p_value = _p_value_from_t(t_stat=t_stat, df=df, alternative=alternative)

    mean_diff = m1 - m2
    if alternative == "two-sided":
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        margin = t_crit * se
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
    elif alternative == "greater":
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_low = mean_diff - t_crit * se
        ci_high = np.inf
    else:
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_low = -np.inf
        ci_high = mean_diff + t_crit * se

    pooled_sd_approx = float(np.sqrt((s1_sq + s2_sq) / 2.0))
    effect_size = mean_diff / pooled_sd_approx  # Approximate Cohen's d

    return TTestResult(
        test_name="Welch two-sample t-test",
        statistic=float(t_stat),
        p_value=float(p_value),
        df=float(df),
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        effect_size=float(effect_size),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def compare_with_scipy(
    one_sample_data: np.ndarray,
    mu0: float,
    group_a: np.ndarray,
    group_b: np.ndarray,
    alternative: Alternative,
) -> None:
    """Assert manual implementations agree with scipy.stats results."""
    manual_one = one_sample_t_test_manual(
        one_sample_data,
        mu0=mu0,
        alpha=0.05,
        alternative=alternative,
    )
    scipy_one = stats.ttest_1samp(one_sample_data, popmean=mu0, alternative=alternative)

    if not np.isclose(manual_one.statistic, float(scipy_one.statistic), atol=1e-10):
        raise RuntimeError("one-sample t statistic mismatch vs scipy")
    if not np.isclose(manual_one.p_value, float(scipy_one.pvalue), atol=1e-10):
        raise RuntimeError("one-sample p-value mismatch vs scipy")

    manual_welch = welch_t_test_manual(group_a, group_b, alpha=0.05, alternative=alternative)
    scipy_welch = stats.ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        alternative=alternative,
    )

    if not np.isclose(manual_welch.statistic, float(scipy_welch.statistic), atol=1e-10):
        raise RuntimeError("Welch t statistic mismatch vs scipy")
    if not np.isclose(manual_welch.p_value, float(scipy_welch.pvalue), atol=1e-10):
        raise RuntimeError("Welch p-value mismatch vs scipy")


def _result_to_row(result: TTestResult) -> dict[str, float | str | bool]:
    return {
        "test": result.test_name,
        "t_stat": result.statistic,
        "df": result.df,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "effect_size_d": result.effect_size,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
    }


def main() -> None:
    print("t-test MVP (MATH-0270)")
    print("=" * 72)

    rng = np.random.default_rng(270)

    # Dataset A: one-sample mean test against mu0
    one_sample_data = rng.normal(loc=51.2, scale=3.5, size=30).astype(np.float64)
    mu0 = 50.0

    # Dataset B: two independent groups for Welch t-test
    group_a = rng.normal(loc=73.0, scale=8.0, size=24).astype(np.float64)
    group_b = rng.normal(loc=68.5, scale=7.5, size=21).astype(np.float64)

    alternative: Alternative = "two-sided"

    one_result = one_sample_t_test_manual(
        one_sample_data,
        mu0=mu0,
        alpha=0.05,
        alternative=alternative,
    )
    welch_result = welch_t_test_manual(
        group_a,
        group_b,
        alpha=0.05,
        alternative=alternative,
    )

    compare_with_scipy(
        one_sample_data=one_sample_data,
        mu0=mu0,
        group_a=group_a,
        group_b=group_b,
        alternative=alternative,
    )

    df = pd.DataFrame([_result_to_row(one_result), _result_to_row(welch_result)])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("=" * 72)
    print("Manual formulas match scipy.stats checks. Run completed successfully.")


if __name__ == "__main__":
    main()
