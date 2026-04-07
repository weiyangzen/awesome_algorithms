"""Minimal runnable MVP for F-test (MATH-0272).

This script implements two F-test scenarios from formulas (non-black-box):
1) Two-sample variance-ratio F-test.
2) One-way ANOVA F-test.

SciPy is used for F distribution utilities and ANOVA result cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Alternative = Literal["two-sided", "greater", "less"]


@dataclass
class VarianceFTestResult:
    """Container for two-sample variance-ratio F-test outputs."""

    test_name: str
    statistic: float
    p_value: float
    df_num: float
    df_den: float
    alpha: float
    reject_h0: bool
    variance_ratio_hat: float
    ci_low: float
    ci_high: float


@dataclass
class AnovaFTestResult:
    """Container for one-way ANOVA outputs."""

    test_name: str
    statistic: float
    p_value: float
    df_num: float
    df_den: float
    alpha: float
    reject_h0: bool
    eta_sq: float


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")


def _validate_alternative(alternative: Alternative) -> None:
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of: two-sided, greater, less")


def _p_value_from_f(f_stat: float, df_num: float, df_den: float, alternative: Alternative) -> float:
    """Convert an F statistic to p-value under the chosen alternative."""
    _validate_alternative(alternative)

    if df_num <= 0 or df_den <= 0:
        raise ValueError("df_num and df_den must be positive")
    if f_stat <= 0:
        raise ValueError("F statistic must be positive")

    if alternative == "greater":
        p = stats.f.sf(f_stat, dfn=df_num, dfd=df_den)
    elif alternative == "less":
        p = stats.f.cdf(f_stat, dfn=df_num, dfd=df_den)
    else:
        left_tail = stats.f.cdf(f_stat, dfn=df_num, dfd=df_den)
        right_tail = stats.f.sf(f_stat, dfn=df_num, dfd=df_den)
        p = 2.0 * min(left_tail, right_tail)

    return float(np.clip(p, 0.0, 1.0))


def variance_f_test_manual(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> VarianceFTestResult:
    """Two-sample variance-ratio F-test implemented from formulas."""
    _validate_alpha(alpha)
    _validate_alternative(alternative)

    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("x1 and x2 must be 1D arrays")
    if x1.size < 2 or x2.size < 2:
        raise ValueError("both samples must have at least 2 observations")

    n1 = x1.size
    n2 = x2.size
    s1_sq = float(np.var(x1, ddof=1))
    s2_sq = float(np.var(x2, ddof=1))

    if s1_sq <= 0.0 or s2_sq <= 0.0:
        raise ValueError("sample variances must be positive")

    variance_ratio_hat = s1_sq / s2_sq
    df_num = float(n1 - 1)
    df_den = float(n2 - 1)
    f_stat = variance_ratio_hat

    p_value = _p_value_from_f(
        f_stat=f_stat,
        df_num=df_num,
        df_den=df_den,
        alternative=alternative,
    )

    if alternative == "two-sided":
        f_upper = float(stats.f.ppf(1.0 - alpha / 2.0, dfn=df_num, dfd=df_den))
        f_lower = float(stats.f.ppf(alpha / 2.0, dfn=df_num, dfd=df_den))
        ci_low = variance_ratio_hat / f_upper
        ci_high = variance_ratio_hat / f_lower
    elif alternative == "greater":
        f_quantile = float(stats.f.ppf(1.0 - alpha, dfn=df_num, dfd=df_den))
        ci_low = variance_ratio_hat / f_quantile
        ci_high = np.inf
    else:
        f_quantile = float(stats.f.ppf(alpha, dfn=df_num, dfd=df_den))
        ci_low = 0.0
        ci_high = variance_ratio_hat / f_quantile

    return VarianceFTestResult(
        test_name="two-sample variance F-test",
        statistic=float(f_stat),
        p_value=float(p_value),
        df_num=float(df_num),
        df_den=float(df_den),
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        variance_ratio_hat=float(variance_ratio_hat),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def one_way_anova_f_test_manual(groups: Sequence[np.ndarray], alpha: float = 0.05) -> AnovaFTestResult:
    """One-way ANOVA F-test implemented from sum-of-squares decomposition."""
    _validate_alpha(alpha)

    if len(groups) < 2:
        raise ValueError("at least two groups are required for one-way ANOVA")

    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("each group must be a 1D array")
    if any(a.size < 2 for a in arrays):
        raise ValueError("each group must have at least 2 observations")

    sizes = np.array([a.size for a in arrays], dtype=np.float64)
    means = np.array([np.mean(a) for a in arrays], dtype=np.float64)

    total_n = int(np.sum(sizes))
    k = len(arrays)
    overall_mean = float(np.sum(sizes * means) / total_n)

    ss_between = float(np.sum(sizes * (means - overall_mean) ** 2))
    ss_within = float(np.sum([np.sum((a - np.mean(a)) ** 2) for a in arrays]))

    df_num = float(k - 1)
    df_den = float(total_n - k)
    if df_den <= 0:
        raise ValueError("invalid ANOVA degrees of freedom")
    if ss_within <= 0.0:
        raise ValueError("within-group variation must be positive")

    ms_between = ss_between / df_num
    ms_within = ss_within / df_den
    f_stat = ms_between / ms_within
    p_value = float(stats.f.sf(f_stat, dfn=df_num, dfd=df_den))

    eta_sq = ss_between / (ss_between + ss_within)

    return AnovaFTestResult(
        test_name="one-way ANOVA F-test",
        statistic=float(f_stat),
        p_value=float(p_value),
        df_num=float(df_num),
        df_den=float(df_den),
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        eta_sq=float(eta_sq),
    )


def compare_anova_with_scipy(groups: Sequence[np.ndarray]) -> None:
    """Assert manual ANOVA implementation agrees with scipy.stats.f_oneway."""
    manual = one_way_anova_f_test_manual(groups=groups, alpha=0.05)
    scipy_result = stats.f_oneway(*groups)

    if not np.isclose(manual.statistic, float(scipy_result.statistic), atol=1e-10):
        raise RuntimeError("ANOVA F statistic mismatch vs scipy")
    if not np.isclose(manual.p_value, float(scipy_result.pvalue), atol=1e-10):
        raise RuntimeError("ANOVA p-value mismatch vs scipy")


def _variance_result_to_row(result: VarianceFTestResult) -> dict[str, float | str | bool]:
    return {
        "test": result.test_name,
        "F_stat": result.statistic,
        "df_num": result.df_num,
        "df_den": result.df_den,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "effect_or_ratio": result.variance_ratio_hat,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
    }


def _anova_result_to_row(result: AnovaFTestResult) -> dict[str, float | str | bool]:
    return {
        "test": result.test_name,
        "F_stat": result.statistic,
        "df_num": result.df_num,
        "df_den": result.df_den,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "effect_or_ratio": result.eta_sq,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }


def main() -> None:
    print("F-test MVP (MATH-0272)")
    print("=" * 78)

    rng = np.random.default_rng(272)

    # Dataset A: variance-ratio F-test (different scales to create a meaningful signal)
    sample_a = rng.normal(loc=100.0, scale=9.0, size=28).astype(np.float64)
    sample_b = rng.normal(loc=101.0, scale=5.0, size=24).astype(np.float64)

    variance_result = variance_f_test_manual(
        sample_a,
        sample_b,
        alpha=0.05,
        alternative="two-sided",
    )

    # Dataset B: one-way ANOVA with three groups
    group_1 = rng.normal(loc=20.0, scale=3.2, size=22).astype(np.float64)
    group_2 = rng.normal(loc=22.6, scale=3.0, size=24).astype(np.float64)
    group_3 = rng.normal(loc=25.1, scale=2.8, size=23).astype(np.float64)
    anova_groups = [group_1, group_2, group_3]

    anova_result = one_way_anova_f_test_manual(anova_groups, alpha=0.05)
    compare_anova_with_scipy(anova_groups)

    summary = pd.DataFrame(
        [
            _variance_result_to_row(variance_result),
            _anova_result_to_row(anova_result),
        ]
    )

    print(
        f"variance test sample variances: s1^2={np.var(sample_a, ddof=1):.6f}, "
        f"s2^2={np.var(sample_b, ddof=1):.6f}"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("=" * 78)
    print("Manual ANOVA implementation matches scipy.stats.f_oneway. Run completed.")


if __name__ == "__main__":
    main()
