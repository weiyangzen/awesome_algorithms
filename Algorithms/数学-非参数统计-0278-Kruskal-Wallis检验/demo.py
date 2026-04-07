"""Minimal runnable MVP for Kruskal-Wallis test (MATH-0278).

This script implements the core Kruskal-Wallis flow from formulas:
1) Merge groups and assign average ranks with tie handling.
2) Compute H statistic with tie correction.
3) Compute p-value from chi-square approximation.

SciPy is used only as a numerical cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KruskalWallisResult:
    """Container for Kruskal-Wallis outputs."""

    case_name: str
    h_statistic: float
    p_value: float
    df: int
    alpha: float
    reject_h0: bool
    epsilon_squared: float
    tie_correction: float
    total_n: int
    group_count: int


def _validate_groups(groups: list[np.ndarray]) -> list[np.ndarray]:
    """Validate and normalize input groups into float64 1D arrays."""
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis requires at least two groups")

    checked: list[np.ndarray] = []
    for i, g in enumerate(groups, start=1):
        arr = np.asarray(g, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"group {i} must be a 1D array")
        if arr.size == 0:
            raise ValueError(f"group {i} must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"group {i} contains NaN or Inf")
        checked.append(arr)

    return checked


def _average_ranks_with_ties(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign average ranks (1-based) and return tie block sizes.

    Returns:
        ranks: rank for each original element.
        tie_counts: array of tie block sizes (>1 only).
    """
    if values.ndim != 1:
        raise ValueError("values must be 1D")

    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    sorted_ranks = np.empty(n, dtype=np.float64)
    tie_counts: list[int] = []

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1

        # Rank index is 1-based: positions [i, j) map to ranks [i+1, j].
        avg_rank = 0.5 * ((i + 1) + j)
        sorted_ranks[i:j] = avg_rank

        tie_size = j - i
        if tie_size > 1:
            tie_counts.append(tie_size)

        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks

    return ranks, np.asarray(tie_counts, dtype=np.int64)


def kruskal_wallis_manual(
    groups: list[np.ndarray],
    case_name: str,
    alpha: float = 0.05,
) -> KruskalWallisResult:
    """Manual Kruskal-Wallis implementation with tie correction."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    checked = _validate_groups(groups)
    k = len(checked)
    sizes = np.array([g.size for g in checked], dtype=np.int64)
    n_total = int(np.sum(sizes))

    if n_total < 2:
        raise ValueError("total sample size must be >= 2")

    merged = np.concatenate(checked)
    ranks, tie_counts = _average_ranks_with_ties(merged)

    # Split rank array back to each group and compute rank sums.
    split_points = np.cumsum(sizes)[:-1]
    rank_groups = np.split(ranks, split_points)
    rank_sums = np.array([np.sum(rg) for rg in rank_groups], dtype=np.float64)

    h_raw = (12.0 / (n_total * (n_total + 1.0))) * np.sum((rank_sums**2) / sizes) - 3.0 * (
        n_total + 1.0
    )

    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    denominator = float(n_total**3 - n_total)
    tie_correction = 1.0 - (tie_term / denominator)

    if tie_correction <= 0.0:
        raise ValueError("tie correction is non-positive; data are degenerate")

    h_stat = float(h_raw / tie_correction)
    df = k - 1
    p_value = float(stats.chi2.sf(h_stat, df=df))

    # A common effect-size approximation for Kruskal-Wallis.
    # Clip at zero because sampling noise may produce tiny negatives.
    if n_total == k:
        epsilon_sq = 0.0
    else:
        epsilon_sq = float(max(0.0, (h_stat - k + 1.0) / (n_total - k)))

    return KruskalWallisResult(
        case_name=case_name,
        h_statistic=h_stat,
        p_value=p_value,
        df=df,
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        epsilon_squared=epsilon_sq,
        tie_correction=float(tie_correction),
        total_n=n_total,
        group_count=k,
    )


def compare_with_scipy(groups: list[np.ndarray], manual: KruskalWallisResult) -> None:
    """Assert manual computation agrees with scipy.stats.kruskal."""
    scipy_result = stats.kruskal(*groups, nan_policy="raise")

    if not np.isclose(manual.h_statistic, float(scipy_result.statistic), atol=1e-10):
        raise RuntimeError(
            f"H mismatch vs scipy for case={manual.case_name}: "
            f"manual={manual.h_statistic}, scipy={float(scipy_result.statistic)}"
        )

    if not np.isclose(manual.p_value, float(scipy_result.pvalue), atol=1e-10):
        raise RuntimeError(
            f"p-value mismatch vs scipy for case={manual.case_name}: "
            f"manual={manual.p_value}, scipy={float(scipy_result.pvalue)}"
        )


def _build_likert_groups(rng: np.random.Generator) -> dict[str, list[np.ndarray]]:
    """Create reproducible integer-score groups (with ties by design)."""
    scores = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    shifted_locations = [
        rng.choice(scores, size=26, p=[0.10, 0.24, 0.34, 0.22, 0.10]),
        rng.choice(scores, size=24, p=[0.04, 0.14, 0.30, 0.32, 0.20]),
        rng.choice(scores, size=25, p=[0.22, 0.34, 0.26, 0.13, 0.05]),
    ]

    null_like = [
        rng.choice(scores, size=22, p=[0.11, 0.24, 0.31, 0.23, 0.11]),
        rng.choice(scores, size=23, p=[0.11, 0.24, 0.31, 0.23, 0.11]),
        rng.choice(scores, size=21, p=[0.11, 0.24, 0.31, 0.23, 0.11]),
    ]

    return {
        "shifted-locations": shifted_locations,
        "null-like": null_like,
    }


def _group_summary(case_name: str, groups: list[np.ndarray]) -> pd.DataFrame:
    """Create a compact per-group summary table."""
    rows: list[dict[str, float | int | str]] = []
    for idx, g in enumerate(groups, start=1):
        rows.append(
            {
                "case": case_name,
                "group": f"G{idx}",
                "n": int(g.size),
                "mean": float(np.mean(g)),
                "median": float(np.median(g)),
                "std": float(np.std(g, ddof=1)) if g.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _result_row(result: KruskalWallisResult) -> dict[str, float | int | bool | str]:
    return {
        "case": result.case_name,
        "H_statistic": result.h_statistic,
        "df": result.df,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "epsilon_squared": result.epsilon_squared,
        "tie_correction": result.tie_correction,
        "total_n": result.total_n,
        "k_groups": result.group_count,
    }


def main() -> None:
    print("Kruskal-Wallis MVP (MATH-0278)")
    print("=" * 72)

    rng = np.random.default_rng(278)
    case_data = _build_likert_groups(rng)

    result_rows: list[dict[str, float | int | bool | str]] = []

    for case_name, groups in case_data.items():
        print(f"\nCase: {case_name}")
        summary_df = _group_summary(case_name, groups)
        print(summary_df.to_string(index=False))

        result = kruskal_wallis_manual(groups, case_name=case_name, alpha=0.05)
        compare_with_scipy(groups, result)
        result_rows.append(_result_row(result))

    print("\nKruskal-Wallis test results")
    result_df = pd.DataFrame(result_rows)
    print(result_df.to_string(index=False))

    print("\nSciPy consistency check: passed")


if __name__ == "__main__":
    main()
