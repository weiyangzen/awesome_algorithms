"""Wilcoxon signed-rank test MVP.

This script provides a small, transparent implementation of the Wilcoxon
signed-rank workflow and compares it against scipy.stats.wilcoxon.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

try:
    from scipy.stats import norm as scipy_norm
    from scipy.stats import rankdata as scipy_rankdata
    from scipy.stats import wilcoxon as scipy_wilcoxon

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


def _prepare_paired_samples(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean paired samples.

    NaN pairs are removed row-wise to keep the example robust for real data.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}.")

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if x_clean.size == 0:
        raise ValueError("No valid paired samples remain after removing NaN.")

    return x_clean, y_clean


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average-tie rankdata implemented with NumPy only."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def _normal_sf(z: float) -> float:
    """Survival function of standard normal distribution."""
    if HAS_SCIPY:
        return float(scipy_norm.sf(z))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank_manual(
    x: np.ndarray,
    y: np.ndarray,
    zero_method: str = "wilcox",
    continuity: bool = True,
) -> Dict[str, Any]:
    """Manual Wilcoxon signed-rank test (two-sided, normal approximation).

    The implementation follows the textbook flow:
    1) paired differences,
    2) remove zero differences (wilcox rule),
    3) rank absolute differences,
    4) sum positive/negative ranks,
    5) compute z and approximate p-value.
    """
    if zero_method != "wilcox":
        raise NotImplementedError("This MVP currently supports zero_method='wilcox' only.")

    x_clean, y_clean = _prepare_paired_samples(x, y)
    diff = x_clean - y_clean

    if zero_method == "wilcox":
        diff_nz = diff[diff != 0.0]
    else:
        diff_nz = diff

    m = diff_nz.size
    if m == 0:
        raise ValueError("All paired differences are zero; Wilcoxon statistic is undefined.")

    abs_diff = np.abs(diff_nz)
    if HAS_SCIPY:
        ranks = scipy_rankdata(abs_diff, method="average")
    else:
        ranks = _rankdata_average(abs_diff)

    w_plus = float(ranks[diff_nz > 0.0].sum())
    w_minus = float(ranks[diff_nz < 0.0].sum())
    t_stat = float(min(w_plus, w_minus))

    # Mean and variance of W+ under H0 with tie correction.
    mu = m * (m + 1.0) / 4.0
    _, counts = np.unique(abs_diff, return_counts=True)
    tie_term = float(np.sum(counts * (counts + 1.0) * (2.0 * counts + 1.0)))
    var_w = (m * (m + 1.0) * (2.0 * m + 1.0) - tie_term) / 24.0

    if var_w <= 0.0:
        z_value = float("nan")
        p_approx = float("nan")
    else:
        cc = 0.0
        if continuity:
            if w_plus > mu:
                cc = 0.5
            elif w_plus < mu:
                cc = -0.5

        z_value = (w_plus - mu - cc) / np.sqrt(var_w)
        p_approx = float(2.0 * _normal_sf(abs(z_value)))

    return {
        "n_total": int(diff.size),
        "n_effective": int(m),
        "diff": diff,
        "diff_nonzero": diff_nz,
        "ranks": ranks,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "t_stat": t_stat,
        "mu_w_plus": float(mu),
        "var_w_plus": float(var_w),
        "z_approx": float(z_value),
        "p_approx": float(p_approx),
    }


def main() -> None:
    # Deterministic paired samples with ties and zero differences.
    # Interpretation: e.g., symptom score before/after intervention.
    x = np.array([15, 14, 10, 8, 12, 11, 9, 13, 7, 10, 12, 8], dtype=float)
    y = np.array([13, 14, 11, 7, 11, 10, 8, 11, 8, 10, 11, 7], dtype=float)

    manual = wilcoxon_signed_rank_manual(x, y, zero_method="wilcox", continuity=True)

    print("=== Wilcoxon符号秩检验 Demo ===")
    print(f"原始样本对数 n_total = {manual['n_total']}")
    print(f"有效非零差值数 n_effective = {manual['n_effective']}")
    print(f"差值 diff = {manual['diff']}")
    print(f"非零差值 diff_nonzero = {manual['diff_nonzero']}")
    print()

    print("[Manual implementation]")
    print(f"W+ = {manual['w_plus']:.6f}")
    print(f"W- = {manual['w_minus']:.6f}")
    print(f"T = min(W+, W-) = {manual['t_stat']:.6f}")
    print(f"mu(W+) = {manual['mu_w_plus']:.6f}")
    print(f"var(W+) = {manual['var_w_plus']:.6f}")
    print(f"z (normal approx) = {manual['z_approx']:.6f}")
    print(f"p (normal approx, two-sided) = {manual['p_approx']:.6f}")
    print()

    alpha = 0.05
    if HAS_SCIPY:
        try:
            scipy_res = scipy_wilcoxon(
                x,
                y,
                zero_method="wilcox",
                correction=True,
                alternative="two-sided",
                mode="approx",
            )
        except TypeError:
            # Compatibility fallback for SciPy versions without `mode`.
            scipy_res = scipy_wilcoxon(
                x,
                y,
                zero_method="wilcox",
                correction=True,
                alternative="two-sided",
            )

        print("[SciPy reference]")
        print(f"statistic = {float(scipy_res.statistic):.6f}")
        print(f"pvalue = {float(scipy_res.pvalue):.6f}")
        print()
        p_for_decision = float(scipy_res.pvalue)
    else:
        print("[SciPy reference]")
        print("SciPy 未安装，跳过库函数对照；使用手工近似 p 值给出结论。")
        print()
        p_for_decision = float(manual["p_approx"])

    if p_for_decision < alpha:
        decision = "拒绝 H0（存在显著差异）"
    else:
        decision = "不拒绝 H0（未见显著差异）"
    print(f"结论 (alpha={alpha}): {decision}")


if __name__ == "__main__":
    main()
