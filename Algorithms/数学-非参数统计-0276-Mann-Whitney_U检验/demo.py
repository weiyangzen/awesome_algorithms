"""Mann-Whitney U test MVP.

This script implements a transparent, minimal Mann-Whitney U workflow and
optionally compares results with scipy.stats.mannwhitneyu.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

try:
    from scipy.stats import mannwhitneyu as scipy_mannwhitneyu
    from scipy.stats import norm as scipy_norm
    from scipy.stats import rankdata as scipy_rankdata

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


def _prepare_independent_samples(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean two independent samples.

    NaN values are removed independently in each group.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]

    if x_clean.size == 0 or y_clean.size == 0:
        raise ValueError("Both groups must contain at least one valid observation.")

    return x_clean, y_clean


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average-tie rank implementation using NumPy only."""
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


def mann_whitney_u_manual(
    x: np.ndarray,
    y: np.ndarray,
    continuity: bool = True,
) -> Dict[str, Any]:
    """Manual Mann-Whitney U test (two-sided, normal approximation)."""
    x_clean, y_clean = _prepare_independent_samples(x, y)
    n1 = int(x_clean.size)
    n2 = int(y_clean.size)

    merged = np.concatenate([x_clean, y_clean])
    group = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    if HAS_SCIPY:
        ranks = scipy_rankdata(merged, method="average")
    else:
        ranks = _rankdata_average(merged)

    r1 = float(ranks[group == 0].sum())
    r2 = float(ranks[group == 1].sum())

    u1 = r1 - n1 * (n1 + 1.0) / 2.0
    u2 = r2 - n2 * (n2 + 1.0) / 2.0
    u_stat = float(min(u1, u2))

    n_total = n1 + n2
    mu_u = n1 * n2 / 2.0

    _, counts = np.unique(merged, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    tie_corr = tie_sum / (n_total * (n_total - 1.0)) if n_total > 1 else 0.0
    var_u = (n1 * n2 / 12.0) * ((n_total + 1.0) - tie_corr)

    if var_u <= 0.0:
        z_value = float("nan")
        p_approx = float("nan")
    else:
        cc = 0.0
        if continuity:
            if u1 > mu_u:
                cc = 0.5
            elif u1 < mu_u:
                cc = -0.5

        z_value = (u1 - mu_u - cc) / math.sqrt(var_u)
        p_approx = float(2.0 * _normal_sf(abs(z_value)))

    return {
        "x_clean": x_clean,
        "y_clean": y_clean,
        "n1": n1,
        "n2": n2,
        "ranks": ranks,
        "r1": r1,
        "r2": r2,
        "u1": float(u1),
        "u2": float(u2),
        "u_stat": u_stat,
        "mu_u": float(mu_u),
        "var_u": float(var_u),
        "z_approx": float(z_value),
        "p_approx": float(p_approx),
    }


def main() -> None:
    # Deterministic independent samples.
    # Example interpretation: two independent teaching methods' exam scores.
    x = np.array([88, 91, 79, 85, 90, 84, 87, 92, 86, 89, np.nan], dtype=float)
    y = np.array([76, 82, 80, 78, 83, 81, 77, 79, 84, 75, np.nan], dtype=float)

    manual = mann_whitney_u_manual(x, y, continuity=True)

    print("=== Mann-Whitney U检验 Demo ===")
    print(f"组1样本量 n1 = {manual['n1']}")
    print(f"组2样本量 n2 = {manual['n2']}")
    print(f"组1样本 x = {manual['x_clean']}")
    print(f"组2样本 y = {manual['y_clean']}")
    print()

    print("[Manual implementation]")
    print(f"R1 (group1 rank sum) = {manual['r1']:.6f}")
    print(f"R2 (group2 rank sum) = {manual['r2']:.6f}")
    print(f"U1 = {manual['u1']:.6f}")
    print(f"U2 = {manual['u2']:.6f}")
    print(f"U = min(U1, U2) = {manual['u_stat']:.6f}")
    print(f"mu(U1) = {manual['mu_u']:.6f}")
    print(f"var(U1) tie-corrected = {manual['var_u']:.6f}")
    print(f"z (normal approx) = {manual['z_approx']:.6f}")
    print(f"p (normal approx, two-sided) = {manual['p_approx']:.6f}")
    print()

    alpha = 0.05
    if HAS_SCIPY:
        try:
            scipy_res = scipy_mannwhitneyu(
                manual["x_clean"],
                manual["y_clean"],
                alternative="two-sided",
                use_continuity=True,
                method="asymptotic",
            )
        except TypeError:
            # Compatibility fallback for older SciPy.
            scipy_res = scipy_mannwhitneyu(
                manual["x_clean"],
                manual["y_clean"],
                alternative="two-sided",
                use_continuity=True,
            )

        print("[SciPy reference]")
        print(f"statistic (U1) = {float(scipy_res.statistic):.6f}")
        print(f"pvalue = {float(scipy_res.pvalue):.6f}")
        print()
        p_for_decision = float(scipy_res.pvalue)
    else:
        print("[SciPy reference]")
        print("SciPy 未安装，跳过库函数对照；使用手工近似 p 值给出结论。")
        print()
        p_for_decision = float(manual["p_approx"])

    if p_for_decision < alpha:
        decision = "拒绝 H0（两组分布存在显著差异）"
    else:
        decision = "不拒绝 H0（未见显著差异）"
    direction = "组1整体偏大于组2" if manual["u1"] > manual["mu_u"] else "组1整体偏小于组2"
    print(f"方向解释: {direction}")
    print(f"结论 (alpha={alpha}): {decision}")


if __name__ == "__main__":
    main()
