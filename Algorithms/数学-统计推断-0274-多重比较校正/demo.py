"""多重比较校正的最小可运行示例。

本示例覆盖三种常见方法：
1) Bonferroni (FWER)
2) Holm-Bonferroni (FWER, step-down)
3) Benjamini-Hochberg (FDR)

运行方式：
    uv run python Algorithms/数学-统计推断-0274-多重比较校正/demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrectionResult:
    """存放一种校正方法的输出。"""

    adjusted_pvalues: np.ndarray
    rejected: np.ndarray


def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni 校正。

    调整后 p 值: p_i^adj = min(m * p_i, 1)
    判定规则: p_i <= alpha / m
    """
    m = pvalues.size
    adjusted = np.minimum(pvalues * m, 1.0)
    rejected = pvalues <= (alpha / m)
    return CorrectionResult(adjusted_pvalues=adjusted, rejected=rejected)


def holm_bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Holm-Bonferroni step-down 校正。

    判定：将 p 值升序排列为 p_(1) <= ... <= p_(m)，
    从最小值开始比较 p_(i) <= alpha / (m - i + 1)，
    一旦某一步不通过，后续全部不拒绝。

    调整后 p 值（按原顺序返回）通过 Holm 公式构造并保持单调性。
    """
    m = pvalues.size
    order = np.argsort(pvalues)
    p_sorted = pvalues[order]

    rejected_sorted = np.zeros(m, dtype=bool)
    for i, p in enumerate(p_sorted):
        threshold = alpha / (m - i)
        if p <= threshold:
            rejected_sorted[i] = True
        else:
            break

    raw_adj = (m - np.arange(m)) * p_sorted
    adj_sorted = np.maximum.accumulate(raw_adj)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    adjusted = adj_sorted[inv]
    rejected = rejected_sorted[inv]

    return CorrectionResult(adjusted_pvalues=adjusted, rejected=rejected)


def benjamini_hochberg_correction(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg (BH) FDR 校正。

    设升序 p_(1) <= ... <= p_(m)，找到最大的 k 满足：
        p_(k) <= (k / m) * alpha
    则前 k 个拒绝。

    调整后 p 值（q-value）：
        q_(i) = min_{j>=i} (m / j) * p_(j)
    """
    m = pvalues.size
    order = np.argsort(pvalues)
    p_sorted = pvalues[order]

    ranks = np.arange(1, m + 1)
    crit = (ranks / m) * alpha
    passed = p_sorted <= crit

    rejected_sorted = np.zeros(m, dtype=bool)
    if np.any(passed):
        k = np.max(np.where(passed)[0])
        rejected_sorted[: k + 1] = True

    raw_adj = (m / ranks) * p_sorted
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    adjusted = adj_sorted[inv]
    rejected = rejected_sorted[inv]

    return CorrectionResult(adjusted_pvalues=adjusted, rejected=rejected)


def simulate_multiple_tests(
    m: int = 24,
    n_per_group: int = 40,
    effect_indices: tuple[int, ...] = (1, 5, 9, 17),
    effect_size: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """构造 m 个双样本检验，返回原始 p 值及真值标签。

    - H0_true=True 表示该假设真实为零效应。
    - H0_true=False 表示该假设存在真实差异。
    """
    rng = np.random.default_rng(seed)

    pvals = []
    h0_true = []

    for i in range(m):
        control = rng.normal(loc=0.0, scale=1.0, size=n_per_group)

        if i in effect_indices:
            treatment = rng.normal(loc=effect_size, scale=1.0, size=n_per_group)
            h0_true.append(False)
        else:
            treatment = rng.normal(loc=0.0, scale=1.0, size=n_per_group)
            h0_true.append(True)

        _, p = stats.ttest_ind(control, treatment, equal_var=False)
        pvals.append(p)

    return pd.DataFrame(
        {
            "hypothesis": [f"H{i:02d}" for i in range(m)],
            "pvalue": np.array(pvals),
            "H0_true": np.array(h0_true),
        }
    )


def attach_corrections(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """对 DataFrame 添加三种校正结果。"""
    pvalues = df["pvalue"].to_numpy()

    bonf = bonferroni_correction(pvalues, alpha=alpha)
    holm = holm_bonferroni_correction(pvalues, alpha=alpha)
    bh = benjamini_hochberg_correction(pvalues, alpha=alpha)

    out = df.copy()
    out["p_bonf"] = bonf.adjusted_pvalues
    out["rej_bonf"] = bonf.rejected

    out["p_holm"] = holm.adjusted_pvalues
    out["rej_holm"] = holm.rejected

    out["p_bh"] = bh.adjusted_pvalues
    out["rej_bh"] = bh.rejected
    return out


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    """统计每种方法的检出数量与误拒数量。"""
    rows = []
    methods = ["bonf", "holm", "bh"]
    for method in methods:
        rej_col = f"rej_{method}"
        rejected_total = int(result[rej_col].sum())

        false_reject = int(((result[rej_col]) & (result["H0_true"])).sum())
        true_reject = int(((result[rej_col]) & (~result["H0_true"])).sum())

        rows.append(
            {
                "method": method,
                "rejected_total": rejected_total,
                "true_discoveries": true_reject,
                "false_discoveries": false_reject,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    alpha = 0.05
    df = simulate_multiple_tests()
    result = attach_corrections(df, alpha=alpha)

    show_cols = [
        "hypothesis",
        "H0_true",
        "pvalue",
        "p_bonf",
        "rej_bonf",
        "p_holm",
        "rej_holm",
        "p_bh",
        "rej_bh",
    ]

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== 多重比较校正 Demo ===")
    print(f"alpha = {alpha}")
    print()

    print("[最小原始 p 值的前 12 个假设]")
    print(result.sort_values("pvalue").head(12)[show_cols].to_string(index=False))
    print()

    print("[方法级统计]")
    print(summarize(result).to_string(index=False))


if __name__ == "__main__":
    main()
