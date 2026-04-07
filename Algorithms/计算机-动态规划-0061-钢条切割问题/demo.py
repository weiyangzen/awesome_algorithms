"""钢条切割问题 MVP：自底向上动态规划 + 记忆化递归校验。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class RodCutResult:
    length: int
    max_revenue: float
    cuts: list[int]


def to_price_array(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"prices must be a 1D sequence, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("prices must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("prices contains non-finite values")
    return arr


def rod_cut_bottom_up(
    prices: Sequence[float] | np.ndarray,
    n: int,
    cut_cost: float = 0.0,
) -> RodCutResult:
    """返回长度 n 的最优收益及一种切割方案。

    prices[i-1] 表示长度 i 的整段售价。
    cut_cost 表示每进行一次实际切割（即分成两段）要支付的成本。
    """
    price_arr = to_price_array(prices)

    if n < 0:
        raise ValueError("n must be non-negative")
    if n > int(price_arr.size):
        raise ValueError("n cannot exceed len(prices) in this MVP")

    best = np.full(n + 1, -np.inf, dtype=float)
    first_cut = np.zeros(n + 1, dtype=int)
    best[0] = 0.0

    for length in range(1, n + 1):
        best_value = -np.inf
        best_cut = 0
        for cut in range(1, length + 1):
            remain = length - cut
            candidate = float(price_arr[cut - 1]) + float(best[remain])
            if remain > 0:
                candidate -= float(cut_cost)
            if candidate > best_value:
                best_value = candidate
                best_cut = cut
        best[length] = best_value
        first_cut[length] = best_cut

    cuts: list[int] = []
    remain = n
    while remain > 0:
        cut = int(first_cut[remain])
        if cut <= 0:
            raise RuntimeError("failed to reconstruct cut plan")
        cuts.append(cut)
        remain -= cut

    return RodCutResult(length=n, max_revenue=float(best[n]), cuts=cuts)


def rod_cut_top_down_revenue(
    prices: Sequence[float] | np.ndarray,
    n: int,
    cut_cost: float = 0.0,
) -> float:
    """记忆化递归基线，只返回最优收益，用于交叉校验。"""
    price_arr = to_price_array(prices)

    if n < 0:
        raise ValueError("n must be non-negative")
    if n > int(price_arr.size):
        raise ValueError("n cannot exceed len(prices) in this MVP")

    @lru_cache(maxsize=None)
    def solve(length: int) -> float:
        if length == 0:
            return 0.0

        ans = -np.inf
        for cut in range(1, length + 1):
            remain = length - cut
            candidate = float(price_arr[cut - 1]) + solve(remain)
            if remain > 0:
                candidate -= float(cut_cost)
            if candidate > ans:
                ans = candidate
        return float(ans)

    return solve(n)


def revenue_from_cuts(
    prices: Sequence[float] | np.ndarray,
    cuts: Sequence[int],
    cut_cost: float,
) -> float:
    """根据切割方案回算收益，作为结果一致性检查。"""
    price_arr = to_price_array(prices)
    total = 0.0
    for c in cuts:
        if c <= 0 or c > int(price_arr.size):
            raise ValueError(f"invalid cut length: {c}")
        total += float(price_arr[c - 1])
    if len(cuts) >= 2:
        total -= float(cut_cost) * float(len(cuts) - 1)
    return total


def run_case(
    name: str,
    prices: Sequence[float],
    n: int,
    cut_cost: float,
    expected_revenue: float | None = None,
) -> None:
    result = rod_cut_bottom_up(prices, n=n, cut_cost=cut_cost)
    baseline = rod_cut_top_down_revenue(prices, n=n, cut_cost=cut_cost)
    reconstructed = revenue_from_cuts(prices, result.cuts, cut_cost)

    print(f"=== {name} ===")
    print(f"prices(1..m) = {list(prices)}")
    print(f"n={n}, cut_cost={cut_cost}")
    print(
        "bottom-up => "
        f"max_revenue={result.max_revenue:.2f}, cuts={result.cuts}, "
        f"sum(cuts)={sum(result.cuts)}"
    )
    print(
        "cross-check => "
        f"top-down={baseline:.2f}, reconstructed={reconstructed:.2f}\n"
    )

    if abs(result.max_revenue - baseline) > 1e-9:
        raise AssertionError("bottom-up and top-down revenue mismatch")
    if abs(result.max_revenue - reconstructed) > 1e-9:
        raise AssertionError("reported revenue and reconstructed revenue mismatch")
    if sum(result.cuts) != n:
        raise AssertionError("cut lengths do not sum to n")
    if expected_revenue is not None and abs(result.max_revenue - expected_revenue) > 1e-9:
        raise AssertionError(
            f"unexpected revenue: got {result.max_revenue}, expected {expected_revenue}"
        )


def randomized_cross_check(
    trials: int = 200,
    max_length: int = 10,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        m = int(rng.integers(1, max_length + 1))
        prices = rng.integers(0, 31, size=m).astype(float)
        n = int(rng.integers(1, m + 1))
        cut_cost = float(rng.integers(0, 4))

        result = rod_cut_bottom_up(prices, n=n, cut_cost=cut_cost)
        baseline = rod_cut_top_down_revenue(prices, n=n, cut_cost=cut_cost)
        reconstructed = revenue_from_cuts(prices, result.cuts, cut_cost)

        if abs(result.max_revenue - baseline) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs top-down mismatch")
        if abs(result.max_revenue - reconstructed) > 1e-9:
            raise AssertionError("random check failed: revenue reconstruction mismatch")
        if sum(result.cuts) != n:
            raise AssertionError("random check failed: invalid cut plan length")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_length={max_length}, seed={seed})."
    )


def main() -> None:
    canonical_prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]

    run_case(
        name="Case 1: textbook (no cut cost)",
        prices=canonical_prices,
        n=8,
        cut_cost=0.0,
        expected_revenue=22.0,
    )

    run_case(
        name="Case 2: textbook (with cut cost = 2)",
        prices=canonical_prices,
        n=8,
        cut_cost=2.0,
        expected_revenue=20.0,
    )

    run_case(
        name="Case 3: non-monotonic table",
        prices=[2, 5, 7, 8, 9, 10, 17, 17],
        n=8,
        cut_cost=1.0,
    )

    randomized_cross_check(trials=200, max_length=10, seed=2026)


if __name__ == "__main__":
    main()
