"""Divide-and-conquer MVP for counting unique BSTs (Catalan numbers)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import time
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class DivideConquerStats:
    recursive_calls: int = 0
    memo_hits: int = 0
    split_evaluations: int = 0
    states_computed: int = 0


def validate_n(n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")


def num_trees_divide_conquer(n: int) -> tuple[int, DivideConquerStats]:
    """Top-down divide-and-conquer with memoization.

    Recurrence:
        f(0) = 1, f(1) = 1
        f(n) = sum(f(left) * f(right)) for left in [0, n-1], right=n-1-left
    """
    validate_n(n)

    memo: dict[int, int] = {0: 1, 1: 1}
    stats = DivideConquerStats()

    def solve(nodes: int) -> int:
        stats.recursive_calls += 1
        if nodes in memo:
            stats.memo_hits += 1
            return memo[nodes]

        stats.states_computed += 1
        total = 0
        for left_nodes in range(nodes):
            right_nodes = nodes - 1 - left_nodes
            stats.split_evaluations += 1
            total += solve(left_nodes) * solve(right_nodes)

        memo[nodes] = total
        return total

    return solve(n), stats


def num_trees_dp(n: int) -> int:
    """Bottom-up dynamic programming with the same recurrence."""
    validate_n(n)
    dp = [0] * (n + 1)
    dp[0] = 1

    for nodes in range(1, n + 1):
        total = 0
        for left_nodes in range(nodes):
            right_nodes = nodes - 1 - left_nodes
            total += dp[left_nodes] * dp[right_nodes]
        dp[nodes] = total

    return dp[n]


def num_trees_catalan_closed_form(n: int) -> int:
    """Catalan closed form: C_n = comb(2n, n) // (n + 1)."""
    validate_n(n)
    return comb(2 * n, n) // (n + 1)


def benchmark_once(fn: Callable[[int], int | tuple[int, DivideConquerStats]], n: int) -> tuple[float, int, DivideConquerStats | None]:
    start = time.perf_counter()
    out = fn(n)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if isinstance(out, tuple):
        value, stats = out
        return elapsed_ms, value, stats
    return elapsed_ms, out, None


def run_fixed_cases() -> None:
    print("=== Fixed Cases ===")
    expected = {
        0: 1,
        1: 1,
        2: 2,
        3: 5,
        4: 14,
        5: 42,
        6: 132,
        7: 429,
        8: 1430,
        9: 4862,
        10: 16796,
    }

    for n, ans in expected.items():
        got, stats = num_trees_divide_conquer(n)
        assert got == ans, f"Mismatch at n={n}: got={got}, expected={ans}"
        print(
            f"n={n:2d}, count={got:6d}, "
            f"calls={stats.recursive_calls:3d}, memo_hits={stats.memo_hits:3d}, "
            f"splits={stats.split_evaluations:3d}"
        )


def run_consistency_regression(max_n: int = 19) -> None:
    print("\n=== Consistency Regression (divide-conquer vs dp vs closed-form) ===")
    for n in range(max_n + 1):
        dc_value, _ = num_trees_divide_conquer(n)
        dp_value = num_trees_dp(n)
        closed_value = num_trees_catalan_closed_form(n)
        assert dc_value == dp_value == closed_value, (
            f"Mismatch at n={n}: divide={dc_value}, dp={dp_value}, closed={closed_value}"
        )
        print(f"n={n:2d}, count={dc_value}")


def run_sequence_summary(max_n: int = 12) -> None:
    print("\n=== Sequence Summary ===")
    counts = [num_trees_dp(n) for n in range(max_n + 1)]
    growth = np.array([np.nan] + [counts[i] / counts[i - 1] for i in range(1, len(counts))], dtype=float)

    df = pd.DataFrame(
        {
            "n": list(range(max_n + 1)),
            "count": counts,
            "ratio_to_prev": np.round(growth, 4),
        }
    )
    print(df.to_string(index=False))
    mean_growth = float(np.nanmean(growth[2:]))
    print(f"mean ratio (n>=2): {mean_growth:.4f}")


def run_benchmark() -> None:
    print("\n=== Micro Benchmark (ms) ===")
    rows: list[dict[str, object]] = []

    for n in [8, 12, 15, 18]:
        dc_ms, dc_value, dc_stats = benchmark_once(num_trees_divide_conquer, n)
        dp_ms, dp_value, _ = benchmark_once(num_trees_dp, n)
        closed_ms, closed_value, _ = benchmark_once(num_trees_catalan_closed_form, n)

        assert dc_value == dp_value == closed_value
        assert dc_stats is not None

        rows.append(
            {
                "n": n,
                "count": dc_value,
                "divide_ms": round(dc_ms, 4),
                "dp_ms": round(dp_ms, 4),
                "closed_ms": round(closed_ms, 4),
                "dc_calls": dc_stats.recursive_calls,
                "dc_memo_hits": dc_stats.memo_hits,
                "dc_splits": dc_stats.split_evaluations,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main() -> None:
    run_fixed_cases()
    run_consistency_regression()
    run_sequence_summary()
    run_benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
