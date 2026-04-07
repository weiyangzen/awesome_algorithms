"""Divide-and-conquer MVP for 'Different Ways to Add Parentheses'."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class DivideConquerStats:
    recursive_calls: int = 0
    memo_hits: int = 0
    combine_ops: int = 0


def tokenize_expression(expression: str) -> tuple[list[int], list[str]]:
    """Parse an expression into numbers and operators.

    Supported grammar: non-negative integer (op non-negative integer)*,
    where op is one of '+', '-', '*'. Spaces are ignored.
    """
    compact = expression.replace(" ", "")
    if not compact:
        raise ValueError("Expression cannot be empty.")

    nums: list[int] = []
    ops: list[str] = []
    i = 0
    n = len(compact)

    while i < n:
        ch = compact[i]
        if ch.isdigit():
            j = i
            while j < n and compact[j].isdigit():
                j += 1
            nums.append(int(compact[i:j]))
            i = j
            continue
        if ch in "+-*":
            if i == 0 or i == n - 1:
                raise ValueError(f"Invalid operator position in expression: {expression!r}")
            ops.append(ch)
            i += 1
            continue
        raise ValueError(f"Unsupported token {ch!r} in expression: {expression!r}")

    if len(nums) != len(ops) + 1:
        raise ValueError(f"Malformed expression: {expression!r}")

    return nums, ops


def apply_op(left: int, right: int, op: str) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(f"Unsupported operator: {op!r}")


def different_ways_to_compute(expression: str) -> tuple[list[int], DivideConquerStats]:
    """Return all results from every valid parenthesization (with duplicates)."""
    nums, ops = tokenize_expression(expression)
    stats = DivideConquerStats()
    memo: dict[tuple[int, int], list[int]] = {}

    def solve(i: int, j: int) -> list[int]:
        stats.recursive_calls += 1
        key = (i, j)
        if key in memo:
            stats.memo_hits += 1
            return memo[key]

        if i == j:
            result = [nums[i]]
            memo[key] = result
            return result

        result: list[int] = []
        for k in range(i, j):
            left_values = solve(i, k)
            right_values = solve(k + 1, j)
            op = ops[k]
            for lv in left_values:
                for rv in right_values:
                    stats.combine_ops += 1
                    result.append(apply_op(lv, rv, op))

        memo[key] = result
        return result

    return solve(0, len(nums) - 1), stats


def different_ways_to_compute_naive(expression: str) -> list[int]:
    """Reference implementation without memoization (for small-size validation)."""
    nums, ops = tokenize_expression(expression)

    def solve(i: int, j: int) -> list[int]:
        if i == j:
            return [nums[i]]
        result: list[int] = []
        for k in range(i, j):
            left_values = solve(i, k)
            right_values = solve(k + 1, j)
            op = ops[k]
            for lv in left_values:
                for rv in right_values:
                    result.append(apply_op(lv, rv, op))
        return result

    return solve(0, len(nums) - 1)


def benchmark_once(fn: Callable[[str], list[int] | tuple[list[int], DivideConquerStats]], expr: str) -> tuple[float, list[int]]:
    start = time.perf_counter()
    out = fn(expr)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if isinstance(out, tuple):
        results = out[0]
    else:
        results = out
    return elapsed_ms, results


def generate_random_expression(num_count: int, rng: random.Random) -> str:
    if num_count < 1:
        raise ValueError("num_count must be positive")
    numbers = [str(rng.randint(0, 9)) for _ in range(num_count)]
    operators = [rng.choice(["+", "-", "*"]) for _ in range(num_count - 1)]
    parts: list[str] = [numbers[0]]
    for i, op in enumerate(operators, start=1):
        parts.append(op)
        parts.append(numbers[i])
    return "".join(parts)


def summarize_results(values: list[int]) -> str:
    arr = np.array(values, dtype=np.int64)
    return (
        f"count={arr.size}, min={arr.min()}, max={arr.max()}, "
        f"mean={arr.mean():.3f}, std={arr.std():.3f}"
    )


def run_fixed_cases() -> None:
    print("=== Fixed Cases ===")
    cases = {
        "2-1-1": [0, 2],
        "2*3-4*5": [-34, -14, -10, -10, 10],
        "11": [11],
        "2*3+4": [10, 14],
    }

    for expr, expected in cases.items():
        got, stats = different_ways_to_compute(expr)
        assert sorted(got) == sorted(expected), (
            f"Fixed case failed for {expr!r}: got={sorted(got)}, expected={sorted(expected)}"
        )
        print(f"expr={expr!r}")
        print(f"  results(sorted) = {sorted(got)}")
        print(f"  stats: calls={stats.recursive_calls}, memo_hits={stats.memo_hits}, combine_ops={stats.combine_ops}")
        print(f"  summary: {summarize_results(got)}")


def run_random_regression(seed: int = 7, trials: int = 12) -> None:
    print("\n=== Random Regression (memo vs naive) ===")
    rng = random.Random(seed)
    for t in range(1, trials + 1):
        num_count = rng.randint(3, 7)
        expr = generate_random_expression(num_count, rng)
        fast, _ = different_ways_to_compute(expr)
        slow = different_ways_to_compute_naive(expr)
        assert sorted(fast) == sorted(slow), (
            f"Random regression mismatch for {expr!r}:\n"
            f"memo={sorted(fast)}\nnaive={sorted(slow)}"
        )
        print(f"trial={t:02d}, num_count={num_count}, expr={expr!r}, result_count={len(fast)}")


def run_benchmark(seed: int = 11) -> None:
    print("\n=== Micro Benchmark (ms) ===")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for num_count in [4, 5, 6, 7]:
        expr = generate_random_expression(num_count, rng)
        memo_ms, memo_results = benchmark_once(different_ways_to_compute, expr)
        naive_ms, naive_results = benchmark_once(different_ways_to_compute_naive, expr)
        assert sorted(memo_results) == sorted(naive_results)

        rows.append(
            {
                "num_count": num_count,
                "expression": expr,
                "result_count": len(memo_results),
                "memo_ms": round(memo_ms, 3),
                "naive_ms": round(naive_ms, 3),
                "speedup(naive/memo)": round(naive_ms / max(memo_ms, 1e-9), 2),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main() -> None:
    run_fixed_cases()
    run_random_regression()
    run_benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
