"""Runnable MVP for Median of Medians (BFPRT) selection algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class BFPRTStats:
    """Execution statistics for deterministic selection."""

    recursive_calls: int = 0
    partition_rounds: int = 0
    max_depth: int = 0
    trace: list[dict[str, int | float]] = field(default_factory=list)


def validate_numeric_sequence(values: Sequence[float]) -> list[float]:
    """Validate input and return a finite 1D float list."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Input must be a numeric sequence, not string/bytes.")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D sequence.")
    if arr.size == 0:
        raise ValueError("Input must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values (NaN or Inf).")

    return arr.tolist()


def _select_bfprt(
    arr: list[float],
    k: int,
    stats: BFPRTStats,
    depth: int,
    group_size: int,
) -> float:
    """Return the k-th smallest value from arr using BFPRT (0-based k)."""
    stats.recursive_calls += 1
    stats.max_depth = max(stats.max_depth, depth)

    n = len(arr)
    if n <= group_size:
        return sorted(arr)[k]

    groups = [arr[i : i + group_size] for i in range(0, n, group_size)]
    medians = [sorted(g)[len(g) // 2] for g in groups]

    pivot = _select_bfprt(
        medians,
        len(medians) // 2,
        stats=stats,
        depth=depth + 1,
        group_size=group_size,
    )

    lows: list[float] = []
    equals: list[float] = []
    highs: list[float] = []
    for x in arr:
        if x < pivot:
            lows.append(x)
        elif x > pivot:
            highs.append(x)
        else:
            equals.append(x)

    stats.partition_rounds += 1
    stats.trace.append(
        {
            "depth": depth,
            "n": n,
            "pivot": pivot,
            "lows": len(lows),
            "equals": len(equals),
            "highs": len(highs),
            "target_k": k,
        }
    )

    if k < len(lows):
        return _select_bfprt(lows, k, stats=stats, depth=depth + 1, group_size=group_size)
    if k < len(lows) + len(equals):
        return pivot
    return _select_bfprt(
        highs,
        k - len(lows) - len(equals),
        stats=stats,
        depth=depth + 1,
        group_size=group_size,
    )


def deterministic_select(
    values: Sequence[float],
    k: int,
    group_size: int = 5,
) -> tuple[float, BFPRTStats]:
    """Select the k-th smallest element with deterministic worst-case O(n)."""
    data = validate_numeric_sequence(values)

    if group_size < 5 or group_size % 2 == 0:
        raise ValueError("group_size must be an odd integer >= 5.")
    if not (0 <= k < len(data)):
        raise IndexError(f"k must be in [0, {len(data) - 1}].")

    stats = BFPRTStats()
    kth = _select_bfprt(data, k, stats=stats, depth=0, group_size=group_size)
    return kth, stats


def run_case(case_name: str, values: Sequence[float], k: int) -> None:
    """Run one deterministic validation case."""
    data = validate_numeric_sequence(values)

    kth, stats = deterministic_select(data, k)
    expected_python = sorted(data)[k]
    expected_numpy = float(np.partition(np.asarray(data, dtype=float), k)[k])

    if kth != expected_python or kth != expected_numpy:
        raise RuntimeError(
            f"{case_name}: mismatch -> bfprt={kth}, py={expected_python}, np={expected_numpy}"
        )

    print(f"\n=== {case_name} ===")
    print(f"n={len(data)}, k={k}")
    print(f"BFPRT result:     {kth}")
    print(f"Python sorted[k]: {expected_python}")
    print(f"NumPy part[k]:    {expected_numpy}")
    print(
        "Stats: "
        f"recursive_calls={stats.recursive_calls}, "
        f"partition_rounds={stats.partition_rounds}, "
        f"max_depth={stats.max_depth}"
    )

    if stats.trace:
        trace_df = pd.DataFrame(stats.trace)
        print("Partition trace (first 12 rows):")
        print(trace_df.head(12).to_string(index=False))


def main() -> None:
    """Execute non-interactive BFPRT demos and checks."""
    fixed = [12, 7, 3, 19, 5, 8, 8, 42, -1, 13, 0, 11, 6]
    run_case("Case 1: fixed mixed integers", fixed, k=6)

    rng = np.random.default_rng(seed=2026)
    random_case = rng.integers(low=-500, high=501, size=101).tolist()
    run_case("Case 2: seeded random integers", random_case, k=50)

    repeated_values = [5, 5, 5, 1, 1, 9, 9, 9, 2, 2, 2, 7, 7]
    run_case("Case 3: many duplicates", repeated_values, k=8)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
