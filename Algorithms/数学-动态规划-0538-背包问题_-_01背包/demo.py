"""01 knapsack MVP: 2D DP with reconstruction + 1D DP/bruteforce checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class KnapsackResult:
    best_value: int
    selected_indices: List[int]
    total_weight: int


def to_nonnegative_int_1d(name: str, data: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}.")
    if arr.size == 0:
        return arr.astype(np.int64)

    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError(f"{name} must contain integer values.")

    arr_i64 = arr.astype(np.int64)
    if np.any(arr_i64 < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return arr_i64


def validate_knapsack_inputs(
    weights: Sequence[int] | np.ndarray,
    values: Sequence[int] | np.ndarray,
    capacity: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    w = to_nonnegative_int_1d("weights", weights)
    v = to_nonnegative_int_1d("values", values)

    if w.size != v.size:
        raise ValueError(f"weights and values must have equal length, got {w.size} and {v.size}.")

    if isinstance(capacity, float) and not float(capacity).is_integer():
        raise ValueError("capacity must be an integer.")

    cap = int(capacity)
    if cap < 0:
        raise ValueError("capacity must be nonnegative.")

    return w, v, cap


def knapsack_01_dp_reconstruct(
    weights: Sequence[int] | np.ndarray,
    values: Sequence[int] | np.ndarray,
    capacity: int,
) -> KnapsackResult:
    """Solve 0/1 knapsack by 2D DP and recover one optimal item set."""
    w, v, cap = validate_knapsack_inputs(weights, values, capacity)
    n = int(w.size)

    dp = np.zeros((n + 1, cap + 1), dtype=np.int64)
    keep = np.zeros((n + 1, cap + 1), dtype=bool)

    for i in range(1, n + 1):
        wi = int(w[i - 1])
        vi = int(v[i - 1])
        for c in range(cap + 1):
            not_take = int(dp[i - 1, c])
            best = not_take

            if wi <= c:
                take = int(dp[i - 1, c - wi]) + vi
                if take > best:
                    best = take
                    keep[i, c] = True

            dp[i, c] = best

    selected_rev: List[int] = []
    c = cap
    for i in range(n, 0, -1):
        if keep[i, c]:
            idx = i - 1
            selected_rev.append(idx)
            c -= int(w[idx])

    selected_indices = list(reversed(selected_rev))
    total_weight = int(w[selected_indices].sum()) if selected_indices else 0
    total_value = int(v[selected_indices].sum()) if selected_indices else 0
    best_value = int(dp[n, cap])

    if total_weight > cap:
        raise AssertionError("Recovered solution exceeds capacity.")
    if total_value != best_value:
        raise AssertionError("Recovered solution value does not match DP optimum.")

    return KnapsackResult(
        best_value=best_value,
        selected_indices=selected_indices,
        total_weight=total_weight,
    )


def knapsack_01_dp_value_1d(
    weights: Sequence[int] | np.ndarray,
    values: Sequence[int] | np.ndarray,
    capacity: int,
) -> int:
    """Space-optimized 0/1 knapsack DP that returns optimal value only."""
    w, v, cap = validate_knapsack_inputs(weights, values, capacity)
    n = int(w.size)

    dp = np.zeros(cap + 1, dtype=np.int64)

    for i in range(n):
        wi = int(w[i])
        vi = int(v[i])
        if wi > cap:
            continue
        for c in range(cap, wi - 1, -1):
            cand = int(dp[c - wi]) + vi
            if cand > int(dp[c]):
                dp[c] = cand

    return int(dp[cap])


def knapsack_bruteforce_small(
    weights: Sequence[int] | np.ndarray,
    values: Sequence[int] | np.ndarray,
    capacity: int,
    max_n: int = 22,
) -> KnapsackResult:
    """Exhaustive solver for small n, used as a correctness oracle in demo."""
    w, v, cap = validate_knapsack_inputs(weights, values, capacity)
    n = int(w.size)
    if n > max_n:
        raise ValueError(f"Bruteforce supports n <= {max_n}, got n={n}.")

    best_value = 0
    best_indices: List[int] = []
    best_weight = 0

    for mask in range(1 << n):
        cur_weight = 0
        cur_value = 0
        cur_indices: List[int] = []

        for i in range(n):
            if (mask >> i) & 1:
                cur_weight += int(w[i])
                if cur_weight > cap:
                    break
                cur_value += int(v[i])
                cur_indices.append(i)
        else:
            if cur_value > best_value:
                best_value = cur_value
                best_indices = cur_indices
                best_weight = cur_weight

    return KnapsackResult(
        best_value=best_value,
        selected_indices=best_indices,
        total_weight=best_weight,
    )


def run_case(name: str, weights: Sequence[int], values: Sequence[int], capacity: int) -> None:
    result_2d = knapsack_01_dp_reconstruct(weights, values, capacity)
    result_1d_value = knapsack_01_dp_value_1d(weights, values, capacity)

    w = np.asarray(weights, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)

    chosen = result_2d.selected_indices
    chosen_weight = int(w[chosen].sum()) if chosen else 0
    chosen_value = int(v[chosen].sum()) if chosen else 0

    weight_ok = chosen_weight <= int(capacity)
    value_ok = chosen_value == result_2d.best_value
    crosscheck_1d = result_2d.best_value == result_1d_value

    bruteforce_ok = True
    if len(weights) <= 20:
        brute = knapsack_bruteforce_small(weights, values, capacity)
        bruteforce_ok = brute.best_value == result_2d.best_value

    print(f"=== {name} ===")
    print(f"Weights : {list(weights)}")
    print(f"Values  : {list(values)}")
    print(f"Capacity: {capacity}")
    print(
        "DP 2D result: "
        f"best_value={result_2d.best_value}, "
        f"selected_indices={result_2d.selected_indices}, "
        f"total_weight={result_2d.total_weight}"
    )
    print(f"DP 1D value : {result_1d_value}")
    print(
        "Checks: "
        f"weight_ok={weight_ok}, "
        f"value_ok={value_ok}, "
        f"crosscheck_1d={crosscheck_1d}, "
        f"bruteforce_ok={bruteforce_ok}\n"
    )

    if not weight_ok:
        raise AssertionError(f"Case '{name}': selected weight exceeds capacity.")
    if not value_ok:
        raise AssertionError(f"Case '{name}': selected value mismatches best value.")
    if not crosscheck_1d:
        raise AssertionError(f"Case '{name}': 2D DP and 1D DP values mismatch.")
    if not bruteforce_ok:
        raise AssertionError(f"Case '{name}': DP value mismatches bruteforce oracle.")


def main() -> None:
    cases = [
        (
            "Case 1: classic",
            [2, 1, 3, 2],
            [12, 10, 20, 15],
            5,
        ),
        (
            "Case 2: zero capacity",
            [2, 3, 4],
            [4, 5, 6],
            0,
        ),
        (
            "Case 3: all overweight",
            [5, 6, 7],
            [10, 11, 13],
            4,
        ),
        (
            "Case 4: medium set",
            [3, 4, 7, 8, 9],
            [4, 5, 10, 11, 13],
            17,
        ),
        (
            "Case 5: with zero-weight item",
            [0, 2, 3, 4],
            [3, 4, 5, 8],
            5,
        ),
        (
            "Case 6: denser choices",
            [1, 2, 5, 6, 7],
            [1, 6, 18, 22, 28],
            11,
        ),
    ]

    for name, weights, values, capacity in cases:
        run_case(name, weights, values, capacity)


if __name__ == "__main__":
    main()
