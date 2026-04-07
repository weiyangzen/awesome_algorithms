"""Quadrangle-inequality optimization (Knuth) MVP for interval merge DP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class MergeDPResult:
    min_cost: float
    dp: np.ndarray
    opt: np.ndarray
    expression: str


def to_1d_nonnegative_array(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"weights must be a 1D array, got shape={arr.shape}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("weights contains non-finite values (nan/inf).")
    if arr.size > 0 and np.any(arr < 0):
        raise ValueError("weights must be non-negative for this merge-cost model.")
    return arr


def build_prefix_sums(arr: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(arr, dtype=float)))


def interval_sum(prefix: np.ndarray, i: int, j: int) -> float:
    return float(prefix[j + 1] - prefix[i])


def reconstruct_expression(opt: np.ndarray, i: int, j: int) -> str:
    if i == j:
        return f"A{i + 1}"
    k = int(opt[i, j])
    left = reconstruct_expression(opt, i, k)
    right = reconstruct_expression(opt, k + 1, j)
    return f"({left}+{right})"


def merge_cost_cubic(weights: Sequence[float] | np.ndarray) -> MergeDPResult:
    """Baseline interval DP: O(n^3)."""
    arr = to_1d_nonnegative_array(weights)
    n = int(arr.size)
    if n == 0:
        return MergeDPResult(0.0, np.zeros((0, 0)), np.zeros((0, 0), dtype=int), "EMPTY")
    if n == 1:
        return MergeDPResult(0.0, np.zeros((1, 1)), np.array([[0]], dtype=int), "A1")

    prefix = build_prefix_sums(arr)
    dp = np.full((n, n), np.inf, dtype=float)
    opt = np.full((n, n), -1, dtype=int)

    for i in range(n):
        dp[i, i] = 0.0
        opt[i, i] = i

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            total = interval_sum(prefix, i, j)
            best_cost = np.inf
            best_k = i
            for k in range(i, j):
                cand = dp[i, k] + dp[k + 1, j] + total
                if cand < best_cost:
                    best_cost = cand
                    best_k = k
            dp[i, j] = best_cost
            opt[i, j] = best_k

    return MergeDPResult(
        min_cost=float(dp[0, n - 1]),
        dp=dp,
        opt=opt,
        expression=reconstruct_expression(opt, 0, n - 1),
    )


def merge_cost_knuth(weights: Sequence[float] | np.ndarray) -> MergeDPResult:
    """Quadrangle-inequality optimization (Knuth): O(n^2)."""
    arr = to_1d_nonnegative_array(weights)
    n = int(arr.size)
    if n == 0:
        return MergeDPResult(0.0, np.zeros((0, 0)), np.zeros((0, 0), dtype=int), "EMPTY")
    if n == 1:
        return MergeDPResult(0.0, np.zeros((1, 1)), np.array([[0]], dtype=int), "A1")

    prefix = build_prefix_sums(arr)
    dp = np.full((n, n), np.inf, dtype=float)
    opt = np.full((n, n), -1, dtype=int)

    for i in range(n):
        dp[i, i] = 0.0
        opt[i, i] = i

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            total = interval_sum(prefix, i, j)

            left_bound = int(opt[i, j - 1])
            right_bound = int(opt[i + 1, j]) if i + 1 <= j else j - 1

            start = max(i, left_bound)
            end = min(j - 1, right_bound)
            if start > end:
                # Robust fallback if bounds are numerically/implementation-wise inconsistent.
                start, end = i, j - 1

            best_cost = np.inf
            best_k = start
            for k in range(start, end + 1):
                cand = dp[i, k] + dp[k + 1, j] + total
                if cand < best_cost:
                    best_cost = cand
                    best_k = k

            dp[i, j] = best_cost
            opt[i, j] = best_k

    return MergeDPResult(
        min_cost=float(dp[0, n - 1]),
        dp=dp,
        opt=opt,
        expression=reconstruct_expression(opt, 0, n - 1),
    )


def verify_opt_monotonicity(opt: np.ndarray) -> bool:
    """Check opt[i][j-1] <= opt[i][j] <= opt[i+1][j] for i < j."""
    n = int(opt.shape[0])
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            left = int(opt[i, j - 1])
            mid = int(opt[i, j])
            right = int(opt[i + 1, j]) if i + 1 <= j else mid
            if not (left <= mid <= right):
                return False
    return True


def check_quadrangle_inequality_for_range_sum(prefix: np.ndarray) -> Tuple[bool, str]:
    """
    Verify w(a,c)+w(b,d) <= w(a,d)+w(b,c) for all a<=b<=c<=d,
    where w(x,y)=sum(weights[x..y]).
    """
    n = int(prefix.size - 1)
    for a in range(n):
        for b in range(a, n):
            for c in range(b, n):
                for d in range(c, n):
                    lhs = interval_sum(prefix, a, c) + interval_sum(prefix, b, d)
                    rhs = interval_sum(prefix, a, d) + interval_sum(prefix, b, c)
                    if lhs > rhs + 1e-12:
                        msg = f"counterexample at (a,b,c,d)=({a},{b},{c},{d}), lhs={lhs}, rhs={rhs}"
                        return False, msg
    return True, "all quadruples passed"


def run_case(name: str, weights: Sequence[float], quadrangle_check_limit: int = 10) -> None:
    arr = to_1d_nonnegative_array(weights)
    cubic = merge_cost_cubic(arr)
    knuth = merge_cost_knuth(arr)

    cost_equal = abs(cubic.min_cost - knuth.min_cost) <= 1e-9
    opt_monotone = verify_opt_monotonicity(knuth.opt) if arr.size >= 2 else True

    quadrangle_ok = True
    quadrangle_msg = "skipped (size too large)"
    if arr.size <= quadrangle_check_limit:
        q_ok, q_msg = check_quadrangle_inequality_for_range_sum(build_prefix_sums(arr))
        quadrangle_ok = q_ok
        quadrangle_msg = q_msg

    print(f"=== {name} ===")
    print(f"weights: {arr.tolist()}")
    print(f"Cubic min_cost: {cubic.min_cost:.1f}")
    print(f"Knuth min_cost: {knuth.min_cost:.1f}")
    print(f"Expression: {knuth.expression}")
    print(
        "Checks: "
        f"cost_equal={cost_equal}, "
        f"opt_monotone={opt_monotone}, "
        f"quadrangle_ok={quadrangle_ok}"
    )
    if arr.size <= quadrangle_check_limit:
        print(f"Quadrangle detail: {quadrangle_msg}")
    print()

    if not cost_equal:
        raise AssertionError(
            f"Cost mismatch in '{name}': cubic={cubic.min_cost}, knuth={knuth.min_cost}"
        )
    if not opt_monotone:
        raise AssertionError(f"Opt monotonicity failed in '{name}'.")
    if not quadrangle_ok:
        raise AssertionError(f"Quadrangle inequality check failed in '{name}': {quadrangle_msg}")


def main() -> None:
    cases = {
        "Case 1: classic": [4, 1, 7, 3, 2],
        "Case 2: balanced": [5, 5, 5, 5],
        "Case 3: with zero": [0, 3, 0, 2, 4],
        "Case 4: single": [9],
        "Case 5: empty": [],
        "Case 6: increasing": [1, 2, 3, 4, 5, 6],
    }

    for name, weights in cases.items():
        run_case(name, weights)

    print("All cases passed.")


if __name__ == "__main__":
    main()
