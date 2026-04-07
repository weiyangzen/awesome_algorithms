"""Digit DP MVP.

Count numbers in [L, R] such that:
1) digit sum % 3 == 0
2) no two adjacent digits are equal
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Optional, Tuple


@dataclass
class RunReport:
    label: str
    left: int
    right: int
    dp_count: int
    brute_count: Optional[int]
    state_count: int
    elapsed_ms: float


def is_valid_number(x: int) -> bool:
    """Check whether x satisfies the demo constraints."""
    if x < 0:
        return False
    s = str(x)
    if sum(int(ch) for ch in s) % 3 != 0:
        return False
    return all(s[i] != s[i - 1] for i in range(1, len(s)))


def brute_force_count(left: int, right: int) -> int:
    """Brute-force counter for verification on small intervals."""
    if left < 0:
        raise ValueError("left must be non-negative in this MVP")
    if left > right:
        raise ValueError("left must be <= right")
    return sum(1 for x in range(left, right + 1) if is_valid_number(x))


def count_up_to(n: int) -> Tuple[int, int]:
    """Return (count, visited_state_count) for numbers in [0, n]."""
    if n < 0:
        return 0, 0

    digits = [int(ch) for ch in str(n)]
    length = len(digits)
    sentinel_prev = 10  # "no previous digit yet"

    @lru_cache(maxsize=None)
    def dfs(pos: int, mod3: int, prev_digit: int, started: bool, tight: bool) -> int:
        if pos == length:
            # started=False corresponds to number 0, which is valid here.
            return 1 if mod3 == 0 else 0

        upper = digits[pos] if tight else 9
        total = 0

        for d in range(upper + 1):
            next_tight = tight and (d == upper)

            if not started and d == 0:
                # Leading zeros do not affect mod/adjacency constraints.
                total += dfs(pos + 1, mod3, sentinel_prev, False, next_tight)
                continue

            if started and d == prev_digit:
                continue

            total += dfs(pos + 1, (mod3 + d) % 3, d, True, next_tight)

        return total

    answer = dfs(0, 0, sentinel_prev, False, True)
    cache_info = dfs.cache_info()
    visited_states = cache_info.misses
    return answer, visited_states


def count_in_range(left: int, right: int) -> Tuple[int, int]:
    """Return (count, state_count) for [left, right]."""
    if left < 0:
        raise ValueError("left must be non-negative in this MVP")
    if left > right:
        raise ValueError("left must be <= right")

    count_r, states_r = count_up_to(right)
    count_lm1, states_lm1 = count_up_to(left - 1)
    return count_r - count_lm1, states_r + states_lm1


def run_case(label: str, left: int, right: int, brute_limit: int = 200_000) -> RunReport:
    start = perf_counter()
    dp_count, state_count = count_in_range(left, right)
    elapsed_ms = (perf_counter() - start) * 1000.0

    brute_count: Optional[int] = None
    if right - left + 1 <= brute_limit:
        brute_count = brute_force_count(left, right)
        if brute_count != dp_count:
            raise AssertionError(
                f"Mismatch in {label}: digit_dp={dp_count}, brute_force={brute_count}"
            )

    return RunReport(
        label=label,
        left=left,
        right=right,
        dp_count=dp_count,
        brute_count=brute_count,
        state_count=state_count,
        elapsed_ms=elapsed_ms,
    )


def main() -> None:
    cases = [
        ("small_exact_1", 0, 200),
        ("small_exact_2", 123, 4567),
        ("medium_exact_3", 10_000, 99_999),
        ("large_dp_only", 0, 10**18 - 1),
    ]

    print("Digit DP demo: count x in [L,R] where sum_digits(x)%3==0 and no equal adjacent digits")
    print("-" * 88)

    for label, left, right in cases:
        report = run_case(label, left, right)
        print(f"Case: {report.label}")
        print(f"  Range                : [{report.left}, {report.right}]")
        print(f"  Digit-DP Count       : {report.dp_count}")
        print(f"  Brute-force Count    : {report.brute_count if report.brute_count is not None else 'skipped'}")
        print(f"  Visited DP States    : {report.state_count}")
        print(f"  Elapsed Time (ms)    : {report.elapsed_ms:.3f}")
        print("-" * 88)

    print("All checks passed.")


if __name__ == "__main__":
    main()
