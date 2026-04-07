"""Inclusion-Exclusion Principle MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from itertools import combinations, permutations, product
from math import comb, factorial
from typing import List, Sequence, Set


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def inclusion_exclusion_union_size(universe: Set[int], events: Sequence[Set[int]]) -> int:
    """Return |A1 ∪ ... ∪ Am| using inclusion-exclusion.

    Args:
        universe: finite universal set U (used for clipping events into U).
        events: list of subsets A_i.
    """
    m = len(events)
    if m == 0:
        return 0

    clipped_events = [set(a) & universe for a in events]
    total = 0

    for r in range(1, m + 1):
        sign = 1 if r % 2 == 1 else -1
        for idxs in combinations(range(m), r):
            intersection = set(universe)
            for idx in idxs:
                intersection &= clipped_events[idx]
                if not intersection:
                    break
            total += sign * len(intersection)

    return total


def direct_union_size(universe: Set[int], events: Sequence[Set[int]]) -> int:
    """Brute-force union size for validation."""
    union_set: Set[int] = set()
    for event in events:
        union_set |= set(event) & universe
    return len(union_set)


def count_surjections_inclusion_exclusion(n: int, m: int) -> int:
    """Count onto functions f:[n] -> [m] by inclusion-exclusion.

    Formula:
        sum_{j=0..m} (-1)^j C(m,j) (m-j)^n
    """
    _validate_non_negative_int("n", n)
    _validate_non_negative_int("m", m)

    if m == 0:
        return 1 if n == 0 else 0
    if m > n:
        return 0

    total = 0
    for j in range(0, m + 1):
        total += ((-1) ** j) * comb(m, j) * ((m - j) ** n)
    return total


def count_surjections_bruteforce(n: int, m: int) -> int:
    """Brute-force count of onto functions for small n,m."""
    _validate_non_negative_int("n", n)
    _validate_non_negative_int("m", m)

    if m == 0:
        return 1 if n == 0 else 0

    count = 0
    codomain = tuple(range(m))
    for image_tuple in product(codomain, repeat=n):
        if len(set(image_tuple)) == m:
            count += 1
    return count


def count_derangements_inclusion_exclusion(n: int) -> int:
    """Count derangements of n labeled elements.

    Formula:
        !n = sum_{k=0..n} (-1)^k C(n,k) (n-k)!
    """
    _validate_non_negative_int("n", n)

    total = 0
    for k in range(0, n + 1):
        total += ((-1) ** k) * comb(n, k) * factorial(n - k)
    return total


def count_derangements_bruteforce(n: int) -> int:
    """Brute-force derangements for small n."""
    _validate_non_negative_int("n", n)

    count = 0
    base = tuple(range(n))
    for p in permutations(base):
        if all(p[i] != i for i in base):
            count += 1
    return count


def _numbers_divisible_by(limit: int, d: int) -> Set[int]:
    _validate_non_negative_int("limit", limit)
    _validate_non_negative_int("d", d)
    if d == 0:
        raise ValueError("d must be positive")
    return {x for x in range(1, limit + 1) if x % d == 0}


def _assert_examples() -> List[str]:
    lines: List[str] = []

    # Example 1: union count on [1..100] for divisibility by 2, 3, 5.
    universe = set(range(1, 101))
    events = [
        _numbers_divisible_by(100, 2),
        _numbers_divisible_by(100, 3),
        _numbers_divisible_by(100, 5),
    ]
    union_ie = inclusion_exclusion_union_size(universe, events)
    union_direct = direct_union_size(universe, events)
    complement_count = len(universe) - union_ie

    assert union_ie == 74
    assert union_ie == union_direct

    lines.append("Example 1: |A2 ∪ A3 ∪ A5| over [1..100]")
    lines.append(f"  Inclusion-Exclusion union size = {union_ie}")
    lines.append(f"  Direct union size            = {union_direct}")
    lines.append(f"  Count in complement          = {complement_count}")

    # Example 2: onto function count.
    n, m = 5, 3
    onto_ie = count_surjections_inclusion_exclusion(n, m)
    onto_bruteforce = count_surjections_bruteforce(n, m)

    assert onto_ie == onto_bruteforce
    assert onto_ie == 150

    lines.append("\nExample 2: onto functions f:[5] -> [3]")
    lines.append(f"  Inclusion-Exclusion count = {onto_ie}")
    lines.append(f"  Brute-force count         = {onto_bruteforce}")

    # Example 3: derangements.
    n_der = 6
    der_ie = count_derangements_inclusion_exclusion(n_der)
    der_bruteforce = count_derangements_bruteforce(n_der)

    assert der_ie == der_bruteforce
    assert der_ie == 265

    lines.append("\nExample 3: derangements of 6 elements")
    lines.append(f"  Inclusion-Exclusion count = {der_ie}")
    lines.append(f"  Brute-force count         = {der_bruteforce}")

    return lines


def main() -> None:
    output_lines = _assert_examples()

    print("Inclusion-Exclusion Principle MVP")
    print("=" * 60)
    for line in output_lines:
        print(line)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
