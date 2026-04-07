"""Burnside lemma MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence, Tuple

Permutation = Tuple[int, ...]
Coloring = Tuple[int, ...]


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _validate_permutation(perm: Permutation) -> None:
    n = len(perm)
    if sorted(perm) != list(range(n)):
        raise ValueError(f"invalid permutation: {perm}")


def rotation_permutation(n: int, shift: int) -> Permutation:
    """Return permutation i -> (i + shift) mod n (old index to new index)."""
    _validate_positive_int("n", n)
    shift %= n
    return tuple((i + shift) % n for i in range(n))


def cyclic_group(n: int) -> List[Permutation]:
    """Return C_n action as permutations on indices [0..n-1]."""
    _validate_positive_int("n", n)
    return [rotation_permutation(n, s) for s in range(n)]


def cycle_count(perm: Permutation) -> int:
    """Count disjoint cycles in a permutation."""
    _validate_permutation(perm)
    n = len(perm)
    visited = [False] * n
    count = 0

    for i in range(n):
        if visited[i]:
            continue
        count += 1
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]

    return count


def fixed_colorings_count(n_colors: int, perm: Permutation) -> int:
    """Number of colorings fixed by perm, equals n_colors^(cycle_count)."""
    _validate_positive_int("n_colors", n_colors)
    return n_colors ** cycle_count(perm)


def burnside_orbit_count(n_colors: int, group: Sequence[Permutation]) -> int:
    """Compute |X/G| by Burnside lemma with explicit fixed-point accumulation."""
    _validate_positive_int("n_colors", n_colors)
    if len(group) == 0:
        raise ValueError("group must be non-empty")

    total_fixed = 0
    for perm in group:
        total_fixed += fixed_colorings_count(n_colors, perm)

    if total_fixed % len(group) != 0:
        raise ArithmeticError("Burnside average must be integer")
    return total_fixed // len(group)


def apply_permutation(coloring: Coloring, perm: Permutation) -> Coloring:
    """Apply old-index -> new-index permutation to a coloring tuple."""
    if len(coloring) != len(perm):
        raise ValueError("coloring length and permutation length must match")

    transformed = [0] * len(coloring)
    for old_idx, new_idx in enumerate(perm):
        transformed[new_idx] = coloring[old_idx]
    return tuple(transformed)


def orbit_representative(coloring: Coloring, group: Iterable[Permutation]) -> Coloring:
    """Canonical representative: lexicographically smallest element in the orbit."""
    return min(apply_permutation(coloring, perm) for perm in group)


def brute_force_orbit_count(n: int, n_colors: int, group: Sequence[Permutation]) -> int:
    """Count orbits by exhaustive enumeration (for small cases only)."""
    _validate_positive_int("n", n)
    _validate_positive_int("n_colors", n_colors)
    if len(group) == 0:
        raise ValueError("group must be non-empty")

    reps = set()
    for coloring in product(range(n_colors), repeat=n):
        reps.add(orbit_representative(coloring, group))
    return len(reps)


def _format_fixed_counts(group: Sequence[Permutation], n_colors: int) -> List[str]:
    lines: List[str] = []
    for idx, perm in enumerate(group):
        cycles = cycle_count(perm)
        fixed = fixed_colorings_count(n_colors, perm)
        lines.append(f"  g[{idx}] cycles={cycles}, fixed={fixed}, perm={perm}")
    return lines


def _run_examples() -> List[str]:
    lines: List[str] = []

    # Case 1: binary necklaces of length 4 under rotation.
    n1, m1 = 4, 2
    group1 = cyclic_group(n1)
    burnside1 = burnside_orbit_count(m1, group1)
    brute1 = brute_force_orbit_count(n1, m1, group1)
    assert burnside1 == brute1 == 6

    lines.append(f"Case 1: n={n1}, m={m1}, |C_n|={len(group1)}")
    lines.append(f"  Burnside orbit count = {burnside1}")
    lines.append(f"  Brute-force count    = {brute1}")

    # Case 2: 3-color necklaces of length 6 under rotation.
    n2, m2 = 6, 3
    group2 = cyclic_group(n2)
    burnside2 = burnside_orbit_count(m2, group2)
    brute2 = brute_force_orbit_count(n2, m2, group2)
    assert burnside2 == brute2 == 130

    lines.append(f"\nCase 2: n={n2}, m={m2}, |C_n|={len(group2)}")
    lines.append(f"  Burnside orbit count = {burnside2}")
    lines.append(f"  Brute-force count    = {brute2}")
    lines.append("  Fixed-coloring details per group element:")
    lines.extend(_format_fixed_counts(group2, m2))

    # Case 3: small grid cross-checks.
    lines.append("\nCase 3: small cross-check table (Burnside == brute force)")
    for n in range(2, 7):
        m = 2
        group = cyclic_group(n)
        b = burnside_orbit_count(m, group)
        t = brute_force_orbit_count(n, m, group)
        assert b == t
        lines.append(f"  n={n}, m={m}: count={b}")

    return lines


def main() -> None:
    lines = _run_examples()

    print("Burnside Lemma MVP (MATH-0555)")
    print("=" * 64)
    for line in lines:
        print(line)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
