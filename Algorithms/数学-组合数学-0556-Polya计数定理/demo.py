"""Polya counting theorem MVP for necklaces and bracelets.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from itertools import product
from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple

Permutation = Tuple[int, ...]
Coloring = Tuple[int, ...]


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def rotation_permutation(n: int, shift: int) -> Permutation:
    """Return permutation i -> (i + shift) mod n (old index to new index)."""
    shift %= n
    return tuple((i + shift) % n for i in range(n))


def reflection_permutation(n: int, shift: int) -> Permutation:
    """Return permutation i -> (shift - i) mod n (old index to new index)."""
    shift %= n
    return tuple((shift - i) % n for i in range(n))


def unique_permutations(perms: Iterable[Permutation]) -> List[Permutation]:
    """Deduplicate permutations while preserving order."""
    return list(dict.fromkeys(perms))


def cyclic_group(n: int) -> List[Permutation]:
    _validate_positive_int("n", n)
    return unique_permutations(rotation_permutation(n, k) for k in range(n))


def dihedral_group(n: int) -> List[Permutation]:
    _validate_positive_int("n", n)
    rotations = [rotation_permutation(n, k) for k in range(n)]
    reflections = [reflection_permutation(n, k) for k in range(n)]
    return unique_permutations(rotations + reflections)


def cycle_lengths(perm: Permutation) -> List[int]:
    n = len(perm)
    visited = [False] * n
    lengths: List[int] = []

    for i in range(n):
        if visited[i]:
            continue
        j = i
        length = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            length += 1
        lengths.append(length)

    return lengths


def cycle_count(perm: Permutation) -> int:
    return len(cycle_lengths(perm))


def apply_permutation(coloring: Coloring, perm: Permutation) -> Coloring:
    """Apply permutation to coloring under old-index -> new-index convention."""
    transformed = [0] * len(coloring)
    for old_idx, new_idx in enumerate(perm):
        transformed[new_idx] = coloring[old_idx]
    return tuple(transformed)


def burnside_orbit_count(n_colors: int, group: Sequence[Permutation]) -> int:
    _validate_positive_int("n_colors", n_colors)
    if len(group) == 0:
        raise ValueError("group must be non-empty")

    total_fixed = 0
    for perm in group:
        total_fixed += n_colors ** cycle_count(perm)

    if total_fixed % len(group) != 0:
        raise ArithmeticError("Burnside average should be integer")
    return total_fixed // len(group)


def necklace_count(n: int, n_colors: int) -> int:
    return burnside_orbit_count(n_colors, cyclic_group(n))


def bracelet_count(n: int, n_colors: int) -> int:
    return burnside_orbit_count(n_colors, dihedral_group(n))


def necklace_closed_form(n: int, n_colors: int) -> int:
    _validate_positive_int("n", n)
    _validate_positive_int("n_colors", n_colors)

    total = 0
    for k in range(n):
        total += n_colors ** gcd(n, k)
    return total // n


def bracelet_closed_form(n: int, n_colors: int) -> int:
    _validate_positive_int("n", n)
    _validate_positive_int("n_colors", n_colors)

    rotation_sum = sum(n_colors ** gcd(n, k) for k in range(n))

    if n % 2 == 1:
        reflection_sum = n * (n_colors ** ((n + 1) // 2))
    else:
        reflection_sum = (n // 2) * (
            (n_colors ** (n // 2 + 1)) + (n_colors ** (n // 2))
        )

    return (rotation_sum + reflection_sum) // (2 * n)


def brute_force_orbit_count(n: int, n_colors: int, group: Sequence[Permutation]) -> int:
    _validate_positive_int("n", n)
    _validate_positive_int("n_colors", n_colors)
    if len(group) == 0:
        raise ValueError("group must be non-empty")

    representatives = set()
    for coloring in product(range(n_colors), repeat=n):
        orbit = [apply_permutation(coloring, perm) for perm in group]
        representatives.add(min(orbit))
    return len(representatives)


def cycle_structure_histogram(group: Sequence[Permutation]) -> Dict[Tuple[Tuple[int, int], ...], int]:
    """Return histogram keyed by sorted (cycle_len, multiplicity) tuples."""
    hist: Dict[Tuple[Tuple[int, int], ...], int] = {}

    for perm in group:
        lengths = cycle_lengths(perm)
        counter: Dict[int, int] = {}
        for length in lengths:
            counter[length] = counter.get(length, 0) + 1
        signature = tuple(sorted(counter.items()))
        hist[signature] = hist.get(signature, 0) + 1

    return hist


def _assert_core_properties() -> None:
    formula_cases = [
        (4, 2),
        (5, 3),
        (6, 2),
        (6, 3),
        (7, 3),
        (8, 2),
    ]

    for n, m in formula_cases:
        assert necklace_count(n, m) == necklace_closed_form(n, m)
        assert bracelet_count(n, m) == bracelet_closed_form(n, m)

    brute_cases = [
        (4, 3),  # 3^4 = 81 colorings
        (5, 2),  # 2^5 = 32 colorings
        (5, 3),  # 3^5 = 243 colorings
    ]

    for n, m in brute_cases:
        c_group = cyclic_group(n)
        d_group = dihedral_group(n)
        assert necklace_count(n, m) == brute_force_orbit_count(n, m, c_group)
        assert bracelet_count(n, m) == brute_force_orbit_count(n, m, d_group)



def _format_histogram(hist: Dict[Tuple[Tuple[int, int], ...], int]) -> List[str]:
    lines: List[str] = []
    for signature, count in sorted(hist.items()):
        parts = [f"{mult}*len{length}" for length, mult in signature]
        lines.append(f"{count} element(s): {' + '.join(parts)}")
    return lines


def main() -> None:
    _assert_core_properties()

    n = 6
    m = 3

    c_group = cyclic_group(n)
    d_group = dihedral_group(n)

    necklace = necklace_count(n, m)
    bracelet = bracelet_count(n, m)

    print("Polya Counting Theorem MVP (MATH-0556)")
    print("=" * 66)
    print(f"n = {n}, colors = {m}")
    print(f"|C_n action set| = {len(c_group)}")
    print(f"|D_n action set| = {len(d_group)}")
    print(f"Necklace count  (rotation only): {necklace}")
    print(f"Bracelet count  (rotation+flip): {bracelet}")

    print("\nClosed-form cross checks:")
    print(f"Necklace closed form : {necklace_closed_form(n, m)}")
    print(f"Bracelet closed form : {bracelet_closed_form(n, m)}")

    print("\nCycle-structure histogram for C_6:")
    for line in _format_histogram(cycle_structure_histogram(c_group)):
        print(f"  {line}")

    print("\nCycle-structure histogram for D_6:")
    for line in _format_histogram(cycle_structure_histogram(d_group)):
        print(f"  {line}")

    print("\nSmall table for m=2, n=1..8:")
    for n_small in range(1, 9):
        nc = necklace_count(n_small, 2)
        bc = bracelet_count(n_small, 2)
        print(f"  n={n_small}: necklace={nc:>3}, bracelet={bc:>3}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
