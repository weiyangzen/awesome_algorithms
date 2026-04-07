"""Farey sequence generation minimal runnable MVP."""

from __future__ import annotations

from functools import cmp_to_key
from math import gcd
from typing import Iterable

Fraction = tuple[int, int]


def generate_farey(order: int) -> list[Fraction]:
    """Generate Farey sequence F_order using adjacent-term recurrence."""
    if order < 1:
        raise ValueError("order must be >= 1")

    a, b = 0, 1
    c, d = 1, order
    sequence: list[Fraction] = [(a, b)]

    while c <= order:
        sequence.append((c, d))
        k = (order + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b

    return sequence


def _compare_fraction(x: Fraction, y: Fraction) -> int:
    left = x[0] * y[1]
    right = y[0] * x[1]
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def generate_farey_by_enumeration(order: int) -> list[Fraction]:
    """Reference implementation: enumerate reduced fractions then sort exactly."""
    if order < 1:
        raise ValueError("order must be >= 1")

    fractions: set[Fraction] = {(0, 1), (1, 1)}
    for den in range(1, order + 1):
        for num in range(1, den):
            if gcd(num, den) == 1:
                fractions.add((num, den))

    return sorted(fractions, key=cmp_to_key(_compare_fraction))


def totients_up_to(n: int) -> list[int]:
    """Euler phi values for 0..n via sieve."""
    phi = list(range(n + 1))
    for i in range(2, n + 1):
        if phi[i] == i:
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    return phi


def farey_length_expected(order: int) -> int:
    """|F_n| = 1 + sum_{m=1..n} phi(m)."""
    if order < 1:
        raise ValueError("order must be >= 1")
    phi = totients_up_to(order)
    return 1 + sum(phi[1:])


def verify_farey_properties(sequence: Iterable[Fraction], order: int) -> tuple[bool, list[str]]:
    """Check boundary, reduced form, monotonicity, and neighbor determinant."""
    seq = list(sequence)
    errors: list[str] = []

    if not seq:
        errors.append("sequence is empty")
        return False, errors

    if seq[0] != (0, 1):
        errors.append("first item must be 0/1")
    if seq[-1] != (1, 1):
        errors.append("last item must be 1/1")

    for idx, (num, den) in enumerate(seq):
        if den <= 0:
            errors.append(f"index {idx}: denominator must be positive")
            continue
        if not (0 <= num <= den <= order):
            errors.append(f"index {idx}: {num}/{den} violates 0<=num<=den<=order")
        if gcd(num, den) != 1:
            errors.append(f"index {idx}: {num}/{den} is not reduced")

    for idx in range(len(seq) - 1):
        a, b = seq[idx]
        c, d = seq[idx + 1]
        if a * d >= c * b:
            errors.append(f"index {idx}->{idx + 1}: sequence not strictly increasing")
        if b * c - a * d != 1:
            errors.append(
                f"index {idx}->{idx + 1}: neighbor determinant != 1 "
                f"(got {b * c - a * d})"
            )

    return len(errors) == 0, errors


def preview_sequence(sequence: list[Fraction], max_items: int = 12) -> str:
    head = sequence[:max_items]
    text = ", ".join(f"{a}/{b}" for a, b in head)
    if len(sequence) > max_items:
        text += ", ..."
    return text


def main() -> None:
    print("=== Farey Sequence MVP Demo ===")

    test_orders = [1, 5, 8, 12]
    for order in test_orders:
        seq = generate_farey(order)
        ok, errors = verify_farey_properties(seq, order)
        expected_len = farey_length_expected(order)

        print(f"\n[order={order}]")
        print(f"length: generated={len(seq)}, expected={expected_len}")
        print(f"property_check: {ok}")
        print(f"preview: {preview_sequence(seq)}")

        if order <= 8:
            ref = generate_farey_by_enumeration(order)
            print(f"cross_check_with_enumeration: {seq == ref}")

        if not ok:
            print("errors:")
            for e in errors[:5]:
                print(f"  - {e}")


if __name__ == "__main__":
    main()
