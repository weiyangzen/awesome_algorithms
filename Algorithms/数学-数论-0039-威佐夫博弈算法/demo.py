"""Minimal runnable MVP for Wythoff's Game algorithm (MATH-0039)."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import List, Optional, Tuple

Position = Tuple[int, int]


def normalize(a: int, b: int) -> Position:
    """Return position in canonical order (small, large)."""
    if a < 0 or b < 0:
        raise ValueError(f"Heap sizes must be non-negative, got ({a}, {b}).")
    return (a, b) if a <= b else (b, a)


def floor_mul_phi(k: int) -> int:
    """Compute floor(k * phi) with Decimal to reduce floating-point misclassification."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    if k == 0:
        return 0

    digits = len(str(k))
    with localcontext() as ctx:
        ctx.prec = digits + 30
        phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
        value = Decimal(k) * phi
        return int(value.to_integral_value(rounding=ROUND_FLOOR))


def cold_position_by_k(k: int) -> Position:
    """Return the k-th cold (P) position of Wythoff's game."""
    a = floor_mul_phi(k)
    return a, a + k


def is_cold_position(a: int, b: int) -> bool:
    """Check whether (a, b) is a P-position using Wythoff/Beatty characterization."""
    x, y = normalize(a, b)
    k = y - x
    return x == floor_mul_phi(k)


def legal_moves(a: int, b: int) -> List[Position]:
    """Enumerate all legal next positions under Wythoff's game rules."""
    x, y = normalize(a, b)
    moves = set()

    # Remove from pile x only.
    for t in range(1, x + 1):
        moves.add((x - t, y))

    # Remove from pile y only.
    for t in range(1, y + 1):
        nx, ny = x, y - t
        moves.add((nx, ny) if nx <= ny else (ny, nx))

    # Remove same amount from both piles.
    for t in range(1, x + 1):
        moves.add((x - t, y - t))

    return sorted(moves)


def winning_move(a: int, b: int) -> Optional[Position]:
    """Return one winning move (to a cold position), or None if already cold."""
    x, y = normalize(a, b)
    if is_cold_position(x, y):
        return None

    for nxt in legal_moves(x, y):
        if is_cold_position(*nxt):
            return nxt

    return None


def verify_theorem_properties(limit: int = 25) -> None:
    """Verify: hot positions have at least one cold successor; cold has none."""
    for a in range(limit + 1):
        for b in range(a, limit + 1):
            cold = is_cold_position(a, b)
            cold_successors = [m for m in legal_moves(a, b) if is_cold_position(*m)]
            if cold:
                assert not cold_successors, (
                    f"Cold position ({a}, {b}) should not move to cold, "
                    f"but got {cold_successors}."
                )
            else:
                assert len(cold_successors) >= 1, (
                    f"Hot position ({a}, {b}) should have at least one cold successor, "
                    f"but got {cold_successors}."
                )


def main() -> None:
    print("Wythoff Game MVP (MATH-0039)")
    print("=" * 72)

    print("First 12 cold positions (k, (a_k, b_k)):")
    first_positions = [(k, cold_position_by_k(k)) for k in range(12)]
    for k, pos in first_positions:
        print(f"k={k:>2} -> {pos}")

    print("=" * 72)
    test_positions = [
        (0, 0),
        (1, 2),
        (2, 3),
        (4, 7),
        (10, 16),
        (8, 9),
        (12, 20),
        (18, 25),
    ]

    print("Position status and one-step strategy:")
    for a, b in test_positions:
        cold = is_cold_position(a, b)
        move = winning_move(a, b)
        state = "P-position(必败态)" if cold else "N-position(必胜态)"
        print(f"({a:>2}, {b:>2}) -> {state}; winning_move={move}")
        if cold:
            assert move is None
        else:
            assert move is not None and is_cold_position(*move)

    print("=" * 72)
    verify_theorem_properties(limit=25)
    print("Verified on grid 0<=a<=b<=25: hot positions have at least one cold successor.")
    print("All checks passed.")


if __name__ == "__main__":
    main()
