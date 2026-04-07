"""Minimal runnable MVP for Stern-Brocot tree traversal.

This script demonstrates:
1. Finite-depth inorder traversal over the Stern-Brocot tree.
2. Path encoding from a reduced positive fraction to L/R moves.
3. Path decoding from L/R moves back to the fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class SBNode:
    """A Stern-Brocot node snapshot for demo output."""

    numerator: int
    denominator: int
    path: str
    depth: int

    def as_fraction(self) -> str:
        return f"{self.numerator}/{self.denominator}"


def _compare_fractions(a_num: int, a_den: int, b_num: int, b_den: int) -> int:
    """Compare a_num/a_den with b_num/b_den using integer arithmetic."""
    left = a_num * b_den
    right = b_num * a_den
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def path_to_fraction(numerator: int, denominator: int, max_steps: int = 4096) -> str:
    """Return the unique Stern-Brocot path (L/R string) for a reduced fraction."""
    if numerator <= 0 or denominator <= 0:
        raise ValueError("numerator and denominator must be positive")
    if gcd(numerator, denominator) != 1:
        raise ValueError("fraction must be reduced (coprime numerator/denominator)")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    # Sentinel boundaries for all positive rationals.
    left_num, left_den = 0, 1
    right_num, right_den = 1, 0
    path: List[str] = []

    for _ in range(max_steps):
        mid_num = left_num + right_num
        mid_den = left_den + right_den
        cmp_result = _compare_fractions(numerator, denominator, mid_num, mid_den)
        if cmp_result == 0:
            return "".join(path)
        if cmp_result < 0:
            # Target is smaller than mediant -> go left; tighten right boundary.
            path.append("L")
            right_num, right_den = mid_num, mid_den
        else:
            # Target is greater than mediant -> go right; tighten left boundary.
            path.append("R")
            left_num, left_den = mid_num, mid_den

    raise RuntimeError("path search exceeded max_steps; check implementation/inputs")


def fraction_from_path(path: str) -> Tuple[int, int]:
    """Decode an L/R path into a reduced positive fraction."""
    left_num, left_den = 0, 1
    right_num, right_den = 1, 0

    for move in path:
        mid_num = left_num + right_num
        mid_den = left_den + right_den
        if move == "L":
            right_num, right_den = mid_num, mid_den
        elif move == "R":
            left_num, left_den = mid_num, mid_den
        else:
            raise ValueError(f"invalid move {move!r}; path must contain only 'L'/'R'")

    return left_num + right_num, left_den + right_den


def inorder_traversal(max_depth: int) -> List[SBNode]:
    """Inorder traversal up to max_depth (root depth = 0)."""
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    output: List[SBNode] = []

    def dfs(
        left_num: int,
        left_den: int,
        right_num: int,
        right_den: int,
        depth: int,
        path: str,
    ) -> None:
        if depth > max_depth:
            return
        mid_num = left_num + right_num
        mid_den = left_den + right_den

        dfs(left_num, left_den, mid_num, mid_den, depth + 1, path + "L")
        output.append(SBNode(mid_num, mid_den, path, depth))
        dfs(mid_num, mid_den, right_num, right_den, depth + 1, path + "R")

    dfs(0, 1, 1, 0, 0, "")
    return output


def is_strictly_increasing(nodes: Sequence[SBNode]) -> bool:
    """Check monotonic increase by exact rational comparison."""
    for i in range(1, len(nodes)):
        prev = nodes[i - 1]
        curr = nodes[i]
        if _compare_fractions(
            prev.numerator,
            prev.denominator,
            curr.numerator,
            curr.denominator,
        ) >= 0:
            return False
    return True


def run_roundtrip_cases() -> List[tuple[int, int, str, tuple[int, int]]]:
    """Run deterministic fraction -> path -> fraction checks."""
    cases = [(1, 1), (5, 2), (13, 8), (7, 5), (11, 4), (8, 13)]
    report: List[tuple[int, int, str, tuple[int, int]]] = []

    for num, den in cases:
        path = path_to_fraction(num, den)
        decoded = fraction_from_path(path)
        report.append((num, den, path, decoded))
        if decoded != (num, den):
            raise RuntimeError(
                f"roundtrip failed for {num}/{den}: path={path}, decoded={decoded}"
            )
    return report


def main() -> None:
    max_depth = 4
    nodes = inorder_traversal(max_depth)
    increasing = is_strictly_increasing(nodes)
    roundtrip = run_roundtrip_cases()

    print("=== Stern-Brocot Tree Traversal MVP ===")
    print(f"max_depth: {max_depth}")
    print(f"node_count: {len(nodes)}")
    print(f"strictly_increasing: {increasing}")
    print("first 18 nodes in inorder (fraction | path | depth):")
    for idx, node in enumerate(nodes[:18], start=1):
        show_path = node.path if node.path else "(root)"
        print(f"  {idx:>2}. {node.as_fraction():>5} | {show_path:<8} | d={node.depth}")

    print("\nRound-trip checks (fraction -> path -> fraction):")
    for num, den, path, decoded in roundtrip:
        show_path = path if path else "(root)"
        print(f"  {num}/{den: <2} -> {show_path:<8} -> {decoded[0]}/{decoded[1]}")


if __name__ == "__main__":
    main()
