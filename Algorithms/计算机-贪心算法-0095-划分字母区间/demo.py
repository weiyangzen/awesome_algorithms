"""Greedy MVP for Partition Labels (划分字母区间).

Run:
    uv run python demo.py

The script performs:
- known example checks
- randomized brute-force cross validation on small strings

No interactive input is required.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List


def partition_labels(s: str) -> List[int]:
    """Return maximum-number partition lengths using greedy boundary closure."""
    if not s:
        return []

    last: Dict[str, int] = {ch: idx for idx, ch in enumerate(s)}
    lengths: List[int] = []

    start = 0
    end = 0
    for idx, ch in enumerate(s):
        end = max(end, last[ch])
        if idx == end:
            lengths.append(end - start + 1)
            start = idx + 1

    return lengths


def lengths_to_segments(s: str, lengths: List[int]) -> List[str]:
    """Convert partition lengths back to segment strings for display."""
    segments: List[str] = []
    left = 0
    for length in lengths:
        right = left + length
        segments.append(s[left:right])
        left = right
    return segments


def is_valid_partition(s: str, lengths: List[int]) -> bool:
    """Check: each character appears in at most one segment."""
    if sum(lengths) != len(s):
        return False

    char_segment: Dict[str, int] = {}
    pos = 0
    for seg_id, length in enumerate(lengths):
        if length <= 0:
            return False
        for _ in range(length):
            ch = s[pos]
            old = char_segment.get(ch)
            if old is None:
                char_segment[ch] = seg_id
            elif old != seg_id:
                return False
            pos += 1
    return True


def brute_force_max_parts(s: str) -> int:
    """Exhaustively compute max number of valid parts (for short strings only)."""
    n = len(s)
    if n == 0:
        return 0

    best = 1
    # Bit i means cut after index i, i in [0, n-2].
    for mask in range(1 << (n - 1)):
        lengths: List[int] = []
        start = 0
        for i in range(n - 1):
            if (mask >> i) & 1:
                lengths.append(i - start + 1)
                start = i + 1
        lengths.append(n - start)

        if is_valid_partition(s, lengths):
            best = max(best, len(lengths))

    return best


def run_known_examples() -> None:
    cases = {
        "ababcbacadefegdehijhklij": [9, 7, 8],
        "eccbbbbdec": [10],
        "qiejxqfnqceocmy": [13, 1, 1],
        "abcd": [1, 1, 1, 1],
        "aaaa": [4],
        "": [],
    }

    print("=== Known Examples ===")
    for s, expected in cases.items():
        got = partition_labels(s)
        assert got == expected, f"case={s!r}, expected={expected}, got={got}"
        assert is_valid_partition(s, got), f"partition validity failed for {s!r}"

        segments = lengths_to_segments(s, got)
        print(f"s={s!r}")
        print(f"  lengths : {got}")
        print(f"  segments: {segments}")


def run_random_verification(seed: int = 2026) -> None:
    """Randomized check against brute-force optimum on small strings."""
    rng = Random(seed)
    alphabet = "abcde"

    checked = 0
    for n in range(1, 11):
        for _ in range(120):
            s = "".join(rng.choice(alphabet) for _ in range(n))
            greedy = partition_labels(s)
            assert is_valid_partition(s, greedy), f"invalid greedy partition for {s!r}: {greedy}"

            best_parts = brute_force_max_parts(s)
            assert len(greedy) == best_parts, (
                f"non-optimal greedy partition for {s!r}: "
                f"greedy_parts={len(greedy)}, best_parts={best_parts}, lengths={greedy}"
            )
            checked += 1

    print("=== Random Verification ===")
    print(f"Checked {checked} random strings (length 1..10).")


def main() -> None:
    run_known_examples()
    run_random_verification(seed=2026)
    print("All checks passed.")


if __name__ == "__main__":
    main()
