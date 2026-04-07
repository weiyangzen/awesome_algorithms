"""Boyer-Moore string matching MVP.

This demo implements classic Boyer-Moore exact matching using:
1) bad-character heuristic
2) good-suffix heuristic (strong suffix + case 2 preprocessing)

It is self-contained and runs without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List


@dataclass
class SearchResult:
    """Container for search output and a lightweight runtime metric."""

    matches: List[int]
    comparisons: int


def build_bad_character_table(pattern: str) -> Dict[str, int]:
    """Map each character to its rightmost index in pattern."""
    last: Dict[str, int] = {}
    for i, ch in enumerate(pattern):
        last[ch] = i
    return last


def build_good_suffix_shift(pattern: str) -> List[int]:
    """Build good-suffix shift table.

    Returns:
        shift: list of length m + 1, where m = len(pattern).
            During mismatch at pattern index j, use shift[j + 1].
            During full match, use shift[0].
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    bpos = [0] * (m + 1)

    i = m
    j = m + 1
    bpos[i] = j

    # Strong good-suffix preprocessing.
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = bpos[j]
        i -= 1
        j -= 1
        bpos[i] = j

    # Case 2 preprocessing.
    j = bpos[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = bpos[j]

    return shift


def boyer_moore_search(text: str, pattern: str) -> SearchResult:
    """Find all exact matches of pattern in text using Boyer-Moore."""
    n = len(text)
    m = len(pattern)

    if m == 0:
        return SearchResult(matches=list(range(n + 1)), comparisons=0)
    if n < m:
        return SearchResult(matches=[], comparisons=0)

    bad_char = build_bad_character_table(pattern)
    good_suffix = build_good_suffix_shift(pattern)

    matches: List[int] = []
    comparisons = 0
    s = 0

    while s <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == text[s + j]:
            comparisons += 1
            j -= 1

        if j < 0:
            matches.append(s)
            s += max(1, good_suffix[0])
            continue

        comparisons += 1  # count the mismatch comparison
        bad_shift = j - bad_char.get(text[s + j], -1)
        good_shift = good_suffix[j + 1]
        s += max(1, bad_shift, good_shift)

    return SearchResult(matches=matches, comparisons=comparisons)


def naive_search(text: str, pattern: str) -> List[int]:
    """Reference implementation for validation."""
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    out: List[int] = []
    for i in range(n - m + 1):
        if text[i : i + m] == pattern:
            out.append(i)
    return out


def run_fixed_case() -> None:
    text = "ABAAABCDABCABCDABCDABDEABCDABD"
    pattern = "ABCDABD"
    bm = boyer_moore_search(text, pattern)
    baseline = naive_search(text, pattern)

    print("=== Boyer-Moore Fixed Case ===")
    print(f"text:    {text}")
    print(f"pattern: {pattern}")
    print(f"matches: {bm.matches}")
    print(f"naive:   {baseline}")
    print(f"char comparisons (BM): {bm.comparisons}")

    assert bm.matches == baseline, "BM result differs from naive baseline."


def run_randomized_checks(num_cases: int = 200, seed: int = 533) -> None:
    """Deterministic property checks against naive search."""
    rng = random.Random(seed)
    alphabet = "ABCD"

    for _ in range(num_cases):
        n = rng.randint(0, 80)
        m = rng.randint(0, 10)
        text = "".join(rng.choice(alphabet) for _ in range(n))
        pattern = "".join(rng.choice(alphabet) for _ in range(m))
        bm = boyer_moore_search(text, pattern).matches
        nv = naive_search(text, pattern)
        if bm != nv:
            raise AssertionError(
                "Randomized check failed.\n"
                f"text={text!r}\npattern={pattern!r}\nbm={bm}\nnaive={nv}"
            )

    print(f"Randomized checks passed: {num_cases} cases.")


def main() -> None:
    run_fixed_case()
    run_randomized_checks()
    print("All checks passed.")


if __name__ == "__main__":
    main()
