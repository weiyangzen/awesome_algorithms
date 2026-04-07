"""Suffix Array MVP: build SA, build LCP, and substring query."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    text: str
    patterns: tuple[str, ...]


def validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if len(text) == 0:
        raise ValueError("text must be non-empty")


def validate_pattern(pattern: str) -> None:
    if not isinstance(pattern, str):
        raise TypeError("pattern must be str")
    if len(pattern) == 0:
        raise ValueError("pattern must be non-empty")


def build_suffix_array(text: str) -> list[int]:
    """
    Build suffix array by doubling algorithm.

    Time: O(n log^2 n) using Python sort each round.
    """
    validate_text(text)
    n = len(text)
    sa = list(range(n))
    rank = [ord(ch) for ch in text]
    k = 1

    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        next_rank = [0] * n
        next_rank[sa[0]] = 0

        for idx in range(1, n):
            prev_i = sa[idx - 1]
            curr_i = sa[idx]
            prev_key = (rank[prev_i], rank[prev_i + k] if prev_i + k < n else -1)
            curr_key = (rank[curr_i], rank[curr_i + k] if curr_i + k < n else -1)
            next_rank[curr_i] = next_rank[prev_i] + (1 if curr_key != prev_key else 0)

        rank = next_rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1

    return sa


def build_lcp_array(text: str, sa: list[int]) -> list[int]:
    """
    Kasai algorithm for LCP.

    lcp[i] means LCP between suffix sa[i] and sa[i-1], and lcp[0] = 0.
    """
    validate_text(text)
    n = len(text)
    if len(sa) != n:
        raise ValueError("sa length must match text length")
    if sorted(sa) != list(range(n)):
        raise ValueError("sa must be a permutation of [0, n)")

    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i

    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1
    return lcp


def find_occurrences(text: str, pattern: str, sa: list[int]) -> list[int]:
    """Find all starting positions of pattern in text via SA binary search."""
    validate_text(text)
    validate_pattern(pattern)
    n = len(text)
    m = len(pattern)
    if len(sa) != n:
        raise ValueError("sa length must match text length")

    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if text[sa[mid] : sa[mid] + m] < pattern:
            left = mid + 1
        else:
            right = mid
    first = left

    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if text[sa[mid] : sa[mid] + m] <= pattern:
            left = mid + 1
        else:
            right = mid
    last = left

    result = [idx for idx in sa[first:last] if text.startswith(pattern, idx)]
    result.sort()
    return result


def naive_suffix_array(text: str) -> list[int]:
    return sorted(range(len(text)), key=lambda i: text[i:])


def naive_lcp_array(text: str, sa: list[int]) -> list[int]:
    lcp = [0] * len(sa)
    for i in range(1, len(sa)):
        a, b = sa[i - 1], sa[i]
        h = 0
        while a + h < len(text) and b + h < len(text) and text[a + h] == text[b + h]:
            h += 1
        lcp[i] = h
    return lcp


def naive_occurrences(text: str, pattern: str) -> list[int]:
    return [i for i in range(0, len(text) - len(pattern) + 1) if text.startswith(pattern, i)]


def run_case(case: Case) -> None:
    text = case.text
    sa = build_suffix_array(text)
    lcp = build_lcp_array(text, sa)

    expected_sa = naive_suffix_array(text)
    expected_lcp = naive_lcp_array(text, sa)
    if sa != expected_sa:
        raise AssertionError(f"suffix array mismatch: {sa} != {expected_sa}")
    if lcp != expected_lcp:
        raise AssertionError(f"lcp array mismatch: {lcp} != {expected_lcp}")

    print("=" * 72)
    print(f"text: {text!r}")
    print(f"length: {len(text)}")
    print(f"suffix array: {sa}")
    print(f"lcp array   : {lcp}")
    print("suffix table (rank, idx, suffix):")
    for rank, idx in enumerate(sa):
        print(f"  {rank:2d}  {idx:2d}  {text[idx:]}")

    print("pattern query results:")
    for pattern in case.patterns:
        found = find_occurrences(text, pattern, sa)
        expected = naive_occurrences(text, pattern)
        if found != expected:
            raise AssertionError(
                f"query mismatch for pattern={pattern!r}: {found} != {expected}"
            )
        print(f"  {pattern!r:12s} -> {found}")


def main() -> None:
    cases = [
        Case(text="banana", patterns=("ana", "na", "ban", "nana", "apple")),
        Case(text="mississippi", patterns=("issi", "ssi", "ppi", "miss", "xyz")),
        Case(text="abracadabra", patterns=("abra", "cad", "ra", "a", "zz")),
    ]

    print("Suffix Array MVP (doubling + Kasai + binary-search query)")
    for case in cases:
        run_case(case)
    print("=" * 72)
    print("All cases passed.")


if __name__ == "__main__":
    main()
