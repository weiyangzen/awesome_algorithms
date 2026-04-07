"""Minimal runnable MVP for KMP string matching."""

from __future__ import annotations

from dataclasses import dataclass
import random


def build_lps(pattern: str) -> list[int]:
    """Build LPS (longest proper prefix that is also suffix) table in O(m)."""
    m = len(pattern)
    if m == 0:
        return []

    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1

    return lps


def kmp_search_all(text: str, pattern: str) -> list[int]:
    """Return all match start indices of pattern in text (supports overlaps)."""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))

    lps = build_lps(pattern)
    matches: list[int] = []

    i = 0  # index in text
    j = 0  # index in pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

            if j == m:
                matches.append(i - m)
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1

    return matches


def naive_search_all(text: str, pattern: str) -> list[int]:
    """Simple baseline matcher for correctness cross-check."""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    out: list[int] = []
    for i in range(n - m + 1):
        if text[i : i + m] == pattern:
            out.append(i)
    return out


@dataclass(frozen=True)
class SearchCase:
    text: str
    pattern: str
    expected: list[int]


def run_deterministic_cases() -> None:
    cases = [
        SearchCase(text="ababcabcabababd", pattern="ababd", expected=[10]),
        SearchCase(text="aaaaa", pattern="aa", expected=[0, 1, 2, 3]),
        SearchCase(text="abcde", pattern="f", expected=[]),
        SearchCase(text="", pattern="", expected=[0]),
        SearchCase(text="", pattern="a", expected=[]),
        SearchCase(text="你好吗你好你", pattern="你好", expected=[0, 3]),
        SearchCase(text="abc", pattern="abcd", expected=[]),
    ]

    print("=== Deterministic Cases ===")
    for idx, case in enumerate(cases, start=1):
        got = kmp_search_all(case.text, case.pattern)
        baseline = naive_search_all(case.text, case.pattern)
        assert got == case.expected, (
            f"Case {idx} expected {case.expected}, got {got}. "
            f"text={case.text!r}, pattern={case.pattern!r}"
        )
        assert got == baseline, (
            f"Case {idx} baseline mismatch: kmp={got}, naive={baseline}."
        )
        print(
            f"Case {idx}: text={case.text!r}, pattern={case.pattern!r}, "
            f"matches={got}"
        )


def run_randomized_regression(num_trials: int = 300, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    alphabet = "abc"

    for _ in range(num_trials):
        text_len = rng.randint(0, 40)
        pattern_len = rng.randint(0, 8)

        text = "".join(rng.choice(alphabet) for _ in range(text_len))
        pattern = "".join(rng.choice(alphabet) for _ in range(pattern_len))

        got = kmp_search_all(text, pattern)
        expected = naive_search_all(text, pattern)
        assert got == expected, (
            "Random regression failed: "
            f"text={text!r}, pattern={pattern!r}, kmp={got}, naive={expected}"
        )

    print(f"Randomized regression passed: {num_trials} / {num_trials}")


def main() -> None:
    sample_pattern = "ababaca"
    print("=== KMP Prefix Table Demo ===")
    print(f"pattern={sample_pattern!r}")
    print(f"lps={build_lps(sample_pattern)}")

    run_deterministic_cases()
    run_randomized_regression()

    print("All KMP checks passed.")


if __name__ == "__main__":
    main()
