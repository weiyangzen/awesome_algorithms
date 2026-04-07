"""通配符匹配 MVP：二维 DP + 一维 DP + 贪心回溯交叉校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class WildcardMatchResult:
    text: str
    pattern: str
    normalized_pattern: str
    matched: bool


def validate_text_and_pattern(text: str, pattern: str) -> tuple[str, str]:
    """Validate that text and pattern are both strings."""
    if not isinstance(text, str):
        raise ValueError(f"text must be str, got {type(text).__name__}")
    if not isinstance(pattern, str):
        raise ValueError(f"pattern must be str, got {type(pattern).__name__}")
    return text, pattern


def normalize_pattern(pattern: str) -> str:
    """Collapse consecutive '*' into a single '*' to reduce redundant states."""
    if not pattern:
        return pattern

    out: list[str] = []
    for ch in pattern:
        if ch == "*" and out and out[-1] == "*":
            continue
        out.append(ch)
    return "".join(out)


def wildcard_match_dp_2d(text: str, pattern: str) -> bool:
    """Full DP table solution: O(mn) time and O(mn) space."""
    text, pattern = validate_text_and_pattern(text, pattern)
    pattern = normalize_pattern(pattern)

    m = len(text)
    n = len(pattern)
    dp = np.zeros((m + 1, n + 1), dtype=bool)
    dp[0, 0] = True

    for j in range(1, n + 1):
        if pattern[j - 1] == "*":
            dp[0, j] = bool(dp[0, j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            token = pattern[j - 1]
            if token == "*":
                # '*' either matches empty (left) or one-more-char (up)
                dp[i, j] = bool(dp[i, j - 1] or dp[i - 1, j])
            elif token == "?" or token == text[i - 1]:
                dp[i, j] = bool(dp[i - 1, j - 1])

    return bool(dp[m, n])


def wildcard_match_dp_1d(text: str, pattern: str) -> bool:
    """Rolling DP solution: O(mn) time and O(n) space."""
    text, pattern = validate_text_and_pattern(text, pattern)
    pattern = normalize_pattern(pattern)

    m = len(text)
    n = len(pattern)
    row = np.zeros(n + 1, dtype=bool)
    row[0] = True

    for j in range(1, n + 1):
        if pattern[j - 1] == "*":
            row[j] = bool(row[j - 1])

    for i in range(1, m + 1):
        prev_diag = bool(row[0])
        row[0] = False

        for j in range(1, n + 1):
            up = bool(row[j])
            token = pattern[j - 1]

            if token == "*":
                row[j] = bool(row[j] or row[j - 1])
            else:
                row[j] = bool(prev_diag and (token == "?" or token == text[i - 1]))

            prev_diag = up

    return bool(row[n])


def wildcard_match_greedy(text: str, pattern: str) -> bool:
    """Two-pointer greedy with star backtracking, used as an independent baseline."""
    text, pattern = validate_text_and_pattern(text, pattern)
    pattern = normalize_pattern(pattern)

    i = 0
    j = 0
    star_j = -1
    match_i = 0

    while i < len(text):
        if j < len(pattern) and (pattern[j] == "?" or pattern[j] == text[i]):
            i += 1
            j += 1
        elif j < len(pattern) and pattern[j] == "*":
            star_j = j
            match_i = i
            j += 1
        elif star_j != -1:
            j = star_j + 1
            match_i += 1
            i = match_i
        else:
            return False

    while j < len(pattern) and pattern[j] == "*":
        j += 1

    return j == len(pattern)


def solve_wildcard_match(text: str, pattern: str) -> WildcardMatchResult:
    """Primary API for this MVP (rolling DP)."""
    text, pattern = validate_text_and_pattern(text, pattern)
    normalized = normalize_pattern(pattern)
    matched = wildcard_match_dp_1d(text, normalized)
    return WildcardMatchResult(
        text=text,
        pattern=pattern,
        normalized_pattern=normalized,
        matched=matched,
    )


def run_case(name: str, text: str, pattern: str, expected: bool | None = None) -> None:
    dp_2d = wildcard_match_dp_2d(text, pattern)
    dp_1d = wildcard_match_dp_1d(text, pattern)
    greedy = wildcard_match_greedy(text, pattern)
    api = solve_wildcard_match(text, pattern)

    print(f"=== {name} ===")
    print(f"text={text!r}, pattern={pattern!r}, normalized={api.normalized_pattern!r}")
    print(f"dp_2d={dp_2d}, dp_1d={dp_1d}, greedy={greedy}, api={api.matched}")
    if expected is not None:
        print(f"expected={expected}")
    print()

    if not (dp_2d == dp_1d == greedy == api.matched):
        raise AssertionError("Inconsistent answers across implementations")
    if expected is not None and api.matched != expected:
        raise AssertionError(f"Unexpected answer: got {api.matched}, expected {expected}")


def random_string(rng: np.random.Generator, alphabet: str, length: int) -> str:
    chars = [alphabet[int(rng.integers(0, len(alphabet)))] for _ in range(length)]
    return "".join(chars)


def random_pattern(rng: np.random.Generator, alphabet: str, length: int) -> str:
    tokens = list(alphabet) + ["?", "*"]
    chars = [tokens[int(rng.integers(0, len(tokens)))] for _ in range(length)]
    return "".join(chars)


def randomized_cross_check(
    trials: int = 300,
    max_text_len: int = 12,
    max_pattern_len: int = 12,
    alphabet: str = "abc",
    seed: int = 2026,
) -> None:
    """Random regression among 2D-DP, 1D-DP and greedy baseline."""
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        text_len = int(rng.integers(0, max_text_len + 1))
        pattern_len = int(rng.integers(0, max_pattern_len + 1))
        text = random_string(rng, alphabet=alphabet, length=text_len)
        pattern = random_pattern(rng, alphabet=alphabet, length=pattern_len)

        a = wildcard_match_dp_2d(text, pattern)
        b = wildcard_match_dp_1d(text, pattern)
        c = wildcard_match_greedy(text, pattern)

        if not (a == b == c):
            raise AssertionError(
                "Random cross-check failed: "
                f"text={text!r}, pattern={pattern!r}, answers=({a}, {b}, {c})"
            )

    print(
        "Randomized cross-check passed: "
        f"trials={trials}, max_text_len={max_text_len}, "
        f"max_pattern_len={max_pattern_len}, alphabet={alphabet!r}, seed={seed}."
    )


def run_batch(cases: Iterable[tuple[str, str, str, bool | None]]) -> None:
    for name, text, pattern, expected in cases:
        run_case(name=name, text=text, pattern=pattern, expected=expected)


def main() -> None:
    cases = [
        ("Case 1: empty-empty", "", "", True),
        ("Case 2: empty-star", "", "*", True),
        ("Case 3: empty-question", "", "?", False),
        ("Case 4: simple false", "aa", "a", False),
        ("Case 5: all-match star", "aa", "*", True),
        ("Case 6: leetcode sample false", "cb", "?a", False),
        ("Case 7: leetcode sample true", "adceb", "*a*b", True),
        ("Case 8: leetcode sample false", "acdcb", "a*c?b", False),
        ("Case 9: mixed wildcards", "abcde", "a*?e", True),
        ("Case 10: redundant stars", "abefcdgiescdfimde", "ab**cd?i*de", True),
    ]

    run_batch(cases)
    randomized_cross_check(trials=300, max_text_len=12, max_pattern_len=12, seed=2026)


if __name__ == "__main__":
    main()
