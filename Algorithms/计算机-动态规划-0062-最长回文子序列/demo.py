"""Longest Palindromic Subsequence (LPS) MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np


@dataclass
class LPSResult:
    length: int
    subsequence: str


def validate_text(text: str) -> str:
    """Validate input text for the MVP solver."""
    if not isinstance(text, str):
        raise TypeError(f"Input must be str, got {type(text).__name__}.")
    return text


def is_palindrome(text: str) -> bool:
    return text == text[::-1]


def is_subsequence(sub: str, text: str) -> bool:
    i = 0
    for ch in text:
        if i < len(sub) and sub[i] == ch:
            i += 1
    return i == len(sub)


def lps_dp_with_reconstruction(text: str) -> LPSResult:
    """Compute one longest palindromic subsequence via interval DP.

    State:
        dp[i, j] = LPS length inside text[i:j+1]
    Transition:
        if text[i] == text[j]:
            dp[i, j] = dp[i+1, j-1] + 2
        else:
            dp[i, j] = max(dp[i+1, j], dp[i, j-1])
    """
    s = validate_text(text)
    n = len(s)
    if n == 0:
        return LPSResult(length=0, subsequence="")

    dp = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        dp[i, i] = 1

    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width - 1
            if s[i] == s[j]:
                inner = int(dp[i + 1, j - 1]) if i + 1 <= j - 1 else 0
                dp[i, j] = inner + 2
            else:
                dp[i, j] = max(int(dp[i + 1, j]), int(dp[i, j - 1]))

    # Reconstruct one valid optimal subsequence.
    i, j = 0, n - 1
    left: list[str] = []
    right: list[str] = []

    while i <= j:
        if i == j:
            if int(dp[i, j]) == 1:
                left.append(s[i])
            break

        if s[i] == s[j]:
            inner = int(dp[i + 1, j - 1]) if i + 1 <= j - 1 else 0
            if int(dp[i, j]) == inner + 2:
                left.append(s[i])
                right.append(s[j])
                i += 1
                j -= 1
                continue

        # Deterministic tie-breaker: move i first when equal.
        if int(dp[i + 1, j]) >= int(dp[i, j - 1]):
            i += 1
        else:
            j -= 1

    subseq = "".join(left + right[::-1])
    return LPSResult(length=int(dp[0, n - 1]), subsequence=subseq)


def lps_via_lcs_length(text: str) -> int:
    """Cross-check length by LCS(text, reversed(text))."""
    s = validate_text(text)
    rev = s[::-1]
    n = len(s)

    # list-of-lists implementation intentionally differs from numpy interval DP.
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        si = s[i - 1]
        for j in range(1, n + 1):
            if si == rev[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][n]


def bruteforce_lps_length(text: str, max_n: int = 16) -> int:
    """Exact exhaustive solver for short strings, used only for tests."""
    s = validate_text(text)
    n = len(s)
    if n > max_n:
        raise ValueError(f"bruteforce supports n <= {max_n}, got {n}.")

    best = 0
    for mask in range(1 << n):
        chars = [s[i] for i in range(n) if (mask >> i) & 1]
        candidate = "".join(chars)
        if len(candidate) > best and is_palindrome(candidate):
            best = len(candidate)
    return best


def run_case(name: str, text: str) -> None:
    result = lps_dp_with_reconstruction(text)
    lcs_len = lps_via_lcs_length(text)
    brute_len = bruteforce_lps_length(text) if len(text) <= 16 else None

    valid_pal = is_palindrome(result.subsequence)
    valid_subseq = is_subsequence(result.subsequence, text)
    valid_length = len(result.subsequence) == result.length

    print(f"=== {name} ===")
    print(f"text               = {text!r}")
    print(
        "dp_result          -> "
        f"length={result.length}, subsequence={result.subsequence!r}"
    )
    print(f"lcs_cross_check    -> length={lcs_len}")
    print(f"bruteforce_check   -> length={brute_len}")
    print(
        "checks             -> "
        f"palindrome={valid_pal}, is_subsequence={valid_subseq}, "
        f"length_match={valid_length}, lcs_match={lcs_len == result.length}, "
        f"bruteforce_match={None if brute_len is None else brute_len == result.length}"
    )
    print()

    if not valid_pal:
        raise AssertionError(f"Result is not palindrome in case: {name}")
    if not valid_subseq:
        raise AssertionError(f"Result is not a subsequence in case: {name}")
    if not valid_length:
        raise AssertionError(f"Length mismatch in case: {name}")
    if lcs_len != result.length:
        raise AssertionError(f"LCS cross-check failed in case: {name}")
    if brute_len is not None and brute_len != result.length:
        raise AssertionError(f"Bruteforce mismatch in case: {name}")


def randomized_regression(seed: int = 2026, rounds: int = 200) -> None:
    """Randomized consistency checks on short strings."""
    rng = Random(seed)
    alphabet = "abcde"

    for _ in range(rounds):
        n = rng.randint(0, 11)
        text = "".join(rng.choice(alphabet) for _ in range(n))

        result = lps_dp_with_reconstruction(text)
        lcs_len = lps_via_lcs_length(text)
        brute_len = bruteforce_lps_length(text, max_n=16)

        assert is_palindrome(result.subsequence)
        assert is_subsequence(result.subsequence, text)
        assert len(result.subsequence) == result.length
        assert lcs_len == result.length
        assert brute_len == result.length

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, n_range=[0,11], alphabet='abcde'"
    )


def main() -> None:
    cases = {
        "Case 1: leetcode-like": "bbbab",
        "Case 2: short": "cbbd",
        "Case 3: empty": "",
        "Case 4: single": "a",
        "Case 5: classic": "agbdba",
        "Case 6: word": "character",
        "Case 7: no repeats": "abcd",
    }

    for name, text in cases.items():
        run_case(name, text)

    randomized_regression()

    print("All LPS checks passed.")


if __name__ == "__main__":
    main()
