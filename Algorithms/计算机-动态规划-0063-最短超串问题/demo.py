"""最短超串问题 MVP：状态压缩 DP + 小规模暴力对拍。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from random import Random

import numpy as np


@dataclass
class ShortestSuperstringResult:
    input_words: list[str]
    normalized_words: list[str]
    superstring: str
    length: int


def validate_words(words: list[str]) -> list[str]:
    """Validate raw input word list."""
    if not isinstance(words, list):
        raise TypeError(f"words must be list[str], got {type(words).__name__}")
    if not words:
        raise ValueError("words must not be empty")
    if any((not isinstance(w, str)) for w in words):
        raise TypeError("each word must be str")
    if any((w == "") for w in words):
        raise ValueError("empty string is not supported in this MVP")
    return words


def normalize_words(words: list[str]) -> list[str]:
    """Deduplicate and remove words fully contained in others.

    This normalization keeps the optimization target unchanged while reducing
    DP state count and improving determinism.
    """
    raw = validate_words(words)
    unique = sorted(set(raw))

    reduced: list[str] = []
    for w in unique:
        contained = False
        for other in unique:
            if w != other and w in other:
                contained = True
                break
        if not contained:
            reduced.append(w)

    return sorted(reduced)


def build_overlap_matrix(words: list[str]) -> np.ndarray:
    """overlap[i, j] = max k s.t. words[i] suffix == words[j] prefix of length k."""
    n = len(words)
    overlap = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_k = min(len(words[i]), len(words[j]))
            for k in range(max_k, 0, -1):
                if words[i].endswith(words[j][:k]):
                    overlap[i, j] = k
                    break

    return overlap


def shortest_superstring_dp(words: list[str]) -> str:
    """Exact shortest superstring via subset DP.

    State:
        dp[mask][j] = best (shortest, lexicographically smallest) superstring
                      covering subset `mask` and ending with words[j].
    """
    normalized = normalize_words(words)
    n = len(normalized)
    if n == 1:
        return normalized[0]

    overlap = build_overlap_matrix(normalized)
    total_masks = 1 << n
    dp: list[list[str | None]] = [[None] * n for _ in range(total_masks)]

    for j in range(n):
        dp[1 << j][j] = normalized[j]

    for mask in range(total_masks):
        for end in range(n):
            cur = dp[mask][end]
            if cur is None:
                continue

            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue

                nmask = mask | bit
                ov = int(overlap[end, nxt])
                cand = cur + normalized[nxt][ov:]
                old = dp[nmask][nxt]

                if old is None or len(cand) < len(old) or (
                    len(cand) == len(old) and cand < old
                ):
                    dp[nmask][nxt] = cand

    full = total_masks - 1
    best: str | None = None
    for end in range(n):
        cand = dp[full][end]
        if cand is None:
            continue
        if best is None or len(cand) < len(best) or (
            len(cand) == len(best) and cand < best
        ):
            best = cand

    if best is None:
        raise RuntimeError("DP failed to produce a superstring")
    return best


def shortest_superstring_length_dp(words: list[str]) -> int:
    """Length-only subset DP, used as independent cross-check."""
    normalized = normalize_words(words)
    n = len(normalized)
    if n == 1:
        return len(normalized[0])

    overlap = build_overlap_matrix(normalized)
    total_masks = 1 << n
    inf = 10**9

    dp = np.full((total_masks, n), inf, dtype=np.int32)
    for j in range(n):
        dp[1 << j, j] = len(normalized[j])

    for mask in range(total_masks):
        for end in range(n):
            cur = int(dp[mask, end])
            if cur >= inf:
                continue
            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                nmask = mask | bit
                cand = cur + len(normalized[nxt]) - int(overlap[end, nxt])
                if cand < int(dp[nmask, nxt]):
                    dp[nmask, nxt] = cand

    full = total_masks - 1
    return int(np.min(dp[full]))


def bruteforce_shortest_superstring(words: list[str], max_n: int = 8) -> str:
    """Exact brute force by permutations, for small n only."""
    normalized = normalize_words(words)
    n = len(normalized)
    if n > max_n:
        raise ValueError(f"bruteforce supports n <= {max_n}, got {n}")
    if n == 1:
        return normalized[0]

    overlap = build_overlap_matrix(normalized)
    best: str | None = None

    for perm in permutations(range(n)):
        s = normalized[perm[0]]
        for k in range(1, n):
            i = perm[k - 1]
            j = perm[k]
            ov = int(overlap[i, j])
            s += normalized[j][ov:]

        if best is None or len(s) < len(best) or (len(s) == len(best) and s < best):
            best = s

    if best is None:
        raise RuntimeError("bruteforce failed to produce a superstring")
    return best


def solve_shortest_superstring(words: list[str]) -> ShortestSuperstringResult:
    raw = validate_words(words)
    normalized = normalize_words(raw)
    best = shortest_superstring_dp(raw)
    return ShortestSuperstringResult(
        input_words=list(raw),
        normalized_words=normalized,
        superstring=best,
        length=len(best),
    )


def run_case(
    name: str,
    words: list[str],
    expected_length: int | None = None,
    expected_superstring: str | None = None,
) -> None:
    result = solve_shortest_superstring(words)

    covers_input = all(w in result.superstring for w in result.input_words)
    covers_normalized = all(w in result.superstring for w in result.normalized_words)
    length_check = shortest_superstring_length_dp(words)

    brute = None
    if len(result.normalized_words) <= 8:
        brute = bruteforce_shortest_superstring(words, max_n=8)

    print(f"=== {name} ===")
    print(f"input_words         = {result.input_words}")
    print(f"normalized_words    = {result.normalized_words}")
    print(f"dp_result           -> {result.superstring!r} (len={result.length})")
    print(f"length_dp_check     -> len={length_check}")
    print(f"bruteforce_check    -> {None if brute is None else repr(brute)}")
    print(
        "checks              -> "
        f"covers_input={covers_input}, covers_normalized={covers_normalized}, "
        f"length_match={result.length == length_check}, "
        f"bruteforce_match={None if brute is None else result.superstring == brute}"
    )
    print()

    if not covers_input or not covers_normalized:
        raise AssertionError(f"substring coverage failed in case: {name}")
    if result.length != length_check:
        raise AssertionError(f"length cross-check failed in case: {name}")
    if brute is not None and result.superstring != brute:
        raise AssertionError(f"bruteforce mismatch in case: {name}")
    if expected_length is not None and result.length != expected_length:
        raise AssertionError(
            f"expected length mismatch in {name}: {result.length} != {expected_length}"
        )
    if expected_superstring is not None and result.superstring != expected_superstring:
        raise AssertionError(
            "expected superstring mismatch in "
            f"{name}: {result.superstring!r} != {expected_superstring!r}"
        )


def random_words(rng: Random, count: int, alphabet: str = "abcd") -> list[str]:
    words: list[str] = []
    for _ in range(count):
        length = rng.randint(1, 5)
        w = "".join(rng.choice(alphabet) for _ in range(length))
        words.append(w)
    return words


def randomized_regression(seed: int = 2026, rounds: int = 160) -> None:
    rng = Random(seed)

    for _ in range(rounds):
        count = rng.randint(2, 7)
        words = random_words(rng, count)

        result = solve_shortest_superstring(words)
        assert all(w in result.superstring for w in result.input_words)
        assert all(w in result.superstring for w in result.normalized_words)

        length_check = shortest_superstring_length_dp(words)
        assert result.length == length_check

        brute = bruteforce_shortest_superstring(words, max_n=8)
        assert result.superstring == brute

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, word_count=[2,7], word_len=[1,5], alphabet='abcd'"
    )


def main() -> None:
    cases = [
        (
            "Case 1: classic leetcode",
            ["alex", "loves", "leetcode"],
            17,
            None,
        ),
        (
            "Case 2: overlap-rich",
            ["catg", "ctaagt", "gcta", "ttca", "atgcatc"],
            16,
            None,
        ),
        ("Case 3: containment", ["abc", "bc", "c"], 3, None),
        ("Case 4: chain", ["ab", "bc", "cd"], 4, None),
        ("Case 5: tie-break", ["ab", "ba"], 3, "aba"),
        ("Case 6: duplicate", ["aba", "bab", "aba"], 4, "abab"),
    ]

    for name, words, expected_len, expected_s in cases:
        run_case(name, words, expected_length=expected_len, expected_superstring=expected_s)

    randomized_regression(seed=2026, rounds=160)
    print("All shortest-superstring checks passed.")


if __name__ == "__main__":
    main()
