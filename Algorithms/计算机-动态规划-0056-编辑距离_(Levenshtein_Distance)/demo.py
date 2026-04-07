"""编辑距离（Levenshtein Distance）最小可运行 MVP。

包含三种实现：
1) 经典二维 DP（返回完整代价表）；
2) 一维滚动数组 DP（空间优化版）；
3) 记忆化递归（仅用于小规模交叉校验）。

脚本无交互输入，直接运行内置样例与随机对拍。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class EditDistanceResult:
    word_a: str
    word_b: str
    distance: int
    normalized_similarity: float


def validate_word(name: str, word: str) -> None:
    if not isinstance(word, str):
        raise TypeError(f"{name} must be str, got {type(word)!r}")


def levenshtein_distance_dp(word_a: str, word_b: str) -> tuple[int, np.ndarray]:
    """经典二维动态规划实现，返回 (distance, dp_table)。"""
    validate_word("word_a", word_a)
    validate_word("word_b", word_b)

    m = len(word_a)
    n = len(word_b)

    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1, dtype=np.int32)
    dp[0, :] = np.arange(n + 1, dtype=np.int32)

    for i in range(1, m + 1):
        a_ch = word_a[i - 1]
        for j in range(1, n + 1):
            b_ch = word_b[j - 1]
            substitution_cost = 0 if a_ch == b_ch else 1

            delete_cost = int(dp[i - 1, j]) + 1
            insert_cost = int(dp[i, j - 1]) + 1
            replace_cost = int(dp[i - 1, j - 1]) + substitution_cost

            dp[i, j] = min(delete_cost, insert_cost, replace_cost)

    return int(dp[m, n]), dp


def levenshtein_distance_optimized(word_a: str, word_b: str) -> int:
    """一维滚动数组版本，空间复杂度 O(min(m, n))。"""
    validate_word("word_a", word_a)
    validate_word("word_b", word_b)

    # 让 second 为较短串，降低空间占用。
    first, second = word_a, word_b
    if len(second) > len(first):
        first, second = second, first

    prev = np.arange(len(second) + 1, dtype=np.int32)

    for i, ch_first in enumerate(first, start=1):
        curr = np.zeros(len(second) + 1, dtype=np.int32)
        curr[0] = i

        for j, ch_second in enumerate(second, start=1):
            substitution_cost = 0 if ch_first == ch_second else 1

            delete_cost = int(prev[j]) + 1
            insert_cost = int(curr[j - 1]) + 1
            replace_cost = int(prev[j - 1]) + substitution_cost

            curr[j] = min(delete_cost, insert_cost, replace_cost)

        prev = curr

    return int(prev[-1])


def levenshtein_distance_memo(word_a: str, word_b: str) -> int:
    """记忆化递归版本（用于小字符串对拍，不用于大规模计算）。"""
    validate_word("word_a", word_a)
    validate_word("word_b", word_b)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i

        substitution_cost = 0 if word_a[i - 1] == word_b[j - 1] else 1
        return min(
            solve(i - 1, j) + 1,
            solve(i, j - 1) + 1,
            solve(i - 1, j - 1) + substitution_cost,
        )

    return solve(len(word_a), len(word_b))


def normalized_similarity(distance: int, len_a: int, len_b: int) -> float:
    max_len = max(len_a, len_b)
    if max_len == 0:
        return 1.0
    return 1.0 - float(distance) / float(max_len)


def format_dp_table(word_a: str, word_b: str, dp: np.ndarray) -> str:
    """将 DP 表格式化为便于查看的纯文本。"""
    row_labels = ["∅"] + list(word_a)
    col_labels = ["∅"] + list(word_b)

    lines: list[str] = []
    header = ["  "] + [f"{c:>3s}" for c in col_labels]
    lines.append("".join(header))

    for i, row_label in enumerate(row_labels):
        row_values = "".join(f"{int(v):3d}" for v in dp[i])
        lines.append(f"{row_label:>2s}{row_values}")

    return "\n".join(lines)


def run_case(word_a: str, word_b: str, expected: int | None = None) -> EditDistanceResult:
    distance_dp, dp_table = levenshtein_distance_dp(word_a, word_b)
    distance_opt = levenshtein_distance_optimized(word_a, word_b)

    # 记忆化递归仅用于小规模，避免不必要开销。
    if len(word_a) <= 10 and len(word_b) <= 10:
        distance_memo = levenshtein_distance_memo(word_a, word_b)
    else:
        distance_memo = distance_dp

    if distance_dp != distance_opt or distance_dp != distance_memo:
        raise AssertionError(
            f"distance mismatch: dp={distance_dp}, optimized={distance_opt}, memo={distance_memo}"
        )

    if expected is not None and distance_dp != expected:
        raise AssertionError(
            f"unexpected distance for ({word_a!r}, {word_b!r}): "
            f"got {distance_dp}, expected {expected}"
        )

    score = normalized_similarity(distance_dp, len(word_a), len(word_b))

    print(f"=== case: {word_a!r} -> {word_b!r} ===")
    print(
        f"distance={distance_dp}, similarity={score:.4f}, "
        f"len_a={len(word_a)}, len_b={len(word_b)}"
    )

    if len(word_a) <= 7 and len(word_b) <= 7:
        print("DP table:")
        print(format_dp_table(word_a, word_b, dp_table))

    print()

    return EditDistanceResult(
        word_a=word_a,
        word_b=word_b,
        distance=distance_dp,
        normalized_similarity=score,
    )


def randomized_cross_check(trials: int = 200, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)
    alphabet = list("abcd")

    for _ in range(trials):
        len_a = int(rng.integers(0, 9))
        len_b = int(rng.integers(0, 9))

        word_a = "".join(rng.choice(alphabet, size=len_a))
        word_b = "".join(rng.choice(alphabet, size=len_b))

        d1, _ = levenshtein_distance_dp(word_a, word_b)
        d2 = levenshtein_distance_optimized(word_a, word_b)
        d3 = levenshtein_distance_memo(word_a, word_b)

        if d1 != d2 or d1 != d3:
            raise AssertionError(
                f"random check failed for ({word_a!r}, {word_b!r}): {d1}, {d2}, {d3}"
            )

    print(f"Randomized cross-check passed: {trials} trials (seed={seed}).")


def main() -> None:
    print("Levenshtein Distance MVP (Dynamic Programming)")

    cases = [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("algorithm", "algorithm", 0),
    ]

    results: list[EditDistanceResult] = []
    for word_a, word_b, expected in cases:
        results.append(run_case(word_a, word_b, expected=expected))

    randomized_cross_check(trials=200, seed=2026)

    max_distance = max(item.distance for item in results)
    min_similarity = min(item.normalized_similarity for item in results)

    print("\n=== summary ===")
    for item in results:
        print(
            f"{item.word_a!r} -> {item.word_b!r}: "
            f"distance={item.distance}, similarity={item.normalized_similarity:.4f}"
        )

    global_ok = (max_distance == 5) and (min_similarity >= 0.0)
    print(f"global checks pass: {global_ok}")
    print(f"aggregate stats: max_distance={max_distance}, min_similarity={min_similarity:.4f}")


if __name__ == "__main__":
    main()
