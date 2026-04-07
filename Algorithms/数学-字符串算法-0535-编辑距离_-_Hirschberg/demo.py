"""Edit distance MVP using Hirschberg divide-and-conquer alignment.

This script provides:
- Linear-space Hirschberg alignment reconstruction (unit costs)
- Full-matrix Wagner-Fischer distance as a correctness baseline
- Fixed test cases, no interactive input
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AlignmentResult:
    distance: int
    aligned_source: str
    aligned_target: str
    operations: List[str]  # each item in {"M", "S", "D", "I"}


def _validate_string(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}.")
    return value


def _last_row_edit_distance(source: str, target: str) -> List[int]:
    """Return DP last row for edit distance(source, target prefixes)."""
    n = len(target)
    prev = list(range(n + 1))

    for i, ch_s in enumerate(source, start=1):
        curr = [i] + [0] * n
        for j, ch_t in enumerate(target, start=1):
            replace_cost = 0 if ch_s == ch_t else 1
            curr[j] = min(
                prev[j] + 1,  # delete
                curr[j - 1] + 1,  # insert
                prev[j - 1] + replace_cost,  # match/substitute
            )
        prev = curr

    return prev


def _small_alignment_dp(source: str, target: str) -> AlignmentResult:
    """Full DP + backtrace for small/base cases."""
    m, n = len(source), len(target)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ch_s = source[i - 1]
        for j in range(1, n + 1):
            ch_t = target[j - 1]
            replace_cost = 0 if ch_s == ch_t else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + replace_cost,
            )

    i, j = m, n
    aligned_s_rev: List[str] = []
    aligned_t_rev: List[str] = []
    ops_rev: List[str] = []

    while i > 0 or j > 0:
        moved = False

        if i > 0 and j > 0:
            replace_cost = 0 if source[i - 1] == target[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + replace_cost:
                aligned_s_rev.append(source[i - 1])
                aligned_t_rev.append(target[j - 1])
                ops_rev.append("M" if replace_cost == 0 else "S")
                i -= 1
                j -= 1
                moved = True

        if not moved and i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_s_rev.append(source[i - 1])
            aligned_t_rev.append("-")
            ops_rev.append("D")
            i -= 1
            moved = True

        if not moved and j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned_s_rev.append("-")
            aligned_t_rev.append(target[j - 1])
            ops_rev.append("I")
            j -= 1
            moved = True

        if not moved:
            raise RuntimeError("Backtrace failed: no valid predecessor found.")

    aligned_source = "".join(reversed(aligned_s_rev))
    aligned_target = "".join(reversed(aligned_t_rev))
    operations = list(reversed(ops_rev))

    return AlignmentResult(
        distance=dp[m][n],
        aligned_source=aligned_source,
        aligned_target=aligned_target,
        operations=operations,
    )


def hirschberg_edit_alignment(source: str, target: str) -> AlignmentResult:
    """Compute edit alignment with Hirschberg divide-and-conquer.

    Costs: insertion=1, deletion=1, substitution=1, match=0.
    Time: O(mn), Space: O(n) excluding recursion output.
    """
    source = _validate_string(source, "source")
    target = _validate_string(target, "target")

    m, n = len(source), len(target)

    if m == 0 or n == 0 or m == 1 or n == 1:
        return _small_alignment_dp(source, target)

    mid = m // 2
    left_source = source[:mid]
    right_source = source[mid:]

    score_left = _last_row_edit_distance(left_source, target)
    score_right_rev = _last_row_edit_distance(right_source[::-1], target[::-1])

    split = 0
    best = float("inf")
    for j in range(n + 1):
        cost = score_left[j] + score_right_rev[n - j]
        if cost < best:
            best = cost
            split = j

    left_result = hirschberg_edit_alignment(left_source, target[:split])
    right_result = hirschberg_edit_alignment(right_source, target[split:])

    return AlignmentResult(
        distance=left_result.distance + right_result.distance,
        aligned_source=left_result.aligned_source + right_result.aligned_source,
        aligned_target=left_result.aligned_target + right_result.aligned_target,
        operations=left_result.operations + right_result.operations,
    )


def wagner_fischer_distance(source: str, target: str) -> int:
    """Classic full-matrix edit distance for verification."""
    source = _validate_string(source, "source")
    target = _validate_string(target, "target")

    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ch_s = source[i - 1]
        for j in range(1, n + 1):
            ch_t = target[j - 1]
            replace_cost = 0 if ch_s == ch_t else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + replace_cost,
            )

    return dp[m][n]


def _op_counts(operations: List[str]) -> Tuple[int, int, int, int]:
    m = sum(1 for op in operations if op == "M")
    s = sum(1 for op in operations if op == "S")
    d = sum(1 for op in operations if op == "D")
    i = sum(1 for op in operations if op == "I")
    return m, s, d, i


def _assert_alignment_valid(result: AlignmentResult, source: str, target: str) -> None:
    if len(result.aligned_source) != len(result.aligned_target):
        raise AssertionError("Aligned strings must have equal length.")
    if len(result.operations) != len(result.aligned_source):
        raise AssertionError("Operation length must match alignment length.")

    rebuilt_source = result.aligned_source.replace("-", "")
    rebuilt_target = result.aligned_target.replace("-", "")
    if rebuilt_source != source:
        raise AssertionError("Aligned source cannot be restored to original source.")
    if rebuilt_target != target:
        raise AssertionError("Aligned target cannot be restored to original target.")

    calc_distance = 0
    for op, ch_s, ch_t in zip(result.operations, result.aligned_source, result.aligned_target):
        if op == "M":
            if ch_s != ch_t:
                raise AssertionError("M operation must align equal characters.")
        elif op == "S":
            if ch_s == ch_t or ch_s == "-" or ch_t == "-":
                raise AssertionError("S operation must be substitution of two different chars.")
            calc_distance += 1
        elif op == "D":
            if ch_t != "-" or ch_s == "-":
                raise AssertionError("D operation must delete one source char.")
            calc_distance += 1
        elif op == "I":
            if ch_s != "-" or ch_t == "-":
                raise AssertionError("I operation must insert one target char.")
            calc_distance += 1
        else:
            raise AssertionError(f"Unknown operation code: {op}")

    if calc_distance != result.distance:
        raise AssertionError(
            f"Distance mismatch by operations: {calc_distance} != {result.distance}"
        )


def run_case(name: str, source: str, target: str) -> None:
    result = hirschberg_edit_alignment(source, target)
    baseline = wagner_fischer_distance(source, target)
    _assert_alignment_valid(result, source, target)

    if result.distance != baseline:
        raise AssertionError(
            f"Distance mismatch in {name}: Hirschberg={result.distance}, baseline={baseline}"
        )

    m_cnt, s_cnt, d_cnt, i_cnt = _op_counts(result.operations)

    print(f"=== {name} ===")
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"distance: {result.distance} (baseline={baseline})")
    print(f"ops count: M={m_cnt}, S={s_cnt}, D={d_cnt}, I={i_cnt}")
    print(f"align S: {result.aligned_source}")
    print(f"align T: {result.aligned_target}")
    print(f"ops   : {''.join(result.operations)}")
    print()


def main() -> None:
    cases = [
        ("Case 1: classic", "kitten", "sitting"),
        ("Case 2: textbook", "intention", "execution"),
        ("Case 3: insertion only", "", "abc"),
        ("Case 4: deletion only", "sunday", ""),
        ("Case 5: mixed", "algorithm", "altruistic"),
        ("Case 6: near-equal", "Hirschberg", "Hirshberg"),
    ]

    for name, source, target in cases:
        run_case(name, source, target)


if __name__ == "__main__":
    main()
