"""LCS MVP: classic O(mn) DP with sequence reconstruction and cross-check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np


@dataclass
class LCSResult:
    length: int
    indices_a: List[int]
    indices_b: List[int]
    subsequence: List[Any]


def to_token_list(data: str | Sequence[Any] | np.ndarray) -> List[Any]:
    """Convert input into a 1D token list."""
    if isinstance(data, str):
        return list(data)
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(f"Input ndarray must be 1D, got shape={data.shape}.")
        return data.tolist()
    try:
        return list(data)
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise ValueError("Input must be a string, 1D ndarray, or sequence.") from exc


def lcs_dynamic_programming(
    seq_a: str | Sequence[Any] | np.ndarray,
    seq_b: str | Sequence[Any] | np.ndarray,
) -> LCSResult:
    """
    Compute one LCS using full DP table and backtracking.

    Returns both sequence indices and token values for inspection.
    """
    a = to_token_list(seq_a)
    b = to_token_list(seq_b)
    m, n = len(a), len(b)

    dp = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                up = dp[i - 1, j]
                left = dp[i, j - 1]
                dp[i, j] = up if up >= left else left

    indices_a_rev: List[int] = []
    indices_b_rev: List[int] = []
    subseq_rev: List[Any] = []
    i, j = m, n

    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            indices_a_rev.append(i - 1)
            indices_b_rev.append(j - 1)
            subseq_rev.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1

    indices_a = list(reversed(indices_a_rev))
    indices_b = list(reversed(indices_b_rev))
    subsequence = list(reversed(subseq_rev))

    return LCSResult(
        length=int(dp[m, n]),
        indices_a=indices_a,
        indices_b=indices_b,
        subsequence=subsequence,
    )


def lcs_length_two_row(
    seq_a: str | Sequence[Any] | np.ndarray,
    seq_b: str | Sequence[Any] | np.ndarray,
) -> int:
    """Space-optimized LCS length in O(min(m, n)) memory."""
    a = to_token_list(seq_a)
    b = to_token_list(seq_b)

    if len(a) <= len(b):
        short_seq = a
        long_seq = b
    else:
        short_seq = b
        long_seq = a

    prev = np.zeros(len(short_seq) + 1, dtype=np.int32)
    cur = np.zeros(len(short_seq) + 1, dtype=np.int32)

    for x in long_seq:
        cur[0] = 0
        for j, y in enumerate(short_seq, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                up = prev[j]
                left = cur[j - 1]
                cur[j] = up if up >= left else left
        prev, cur = cur, prev

    return int(prev[-1])


def is_subsequence(candidate: Sequence[Any], base: Sequence[Any]) -> bool:
    """Check whether candidate is a subsequence of base."""
    if not candidate:
        return True
    pos = 0
    for token in base:
        if token == candidate[pos]:
            pos += 1
            if pos == len(candidate):
                return True
    return False


def format_tokens(tokens: Sequence[Any]) -> str:
    if tokens and all(isinstance(tok, str) and len(tok) == 1 for tok in tokens):
        return "".join(str(tok) for tok in tokens)
    return str(list(tokens))


def run_case(name: str, seq_a: Sequence[Any] | str, seq_b: Sequence[Any] | str) -> None:
    a = to_token_list(seq_a)
    b = to_token_list(seq_b)

    full = lcs_dynamic_programming(a, b)
    len_two_row = lcs_length_two_row(a, b)

    in_a = is_subsequence(full.subsequence, a)
    in_b = is_subsequence(full.subsequence, b)
    len_ok = full.length == len_two_row
    index_ok = (
        len(full.indices_a) == full.length
        and len(full.indices_b) == full.length
        and all(0 <= idx < len(a) for idx in full.indices_a)
        and all(0 <= idx < len(b) for idx in full.indices_b)
    )

    print(f"=== {name} ===")
    print(f"A: {format_tokens(a)}")
    print(f"B: {format_tokens(b)}")
    print(f"LCS length (full table): {full.length}")
    print(f"LCS length (two-row):    {len_two_row}")
    print(f"LCS indices in A: {full.indices_a}")
    print(f"LCS indices in B: {full.indices_b}")
    print(f"LCS subsequence:  {format_tokens(full.subsequence)}")
    print(
        "Checks: "
        f"length_equal={len_ok}, "
        f"subseq_in_a={in_a}, "
        f"subseq_in_b={in_b}, "
        f"indices_well_formed={index_ok}"
    )
    print()

    if not len_ok:
        raise AssertionError(
            f"LCS length mismatch in {name}: full={full.length}, two-row={len_two_row}"
        )
    if not (in_a and in_b and index_ok):
        raise AssertionError(f"LCS reconstruction failed checks in {name}.")


def main() -> None:
    cases = [
        ("Case 1: classic strings", "ABCBDAB", "BDCABA"),
        ("Case 2: repeated chars", "AAAAAB", "BAAAAC"),
        ("Case 3: integer tokens", [1, 3, 4, 1, 2, 3], [3, 4, 1, 2, 1, 3]),
        ("Case 4: no common token", "XYZ", "ABC"),
        ("Case 5: one empty input", "", "ABCDE"),
    ]
    for name, a, b in cases:
        run_case(name, a, b)


if __name__ == "__main__":
    main()
