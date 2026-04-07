"""Interleaving String MVP: DP with path reconstruction and cross-checks.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random

import numpy as np


@dataclass
class InterleaveResult:
    is_interleaving: bool
    source_path: str
    reconstructed: str


def validate_text(text: str, name: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be str, got {type(text).__name__}.")
    return text


def build_interleaving_from_path(s1: str, s2: str, path: str) -> str:
    i = 0
    j = 0
    out: list[str] = []

    for step in path:
        if step == "1":
            if i >= len(s1):
                raise ValueError("path consumes s1 out of bounds")
            out.append(s1[i])
            i += 1
        elif step == "2":
            if j >= len(s2):
                raise ValueError("path consumes s2 out of bounds")
            out.append(s2[j])
            j += 1
        else:
            raise ValueError(f"invalid path token: {step!r}")

    if i != len(s1) or j != len(s2):
        raise ValueError("path does not consume all of s1/s2")

    return "".join(out)


def interleave_dp_with_path(s1: str, s2: str, s3: str) -> InterleaveResult:
    a = validate_text(s1, "s1")
    b = validate_text(s2, "s2")
    c = validate_text(s3, "s3")

    m = len(a)
    n = len(b)
    t = len(c)

    if m + n != t:
        return InterleaveResult(is_interleaving=False, source_path="", reconstructed="")

    dp = np.zeros((m + 1, n + 1), dtype=bool)
    parent = np.full((m + 1, n + 1), fill_value=-1, dtype=np.int8)
    dp[0, 0] = True
    parent[0, 0] = 0

    for i in range(0, m + 1):
        for j in range(0, n + 1):
            if i == 0 and j == 0:
                continue

            k = i + j - 1
            from_s1 = i > 0 and bool(dp[i - 1, j]) and a[i - 1] == c[k]
            from_s2 = j > 0 and bool(dp[i, j - 1]) and b[j - 1] == c[k]

            if from_s1 or from_s2:
                dp[i, j] = True
                # Deterministic tie-break: prefer taking from s1 first.
                parent[i, j] = 1 if from_s1 else 2

    if not bool(dp[m, n]):
        return InterleaveResult(is_interleaving=False, source_path="", reconstructed="")

    rev_path: list[str] = []
    i = m
    j = n
    while i > 0 or j > 0:
        p = int(parent[i, j])
        if p == 1:
            rev_path.append("1")
            i -= 1
        elif p == 2:
            rev_path.append("2")
            j -= 1
        else:
            raise RuntimeError(f"invalid parent at state ({i}, {j})")

    path = "".join(reversed(rev_path))
    reconstructed = build_interleaving_from_path(a, b, path)
    return InterleaveResult(is_interleaving=True, source_path=path, reconstructed=reconstructed)


def interleave_memoized(s1: str, s2: str, s3: str) -> bool:
    a = validate_text(s1, "s1")
    b = validate_text(s2, "s2")
    c = validate_text(s3, "s3")

    if len(a) + len(b) != len(c):
        return False

    @lru_cache(maxsize=None)
    def dfs(i: int, j: int) -> bool:
        k = i + j
        if k == len(c):
            return i == len(a) and j == len(b)

        ok1 = i < len(a) and a[i] == c[k] and dfs(i + 1, j)
        if ok1:
            return True

        ok2 = j < len(b) and b[j] == c[k] and dfs(i, j + 1)
        return ok2

    return dfs(0, 0)


def interleave_bruteforce(s1: str, s2: str, s3: str, max_total: int = 18) -> bool:
    """Exact DFS without memoization for small strings only."""
    a = validate_text(s1, "s1")
    b = validate_text(s2, "s2")
    c = validate_text(s3, "s3")

    if len(a) + len(b) != len(c):
        return False
    if len(c) > max_total:
        raise ValueError(f"bruteforce supports len(s3) <= {max_total}, got {len(c)}")

    def dfs(i: int, j: int) -> bool:
        k = i + j
        if k == len(c):
            return i == len(a) and j == len(b)

        ok1 = i < len(a) and a[i] == c[k] and dfs(i + 1, j)
        ok2 = j < len(b) and b[j] == c[k] and dfs(i, j + 1)
        return ok1 or ok2

    return dfs(0, 0)


def run_case(
    name: str,
    s1: str,
    s2: str,
    s3: str,
    expected: bool | None = None,
) -> None:
    dp_result = interleave_dp_with_path(s1, s2, s3)
    memoized = interleave_memoized(s1, s2, s3)
    brute = interleave_bruteforce(s1, s2, s3) if len(s3) <= 18 else None

    path_valid = True
    reconstructed_match = not dp_result.is_interleaving
    if dp_result.is_interleaving:
        try:
            reconstructed = build_interleaving_from_path(s1, s2, dp_result.source_path)
            path_valid = True
            reconstructed_match = reconstructed == s3 == dp_result.reconstructed
        except ValueError:
            path_valid = False
            reconstructed_match = False

    print(f"=== {name} ===")
    print(f"s1={s1!r}, s2={s2!r}, s3={s3!r}")
    print(
        "dp_result        -> "
        f"is_interleaving={dp_result.is_interleaving}, "
        f"path={dp_result.source_path!r}, reconstructed={dp_result.reconstructed!r}"
    )
    print(f"memoized_check   -> {memoized}")
    print(f"bruteforce_check -> {brute}")
    print(
        "checks           -> "
        f"path_valid={path_valid}, reconstructed_match={reconstructed_match}, "
        f"memoized_match={memoized == dp_result.is_interleaving}, "
        f"bruteforce_match={None if brute is None else brute == dp_result.is_interleaving}"
    )
    print()

    if not path_valid:
        raise AssertionError(f"invalid source_path in case: {name}")
    if not reconstructed_match:
        raise AssertionError(f"reconstruction mismatch in case: {name}")
    if memoized != dp_result.is_interleaving:
        raise AssertionError(f"memoized mismatch in case: {name}")
    if brute is not None and brute != dp_result.is_interleaving:
        raise AssertionError(f"bruteforce mismatch in case: {name}")
    if expected is not None and dp_result.is_interleaving != expected:
        raise AssertionError(
            f"expected={expected}, got={dp_result.is_interleaving} in case: {name}"
        )


def random_interleaving_target(rng: Random, s1: str, s2: str) -> str:
    i = 0
    j = 0
    merged: list[str] = []
    while i < len(s1) or j < len(s2):
        if i == len(s1):
            take_s1 = False
        elif j == len(s2):
            take_s1 = True
        else:
            take_s1 = bool(rng.randint(0, 1))

        if take_s1:
            merged.append(s1[i])
            i += 1
        else:
            merged.append(s2[j])
            j += 1

    return "".join(merged)


def randomized_regression(seed: int = 2026, rounds: int = 240) -> None:
    rng = Random(seed)
    alphabet = "abc"

    for _ in range(rounds):
        len1 = rng.randint(0, 6)
        len2 = rng.randint(0, 6)
        s1 = "".join(rng.choice(alphabet) for _ in range(len1))
        s2 = "".join(rng.choice(alphabet) for _ in range(len2))

        if rng.random() < 0.5:
            s3 = random_interleaving_target(rng, s1, s2)
        else:
            s3 = "".join(rng.choice(alphabet) for _ in range(len1 + len2))

        dp_result = interleave_dp_with_path(s1, s2, s3)
        memoized = interleave_memoized(s1, s2, s3)
        brute = interleave_bruteforce(s1, s2, s3, max_total=18)

        assert memoized == dp_result.is_interleaving
        assert brute == dp_result.is_interleaving

        if dp_result.is_interleaving:
            reconstructed = build_interleaving_from_path(s1, s2, dp_result.source_path)
            assert reconstructed == s3 == dp_result.reconstructed
        else:
            assert dp_result.source_path == ""
            assert dp_result.reconstructed == ""

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, len_range=[0,6], alphabet='abc'"
    )


def main() -> None:
    run_case(
        name="Case 1: classic true",
        s1="aabcc",
        s2="dbbca",
        s3="aadbbcbcac",
        expected=True,
    )

    run_case(
        name="Case 2: classic false",
        s1="aabcc",
        s2="dbbca",
        s3="aadbbbaccc",
        expected=False,
    )

    run_case(
        name="Case 3: empty all",
        s1="",
        s2="",
        s3="",
        expected=True,
    )

    run_case(
        name="Case 4: length mismatch",
        s1="",
        s2="",
        s3="a",
        expected=False,
    )

    run_case(
        name="Case 5: only s1",
        s1="abc",
        s2="",
        s3="abc",
        expected=True,
    )

    run_case(
        name="Case 6: only s2",
        s1="",
        s2="xyz",
        s3="xyz",
        expected=True,
    )

    run_case(
        name="Case 7: repeated characters",
        s1="aa",
        s2="ab",
        s3="aaba",
        expected=True,
    )

    randomized_regression(seed=2026, rounds=240)
    print("All interleaving-string checks passed.")


if __name__ == "__main__":
    main()
