"""Regular expression matching (supports '.' and '*') via dynamic programming."""

from __future__ import annotations

import numpy as np


def char_match(sc: str, pc: str) -> bool:
    """Return whether one text char matches one pattern char."""
    return pc == "." or sc == pc


def is_match_dp(s: str, p: str) -> bool:
    """Return True iff pattern p fully matches string s."""
    m, n = len(s), len(p)
    dp = np.zeros((m + 1, n + 1), dtype=bool)
    dp[0, 0] = True

    # Empty string can be matched by patterns like a*b*c*...
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0, j] = dp[0, j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pj = p[j - 1]
            if pj == "*":
                # Invalid leading '*': keep False.
                if j < 2:
                    continue

                # Case 1: use the x* as zero occurrence.
                dp[i, j] = dp[i, j - 2]

                # Case 2: use x* as one or more occurrences.
                if char_match(s[i - 1], p[j - 2]):
                    dp[i, j] = dp[i, j] or dp[i - 1, j]
            else:
                if char_match(s[i - 1], pj):
                    dp[i, j] = dp[i - 1, j - 1]

    return bool(dp[m, n])


def run_demo_cases() -> None:
    """Run a small deterministic test suite."""
    cases: list[tuple[str, str, bool]] = [
        ("", "", True),
        ("", "a*", True),
        ("", ".", False),
        ("aa", "a", False),
        ("aa", "a*", True),
        ("ab", ".*", True),
        ("aab", "c*a*b", True),
        ("mississippi", "mis*is*p*.", False),
        ("aaa", "ab*a*c*a", True),
        ("abcd", "d*", False),
    ]

    all_passed = True
    for idx, (s, p, expected) in enumerate(cases, start=1):
        got = is_match_dp(s, p)
        ok = got == expected
        all_passed = all_passed and ok
        print(
            f"[{idx:02d}] s={s!r}, p={p!r} -> got={got}, "
            f"expected={expected}, pass={ok}"
        )

    if not all_passed:
        raise AssertionError("Some demo cases failed.")

    print("All demo cases passed.")


def main() -> None:
    run_demo_cases()


if __name__ == "__main__":
    main()
