"""Minimal runnable MVP for Manacher algorithm.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ManacherResult:
    """Algorithm output bundle for one input text."""

    text: str
    transformed: str
    radii: list[int]
    longest: str
    start: int
    length: int
    palindrome_count: int


def transform_text(text: str) -> str:
    """Convert text to unified form for odd/even palindrome handling."""
    return "^#" + "#".join(text) + "#$"


def manacher(text: str) -> ManacherResult:
    """Run Manacher in O(n) and return longest palindrome + radius data."""
    transformed = transform_text(text)
    n = len(transformed)
    radii = [0] * n

    center = 0
    right = 0

    for i in range(1, n - 1):
        mirror = 2 * center - i

        if i < right:
            radii[i] = min(right - i, radii[mirror])

        while transformed[i + 1 + radii[i]] == transformed[i - 1 - radii[i]]:
            radii[i] += 1

        if i + radii[i] > right:
            center = i
            right = i + radii[i]

    best_center = 0
    best_len = 0
    best_start = 0

    for i, radius in enumerate(radii):
        if radius == 0:
            continue
        start = (i - radius) // 2
        if radius > best_len or (radius == best_len and start < best_start):
            best_len = radius
            best_center = i
            best_start = start

    if best_len == 0:
        longest = ""
        best_start = 0
    else:
        longest = text[best_start : best_start + best_len]

    palindrome_count = sum((radius + 1) // 2 for radius in radii)

    return ManacherResult(
        text=text,
        transformed=transformed,
        radii=radii,
        longest=longest,
        start=best_start,
        length=best_len,
        palindrome_count=palindrome_count,
    )


def brute_force_longest_palindrome(text: str) -> tuple[str, int, int]:
    """Return (longest_substring, start_index, length) with earliest-start tie break."""
    n = len(text)
    best_start = 0
    best_len = 0

    for i in range(n):
        for j in range(i, n):
            sub = text[i : j + 1]
            if sub == sub[::-1]:
                cur_len = j - i + 1
                if cur_len > best_len or (cur_len == best_len and i < best_start):
                    best_len = cur_len
                    best_start = i

    if best_len == 0:
        return "", 0, 0
    return text[best_start : best_start + best_len], best_start, best_len


def brute_force_palindrome_count(text: str) -> int:
    """Count palindromic substrings with O(n^3) brute force for validation."""
    n = len(text)
    count = 0
    for i in range(n):
        for j in range(i, n):
            sub = text[i : j + 1]
            if sub == sub[::-1]:
                count += 1
    return count


def run_deterministic_cases() -> None:
    cases = [
        "",
        "a",
        "abba",
        "babad",
        "cbbd",
        "aaaaa",
        "abaxyzzyxf",
        "forgeeksskeegfor",
        "上海自来水来自海上",
    ]

    print("=== Deterministic Cases ===")
    for idx, text in enumerate(cases, start=1):
        result = manacher(text)
        brute_longest, brute_start, brute_len = brute_force_longest_palindrome(text)
        brute_count = brute_force_palindrome_count(text)

        assert result.longest == brute_longest, (
            f"Case {idx} longest mismatch: text={text!r}, "
            f"manacher={result.longest!r}, brute={brute_longest!r}"
        )
        assert result.start == brute_start and result.length == brute_len, (
            f"Case {idx} index/len mismatch: text={text!r}, "
            f"manacher=(start={result.start}, len={result.length}), "
            f"brute=(start={brute_start}, len={brute_len})"
        )
        assert result.palindrome_count == brute_count, (
            f"Case {idx} count mismatch: text={text!r}, "
            f"manacher={result.palindrome_count}, brute={brute_count}"
        )

        preview = result.radii[: min(16, len(result.radii))]
        print(
            f"Case {idx}: text={text!r}, longest={result.longest!r}, "
            f"start={result.start}, len={result.length}, "
            f"pal_count={result.palindrome_count}, radius_preview={preview}"
        )


def run_randomized_regression(num_trials: int = 400, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    alphabet = "abca"

    for _ in range(num_trials):
        length = rng.randint(0, 30)
        text = "".join(rng.choice(alphabet) for _ in range(length))

        result = manacher(text)
        brute_longest, brute_start, brute_len = brute_force_longest_palindrome(text)
        brute_count = brute_force_palindrome_count(text)

        assert result.longest == brute_longest, (
            "Random longest mismatch: "
            f"text={text!r}, manacher={result.longest!r}, brute={brute_longest!r}"
        )
        assert (result.start, result.length) == (brute_start, brute_len), (
            "Random index/len mismatch: "
            f"text={text!r}, manacher={(result.start, result.length)}, "
            f"brute={(brute_start, brute_len)}"
        )
        assert result.palindrome_count == brute_count, (
            "Random count mismatch: "
            f"text={text!r}, manacher={result.palindrome_count}, brute={brute_count}"
        )

    print(f"Randomized regression passed: {num_trials} / {num_trials}")


def main() -> None:
    run_deterministic_cases()
    run_randomized_regression()
    print("All Manacher checks passed.")


if __name__ == "__main__":
    main()
