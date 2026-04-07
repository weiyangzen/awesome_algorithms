"""Minimal runnable MVP for Z-function and Extended KMP (ExKMP)."""

from __future__ import annotations

import random
from typing import List


def z_function(s: str) -> List[int]:
    """Compute Z-array in O(n), with convention z[0] = n."""
    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    z[0] = n

    l = 0
    r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1

    return z


def exkmp_extend(text: str, pattern: str) -> List[int]:
    """
    Compute extend array in O(n + m):
    extend[i] = LCP(text[i:], pattern)
    """
    if pattern == "":
        raise ValueError("pattern must not be empty in this MVP")

    n = len(text)
    m = len(pattern)
    if n == 0:
        return []

    nxt = z_function(pattern)
    extend = [0] * n

    while extend[0] < n and extend[0] < m and text[extend[0]] == pattern[extend[0]]:
        extend[0] += 1

    l = 0
    r = extend[0] - 1
    for i in range(1, n):
        if i <= r:
            k = i - l
            extend[i] = min(r - i + 1, nxt[k])

        while (
            i + extend[i] < n
            and extend[i] < m
            and text[i + extend[i]] == pattern[extend[i]]
        ):
            extend[i] += 1

        if i + extend[i] - 1 > r:
            l = i
            r = i + extend[i] - 1

    return extend


def find_occurrences(text: str, pattern: str) -> List[int]:
    """Return all start indices where pattern appears in text."""
    ext = exkmp_extend(text, pattern)
    m = len(pattern)
    return [i for i, v in enumerate(ext) if v == m]


def naive_z(s: str) -> List[int]:
    n = len(s)
    if n == 0:
        return []

    out = [0] * n
    out[0] = n
    for i in range(1, n):
        k = 0
        while i + k < n and s[k] == s[i + k]:
            k += 1
        out[i] = k
    return out


def naive_extend(text: str, pattern: str) -> List[int]:
    if pattern == "":
        raise ValueError("pattern must not be empty in this MVP")

    n = len(text)
    m = len(pattern)
    out = [0] * n
    for i in range(n):
        k = 0
        while i + k < n and k < m and text[i + k] == pattern[k]:
            k += 1
        out[i] = k
    return out


def run_fixed_demo() -> None:
    pattern = "aba"
    text = "abacabaaba"

    z_arr = z_function(pattern)
    ext = exkmp_extend(text, pattern)
    hits = find_occurrences(text, pattern)

    if z_arr != naive_z(pattern):
        raise AssertionError("fixed demo: z_function mismatch")
    if ext != naive_extend(text, pattern):
        raise AssertionError("fixed demo: exkmp_extend mismatch")

    print("=== Z Function / ExKMP Fixed Demo ===")
    print(f"pattern: {pattern}")
    print(f"text:    {text}")
    print(f"Z(pattern): {z_arr}")
    print(f"extend:     {ext}")
    print(f"match starts: {hits}")
    print("fixed demo check: PASS")


def run_random_regression(rounds: int = 200, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    alphabet = "abcd"

    for r in range(rounds):
        m = rng.randint(1, 12)
        n = rng.randint(0, 80)
        pattern = "".join(rng.choice(alphabet) for _ in range(m))
        text = "".join(rng.choice(alphabet) for _ in range(n))

        z_fast = z_function(pattern)
        z_slow = naive_z(pattern)
        if z_fast != z_slow:
            raise AssertionError(
                f"z regression failed at round={r}\n"
                f"pattern={pattern}\nfast={z_fast}\nslow={z_slow}"
            )

        ext_fast = exkmp_extend(text, pattern)
        ext_slow = naive_extend(text, pattern)
        if ext_fast != ext_slow:
            raise AssertionError(
                f"extend regression failed at round={r}\n"
                f"pattern={pattern}\ntext={text}\n"
                f"fast={ext_fast}\nslow={ext_slow}"
            )

        hit_fast = [i for i, v in enumerate(ext_fast) if v == len(pattern)]
        hit_slow = [i for i in range(len(text) - len(pattern) + 1) if text[i : i + len(pattern)] == pattern]
        if hit_fast != hit_slow:
            raise AssertionError(
                f"occurrence regression failed at round={r}\n"
                f"pattern={pattern}\ntext={text}\n"
                f"fast_hits={hit_fast}\nslow_hits={hit_slow}"
            )

    print(f"random regression: PASS ({rounds} rounds, seed={seed})")


def main() -> None:
    run_fixed_demo()
    run_random_regression()


if __name__ == "__main__":
    main()
