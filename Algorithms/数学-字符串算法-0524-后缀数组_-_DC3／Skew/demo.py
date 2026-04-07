"""Suffix Array (DC3/Skew) minimal runnable MVP.

This script implements the linear-time DC3/Skew suffix array algorithm from scratch,
plus a small Kasai LCP routine and deterministic correctness checks.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import random


def _leq_pair(a1: int, a2: int, b1: int, b2: int) -> bool:
    return a1 < b1 or (a1 == b1 and a2 <= b2)


def _leq_triple(a1: int, a2: int, a3: int, b1: int, b2: int, b3: int) -> bool:
    return a1 < b1 or (a1 == b1 and _leq_pair(a2, a3, b2, b3))


def _radix_pass(indices: Sequence[int], s: Sequence[int], offset: int, k: int) -> List[int]:
    """Stable counting sort of indices by key s[i + offset]."""
    count = [0] * (k + 2)
    out = [0] * len(indices)

    for idx in indices:
        count[s[idx + offset] + 1] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    for idx in indices:
        key = s[idx + offset]
        out[count[key]] = idx
        count[key] += 1

    return out


def _suffix_array_dc3_int(s: List[int], n: int, k: int) -> List[int]:
    """DC3/Skew for integer alphabet, values in [0, k], with 3 trailing zeros available."""
    n0 = (n + 2) // 3
    n1 = (n + 1) // 3
    n2 = n // 3
    n02 = n0 + n2

    # Positions i % 3 != 0.
    s12 = [0] * (n02 + 3)
    sa12 = [0] * (n02 + 3)
    s0 = [0] * n0

    j = 0
    for i in range(n + (n0 - n1)):
        if i % 3 != 0:
            s12[j] = i
            j += 1

    # Sort sampled suffixes by 3-char tuples.
    sa12_sorted = _radix_pass(s12[:n02], s, 2, k)
    sa12_sorted = _radix_pass(sa12_sorted, s, 1, k)
    sa12_sorted = _radix_pass(sa12_sorted, s, 0, k)
    sa12[:n02] = sa12_sorted

    # Assign lexicographic names.
    name = 0
    c0 = c1 = c2 = -1
    for pos in sa12[:n02]:
        if s[pos] != c0 or s[pos + 1] != c1 or s[pos + 2] != c2:
            name += 1
            c0, c1, c2 = s[pos], s[pos + 1], s[pos + 2]

        if pos % 3 == 1:
            s12[pos // 3] = name
        else:
            s12[pos // 3 + n0] = name

    # Recurse if names are not unique.
    if name < n02:
        sa12_rec = _suffix_array_dc3_int(s12, n02, name)
        sa12[:n02] = sa12_rec
        for rank, idx in enumerate(sa12[:n02], start=1):
            s12[idx] = rank
    else:
        for i in range(n02):
            sa12[s12[i] - 1] = i

    # Sort mod-0 suffixes by first character.
    j = 0
    for idx in sa12[:n02]:
        if idx < n0:
            s0[j] = 3 * idx
            j += 1

    sa0 = _radix_pass(s0[:n0], s, 0, k)

    # Merge SA0 and SA12.
    sa = [0] * n
    p = 0
    t = n0 - n1
    out = 0

    while out < n:
        if t == n02:
            sa[out:] = sa0[p:]
            break
        if p == n0:
            for idx in sa12[t:n02]:
                sa[out] = 3 * idx + 1 if idx < n0 else 3 * (idx - n0) + 2
                out += 1
            break

        i = 3 * sa12[t] + 1 if sa12[t] < n0 else 3 * (sa12[t] - n0) + 2
        j0 = sa0[p]

        if sa12[t] < n0:
            take_i = _leq_pair(s[i], s12[sa12[t] + n0], s[j0], s12[j0 // 3])
        else:
            take_i = _leq_triple(
                s[i],
                s[i + 1],
                s12[sa12[t] - n0 + 1],
                s[j0],
                s[j0 + 1],
                s12[j0 // 3 + n0],
            )

        if take_i:
            sa[out] = i
            t += 1
        else:
            sa[out] = j0
            p += 1

        out += 1

    return sa


def suffix_array_dc3(text: str) -> List[int]:
    """Build suffix array of `text` using DC3/Skew."""
    n = len(text)
    if n == 0:
        return []

    seq = [ord(ch) + 1 for ch in text]
    k = max(seq)
    padded = seq + [0, 0, 0]
    return _suffix_array_dc3_int(padded, n, k)


def lcp_kasai(text: str, sa: Sequence[int]) -> List[int]:
    """Kasai LCP array where lcp[i] = LCP(sa[i], sa[i-1]), and lcp[0] = 0."""
    n = len(text)
    if n == 0:
        return []

    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i

    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1
    return lcp


def _naive_sa(text: str) -> List[int]:
    return sorted(range(len(text)), key=lambda i: text[i:])


def _run_self_checks() -> None:
    fixed_cases = [
        "banana",
        "mississippi",
        "abracadabra",
        "aaaaaa",
        "zxyxzyzx",
        "",  # edge case
        "a",
    ]

    for s in fixed_cases:
        assert suffix_array_dc3(s) == _naive_sa(s), f"fixed-case mismatch: {s!r}"

    rng = random.Random(20260407)
    alphabet = "abcde"
    for _ in range(120):
        n = rng.randint(0, 40)
        s = "".join(rng.choice(alphabet) for _ in range(n))
        assert suffix_array_dc3(s) == _naive_sa(s), f"random mismatch: {s!r}"


def _format_suffix_table(text: str, sa: Sequence[int], lcp: Sequence[int]) -> List[Tuple[int, int, int, str]]:
    rows: List[Tuple[int, int, int, str]] = []
    for rank, pos in enumerate(sa):
        rows.append((rank, pos, lcp[rank] if rank < len(lcp) else 0, text[pos:]))
    return rows


def main() -> None:
    _run_self_checks()

    samples = ["banana", "mississippi", "panamabananas"]
    for text in samples:
        sa = suffix_array_dc3(text)
        lcp = lcp_kasai(text, sa)

        print(f"Text: {text}")
        print(f"Length: {len(text)}")
        print(f"SA: {sa}")
        print(f"LCP: {lcp}")
        print("rank\tpos\tlcp\tsuffix")

        for rank, pos, lcp_val, suffix in _format_suffix_table(text, sa, lcp):
            print(f"{rank}\t{pos}\t{lcp_val}\t{suffix}")

        print("-" * 60)

    print("All DC3 checks passed.")


if __name__ == "__main__":
    main()
