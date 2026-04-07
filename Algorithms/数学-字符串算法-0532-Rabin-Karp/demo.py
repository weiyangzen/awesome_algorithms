"""Rabin-Karp 字符串匹配最小可运行示例。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingHashParams:
    """滚动哈希参数。"""

    base: int = 257
    mod: int = 1_000_000_007


def _char_value(ch: str) -> int:
    """字符映射到正整数，兼容 Unicode。"""
    return ord(ch) + 1


def _poly_hash(s: str, params: RollingHashParams) -> int:
    """按多项式定义计算字符串哈希。"""
    h = 0
    for ch in s:
        h = (h * params.base + _char_value(ch)) % params.mod
    return h


def brute_force_search(text: str, pattern: str) -> list[int]:
    """暴力匹配，仅用于小样例正确性对拍。"""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []
    matches: list[int] = []
    for i in range(n - m + 1):
        if text[i : i + m] == pattern:
            matches.append(i)
    return matches


def rabin_karp_search(
    text: str,
    pattern: str,
    params: RollingHashParams | None = None,
) -> tuple[list[int], dict[str, int]]:
    """Rabin-Karp 单模式匹配。

    返回：
    - matches: 所有匹配起始下标
    - stats: 过程统计（窗口数、哈希相等触发次数、最终确认匹配次数）
    """
    if params is None:
        params = RollingHashParams()

    n, m = len(text), len(pattern)
    if m == 0:
        matches = list(range(n + 1))
        return matches, {
            "windows_scanned": n + 1,
            "hash_equal_checks": n + 1,
            "verified_matches": n + 1,
        }
    if m > n:
        return [], {"windows_scanned": 0, "hash_equal_checks": 0, "verified_matches": 0}

    high_base = pow(params.base, m - 1, params.mod)
    pattern_hash = _poly_hash(pattern, params)
    window_hash = _poly_hash(text[:m], params)

    matches: list[int] = []
    hash_equal_checks = 0
    verified_matches = 0
    windows = n - m + 1

    for start in range(windows):
        if window_hash == pattern_hash:
            hash_equal_checks += 1
            if text[start : start + m] == pattern:
                matches.append(start)
                verified_matches += 1

        if start < windows - 1:
            left_val = _char_value(text[start])
            right_val = _char_value(text[start + m])
            window_hash = (window_hash - left_val * high_base) % params.mod
            window_hash = (window_hash * params.base + right_val) % params.mod

    return matches, {
        "windows_scanned": windows,
        "hash_equal_checks": hash_equal_checks,
        "verified_matches": verified_matches,
    }


def validate_case(
    text: str,
    pattern: str,
    params: RollingHashParams,
) -> dict[str, object]:
    """单样例对拍验证，失败时抛出异常。"""
    rk_matches, stats = rabin_karp_search(text, pattern, params)
    brute_matches = brute_force_search(text, pattern)
    if rk_matches != brute_matches:
        raise AssertionError(
            f"Mismatch for text={text!r}, pattern={pattern!r}, "
            f"rabin_karp={rk_matches}, brute_force={brute_matches}"
        )
    return {
        "text": text,
        "pattern": pattern,
        "matches": rk_matches,
        "expected": brute_matches,
        **stats,
    }


def run_demo_samples() -> None:
    """运行内置样例并打印结果。"""
    params = RollingHashParams(base=257, mod=1_000_000_007)
    samples: list[tuple[str, str]] = [
        ("abracadabra", "abra"),
        ("aaaaaa", "aaa"),
        ("banananobano", "nano"),
        ("abcxabcdabxabcdabcdabcy", "abcdabcy"),
        ("我爱算法，算法爱我，算法真有趣", "算法"),
        ("short", "longpattern"),
        ("edge", ""),
    ]

    for idx, (text, pattern) in enumerate(samples, start=1):
        result = validate_case(text, pattern, params)
        print("=" * 76)
        print(f"sample #{idx}")
        print(f"text: {result['text']!r}")
        print(f"pattern: {result['pattern']!r}")
        print(f"matches: {result['matches']}")
        print(f"expected: {result['expected']}")
        print(
            "stats: "
            f"windows_scanned={result['windows_scanned']}, "
            f"hash_equal_checks={result['hash_equal_checks']}, "
            f"verified_matches={result['verified_matches']}"
        )

    print("=" * 76)
    print("All demo samples passed Rabin-Karp vs brute-force validation.")


def main() -> None:
    run_demo_samples()


if __name__ == "__main__":
    main()
