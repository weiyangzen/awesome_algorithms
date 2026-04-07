"""Generating Function MVP for combinatorial counting.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None  # type: ignore[assignment]


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def convolve_truncated_manual(a: Sequence[int], b: Sequence[int], max_degree: int) -> List[int]:
    """Return truncated polynomial product c = a * b, only keeping degrees <= max_degree.

    If a(x)=sum a_i x^i and b(x)=sum b_j x^j, then
    c_k = sum_{i+j=k} a_i*b_j.
    """
    _validate_non_negative_int("max_degree", max_degree)

    if not a or not b:
        return [0] * (max_degree + 1)

    out = [0] * (max_degree + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        if i > max_degree:
            break
        max_j = min(len(b) - 1, max_degree - i)
        for j in range(max_j + 1):
            bj = b[j]
            if bj:
                out[i + j] += ai * bj
    return out


def convolve_truncated_numpy(a: Sequence[int], b: Sequence[int], max_degree: int) -> List[int]:
    """Truncated convolution with numpy when available, else fallback to manual."""
    _validate_non_negative_int("max_degree", max_degree)

    if np is None:
        return convolve_truncated_manual(a, b, max_degree)

    arr_a = np.array(a, dtype=np.int64)
    arr_b = np.array(b, dtype=np.int64)
    full = np.convolve(arr_a, arr_b)
    out = np.zeros(max_degree + 1, dtype=np.int64)
    keep = min(max_degree + 1, full.shape[0])
    out[:keep] = full[:keep]
    return out.tolist()


def coefficient_bounded_selection(caps: Sequence[int], target: int, use_numpy: bool = False) -> int:
    """Compute [x^target] prod_i (1 + x + ... + x^{caps_i})."""
    _validate_non_negative_int("target", target)
    for idx, cap in enumerate(caps):
        _validate_non_negative_int(f"caps[{idx}]", cap)

    poly: List[int] = [1] + [0] * target
    convolver = convolve_truncated_numpy if use_numpy else convolve_truncated_manual

    for cap in caps:
        factor = [1] * (cap + 1)
        poly = convolver(poly, factor, target)

    return poly[target]


def coefficient_bounded_selection_bruteforce(caps: Sequence[int], target: int) -> int:
    """Brute-force cross-check for bounded selection."""
    _validate_non_negative_int("target", target)
    for idx, cap in enumerate(caps):
        _validate_non_negative_int(f"caps[{idx}]", cap)

    ranges: Iterable[range] = [range(cap + 1) for cap in caps]
    count = 0
    for choices in product(*ranges):
        if sum(choices) == target:
            count += 1
    return count


def count_dice_sum_gf(num_dice: int, target_sum: int, use_numpy: bool = False) -> int:
    """Compute [x^target_sum] (x + x^2 + ... + x^6)^num_dice."""
    _validate_non_negative_int("num_dice", num_dice)
    _validate_non_negative_int("target_sum", target_sum)

    poly: List[int] = [1] + [0] * target_sum
    factor = [0, 1, 1, 1, 1, 1, 1]  # x^1 ... x^6
    convolver = convolve_truncated_numpy if use_numpy else convolve_truncated_manual

    for _ in range(num_dice):
        poly = convolver(poly, factor, target_sum)

    return poly[target_sum]


def count_dice_sum_dp(num_dice: int, target_sum: int) -> int:
    """Independent DP check for dice sum counts."""
    _validate_non_negative_int("num_dice", num_dice)
    _validate_non_negative_int("target_sum", target_sum)

    dp = [0] * (target_sum + 1)
    dp[0] = 1

    for _ in range(num_dice):
        nxt = [0] * (target_sum + 1)
        for s in range(target_sum + 1):
            if dp[s] == 0:
                continue
            for face in range(1, 7):
                ns = s + face
                if ns <= target_sum:
                    nxt[ns] += dp[s]
        dp = nxt

    return dp[target_sum]


def count_coin_change_gf(denominations: Sequence[int], target: int, use_numpy: bool = False) -> int:
    """Compute [x^target] prod_d (1 + x^d + x^{2d} + ...), truncated to target."""
    _validate_non_negative_int("target", target)

    for idx, d in enumerate(denominations):
        _validate_non_negative_int(f"denominations[{idx}]", d)
        if d == 0:
            raise ValueError("coin denomination must be positive")

    poly: List[int] = [1] + [0] * target
    convolver = convolve_truncated_numpy if use_numpy else convolve_truncated_manual

    for d in denominations:
        factor = [0] * (target + 1)
        for amount in range(0, target + 1, d):
            factor[amount] = 1
        poly = convolver(poly, factor, target)

    return poly[target]


def count_coin_change_dp(denominations: Sequence[int], target: int) -> int:
    """Independent DP check for unordered unlimited coin change."""
    _validate_non_negative_int("target", target)

    for idx, d in enumerate(denominations):
        _validate_non_negative_int(f"denominations[{idx}]", d)
        if d == 0:
            raise ValueError("coin denomination must be positive")

    dp = [0] * (target + 1)
    dp[0] = 1
    for d in denominations:
        for amount in range(d, target + 1):
            dp[amount] += dp[amount - d]
    return dp[target]


def _collect_report_lines() -> List[str]:
    lines: List[str] = []

    # Example 1: bounded selection count.
    caps = [2, 3, 1, 4]
    target = 6
    by_gf_manual = coefficient_bounded_selection(caps, target, use_numpy=False)
    by_bruteforce = coefficient_bounded_selection_bruteforce(caps, target)

    assert by_gf_manual == by_bruteforce

    lines.append("Example 1: bounded selection")
    lines.append(f"  caps={caps}, target={target}")
    lines.append(f"  GF(manual convolution) = {by_gf_manual}")
    lines.append(f"  brute force            = {by_bruteforce}")

    if np is not None:
        by_gf_numpy = coefficient_bounded_selection(caps, target, use_numpy=True)
        assert by_gf_numpy == by_gf_manual
        lines.append(f"  GF(numpy.convolve)     = {by_gf_numpy}")

    # Example 2: dice sum distribution coefficient.
    num_dice, target_sum = 4, 14
    dice_gf_manual = count_dice_sum_gf(num_dice, target_sum, use_numpy=False)
    dice_dp = count_dice_sum_dp(num_dice, target_sum)

    assert dice_gf_manual == dice_dp

    lines.append("\nExample 2: dice sum coefficient")
    lines.append(f"  (x+...+x^6)^{num_dice}, target_sum={target_sum}")
    lines.append(f"  GF(manual convolution) = {dice_gf_manual}")
    lines.append(f"  DP check               = {dice_dp}")

    if np is not None:
        dice_gf_numpy = count_dice_sum_gf(num_dice, target_sum, use_numpy=True)
        assert dice_gf_numpy == dice_gf_manual
        lines.append(f"  GF(numpy.convolve)     = {dice_gf_numpy}")

    # Example 3: unlimited coin change coefficient.
    denominations = [1, 2, 5]
    target_coin = 10
    coin_gf_manual = count_coin_change_gf(denominations, target_coin, use_numpy=False)
    coin_dp = count_coin_change_dp(denominations, target_coin)

    assert coin_gf_manual == coin_dp

    lines.append("\nExample 3: coin change coefficient")
    lines.append(f"  denominations={denominations}, target={target_coin}")
    lines.append(f"  GF(manual convolution) = {coin_gf_manual}")
    lines.append(f"  DP check               = {coin_dp}")

    if np is not None:
        coin_gf_numpy = count_coin_change_gf(denominations, target_coin, use_numpy=True)
        assert coin_gf_numpy == coin_gf_manual
        lines.append(f"  GF(numpy.convolve)     = {coin_gf_numpy}")

    return lines


def main() -> None:
    report_lines = _collect_report_lines()

    print("Generating Function MVP")
    print("=" * 60)
    for line in report_lines:
        print(line)
    if np is None:
        print("\nNote: numpy is unavailable; numpy-based cross-checks were skipped.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
