"""离散对数（Baby-step Giant-step）最小可运行示例。

目标：求解 x，使得 a^x ≡ b (mod m)。
说明：本 MVP 实现标准 BSGS，要求 gcd(a, m) = 1。
"""

from __future__ import annotations

from math import gcd, isqrt
from typing import Optional


def bsgs_discrete_log(base: int, target: int, modulus: int) -> Optional[int]:
    """Solve base^x == target (mod modulus) using Baby-step Giant-step.

    Returns:
        最小非负整数解 x；若无解返回 None。

    Raises:
        ValueError: 当 modulus <= 1 或 gcd(base, modulus) != 1 时抛出。
    """
    if modulus <= 1:
        raise ValueError("modulus must be > 1")

    base %= modulus
    target %= modulus

    if target == 1:
        return 0

    if gcd(base, modulus) != 1:
        raise ValueError(
            "standard BSGS requires gcd(base, modulus) = 1; "
            "use extended BSGS for non-coprime cases"
        )

    # n ~= sqrt(group_size). 对常见 prime modulus，有 group_size = modulus - 1。
    n = isqrt(modulus - 1) + 1

    # Baby steps: 记录 base^j -> j。
    baby_steps: dict[int, int] = {}
    cur = 1
    for j in range(n):
        if cur not in baby_steps:
            baby_steps[cur] = j
        cur = (cur * base) % modulus

    # Giant steps: target * (base^{-n})^i，查表匹配 baby step。
    factor = pow(base, -n, modulus)  # modular inverse power
    gamma = target
    for i in range(n + 1):
        j = baby_steps.get(gamma)
        if j is not None:
            x = i * n + j
            if pow(base, x, modulus) == target:
                return x
        gamma = (gamma * factor) % modulus

    return None


def _format_case_result(base: int, target: int, modulus: int) -> str:
    """Run one case and return a readable result line."""
    try:
        x = bsgs_discrete_log(base, target, modulus)
    except ValueError as exc:
        return f"a={base:>2}, b={target:>2}, m={modulus:>2} -> ERROR: {exc}"

    if x is None:
        return f"a={base:>2}, b={target:>2}, m={modulus:>2} -> no solution"

    verified = pow(base, x, modulus) == (target % modulus)
    return (
        f"a={base:>2}, b={target:>2}, m={modulus:>2} -> "
        f"x={x:<3} | check: {base}^{x} mod {modulus} = {pow(base, x, modulus)} "
        f"| verified={verified}"
    )


def main() -> None:
    """Run deterministic demo cases without interactive input."""
    cases = [
        # 有解案例
        (2, 22, 29),  # x = 26
        (5, 8, 23),   # x = 6
        (10, 17, 19), # x = 8
        (3, 13, 17),  # x = 4
        # 无解案例（在该循环子群里不存在）
        (2, 3, 7),
        # 非互素案例：标准 BSGS 不适用
        (6, 9, 15),
    ]

    print("=== Baby-step Giant-step (Discrete Log) Demo ===")
    for base, target, modulus in cases:
        print(_format_case_result(base, target, modulus))


if __name__ == "__main__":
    main()
