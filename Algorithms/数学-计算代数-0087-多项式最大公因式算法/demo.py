"""MATH-0087: 多项式最大公因式算法 (Polynomial GCD) 的最小可运行 MVP.

实现目标:
1) 以有理数域上的欧几里得算法计算一元多项式 gcd;
2) 不调用黑盒 CAS 的 gcd() 接口，显式实现长除与余式;
3) 输出规范化的 monic gcd（最高次项系数为 1）。
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import List, Sequence, Tuple


def _as_fraction_list(coeffs: Sequence[int | Fraction]) -> List[Fraction]:
    if not coeffs:
        raise ValueError("empty polynomial")
    return [c if isinstance(c, Fraction) else Fraction(c) for c in coeffs]


def _trim_leading_zeros(coeffs: Sequence[Fraction]) -> List[Fraction]:
    """去掉首部 0，保持降幂表示。"""
    if not coeffs:
        raise ValueError("empty polynomial")
    i = 0
    while i < len(coeffs) - 1 and coeffs[i] == 0:
        i += 1
    return list(coeffs[i:])


def _is_zero_poly(coeffs: Sequence[Fraction]) -> bool:
    c = _trim_leading_zeros(coeffs)
    return len(c) == 1 and c[0] == 0


def degree(coeffs: Sequence[Fraction]) -> int:
    return len(_trim_leading_zeros(coeffs)) - 1


def poly_divmod(
    dividend: Sequence[int | Fraction], divisor: Sequence[int | Fraction]
) -> Tuple[List[Fraction], List[Fraction]]:
    """多项式长除（降幂表示），返回 (quotient, remainder)。"""
    a = _trim_leading_zeros(_as_fraction_list(dividend))
    b = _trim_leading_zeros(_as_fraction_list(divisor))

    if _is_zero_poly(b):
        raise ZeroDivisionError("polynomial division by zero")

    da = degree(a)
    db = degree(b)
    if da < db:
        return [Fraction(0)], a

    q_degree = da - db
    q = [Fraction(0) for _ in range(q_degree + 1)]
    r = a[:]

    while not _is_zero_poly(r) and degree(r) >= db:
        dr = degree(r)
        shift = dr - db
        lead_factor = r[0] / b[0]

        # q 的最高次数是 q_degree，当前项次数是 shift（都指真实次数）
        q_index = q_degree - shift
        q[q_index] += lead_factor

        scaled_b = [lead_factor * x for x in b] + [Fraction(0)] * shift
        r = [x - y for x, y in zip(r, scaled_b)]
        r = _trim_leading_zeros(r)

    return _trim_leading_zeros(q), _trim_leading_zeros(r)


def poly_gcd(
    p: Sequence[int | Fraction], q: Sequence[int | Fraction]
) -> List[Fraction]:
    """欧几里得算法计算多项式 gcd，返回 monic 形式。"""
    a = _trim_leading_zeros(_as_fraction_list(p))
    b = _trim_leading_zeros(_as_fraction_list(q))

    if _is_zero_poly(a) and _is_zero_poly(b):
        return [Fraction(0)]

    while not _is_zero_poly(b):
        _, r = poly_divmod(a, b)
        a, b = b, r

    if _is_zero_poly(a):
        return [Fraction(0)]

    lc = a[0]
    return [x / lc for x in a]


def to_integer_primitive(coeffs: Sequence[Fraction]) -> List[int]:
    """把有理系数多项式转成整系数 primitive 代表（仅用于展示）。"""
    c = _trim_leading_zeros(coeffs)
    if len(c) == 1 and c[0] == 0:
        return [0]

    lcm_den = 1
    for x in c:
        lcm_den = lcm_den * x.denominator // gcd(lcm_den, x.denominator)

    ints = [int(x * lcm_den) for x in c]
    content = 0
    for v in ints:
        content = gcd(content, abs(v))
    if content > 1:
        ints = [v // content for v in ints]
    if ints[0] < 0:
        ints = [-v for v in ints]
    return ints


def format_poly(coeffs: Sequence[int | Fraction]) -> str:
    """把降幂系数格式化为可读多项式字符串。"""
    c = _trim_leading_zeros(_as_fraction_list(coeffs))
    d = len(c) - 1
    parts: List[str] = []

    for i, a in enumerate(c):
        power = d - i
        if a == 0:
            continue

        sign = "+" if a > 0 else "-"
        abs_a = abs(a)
        coef = f"{abs_a.numerator}/{abs_a.denominator}" if abs_a.denominator != 1 else f"{abs_a.numerator}"

        if power == 0:
            body = coef
        elif power == 1:
            body = "x" if abs_a == 1 else f"{coef}x"
        else:
            body = f"x^{power}" if abs_a == 1 else f"{coef}x^{power}"

        if not parts:
            parts.append(body if a > 0 else f"-{body}")
        else:
            parts.append(f" {sign} {body}")

    return "".join(parts) if parts else "0"


def run_case(
    name: str,
    p: Sequence[int | Fraction],
    q: Sequence[int | Fraction],
    expected_monic: Sequence[int | Fraction],
) -> bool:
    g = poly_gcd(p, q)
    ok = g == _trim_leading_zeros(_as_fraction_list(expected_monic))
    g_int = to_integer_primitive(g)
    print(f"[{name}]")
    print(f"  p(x) = {format_poly(p)}")
    print(f"  q(x) = {format_poly(q)}")
    print(f"  gcd_monic = {format_poly(g)}")
    print(f"  gcd_primitive_Z[x] = {g_int}")
    print(f"  check = {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    print("=== Polynomial GCD MVP Demo (MATH-0087) ===")

    cases = [
        (
            "共享二次因子",
            [1, 0, -7, 6],
            [1, 1, -6],
            [1, 1, -6],
        ),
        (
            "非首一输入（提取内容后同 gcd）",
            [2, 0, -6, 4],
            [3, 9, 0, -12],
            [1, 1, -2],
        ),
        (
            "互素多项式",
            [1, 0, 0, -2],
            [1, 0, 1],
            [1],
        ),
        (
            "与零多项式求 gcd",
            [0],
            [-2, 4, -2],
            [1, -2, 1],
        ),
        (
            "双零边界",
            [0],
            [0],
            [0],
        ),
    ]

    all_ok = True
    for name, p, q, expected in cases:
        all_ok = run_case(name, p, q, expected) and all_ok

    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
