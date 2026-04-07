"""MATH-0088: 结式计算 (Resultant) 的最小可运行 MVP.

实现目标:
1) 不依赖黑盒 CAS 的 resultant() 调用;
2) 通过 Sylvester 矩阵 + Bareiss 消元求行列式，得到精确整数结式;
3) 给出若干可验证样例，说明“有公共根 <=> 结式为 0”。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _trim_leading_zeros(coeffs: Sequence[int]) -> List[int]:
    """去掉首部 0，保证最高次项系数非零。"""
    if not coeffs:
        raise ValueError("empty polynomial")
    i = 0
    while i < len(coeffs) - 1 and coeffs[i] == 0:
        i += 1
    return list(coeffs[i:])


def degree(coeffs: Sequence[int]) -> int:
    """返回多项式次数（系数按降幂给出）。"""
    c = _trim_leading_zeros(coeffs)
    return len(c) - 1


def eval_poly(coeffs: Sequence[int], x: int) -> int:
    """Horner 法计算 p(x)。"""
    acc = 0
    for a in coeffs:
        acc = acc * x + a
    return acc


def sylvester_matrix(p: Sequence[int], q: Sequence[int]) -> List[List[int]]:
    """构造 Sylvester 矩阵 S(p, q).

    约定:
    - p, q 均为降幂系数列表，且次数分别为 m, n;
    - S 是 (m+n) x (m+n) 整数矩阵:
      前 n 行由 p 的系数右移填充，后 m 行由 q 的系数右移填充。
    """
    p = _trim_leading_zeros(p)
    q = _trim_leading_zeros(q)
    m = len(p) - 1
    n = len(q) - 1
    if m < 0 or n < 0:
        raise ValueError("polynomial degree must be >= 0")
    if m == 0 and n == 0:
        # 常数与常数的结式按定义可以取 1（这里在上层直接处理）
        return [[1]]

    size = m + n
    mat: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]

    for row in range(n):
        for j, v in enumerate(p):
            mat[row][row + j] = int(v)

    for i in range(m):
        row = n + i
        for j, v in enumerate(q):
            mat[row][i + j] = int(v)

    return mat


def bareiss_det_int(matrix: Sequence[Sequence[int]]) -> int:
    """Bareiss fraction-free 消元，精确计算整数矩阵行列式。"""
    n = len(matrix)
    if n == 0:
        return 1
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

    a = [list(map(int, row)) for row in matrix]
    sign = 1
    denom = 1

    for k in range(n - 1):
        if a[k][k] == 0:
            pivot_row = None
            for r in range(k + 1, n):
                if a[r][k] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                return 0
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign *= -1

        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = pivot * a[i][j] - a[i][k] * a[k][j]
                if denom != 1:
                    num //= denom
                a[i][j] = num
            a[i][k] = 0
        denom = pivot

    return sign * a[n - 1][n - 1]


def resultant(p: Sequence[int], q: Sequence[int]) -> int:
    """计算 Res(p, q).

    这里处理了常数多项式的简单情形:
    - p 常数 c, deg(q)=n: Res(p, q)=c^n
    - q 常数 d, deg(p)=m: Res(p, q)=d^m
    """
    p = _trim_leading_zeros(p)
    q = _trim_leading_zeros(q)
    m = len(p) - 1
    n = len(q) - 1

    if m == 0 and n == 0:
        return 1
    if m == 0:
        return int(p[0]) ** n
    if n == 0:
        return int(q[0]) ** m

    s = sylvester_matrix(p, q)
    return bareiss_det_int(s)


def format_poly(coeffs: Sequence[int]) -> str:
    """把降幂系数转为可读字符串。"""
    coeffs = _trim_leading_zeros(coeffs)
    d = len(coeffs) - 1
    parts: List[str] = []
    for i, a in enumerate(coeffs):
        power = d - i
        if a == 0:
            continue
        sign = "+" if a > 0 else "-"
        abs_a = abs(a)
        if power == 0:
            body = f"{abs_a}"
        elif power == 1:
            body = "x" if abs_a == 1 else f"{abs_a}x"
        else:
            body = f"x^{power}" if abs_a == 1 else f"{abs_a}x^{power}"
        if not parts:
            parts.append(body if a > 0 else f"-{body}")
        else:
            parts.append(f" {sign} {body}")
    return "".join(parts) if parts else "0"


def run_case(name: str, p: Sequence[int], q: Sequence[int]) -> Tuple[str, int]:
    r = resultant(p, q)
    line = (
        f"[{name}] p(x)={format_poly(p)}, q(x)={format_poly(q)}, "
        f"Res(p,q)={r}"
    )
    return line, r


def main() -> None:
    cases = [
        (
            "共享根案例",
            [1, -3, 2],   # (x-1)(x-2)
            [1, -5, 6],   # (x-2)(x-3)
            0,
        ),
        (
            "无公共根案例A",
            [1, 0, -1],   # x^2 - 1
            [1, -2],      # x - 2
            3,            # p(2)=3
        ),
        (
            "无公共根案例B",
            [2, -3, 1],   # 2x^2 - 3x + 1
            [3, 1],       # 3x + 1
            20,           # 3^2 * p(-1/3) = 20/9 * 9 = 20
        ),
    ]

    print("=== Resultant MVP Demo (MATH-0088) ===")
    for name, p, q, expected in cases:
        line, r = run_case(name, p, q)
        ok = "OK" if r == expected else f"MISMATCH(expected={expected})"
        print(f"{line} -> {ok}")

    # 额外验证: 若 q(x)=x-a，则 Res(p,q)=(-1)^deg(p) * p(a)
    p = [1, -6, 11, -6]  # (x-1)(x-2)(x-3)
    a = 4
    q = [1, -a]
    r = resultant(p, q)
    pa = eval_poly(p, a)
    signed_pa = ((-1) ** degree(p)) * pa
    print(
        "线性因子检验: "
        f"Res(p, x-{a})={r}, (-1)^deg(p)*p({a})={signed_pa}, "
        f"check={'OK' if r == signed_pa else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
