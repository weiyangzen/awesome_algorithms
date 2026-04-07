"""MATH-0029: 筛法求积性函数前缀和.

运行方式:
    python3 demo.py
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple


PrimePowerValue = Callable[[int, int], int]


def sieve_prefix_multiplicative(
    limit: int, prime_power_value: PrimePowerValue
) -> Tuple[List[int], List[int]]:
    """在线性筛框架下计算积性函数值和前缀和.

    参数:
        limit: 计算到 1..limit
        prime_power_value: 回调, 输入 (p, e) 返回 f(p^e)
    返回:
        f, prefix
        其中 f[n] = f(n), prefix[n] = sum_{k=1..n} f(k)
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")

    lp = [0] * (limit + 1)  # least prime factor
    exp = [0] * (limit + 1)  # exponent of lp in n
    rest = [0] * (limit + 1)  # n / lp^exp
    primes: List[int] = []

    for i in range(2, limit + 1):
        if lp[i] == 0:
            lp[i] = i
            exp[i] = 1
            rest[i] = 1
            primes.append(i)
        for p in primes:
            x = i * p
            if x > limit:
                break
            lp[x] = p
            if p == lp[i]:
                exp[x] = exp[i] + 1
                rest[x] = rest[i]
                break
            exp[x] = 1
            rest[x] = i

    f = [0] * (limit + 1)
    prefix = [0] * (limit + 1)
    f[1] = 1
    prefix[1] = 1

    for n in range(2, limit + 1):
        p = lp[n]
        e = exp[n]
        r = rest[n]
        f[n] = f[r] * prime_power_value(p, e)

    for n in range(2, limit + 1):
        prefix[n] = prefix[n - 1] + f[n]

    return f, prefix


def phi_prime_power(p: int, e: int) -> int:
    return (p - 1) * (p ** (e - 1))


def mu_prime_power(_: int, e: int) -> int:
    return -1 if e == 1 else 0


def tau_prime_power(_: int, e: int) -> int:
    return e + 1


def factor_exponents(n: int) -> Dict[int, int]:
    d: Dict[int, int] = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            d[p] = d.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d


def phi_naive(n: int) -> int:
    if n == 1:
        return 1
    result = n
    for p in factor_exponents(n):
        result -= result // p
    return result


def mu_naive(n: int) -> int:
    if n == 1:
        return 1
    factors = factor_exponents(n)
    for e in factors.values():
        if e >= 2:
            return 0
    return -1 if len(factors) % 2 == 1 else 1


def tau_naive(n: int) -> int:
    if n == 1:
        return 1
    ans = 1
    for e in factor_exponents(n).values():
        ans *= e + 1
    return ans


def verify_prefix(prefix: List[int], naive_fn: Callable[[int], int], upto: int) -> None:
    running = 0
    for i in range(1, upto + 1):
        running += naive_fn(i)
        if prefix[i] != running:
            raise AssertionError(
                f"Prefix mismatch at i={i}: got {prefix[i]}, expected {running}"
            )


def main() -> None:
    limit = 100_000

    _, phi_prefix = sieve_prefix_multiplicative(limit, phi_prime_power)
    _, mu_prefix = sieve_prefix_multiplicative(limit, mu_prime_power)
    _, tau_prefix = sieve_prefix_multiplicative(limit, tau_prime_power)

    verify_upto = 300
    verify_prefix(phi_prefix, phi_naive, verify_upto)
    verify_prefix(mu_prefix, mu_naive, verify_upto)
    verify_prefix(tau_prefix, tau_naive, verify_upto)

    print(f"N = {limit}")
    print("Verified with naive factorization on range [1, 300].")
    print(f"sum_{{k<=N}} phi(k) = {phi_prefix[limit]}")
    print(f"sum_{{k<=N}} mu(k)  = {mu_prefix[limit]}")
    print(f"sum_{{k<=N}} tau(k) = {tau_prefix[limit]}")
    print()
    print("Sample prefix values:")
    sample_points = [1, 10, 100, 1000, limit]
    for x in sample_points:
        print(
            f"x={x:>6} | "
            f"PhiPrefix={phi_prefix[x]:>12} | "
            f"MuPrefix={mu_prefix[x]:>7} | "
            f"TauPrefix={tau_prefix[x]:>10}"
        )


if __name__ == "__main__":
    main()
