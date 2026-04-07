"""Euler theorem and Euler totient function MVP demo."""

from math import gcd
from typing import List


def euler_phi_trial_division(n: int) -> int:
    """Compute phi(n) by trial division factorization."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return 1

    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            result -= result // p
            while x % p == 0:
                x //= p
        p += 1

    if x > 1:
        result -= result // x
    return result


def euler_phi_linear_sieve(limit: int) -> List[int]:
    """Compute phi(0..limit) using linear sieve."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    phi = [0] * (limit + 1)
    is_composite = [False] * (limit + 1)
    primes: List[int] = []
    phi[1] = 1

    for i in range(2, limit + 1):
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            v = i * p
            if v > limit:
                break
            is_composite[v] = True
            if i % p == 0:
                phi[v] = phi[i] * p
                break
            phi[v] = phi[i] * (p - 1)

    return phi


def verify_euler_theorem_for_n(n: int) -> bool:
    """Check Euler theorem for all a in [1, n-1] that are coprime to n."""
    if n < 2:
        return True
    phi_n = euler_phi_trial_division(n)
    for a in range(1, n):
        if gcd(a, n) == 1 and pow(a, phi_n, n) != 1:
            return False
    return True


def main() -> None:
    sample_ns = [1, 2, 5, 8, 9, 10, 12, 36, 97, 1001]
    print("=== Euler Totient via Trial Division ===")
    for n in sample_ns:
        print(f"phi({n}) = {euler_phi_trial_division(n)}")

    print("\n=== Cross-check with Linear Sieve (1..200) ===")
    limit = 200
    phi_table = euler_phi_linear_sieve(limit)
    for n in range(1, limit + 1):
        assert phi_table[n] == euler_phi_trial_division(n), (
            f"Mismatch at n={n}: sieve={phi_table[n]}, "
            f"trial={euler_phi_trial_division(n)}"
        )
    print("All values matched for n in [1, 200].")

    print("\n=== Euler Theorem Verification ===")
    for n in [2, 5, 8, 9, 10, 12, 36]:
        ok = verify_euler_theorem_for_n(n)
        print(f"n={n:>2}, phi(n)={euler_phi_trial_division(n):>2}, theorem_holds={ok}")

    n = 36
    a = 5
    exponent = 12345
    phi_n = euler_phi_trial_division(n)
    reduced_exponent = exponent % phi_n
    direct = pow(a, exponent, n)
    reduced = pow(a, reduced_exponent, n)
    print("\n=== Exponent Reduction Example ===")
    print(
        f"a={a}, n={n}, exponent={exponent}, phi(n)={phi_n}, "
        f"exponent % phi(n)={reduced_exponent}"
    )
    print(f"pow(a, exponent, n)={direct}, pow(a, exponent % phi(n), n)={reduced}")


if __name__ == "__main__":
    main()
