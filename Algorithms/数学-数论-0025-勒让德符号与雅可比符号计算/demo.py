"""Legendre symbol and Jacobi symbol MVP demo."""

from __future__ import annotations

from math import isqrt
from typing import Dict, List, Tuple


def is_odd_prime(p: int) -> bool:
    """Return True if p is an odd prime."""
    if p < 3 or p % 2 == 0:
        return False
    limit = isqrt(p)
    d = 3
    while d <= limit:
        if p % d == 0:
            return False
        d += 2
    return True


def legendre_symbol(a: int, p: int) -> int:
    """Compute Legendre symbol (a/p) for odd prime p.

    Returns one of {-1, 0, 1}.
    """
    if not is_odd_prime(p):
        raise ValueError("p must be an odd prime")

    a %= p
    if a == 0:
        return 0

    value = pow(a, (p - 1) // 2, p)
    if value == 1:
        return 1
    if value == p - 1:
        return -1
    raise RuntimeError("Unexpected value in Euler criterion")


def jacobi_symbol(a: int, n: int) -> int:
    """Compute Jacobi symbol (a/n) for positive odd n.

    Returns one of {-1, 0, 1}.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")

    a %= n
    result = 1

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result

        a, n = n, a

        if a % 4 == 3 and n % 4 == 3:
            result = -result

        a %= n

    return result if n == 1 else 0


def prime_factorization(n: int) -> Dict[int, int]:
    """Trial-division factorization, used only for small cross-checks."""
    if n <= 0:
        raise ValueError("n must be positive")

    factors: Dict[int, int] = {}
    x = n

    while x % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        x //= 2

    d = 3
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 2

    if x > 1:
        factors[x] = factors.get(x, 0) + 1

    return factors


def jacobi_from_factorization(a: int, n: int) -> int:
    """Compute Jacobi by its multiplicative definition via factorization."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")

    factors = prime_factorization(n)
    result = 1
    for p, exp in factors.items():
        if p == 2:
            raise ValueError("n must be odd")
        lp = legendre_symbol(a, p)
        result *= lp ** exp
    return result


def quadratic_residue_set_mod_prime(p: int) -> set[int]:
    """Return {x^2 mod p | x in [0, p-1]} for prime p."""
    return {pow(x, 2, p) for x in range(p)}


def has_square_solution_mod_n(a: int, n: int) -> bool:
    """Brute-force whether x^2 ≡ a (mod n) has a solution."""
    target = a % n
    for x in range(n):
        if (x * x) % n == target:
            return True
    return False


def verify_quadratic_reciprocity(primes: List[int]) -> List[Tuple[int, int, int, int]]:
    """Return records: (p, q, lhs, rhs) for quadratic reciprocity checks.

    lhs = (p/q) * (q/p)
    rhs = (-1)^(((p-1)/2)*((q-1)/2))
    """
    records: List[Tuple[int, int, int, int]] = []
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            lhs = legendre_symbol(p, q) * legendre_symbol(q, p)
            rhs = -1 if (((p - 1) // 2) * ((q - 1) // 2)) % 2 == 1 else 1
            records.append((p, q, lhs, rhs))
    return records


def main() -> None:
    print("=== Legendre Symbol Table (mod p=23) ===")
    p = 23
    residues = quadratic_residue_set_mod_prime(p)
    for a in range(p):
        ls = legendre_symbol(a, p)
        expected = 0 if a % p == 0 else (1 if (a % p) in residues else -1)
        assert ls == expected, f"Legendre mismatch at a={a}, p={p}"
        print(f"(a/p)=({a:>2}/{p}) -> {ls:+d}")
    print("Legendre table cross-check passed for all a in [0, p-1].")

    print("\n=== Jacobi Symbol Cross-check (algorithm vs factorization) ===")
    sample_ns = [9, 15, 21, 45, 77, 91, 99]
    sample_as = [-7, 2, 5, 10, 19, 37]
    for n in sample_ns:
        for a in sample_as:
            j1 = jacobi_symbol(a, n)
            j2 = jacobi_from_factorization(a, n)
            assert j1 == j2, f"Jacobi mismatch for a={a}, n={n}: {j1} vs {j2}"
            print(f"(a/n)=({a:>3}/{n:>2}) -> {j1:+d}")
    print("All sample Jacobi values matched multiplicative definition.")

    print("\n=== Counterexample: Jacobi=1 but no square root ===")
    a, n = 2, 15
    jac = jacobi_symbol(a, n)
    solvable = has_square_solution_mod_n(a, n)
    print(f"(a/n)=({a}/{n}) = {jac:+d}")
    print(f"x^2 ≡ {a} (mod {n}) solvable? {solvable}")
    assert jac == 1 and not solvable
    print("Counterexample confirmed.")

    print("\n=== Quadratic Reciprocity Sanity Check ===")
    prime_list = [3, 5, 7, 11, 13, 17, 19]
    reciprocity_records = verify_quadratic_reciprocity(prime_list)
    for p1, p2, lhs, rhs in reciprocity_records:
        print(f"p={p1:>2}, q={p2:>2}, lhs={lhs:+d}, rhs={rhs:+d}")
        assert lhs == rhs, f"Quadratic reciprocity failed for p={p1}, q={p2}"
    print("Quadratic reciprocity checks passed.")

    print("\n=== Batch Consistency Test (odd n <= 199, a in [-30,30]) ===")
    for n in range(3, 200, 2):
        for a in range(-30, 31):
            assert jacobi_symbol(a, n) == jacobi_from_factorization(a, n)
    print("Batch consistency passed.")

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
