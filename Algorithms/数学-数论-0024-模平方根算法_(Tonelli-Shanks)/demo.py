"""Tonelli-Shanks algorithm MVP (odd prime modulus)."""

from __future__ import annotations

from typing import Optional, Tuple


def is_prime_trial(n: int) -> bool:
    """Simple trial-division primality check for small demo inputs."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def legendre_symbol(a: int, p: int) -> int:
    """Return Legendre symbol (a|p) in {-1, 0, 1}, with p an odd prime."""
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def tonelli_shanks_one_root(n: int, p: int) -> Optional[int]:
    """Return one root r such that r^2 ≡ n (mod p), or None if no root exists.

    Preconditions:
    - p is an odd prime.
    """
    n %= p
    if n == 0:
        return 0

    if legendre_symbol(n, p) != 1:
        return None

    # Fast path for p % 4 == 3.
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Factor p - 1 = q * 2^s, q odd.
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a quadratic non-residue z.
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while t != 1:
        # Find smallest i in [1, m) such that t^(2^i) == 1.
        i = 1
        t2i = (t * t) % p
        while i < m and t2i != 1:
            t2i = (t2i * t2i) % p
            i += 1

        if i == m:
            # Should not happen for valid odd prime modulus and quadratic residue.
            raise RuntimeError("Tonelli-Shanks invariant broken: i == m")

        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        b2 = (b * b) % p
        t = (t * b2) % p
        c = b2
        m = i

    return r


def sqrt_mod_prime(n: int, p: int) -> Tuple[int, ...]:
    """Return all roots of x^2 ≡ n (mod p), sorted and deduplicated."""
    if p == 2:
        return (n & 1,)
    if not is_prime_trial(p) or p % 2 == 0:
        raise ValueError("This MVP only supports odd prime modulus p")

    n %= p
    if n == 0:
        return (0,)

    root = tonelli_shanks_one_root(n, p)
    if root is None:
        return ()

    other = (-root) % p
    if other == root:
        return (root,)
    return tuple(sorted((root, other)))


def brute_force_roots(n: int, p: int) -> Tuple[int, ...]:
    """Reference implementation for verification on small p."""
    n %= p
    roots = [x for x in range(p) if (x * x) % p == n]
    return tuple(roots)


def run_showcase() -> None:
    print("Tonelli-Shanks 模平方根算法 MVP")
    print("=" * 52)

    examples = [
        (10, 13),
        (5, 41),
        (56, 101),
        (2, 11),  # non-residue: should have no root
        (0, 97),
    ]

    for n, p in examples:
        roots = sqrt_mod_prime(n, p)
        if not roots:
            print(f"n={n:>3}, p={p:>3} -> 无解")
            continue
        checks = [pow(r, 2, p) for r in roots]
        print(f"n={n:>3}, p={p:>3} -> roots={roots}, verify={checks}")


def run_self_tests() -> None:
    # Cross-check Tonelli-Shanks against brute force on a range of small odd primes.
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47]
    for p in primes:
        for n in range(p):
            got = sqrt_mod_prime(n, p)
            expected = brute_force_roots(n, p)
            assert got == expected, (
                f"Mismatch at p={p}, n={n}: got={got}, expected={expected}"
            )

    print("\n[OK] 小素数全量交叉验证通过。")


def main() -> None:
    run_showcase()
    run_self_tests()


if __name__ == "__main__":
    main()
