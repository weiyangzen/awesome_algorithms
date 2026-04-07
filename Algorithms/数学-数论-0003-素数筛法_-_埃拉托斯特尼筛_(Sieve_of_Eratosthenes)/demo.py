"""MVP: Sieve of Eratosthenes."""

from __future__ import annotations

from math import isqrt
from time import perf_counter


def sieve_of_eratosthenes(n: int) -> list[int]:
    """Return all primes p where 2 <= p <= n."""
    if n < 2:
        return []

    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0

    limit = isqrt(n)
    for p in range(2, limit + 1):
        if is_prime[p]:
            start = p * p
            steps = ((n - start) // p) + 1
            is_prime[start : n + 1 : p] = b"\x00" * steps

    return [x for x in range(2, n + 1) if is_prime[x]]


def is_prime_naive(x: int) -> bool:
    """Slow primality check used only for correctness validation."""
    if x < 2:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False

    r = isqrt(x)
    for d in range(3, r + 1, 2):
        if x % d == 0:
            return False
    return True


def verify_small_range(max_n: int = 300) -> None:
    """Cross-check sieve output with naive primality results on small range."""
    for n in range(max_n + 1):
        expected = [x for x in range(2, n + 1) if is_prime_naive(x)]
        got = sieve_of_eratosthenes(n)
        assert got == expected, f"Mismatch at n={n}: got={got}, expected={expected}"


def run_case(n: int, expected_count: int | None = None) -> None:
    start = perf_counter()
    primes = sieve_of_eratosthenes(n)
    elapsed = perf_counter() - start

    if expected_count is not None:
        assert len(primes) == expected_count, (
            f"Prime count mismatch for n={n}: got={len(primes)}, "
            f"expected={expected_count}"
        )

    assert all(primes[i] < primes[i + 1] for i in range(len(primes) - 1))
    assert all(is_prime_naive(p) for p in primes[: min(30, len(primes))])

    head = primes[:10]
    tail = primes[-10:] if len(primes) > 10 else primes
    print(
        f"n={n:<7d} count={len(primes):<6d} elapsed={elapsed:.6f}s\n"
        f"  head={head}\n"
        f"  tail={tail}"
    )


def main() -> None:
    verify_small_range(300)
    print("Small-range verification passed (n <= 300).")

    # Known values of prime counting function pi(n).
    run_case(30, expected_count=10)
    run_case(100, expected_count=25)
    run_case(100_000, expected_count=9592)


if __name__ == "__main__":
    main()
