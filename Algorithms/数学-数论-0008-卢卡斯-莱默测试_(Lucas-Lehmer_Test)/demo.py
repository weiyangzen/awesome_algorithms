"""Lucas-Lehmer test MVP for Mersenne numbers.

Run:
    python3 demo.py
"""

from __future__ import annotations

from time import perf_counter


def is_prime_small(n: int) -> bool:
    """Simple deterministic primality check for small/medium integers."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def lucas_lehmer_test(p: int, trace: bool = False) -> tuple[bool, list[int], int]:
    """Test whether M_p = 2^p - 1 is prime (requires prime exponent p).

    Returns:
        (is_mersenne_prime, sequence, final_residue)
        - sequence records s_0..s_{p-2} when trace=True, otherwise [].
    """
    if p < 2:
        return (False, [], -1)
    if not is_prime_small(p):
        return (False, [], -1)
    if p == 2:
        # M_2 = 3 is prime; LL recurrence has zero iterations here.
        return (True, [4] if trace else [], 0)

    m = (1 << p) - 1
    s = 4
    sequence = [s] if trace else []
    for _ in range(p - 2):
        s = (s * s - 2) % m
        if trace:
            sequence.append(s)
    return (s == 0, sequence, s)


def main() -> None:
    # Include both known prime and composite Mersenne cases.
    candidate_exponents = [2, 3, 5, 7, 11, 13, 17, 19, 31]
    known_mersenne_prime_exponents = {2, 3, 5, 7, 13, 17, 19, 31}

    print("Lucas-Lehmer Test Demo")
    print("=" * 80)
    print(
        f"{'p':>3} | {'exp_prime':>9} | {'M_p prime':>9} | {'iterations':>10} | {'final_s':>8} | {'time_ms':>8}"
    )
    print("-" * 80)

    for p in candidate_exponents:
        t0 = perf_counter()
        exp_prime = is_prime_small(p)
        is_mp, _, final_s = lucas_lehmer_test(p, trace=False)
        dt_ms = (perf_counter() - t0) * 1000.0

        iterations = max(0, p - 2) if exp_prime else 0
        final_s_text = str(final_s) if exp_prime else "N/A"
        print(
            f"{p:>3} | {str(exp_prime):>9} | {str(is_mp):>9} | {iterations:>10} | {final_s_text:>8} | {dt_ms:>8.3f}"
        )

        # Self-check only for the known set used in this demo.
        expected = p in known_mersenne_prime_exponents
        if exp_prime:
            assert is_mp == expected, f"Unexpected result for p={p}"

    print("-" * 80)
    print("Trace examples:")
    for p in (7, 11):
        is_mp, seq, final_s = lucas_lehmer_test(p, trace=True)
        print(f"p={p}, M_p prime? {is_mp}, final_s={final_s}, sequence={seq}")


if __name__ == "__main__":
    main()
