"""Montgomery multiplication MVP for MATH-0016.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time


def mod_inverse(a: int, m: int) -> int:
    """Return x such that (a * x) % m == 1, assuming gcd(a, m) == 1."""
    if m <= 0:
        raise ValueError("Modulus m must be positive.")

    t, new_t = 0, 1
    r, new_r = m, a % m
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r

    if r != 1:
        raise ValueError("Inverse does not exist because gcd(a, m) != 1.")

    return t % m


@dataclass(frozen=True)
class MontgomeryContext:
    """Context for Montgomery arithmetic under a fixed odd modulus."""

    modulus: int
    k: int
    R: int
    mask: int
    n_prime: int
    r2: int

    @classmethod
    def create(cls, modulus: int) -> "MontgomeryContext":
        if modulus <= 1:
            raise ValueError("modulus must be > 1")
        if modulus % 2 == 0:
            raise ValueError("modulus must be odd for R = 2^k Montgomery form")

        k = modulus.bit_length()
        R = 1 << k
        mask = R - 1

        # n_prime = -n^{-1} mod R
        inv = mod_inverse(modulus, R)
        n_prime = (-inv) & mask

        # R^2 mod n is used for fast conversion to Montgomery domain.
        r2 = (R * R) % modulus
        return cls(
            modulus=modulus,
            k=k,
            R=R,
            mask=mask,
            n_prime=n_prime,
            r2=r2,
        )

    def redc(self, t: int) -> int:
        """REDC reduction: returns t * R^{-1} (mod modulus)."""
        m = (t * self.n_prime) & self.mask
        u = (t + m * self.modulus) >> self.k
        if u >= self.modulus:
            u -= self.modulus
        return u

    def to_mont(self, x: int) -> int:
        """Map x from standard domain to Montgomery domain."""
        return self.redc((x % self.modulus) * self.r2)

    def from_mont(self, x_bar: int) -> int:
        """Map x from Montgomery domain to standard domain."""
        return self.redc(x_bar)

    def mont_mul(self, a_bar: int, b_bar: int) -> int:
        """Multiply two Montgomery-domain values; result stays in Montgomery domain."""
        return self.redc(a_bar * b_bar)

    def mul_mod(self, a: int, b: int) -> int:
        """Compute (a * b) % modulus using Montgomery arithmetic."""
        a_bar = self.to_mont(a)
        b_bar = self.to_mont(b)
        c_bar = self.mont_mul(a_bar, b_bar)
        return self.from_mont(c_bar)

    def pow_mod(self, base: int, exponent: int) -> int:
        """Compute pow(base, exponent, modulus) via Montgomery + binary exponentiation."""
        if exponent < 0:
            raise ValueError("exponent must be non-negative")

        result_bar = self.to_mont(1)
        base_bar = self.to_mont(base)
        e = exponent
        while e > 0:
            if e & 1:
                result_bar = self.mont_mul(result_bar, base_bar)
            base_bar = self.mont_mul(base_bar, base_bar)
            e >>= 1

        return self.from_mont(result_bar)


def self_test(ctx: MontgomeryContext, mul_cases: int = 800, pow_cases: int = 200) -> None:
    """Validate against Python built-ins for random inputs."""
    rng = random.Random(42)

    for _ in range(mul_cases):
        a = rng.randrange(0, ctx.modulus)
        b = rng.randrange(0, ctx.modulus)
        got = ctx.mul_mod(a, b)
        expected = (a * b) % ctx.modulus
        if got != expected:
            raise AssertionError(f"mul mismatch: got={got}, expected={expected}")

    for _ in range(pow_cases):
        a = rng.randrange(0, ctx.modulus)
        e = rng.randrange(0, 1 << 16)
        got = ctx.pow_mod(a, e)
        expected = pow(a, e, ctx.modulus)
        if got != expected:
            raise AssertionError(f"pow mismatch: got={got}, expected={expected}")


def benchmark(ctx: MontgomeryContext, rounds: int = 15000) -> None:
    """Simple micro-benchmark for demonstration (not a rigorous performance study)."""
    rng = random.Random(7)
    pairs = [(rng.randrange(ctx.modulus), rng.randrange(ctx.modulus)) for _ in range(rounds)]

    t0 = time.perf_counter()
    checksum_naive = 0
    for a, b in pairs:
        checksum_naive ^= (a * b) % ctx.modulus
    naive_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    checksum_mont_full = 0
    for a, b in pairs:
        checksum_mont_full ^= ctx.mul_mod(a, b)
    mont_full_time = time.perf_counter() - t0

    a_bars = [ctx.to_mont(a) for a, _ in pairs]
    b_bars = [ctx.to_mont(b) for _, b in pairs]

    t0 = time.perf_counter()
    checksum_mont_domain = 0
    for a_bar, b_bar in zip(a_bars, b_bars):
        checksum_mont_domain ^= ctx.from_mont(ctx.mont_mul(a_bar, b_bar))
    mont_domain_time = time.perf_counter() - t0

    print("\n[Benchmark]")
    print(f"rounds={rounds}, bits={ctx.modulus.bit_length()}")
    print(f"naive (a*b)%n          : {naive_time:.6f}s, checksum={checksum_naive}")
    print(f"montgomery full path   : {mont_full_time:.6f}s, checksum={checksum_mont_full}")
    print(f"montgomery in-domain   : {mont_domain_time:.6f}s, checksum={checksum_mont_domain}")


def main() -> None:
    # NIST P-256 prime modulus (odd, fixed-size finite field used in ECC).
    modulus = int(
        "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF", 16
    )
    ctx = MontgomeryContext.create(modulus)

    a = int("1234567890ABCDEF1234567890ABCDEF", 16)
    b = int("FEDCBA0987654321FEDCBA0987654321", 16)

    print("Montgomery Multiplication MVP")
    print(f"modulus bits: {ctx.modulus.bit_length()}")
    print(f"modulus: {ctx.modulus}")
    print()

    result = ctx.mul_mod(a, b)
    expected = (a * b) % ctx.modulus
    print("[Single multiplication]")
    print(f"a*b mod n (montgomery): {result}")
    print(f"a*b mod n (builtin)   : {expected}")

    exp_result = ctx.pow_mod(a, 65537)
    exp_expected = pow(a, 65537, ctx.modulus)
    print("\n[Single exponentiation]")
    print(f"a^65537 mod n (montgomery): {exp_result}")
    print(f"a^65537 mod n (builtin)   : {exp_expected}")

    self_test(ctx)
    print("\nSelf-test passed: Montgomery results match Python built-ins.")

    benchmark(ctx)


if __name__ == "__main__":
    main()
