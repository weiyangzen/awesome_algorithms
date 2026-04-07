"""Barrett reduction minimal runnable MVP.

This script demonstrates:
1) Barrett precomputation for a fixed modulus
2) Modular reduction and modular multiplication via Barrett
3) Modular exponentiation based on repeated squaring
4) Correctness checks against Python built-ins and a small benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time


@dataclass
class BarrettReducer:
    """Reducer for fixed-modulus Barrett reduction (binary base)."""

    modulus: int

    def __post_init__(self) -> None:
        if self.modulus <= 1:
            raise ValueError(f"modulus must be > 1, got {self.modulus}")

        self.k = self.modulus.bit_length()
        self.shift = 2 * self.k
        self.limit = 1 << self.shift
        self.mu = self.limit // self.modulus

    def reduce(self, x: int) -> int:
        """Return x mod modulus using Barrett reduction.

        Current MVP range contract: 0 <= x < 2**(2k), where k = bit_length(modulus).
        """
        if x < 0:
            raise ValueError(f"x must be non-negative, got {x}")
        if x >= self.limit:
            raise ValueError(
                "x out of supported range for this MVP: "
                f"x={x}, limit={self.limit} (need x < 2**(2k))"
            )

        q_hat = (x * self.mu) >> self.shift
        r = x - q_hat * self.modulus

        while r >= self.modulus:
            r -= self.modulus
        while r < 0:
            r += self.modulus
        return r

    def mul_mod(self, a: int, b: int) -> int:
        """Return (a * b) % modulus using Barrett reduction."""
        a_mod = a % self.modulus
        b_mod = b % self.modulus
        x = a_mod * b_mod
        return self.reduce(x)

    def pow_mod(self, base: int, exp: int) -> int:
        """Return (base ** exp) % modulus using repeated squaring + Barrett mul."""
        if exp < 0:
            raise ValueError(f"negative exponent is not supported, got {exp}")

        result = 1 % self.modulus
        factor = base % self.modulus
        e = exp

        while e > 0:
            if e & 1:
                result = self.mul_mod(result, factor)
            factor = self.mul_mod(factor, factor)
            e >>= 1

        return result


def run_correctness_suite(reducer: BarrettReducer, seed: int = 20260407) -> None:
    """Randomized checks versus Python ground truth."""
    rng = random.Random(seed)
    n = reducer.modulus

    for _ in range(3000):
        x = rng.randrange(0, n * n)
        got = reducer.reduce(x)
        expect = x % n
        if got != expect:
            raise AssertionError(f"reduce mismatch: x={x}, got={got}, expect={expect}")

    for _ in range(3000):
        a = rng.randrange(-n, n)
        b = rng.randrange(-n, n)
        got = reducer.mul_mod(a, b)
        expect = (a * b) % n
        if got != expect:
            raise AssertionError(f"mul mismatch: a={a}, b={b}, got={got}, expect={expect}")

    for _ in range(600):
        base = rng.randrange(-n, n)
        exp = rng.randrange(0, 5000)
        got = reducer.pow_mod(base, exp)
        expect = pow(base, exp, n)
        if got != expect:
            raise AssertionError(
                f"pow mismatch: base={base}, exp={exp}, got={got}, expect={expect}"
            )


def run_micro_benchmark(reducer: BarrettReducer, rounds: int = 40000) -> tuple[float, float]:
    """Compare pure-Python Barrett mul with direct % baseline.

    Returns:
        (barrett_seconds, builtin_seconds)
    """
    rng = random.Random(123456)
    n = reducer.modulus
    pairs = [(rng.randrange(0, n), rng.randrange(0, n)) for _ in range(rounds)]

    start = time.perf_counter()
    acc1 = 0
    for a, b in pairs:
        acc1 = (acc1 + reducer.mul_mod(a, b)) % n
    t_barrett = time.perf_counter() - start

    start = time.perf_counter()
    acc2 = 0
    for a, b in pairs:
        acc2 = (acc2 + (a * b) % n) % n
    t_builtin = time.perf_counter() - start

    if acc1 != acc2:
        raise AssertionError("benchmark accumulator mismatch")

    return t_barrett, t_builtin


def main() -> None:
    modulus = 1_000_000_007
    reducer = BarrettReducer(modulus)

    print("=== Barrett Reduction MVP ===")
    print(f"modulus={reducer.modulus}, k={reducer.k}, mu={reducer.mu}")

    x_demo = 123_456_789_123_456_789
    x_demo = x_demo % (modulus * modulus)
    barrett_r = reducer.reduce(x_demo)
    builtin_r = x_demo % modulus
    print(
        f"demo reduce: x={x_demo}, barrett={barrett_r}, builtin={builtin_r}, "
        f"equal={barrett_r == builtin_r}"
    )

    pow_demo = reducer.pow_mod(123456789, 12345)
    print(f"demo pow_mod: {pow_demo} (verified={pow_demo == pow(123456789, 12345, modulus)})")

    run_correctness_suite(reducer)
    print("random correctness suite: PASS")

    t_barrett, t_builtin = run_micro_benchmark(reducer)
    print(
        "micro benchmark (pure Python, lower is better): "
        f"barrett={t_barrett:.6f}s, builtin_mod={t_builtin:.6f}s"
    )


if __name__ == "__main__":
    main()
