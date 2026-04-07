"""Minimal runnable MVP for big integer multiplication via divide and conquer.

This demo implements Karatsuba multiplication on decimal strings and keeps
all arithmetic transparent (no third-party big-number package).
"""

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import Tuple


def strip_leading_zeros(num: str) -> str:
    """Return canonical non-negative decimal string without leading zeros."""
    stripped = num.lstrip("0")
    return stripped if stripped else "0"


def parse_signed_decimal(text: str) -> Tuple[int, str]:
    """Parse a decimal string into (sign, abs_digits).

    Returns:
        sign: either +1 or -1
        abs_digits: canonical non-negative decimal string ("0" has sign +1)
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty input is not a valid integer")

    sign = 1
    if s[0] == "-":
        sign = -1
        s = s[1:]
    elif s[0] == "+":
        s = s[1:]

    if not s or not s.isdigit():
        raise ValueError(f"Invalid integer literal: {text!r}")

    abs_digits = strip_leading_zeros(s)
    if abs_digits == "0":
        sign = 1
    return sign, abs_digits


def compare_abs(a: str, b: str) -> int:
    """Compare two non-negative decimal strings.

    Returns -1 if a<b, 0 if a==b, +1 if a>b.
    """
    a = strip_leading_zeros(a)
    b = strip_leading_zeros(b)
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def add_abs(a: str, b: str) -> str:
    """Add two non-negative decimal strings."""
    i = len(a) - 1
    j = len(b) - 1
    carry = 0
    out = []

    while i >= 0 or j >= 0 or carry:
        da = ord(a[i]) - ord("0") if i >= 0 else 0
        db = ord(b[j]) - ord("0") if j >= 0 else 0
        total = da + db + carry
        out.append(chr((total % 10) + ord("0")))
        carry = total // 10
        i -= 1
        j -= 1

    out.reverse()
    return "".join(out)


def sub_abs(a: str, b: str) -> str:
    """Compute a-b for non-negative decimal strings under precondition a>=b."""
    if compare_abs(a, b) < 0:
        raise ValueError("sub_abs requires a >= b")

    i = len(a) - 1
    j = len(b) - 1
    borrow = 0
    out = []

    while i >= 0:
        da = (ord(a[i]) - ord("0")) - borrow
        db = ord(b[j]) - ord("0") if j >= 0 else 0
        if da < db:
            da += 10
            borrow = 1
        else:
            borrow = 0
        out.append(chr((da - db) + ord("0")))
        i -= 1
        j -= 1

    out.reverse()
    return strip_leading_zeros("".join(out))


def shift_digits(num: str, k: int) -> str:
    """Multiply a non-negative decimal string by 10^k."""
    if num == "0":
        return "0"
    return num + ("0" * k)


def schoolbook_mul_abs(a: str, b: str) -> str:
    """Grade-school multiplication for non-negative decimal strings."""
    a = strip_leading_zeros(a)
    b = strip_leading_zeros(b)
    if a == "0" or b == "0":
        return "0"

    ra = [ord(ch) - ord("0") for ch in reversed(a)]
    rb = [ord(ch) - ord("0") for ch in reversed(b)]
    out = [0] * (len(ra) + len(rb))

    for i, da in enumerate(ra):
        carry = 0
        for j, db in enumerate(rb):
            pos = i + j
            total = out[pos] + da * db + carry
            out[pos] = total % 10
            carry = total // 10

        pos = i + len(rb)
        while carry:
            total = out[pos] + carry
            out[pos] = total % 10
            carry = total // 10
            pos += 1

    while len(out) > 1 and out[-1] == 0:
        out.pop()

    return "".join(str(d) for d in reversed(out))


def karatsuba_mul_abs(a: str, b: str, threshold: int = 32) -> str:
    """Karatsuba multiplication for non-negative decimal strings."""
    a = strip_leading_zeros(a)
    b = strip_leading_zeros(b)

    if a == "0" or b == "0":
        return "0"

    n = max(len(a), len(b))
    if n <= threshold:
        return schoolbook_mul_abs(a, b)

    if n % 2 == 1:
        n += 1

    a = a.zfill(n)
    b = b.zfill(n)
    m = n // 2

    a_high, a_low = a[:-m], a[-m:]
    b_high, b_low = b[:-m], b[-m:]

    z0 = karatsuba_mul_abs(a_low, b_low, threshold)
    z2 = karatsuba_mul_abs(a_high, b_high, threshold)

    sum_a = add_abs(a_high, a_low)
    sum_b = add_abs(b_high, b_low)
    z1 = karatsuba_mul_abs(sum_a, sum_b, threshold)

    # z1 - z2 - z0 = ad + bc
    middle = sub_abs(sub_abs(z1, z2), z0)

    return strip_leading_zeros(
        add_abs(
            add_abs(shift_digits(z2, 2 * m), shift_digits(middle, m)),
            z0,
        )
    )


def multiply_bigint(lhs: str, rhs: str, threshold: int = 32) -> str:
    """Signed big integer multiplication based on Karatsuba."""
    sign_l, abs_l = parse_signed_decimal(lhs)
    sign_r, abs_r = parse_signed_decimal(rhs)

    abs_result = karatsuba_mul_abs(abs_l, abs_r, threshold=threshold)
    sign_out = sign_l * sign_r

    if abs_result == "0":
        return "0"
    return abs_result if sign_out > 0 else f"-{abs_result}"


def multiply_bigint_schoolbook(lhs: str, rhs: str) -> str:
    """Signed big integer multiplication using grade-school method."""
    sign_l, abs_l = parse_signed_decimal(lhs)
    sign_r, abs_r = parse_signed_decimal(rhs)

    abs_result = schoolbook_mul_abs(abs_l, abs_r)
    sign_out = sign_l * sign_r

    if abs_result == "0":
        return "0"
    return abs_result if sign_out > 0 else f"-{abs_result}"


def random_decimal(rng: Random, digits: int, allow_negative: bool = True) -> str:
    """Generate a random decimal string with exactly *digits* digits (unless zero)."""
    if digits <= 0:
        return "0"

    first = str(rng.randint(1, 9))
    tail = "".join(str(rng.randint(0, 9)) for _ in range(digits - 1))
    value = first + tail

    if allow_negative and rng.random() < 0.5:
        value = "-" + value
    return value


def run_basic_cases() -> None:
    print("[Case 1] 基础样例")
    examples = [
        ("123456789", "987654321"),
        ("-123456", "7890"),
        ("0000123", "-0000456"),
        ("0", "999999999999999999999"),
        ("3141592653589793238462643383279", "2718281828459045235360287471352"),
    ]

    for lhs, rhs in examples:
        got = multiply_bigint(lhs, rhs)
        expected = str(int(lhs) * int(rhs))
        print(f"{lhs} * {rhs} = {got}")
        assert got == expected, f"Mismatch for {lhs} * {rhs}"


def run_random_regression(seed: int = 17, rounds: int = 40) -> None:
    print("\n[Case 2] 随机回归")
    rng = Random(seed)

    for _ in range(rounds):
        digits_a = rng.randint(1, 180)
        digits_b = rng.randint(1, 180)
        lhs = random_decimal(rng, digits_a, allow_negative=True)
        rhs = random_decimal(rng, digits_b, allow_negative=True)

        got = multiply_bigint(lhs, rhs)
        expected = str(int(lhs) * int(rhs))
        assert got == expected, "Random regression failed"

    print(f"random rounds={rounds}, seed={seed}: passed")


def benchmark(seed: int = 2026, digits: int = 800) -> None:
    print("\n[Case 3] 小规模性能对照（同样正确性）")
    rng = Random(seed)
    lhs = random_decimal(rng, digits, allow_negative=False)
    rhs = random_decimal(rng, digits, allow_negative=False)

    t0 = perf_counter()
    out_school = multiply_bigint_schoolbook(lhs, rhs)
    t1 = perf_counter()

    t2 = perf_counter()
    out_karatsuba = multiply_bigint(lhs, rhs)
    t3 = perf_counter()

    assert out_school == out_karatsuba, "Benchmark correctness mismatch"

    print(f"digits(lhs)=digits(rhs)={digits}")
    print(f"schoolbook: {t1 - t0:.6f}s")
    print(f"karatsuba : {t3 - t2:.6f}s")


def main() -> None:
    run_basic_cases()
    run_random_regression()
    benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
