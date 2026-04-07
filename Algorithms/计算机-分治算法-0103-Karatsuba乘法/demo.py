"""Minimal runnable MVP for Karatsuba multiplication.

The implementation keeps arithmetic transparent with decimal-string helpers.
No third-party big integer package is used.
"""

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import Tuple


def strip_leading_zeros(num: str) -> str:
    """Normalize a non-negative decimal string."""
    stripped = num.lstrip("0")
    return stripped if stripped else "0"


def parse_signed_decimal(text: str) -> Tuple[int, str]:
    """Parse decimal text into (sign, abs_digits)."""
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

    Returns:
        -1 if a < b, 0 if a == b, +1 if a > b
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
    """Compute a-b for non-negative decimals under precondition a >= b."""
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


def shift_digits(num: str, places: int) -> str:
    """Multiply a non-negative decimal string by 10^places."""
    if num == "0":
        return "0"
    return num + ("0" * places)


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

    z2 = karatsuba_mul_abs(a_high, b_high, threshold)
    z0 = karatsuba_mul_abs(a_low, b_low, threshold)

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


def multiply_karatsuba(lhs: str, rhs: str, threshold: int = 32) -> str:
    """Multiply two signed decimal strings using Karatsuba."""
    sign_l, abs_l = parse_signed_decimal(lhs)
    sign_r, abs_r = parse_signed_decimal(rhs)

    abs_result = karatsuba_mul_abs(abs_l, abs_r, threshold=threshold)
    sign_out = sign_l * sign_r

    if abs_result == "0":
        return "0"
    return abs_result if sign_out > 0 else f"-{abs_result}"


def multiply_schoolbook(lhs: str, rhs: str) -> str:
    """Reference multiplication using manual grade-school method."""
    sign_l, abs_l = parse_signed_decimal(lhs)
    sign_r, abs_r = parse_signed_decimal(rhs)

    abs_result = schoolbook_mul_abs(abs_l, abs_r)
    sign_out = sign_l * sign_r

    if abs_result == "0":
        return "0"
    return abs_result if sign_out > 0 else f"-{abs_result}"


def random_decimal(rng: Random, digits: int, allow_negative: bool = True) -> str:
    """Generate a random decimal string with exact digit length (except zero case)."""
    if digits <= 0:
        return "0"

    first = str(rng.randint(1, 9))
    tail = "".join(str(rng.randint(0, 9)) for _ in range(digits - 1))
    value = first + tail

    if allow_negative and rng.random() < 0.5:
        return "-" + value
    return value


def run_fixed_cases() -> None:
    print("[Case 1] Fixed examples")
    cases = [
        ("1234", "5678"),
        ("-123456", "7890"),
        ("+0000123", "-0000456"),
        ("0", "98765432109876543210"),
        ("3141592653589793238462643383279", "2718281828459045235360287471352"),
    ]

    for lhs, rhs in cases:
        got = multiply_karatsuba(lhs, rhs)
        expected = str(int(lhs) * int(rhs))
        print(f"{lhs} * {rhs} = {got}")
        assert got == expected, f"Mismatch for {lhs} * {rhs}"


def run_random_regression(seed: int = 20260407, rounds: int = 64) -> None:
    print("\n[Case 2] Random regression")
    rng = Random(seed)

    for _ in range(rounds):
        da = rng.randint(1, 220)
        db = rng.randint(1, 220)

        # Inject zero cases occasionally to exercise sign/zero normalization.
        lhs = "0" if rng.random() < 0.08 else random_decimal(rng, da)
        rhs = "0" if rng.random() < 0.08 else random_decimal(rng, db)

        got = multiply_karatsuba(lhs, rhs)
        expected = str(int(lhs) * int(rhs))
        assert got == expected, "Random regression failed"

    print(f"rounds={rounds}, seed={seed}: passed")


def run_micro_benchmark(seed: int = 7) -> None:
    print("\n[Case 3] Micro benchmark (same inputs)")
    rng = Random(seed)
    sizes = [32, 64, 128, 256]

    for digits in sizes:
        lhs = random_decimal(rng, digits, allow_negative=False)
        rhs = random_decimal(rng, digits, allow_negative=False)

        t0 = perf_counter()
        ans_school = multiply_schoolbook(lhs, rhs)
        t1 = perf_counter()

        t2 = perf_counter()
        ans_karatsuba = multiply_karatsuba(lhs, rhs)
        t3 = perf_counter()

        assert ans_school == ans_karatsuba

        school_ms = (t1 - t0) * 1000
        kara_ms = (t3 - t2) * 1000
        print(
            f"digits={digits:>3} | schoolbook={school_ms:8.3f} ms | "
            f"karatsuba={kara_ms:8.3f} ms"
        )


def main() -> None:
    run_fixed_cases()
    run_random_regression()
    run_micro_benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
