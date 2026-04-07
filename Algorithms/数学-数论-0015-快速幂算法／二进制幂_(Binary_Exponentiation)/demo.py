"""Minimal runnable MVP for Binary Exponentiation."""

from __future__ import annotations



def binary_exponentiation(base: int, exp: int) -> int:
    """Compute base^exp for exp >= 0 using binary exponentiation."""
    if exp < 0:
        raise ValueError("exp must be non-negative")

    result = 1
    cur = base
    e = exp

    while e > 0:
        if e & 1:
            result *= cur
        cur *= cur
        e >>= 1

    return result



def mod_binary_exponentiation(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) % mod for exp >= 0 and mod > 0."""
    if exp < 0:
        raise ValueError("exp must be non-negative")
    if mod <= 0:
        raise ValueError("mod must be positive")

    result = 1 % mod
    cur = base % mod
    e = exp

    while e > 0:
        if e & 1:
            result = (result * cur) % mod
        cur = (cur * cur) % mod
        e >>= 1

    return result



def run_demo_cases() -> None:
    non_mod_cases = [
        (2, 10),
        (5, 0),
        (-2, 7),
        (3, 45),
        (7, 64),
    ]
    mod_cases = [
        (2, 1000, 1_000_000_007),
        (123456789, 12345, 998244353),
        (-7, 255, 97),
        (0, 0, 13),
        (42, 999999, 1),
    ]

    total = 0
    passed = 0

    print("=== Binary Exponentiation Demo ===")
    print("[Non-mod exponentiation]")
    for base, exp in non_mod_cases:
        got = binary_exponentiation(base, exp)
        ref = pow(base, exp)
        ok = got == ref
        total += 1
        passed += int(ok)
        print(
            f"base={base:>11}, exp={exp:>8} | got={got} | ref={ref} | {'PASS' if ok else 'FAIL'}"
        )

    print("\n[Modular exponentiation]")
    for base, exp, mod in mod_cases:
        got = mod_binary_exponentiation(base, exp, mod)
        ref = pow(base, exp, mod)
        ok = got == ref
        total += 1
        passed += int(ok)
        print(
            f"base={base:>11}, exp={exp:>8}, mod={mod:>10} | got={got} | ref={ref} | {'PASS' if ok else 'FAIL'}"
        )

    print(f"\nSummary: {passed}/{total} cases passed.")
    if passed != total:
        raise RuntimeError("Some demo cases failed")



def main() -> None:
    run_demo_cases()


if __name__ == "__main__":
    main()
