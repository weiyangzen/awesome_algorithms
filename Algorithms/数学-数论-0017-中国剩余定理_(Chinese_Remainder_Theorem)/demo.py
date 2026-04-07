"""Chinese Remainder Theorem (CRT) minimal runnable MVP.

This script demonstrates:
1) Pairwise-coprime CRT
2) Generalized CRT for non-coprime moduli
3) Inconsistency detection
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def mod_inverse(a: int, m: int) -> int:
    """Return inverse of a modulo m, assuming gcd(a, m) == 1."""
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for a={a}, m={m} (gcd={g})")
    return x % m


def _normalize_inputs(
    remainders: Sequence[int], moduli: Sequence[int]
) -> Tuple[List[int], List[int]]:
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    if not remainders:
        raise ValueError("empty system is not supported")
    normalized_remainders: List[int] = []
    normalized_moduli: List[int] = []
    for a, m in zip(remainders, moduli):
        if m <= 0:
            raise ValueError(f"modulus must be positive, got {m}")
        normalized_moduli.append(m)
        normalized_remainders.append(a % m)
    return normalized_remainders, normalized_moduli


def crt_pairwise(remainders: Sequence[int], moduli: Sequence[int]) -> Tuple[int, int]:
    """Solve CRT when all moduli are pairwise coprime.

    Returns:
        (x, M), meaning x is the unique solution modulo M.
    """
    a, m = _normalize_inputs(remainders, moduli)
    total_modulus = 1
    for mi in m:
        total_modulus *= mi

    x = 0
    for ai, mi in zip(a, m):
        Mi = total_modulus // mi
        inv = mod_inverse(Mi % mi, mi)
        x = (x + ai * Mi * inv) % total_modulus
    return x, total_modulus


def crt_general(remainders: Sequence[int], moduli: Sequence[int]) -> Tuple[int, int]:
    """Solve general CRT (moduli may be non-coprime).

    Raises:
        ValueError if the congruence system is inconsistent.
    """
    a, m = _normalize_inputs(remainders, moduli)
    x = 0
    current_modulus = 1

    for ai, mi in zip(a, m):
        g, p, _ = extended_gcd(current_modulus, mi)
        delta = ai - x
        if delta % g != 0:
            raise ValueError(
                "Inconsistent system: "
                f"x ≡ {x} (mod {current_modulus}) and x ≡ {ai} (mod {mi})"
            )

        reduced_modulus = mi // g
        t = ((delta // g) * p) % reduced_modulus
        lcm = (current_modulus // g) * mi
        x = (x + current_modulus * t) % lcm
        current_modulus = lcm

    return x, current_modulus


def verify_solution(x: int, remainders: Iterable[int], moduli: Iterable[int]) -> bool:
    """Check x satisfies all congruences."""
    return all(x % m == a % m for a, m in zip(remainders, moduli))


def main() -> None:
    print("=== Chinese Remainder Theorem MVP ===")

    # Case 1: pairwise-coprime system
    remainders1 = [2, 3, 2]
    moduli1 = [3, 5, 7]
    x1, M1 = crt_pairwise(remainders1, moduli1)
    print(f"[Pairwise] x ≡ {x1} (mod {M1}), valid={verify_solution(x1, remainders1, moduli1)}")

    # Case 2: non-coprime but consistent system
    remainders2 = [1, 3]
    moduli2 = [4, 6]
    x2, M2 = crt_general(remainders2, moduli2)
    print(f"[General ] x ≡ {x2} (mod {M2}), valid={verify_solution(x2, remainders2, moduli2)}")

    # Case 3: non-coprime and inconsistent system
    remainders3 = [1, 0]
    moduli3 = [2, 4]
    try:
        x3, M3 = crt_general(remainders3, moduli3)
        print(
            f"[Invalid ] Unexpected solution x ≡ {x3} (mod {M3}), "
            f"valid={verify_solution(x3, remainders3, moduli3)}"
        )
    except ValueError as exc:
        print(f"[Invalid ] correctly detected inconsistency: {exc}")


if __name__ == "__main__":
    main()
