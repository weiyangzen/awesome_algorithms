"""Minimal runnable MVP for continued-fraction convergents (MATH-0035)."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Sequence, Tuple


def convergents_from_terms(terms: Sequence[int]) -> List[Tuple[int, int]]:
    """Return convergents (p_k, q_k) of a simple continued fraction.

    Recurrence:
    - p_{-2}=0, p_{-1}=1, p_k = a_k*p_{k-1} + p_{k-2}
    - q_{-2}=1, q_{-1}=0, q_k = a_k*q_{k-1} + q_{k-2}
    """
    if not terms:
        raise ValueError("terms must be non-empty")

    for i, a in enumerate(terms):
        if not isinstance(a, int):
            raise TypeError(f"term[{i}] must be int, got {type(a)}")

    p_nm2, p_nm1 = 0, 1
    q_nm2, q_nm1 = 1, 0
    out: List[Tuple[int, int]] = []

    for a_k in terms:
        p_k = a_k * p_nm1 + p_nm2
        q_k = a_k * q_nm1 + q_nm2
        out.append((p_k, q_k))
        p_nm2, p_nm1 = p_nm1, p_k
        q_nm2, q_nm1 = q_nm1, q_k

    return out


def continued_fraction_of_real(x: float, max_terms: int = 12, tol: float = 1e-15) -> List[int]:
    """Approximate real x with finite simple continued fraction terms."""
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    if max_terms <= 0:
        raise ValueError("max_terms must be positive")

    terms: List[int] = []
    value = x

    for _ in range(max_terms):
        a = math.floor(value)
        terms.append(int(a))
        frac_part = value - a
        if abs(frac_part) < tol:
            break
        value = 1.0 / frac_part

    return terms


def fraction_from_terms(terms: Sequence[int]) -> Fraction:
    """Evaluate a finite continued fraction exactly as Fraction."""
    if not terms:
        raise ValueError("terms must be non-empty")

    value = Fraction(terms[-1], 1)
    for a in reversed(terms[:-1]):
        value = Fraction(a, 1) + Fraction(1, value)
    return value


def verify_determinant_identity(convergents: Sequence[Tuple[int, int]]) -> bool:
    """Check p_k*q_{k-1} - p_{k-1}*q_k = (-1)^(k-1) for k>=1."""
    for k in range(1, len(convergents)):
        p_k, q_k = convergents[k]
        p_prev, q_prev = convergents[k - 1]
        lhs = p_k * q_prev - p_prev * q_k
        rhs = -1 if (k - 1) % 2 else 1
        if lhs != rhs:
            return False
    return True


def print_case(name: str, x: float, terms: Sequence[int]) -> None:
    """Print convergent table and run internal consistency checks."""
    convs = convergents_from_terms(terms)
    assert verify_determinant_identity(convs), "determinant identity failed"

    print(f"\nCase: {name}")
    print(f"terms = {list(terms)}")
    print("k | convergent p_k/q_k | approx               | abs_error           | bound 1/(q_k*q_{k+1})")
    print("-" * 98)

    for k, (p_k, q_k) in enumerate(convs):
        exact_prefix = fraction_from_terms(terms[: k + 1])
        assert exact_prefix.numerator == p_k and exact_prefix.denominator == q_k

        approx = p_k / q_k
        err = abs(x - approx)

        bound_text = "N/A"
        if k + 1 < len(convs):
            q_next = convs[k + 1][1]
            bound = 1.0 / (q_k * q_next)
            # Floating calculations have tiny representation errors; keep a small slack.
            assert err <= bound + 1e-12, "error bound check failed"
            bound_text = f"{bound:.12e}"

        print(
            f"{k:1d} | {p_k:>8d}/{q_k:<8d} | {approx: .15f} | {err: .12e} | {bound_text}"
        )

    final_p, final_q = convs[-1]
    print(f"final convergent = {final_p}/{final_q} = {final_p / final_q:.15f}")


def main() -> None:
    print("Continued-Fraction Convergents MVP (MATH-0035)")
    print("=" * 98)

    sqrt2_terms = [1] + [2] * 8
    pi_terms = [3, 7, 15, 1, 292]
    e_terms = [2, 1, 2, 1, 1, 4, 1, 1, 6]

    print_case("sqrt(2)", math.sqrt(2.0), sqrt2_terms)
    print_case("pi", math.pi, pi_terms)
    print_case("e", math.e, e_terms)

    auto_terms = continued_fraction_of_real(math.pi, max_terms=8)
    auto_convs = convergents_from_terms(auto_terms)
    auto_p, auto_q = auto_convs[-1]
    auto_err = abs(math.pi - auto_p / auto_q)

    print("\nAuto-generated terms for pi (from float):", auto_terms)
    print(f"last convergent = {auto_p}/{auto_q}, abs_error = {auto_err:.12e}")

    print("=" * 98)
    print("All checks passed.")


if __name__ == "__main__":
    main()
