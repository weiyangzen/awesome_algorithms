"""Brent's Method minimal runnable MVP.

This demo implements a scalar root finder with bracketing:
- bisection for robustness,
- secant / inverse quadratic interpolation (IQI) for speed,
- automatic fallback when interpolation is unsafe.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Callable, List, Tuple


EPS = 2.220446049250313e-16  # IEEE-754 double epsilon


@dataclass
class IterationRecord:
    iteration: int
    a: float
    b: float
    c: float
    fa: float
    fb: float
    fc: float
    half_interval: float
    step: str


@dataclass
class BrentResult:
    root: float
    f_at_root: float
    iterations: int
    converged: bool
    history: List[IterationRecord]


def _ensure_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")


def brent_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> BrentResult:
    """Find a root of f on [a, b] with Brent's method.

    Preconditions:
    - f(a) and f(b) must have opposite signs unless one is exactly zero.
    - tol > 0 and max_iter >= 1.
    """

    _ensure_finite_scalar("a", a)
    _ensure_finite_scalar("b", b)
    _ensure_finite_scalar("tol", tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    a = float(a)
    b = float(b)
    if a == b:
        raise ValueError("a and b must be different")

    fa = float(f(a))
    fb = float(f(b))
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("f(a) and f(b) must be finite")

    history: List[IterationRecord] = []

    if fa == 0.0:
        history.append(IterationRecord(0, a, a, b, fa, fa, fb, abs(b - a) * 0.5, "exact-endpoint"))
        return BrentResult(root=a, f_at_root=fa, iterations=0, converged=True, history=history)
    if fb == 0.0:
        history.append(IterationRecord(0, a, b, a, fa, fb, fa, abs(b - a) * 0.5, "exact-endpoint"))
        return BrentResult(root=b, f_at_root=fb, iterations=0, converged=True, history=history)

    if fa * fb > 0.0:
        raise ValueError("Root is not bracketed: f(a) and f(b) must have opposite signs")

    # Keep |fb| <= |fa| so b is the best current estimate.
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    history.append(IterationRecord(0, a, b, c, fa, fb, fc, abs(c - b) * 0.5, "initial"))

    for it in range(1, max_iter + 1):
        # Ensure [b, c] brackets the root.
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d

        # Make b the point with smallest |f| among (b, c).
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2.0 * EPS * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)

        if abs(m) <= tol1 or fb == 0.0:
            history.append(IterationRecord(it, a, b, c, fa, fb, fc, abs(c - b) * 0.5, "converged"))
            return BrentResult(root=b, f_at_root=fb, iterations=it - 1, converged=True, history=history)

        step_kind = "bisection"

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa

            # Interpolation proposal: secant (2-point) or IQI (3-point).
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
                proposal = "secant"
            else:
                q_ac = fa / fc
                r_bc = fb / fc
                p = s * (2.0 * m * q_ac * (q_ac - r_bc) - (b - a) * (r_bc - 1.0))
                q = (q_ac - 1.0) * (r_bc - 1.0) * (s - 1.0)
                proposal = "iqi"

            if p > 0.0:
                q = -q
            p = abs(p)

            accept_interp = False
            if q != 0.0:
                bound_1 = 3.0 * m * q - abs(tol1 * q)
                bound_2 = abs(e * q)
                if 2.0 * p < min(bound_1, bound_2):
                    accept_interp = True

            if accept_interp:
                e = d
                d = p / q
                step_kind = proposal
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a = b
        fa = fb

        if abs(d) > tol1:
            b = b + d
        else:
            # Keep movement when interpolation shrinks to zero under rounding.
            b = b + math.copysign(tol1, m)
            step_kind = f"{step_kind}+guard"

        fb = float(f(b))
        if not math.isfinite(fb):
            raise RuntimeError("Encountered non-finite function value during iterations")

        history.append(IterationRecord(it, a, b, c, fa, fb, fc, abs(c - b) * 0.5, step_kind))

    return BrentResult(root=b, f_at_root=fb, iterations=max_iter, converged=False, history=history)


def relative_error(estimate: float, reference: float) -> float:
    denom = max(1.0, abs(reference))
    return abs(estimate - reference) / denom


def print_history(history: List[IterationRecord], max_lines: int = 10) -> None:
    header = "iter | step              | b (estimate)         | f(b)                 | |c-b|/2"
    print(header)
    print("-" * len(header))

    total = len(history)
    for idx, rec in enumerate(history):
        if idx < max_lines or idx == total - 1:
            print(
                f"{rec.iteration:>4d} | {rec.step:<17s} | "
                f"{rec.b: .15e} | {rec.fb: .15e} | {rec.half_interval: .15e}"
            )
        elif idx == max_lines:
            hidden = total - max_lines - 1
            if hidden > 0:
                print(f"... ({hidden} lines omitted) ...")


def run_case(
    case_name: str,
    f: Callable[[float], float],
    interval: Tuple[float, float],
    reference: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> None:
    left, right = interval
    result = brent_root(f=f, a=left, b=right, tol=tol, max_iter=max_iter)

    print(f"\n=== {case_name} ===")
    print(f"interval      : [{left}, {right}]")
    print(f"converged     : {result.converged}")
    print(f"iterations    : {result.iterations}")
    print(f"estimate root : {result.root:.16f}")
    print(f"reference     : {reference:.16f}")
    print(f"abs error     : {abs(result.root - reference):.3e}")
    print(f"rel error     : {relative_error(result.root, reference):.3e}")
    print(f"|f(root)|     : {abs(result.f_at_root):.3e}")

    step_counts = Counter(rec.step for rec in result.history)
    print(f"step counts   : {dict(step_counts)}")

    print_history(result.history, max_lines=8)


def main() -> None:
    # Three standard scalar equations with known roots.
    cases = [
        (
            "f(x)=cos(x)-x",
            lambda x: math.cos(x) - x,
            (0.0, 1.0),
            0.7390851332151607,
        ),
        (
            "f(x)=x^3-x-2",
            lambda x: x * x * x - x - 2.0,
            (1.0, 2.0),
            1.5213797068045676,
        ),
        (
            "f(x)=exp(-x)-x",
            lambda x: math.exp(-x) - x,
            (0.0, 1.0),
            0.5671432904097838,
        ),
    ]

    for name, func, interval, ref in cases:
        run_case(name, func, interval, ref, tol=1e-12, max_iter=100)


if __name__ == "__main__":
    main()
