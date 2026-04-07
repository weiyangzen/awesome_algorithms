"""Minimal runnable MVP for False Position (Regula Falsi)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

try:
    import numpy as np  # Optional: used only for light statistics in report.
except Exception:  # pragma: no cover - demo fallback path
    np = None


@dataclass
class IterationRecord:
    iteration: int
    a: float
    b: float
    x: float
    fx: float
    interval_width: float


@dataclass
class RootResult:
    method: str
    root: float
    f_root: float
    iterations: int
    converged: bool
    history: List[IterationRecord]


def false_position(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> RootResult:
    """Classic Regula Falsi root finding on a sign-changing interval."""
    if a >= b:
        raise ValueError("Expected a < b.")

    fa = f(a)
    fb = f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("Endpoint function values must be finite.")

    history: List[IterationRecord] = []

    if fa == 0.0:
        return RootResult("false_position", a, 0.0, 0, True, history)
    if fb == 0.0:
        return RootResult("false_position", b, 0.0, 0, True, history)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    x = a
    fx = fa
    for it in range(1, max_iter + 1):
        denom = fb - fa
        if denom == 0.0:
            break

        x = (a * fb - b * fa) / denom
        fx = f(x)
        width = b - a
        history.append(IterationRecord(it, a, b, x, fx, width))

        if abs(fx) <= tol:
            return RootResult("false_position", x, fx, it, True, history)
        if width <= tol * max(1.0, abs(x)):
            return RootResult("false_position", x, fx, it, True, history)

        if fa * fx < 0:
            b, fb = x, fx
        elif fb * fx < 0:
            a, fa = x, fx
        else:
            # Very close to root or numerical tie.
            return RootResult("false_position", x, fx, it, True, history)

    return RootResult("false_position", x, fx, len(history), False, history)


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> RootResult:
    """Reference baseline method for cross-checking."""
    if a >= b:
        raise ValueError("Expected a < b.")

    fa = f(a)
    fb = f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("Endpoint function values must be finite.")

    history: List[IterationRecord] = []

    if fa == 0.0:
        return RootResult("bisection", a, 0.0, 0, True, history)
    if fb == 0.0:
        return RootResult("bisection", b, 0.0, 0, True, history)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    x = a
    fx = fa
    for it in range(1, max_iter + 1):
        x = 0.5 * (a + b)
        fx = f(x)
        width = b - a
        history.append(IterationRecord(it, a, b, x, fx, width))

        if abs(fx) <= tol:
            return RootResult("bisection", x, fx, it, True, history)
        if width <= tol * max(1.0, abs(x)):
            return RootResult("bisection", x, fx, it, True, history)

        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

    return RootResult("bisection", x, fx, len(history), False, history)


def mean_abs_residual(records: Sequence[IterationRecord]) -> float:
    values = [abs(r.fx) for r in records]
    if not values:
        return 0.0
    if np is not None:
        return float(np.mean(np.asarray(values, dtype=float)))
    return sum(values) / len(values)


def run_case(
    name: str,
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> None:
    fp = false_position(f, a, b, tol=tol, max_iter=max_iter)
    bi = bisection(f, a, b, tol=tol, max_iter=max_iter)

    assert fp.converged, f"False position did not converge for case: {name}"
    assert bi.converged, f"Bisection did not converge for case: {name}"
    assert a <= fp.root <= b, f"False position root out of bracket for case: {name}"
    assert abs(fp.root - bi.root) < 1e-7, f"Methods disagree too much for case: {name}"

    print(f"[Case] {name}")
    print(
        "  false_position: "
        f"root={fp.root:.12f}, |f(root)|={abs(fp.f_root):.3e}, iters={fp.iterations}, "
        f"mean|f|={mean_abs_residual(fp.history):.3e}"
    )
    print(
        "  bisection     : "
        f"root={bi.root:.12f}, |f(root)|={abs(bi.f_root):.3e}, iters={bi.iterations}, "
        f"mean|f|={mean_abs_residual(bi.history):.3e}"
    )
    print(f"  agreement     : |delta_root|={abs(fp.root - bi.root):.3e}")
    print()


def main() -> None:
    cases = [
        ("x^3 - x - 2 on [1,2]", lambda x: x**3 - x - 2.0, 1.0, 2.0),
        ("cos(x) - x on [0,1]", lambda x: math.cos(x) - x, 0.0, 1.0),
        ("exp(-x) - x on [0,1]", lambda x: math.exp(-x) - x, 0.0, 1.0),
    ]

    print("False Position (Regula Falsi) MVP demo")
    print("tolerance=1e-12, max_iter=200")
    print("=" * 60)

    for name, f, a, b in cases:
        run_case(name, f, a, b, tol=1e-12, max_iter=200)

    print("All checks passed.")


if __name__ == "__main__":
    main()
