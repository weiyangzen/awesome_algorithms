"""Minimal runnable MVP for Steffensen iteration (derivative-free root finding)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class IterationRecord:
    """One iteration snapshot for debugging and convergence inspection."""

    k: int
    x: float
    fx: float
    denom: float
    step: float


@dataclass
class SteffensenResult:
    """Final outcome and trace of a Steffensen run."""

    root: float
    converged: bool
    iterations: int
    final_residual: float
    stop_reason: str
    history: List[IterationRecord]


@dataclass(frozen=True)
class DemoCase:
    """Static configuration for one demo equation."""

    name: str
    func: Callable[[float], float]
    x0: float
    bracket: Tuple[float, float]


def check_finite(value: float, name: str) -> None:
    """Validate that a floating-point value is finite."""
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got {value!r}")


def steffensen(
    f: Callable[[float], float],
    x0: float,
    max_iter: int = 30,
    tol: float = 1e-12,
    denom_tol: float = 1e-14,
) -> SteffensenResult:
    """Solve f(x)=0 using Steffensen iteration.

    Update formula:
        x_{k+1} = x_k - f(x_k)^2 / (f(x_k + f(x_k)) - f(x_k))
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if denom_tol <= 0.0:
        raise ValueError("denom_tol must be positive")

    x = float(x0)
    check_finite(x, "x0")

    history: List[IterationRecord] = []

    for k in range(max_iter):
        fx = f(x)
        check_finite(fx, "f(x)")

        if abs(fx) <= tol:
            history.append(IterationRecord(k=k, x=x, fx=fx, denom=float("nan"), step=0.0))
            return SteffensenResult(
                root=x,
                converged=True,
                iterations=len(history),
                final_residual=abs(fx),
                stop_reason="residual tolerance reached",
                history=history,
            )

        x_shift = x + fx
        check_finite(x_shift, "x + f(x)")
        fx_shift = f(x_shift)
        check_finite(fx_shift, "f(x + f(x))")

        denom = fx_shift - fx
        scale = max(1.0, abs(fx_shift), abs(fx))
        if abs(denom) <= denom_tol * scale:
            history.append(IterationRecord(k=k, x=x, fx=fx, denom=denom, step=0.0))
            return SteffensenResult(
                root=x,
                converged=False,
                iterations=len(history),
                final_residual=abs(fx),
                stop_reason="degenerate denominator",
                history=history,
            )

        step = (fx * fx) / denom
        x_next = x - step
        check_finite(x_next, "x_next")

        history.append(IterationRecord(k=k, x=x, fx=fx, denom=denom, step=step))

        step_small = abs(x_next - x) <= tol * max(1.0, abs(x_next))
        x = x_next

        if step_small:
            fx_new = f(x)
            check_finite(fx_new, "f(x_next)")
            history.append(IterationRecord(k=k + 1, x=x, fx=fx_new, denom=float("nan"), step=0.0))
            converged = abs(fx_new) <= 10.0 * tol
            return SteffensenResult(
                root=x,
                converged=converged,
                iterations=len(history),
                final_residual=abs(fx_new),
                stop_reason=(
                    "step tolerance reached and residual verified"
                    if converged
                    else "step tolerance reached but residual not small enough"
                ),
                history=history,
            )

    final_fx = f(x)
    check_finite(final_fx, "final f(x)")
    return SteffensenResult(
        root=x,
        converged=abs(final_fx) <= tol,
        iterations=len(history),
        final_residual=abs(final_fx),
        stop_reason="max_iter reached",
        history=history,
    )


def bisection_reference(
    f: Callable[[float], float],
    left: float,
    right: float,
    max_iter: int = 200,
    tol: float = 1e-14,
) -> float:
    """Compute a high-precision reference root on a sign-change interval."""
    if left >= right:
        raise ValueError("left must be smaller than right")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    fl = f(left)
    fr = f(right)
    check_finite(fl, "f(left)")
    check_finite(fr, "f(right)")

    if fl == 0.0:
        return left
    if fr == 0.0:
        return right
    if fl * fr > 0.0:
        raise ValueError("f(left) and f(right) must have opposite signs")

    lo, hi = left, right

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        check_finite(fm, "f(mid)")

        if abs(fm) <= tol or 0.5 * (hi - lo) <= tol * max(1.0, abs(mid)):
            return mid

        if fl * fm < 0.0:
            hi = mid
            fr = fm
        else:
            lo = mid
            fl = fm

    return 0.5 * (lo + hi)


def relative_error(estimate: float, reference: float) -> float:
    """Compute relative error with zero-safe branch."""
    if reference == 0.0:
        return abs(estimate)
    return abs(estimate - reference) / abs(reference)


def print_history(history: List[IterationRecord], max_lines: int = 6) -> None:
    """Print the first several iteration records."""
    for rec in history[:max_lines]:
        denom_str = "n/a" if not math.isfinite(rec.denom) else f"{rec.denom:.3e}"
        print(
            f"    k={rec.k:2d}  x={rec.x:.16e}  "
            f"|f(x)|={abs(rec.fx):.3e}  denom={denom_str:>10}  step={rec.step:.3e}"
        )

    if len(history) > max_lines:
        print(f"    ... ({len(history) - max_lines} more records)")


def run_case(case: DemoCase, max_iter: int, tol: float, denom_tol: float) -> None:
    """Run one configured equation and print summary statistics."""
    result = steffensen(case.func, case.x0, max_iter=max_iter, tol=tol, denom_tol=denom_tol)
    reference = bisection_reference(case.func, case.bracket[0], case.bracket[1])
    err = relative_error(result.root, reference)

    print("=" * 88)
    print(f"Case: {case.name}")
    print(f"x0={case.x0:.10g}, bracket={case.bracket}")
    print(f"estimate       = {result.root:.16e}")
    print(f"reference      = {reference:.16e}")
    print(f"residual       = {result.final_residual:.3e}")
    print(f"relative_error = {err:.3e}")
    print(f"iterations     = {result.iterations}")
    print(f"converged      = {result.converged}")
    print(f"stop_reason    = {result.stop_reason}")
    print("trace:")
    print_history(result.history)


def f_cos_minus_x(x: float) -> float:
    return math.cos(x) - x


def f_cubic(x: float) -> float:
    return x * x * x - x - 2.0


def f_exp_minus_x(x: float) -> float:
    return math.exp(-x) - x


def main() -> None:
    max_iter = 30
    tol = 1e-12
    denom_tol = 1e-14

    cases = [
        DemoCase("cos(x) - x = 0", f_cos_minus_x, x0=1.0, bracket=(0.0, 1.0)),
        DemoCase("x^3 - x - 2 = 0", f_cubic, x0=1.5, bracket=(1.0, 2.0)),
        DemoCase("exp(-x) - x = 0", f_exp_minus_x, x0=0.5, bracket=(0.0, 1.0)),
    ]

    print("Steffensen Iteration Demo (Derivative-Free Root Finding)")
    print(f"max_iter={max_iter}, tol={tol}, denom_tol={denom_tol}")

    for case in cases:
        run_case(case, max_iter=max_iter, tol=tol, denom_tol=denom_tol)


if __name__ == "__main__":
    main()
