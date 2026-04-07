"""Minimal runnable MVP for Newton's method (scalar nonlinear equation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass
class NewtonStep:
    """One iteration snapshot for traceability."""

    iter_id: int
    x: float
    fx: float
    dfx: float
    step: float


@dataclass
class NewtonResult:
    """Final result of Newton iteration."""

    root: float
    converged: bool
    iterations: int
    reason: str
    trace: List[NewtonStep]


@dataclass
class Case:
    """Deterministic test case."""

    name: str
    func: Callable[[float], float]
    deriv: Optional[Callable[[float], float]]
    x0: float
    expected: float


def safe_call(func: Callable[[float], float], x: float, tag: str) -> float:
    """Evaluate function and ensure the value is finite."""
    y = float(func(float(x)))
    if not np.isfinite(y):
        raise RuntimeError(f"{tag} returned non-finite value at x={x}: {y}")
    return y


def finite_diff_derivative(
    func: Callable[[float], float],
    x: float,
    diff_eps: float,
) -> float:
    """Central-difference derivative approximation."""
    h = diff_eps * max(1.0, abs(x))
    fp = safe_call(func, x + h, "f")
    fm = safe_call(func, x - h, "f")
    return (fp - fm) / (2.0 * h)


def newton_solve(
    func: Callable[[float], float],
    x0: float,
    deriv: Optional[Callable[[float], float]] = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 40,
    damping: float = 1.0,
    diff_eps: float = 1e-6,
    min_deriv: float = 1e-14,
) -> NewtonResult:
    """Solve f(x)=0 with Newton iterations and robust stop conditions."""
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if diff_eps <= 0.0:
        raise ValueError("diff_eps must be positive")
    if min_deriv <= 0.0:
        raise ValueError("min_deriv must be positive")

    x = float(x0)
    trace: List[NewtonStep] = []

    for k in range(1, max_iter + 1):
        fx = safe_call(func, x, "f")
        if abs(fx) <= tol:
            return NewtonResult(
                root=x,
                converged=True,
                iterations=k - 1,
                reason="residual_converged",
                trace=trace,
            )

        if deriv is None:
            dfx = finite_diff_derivative(func, x, diff_eps=diff_eps)
        else:
            dfx = safe_call(deriv, x, "df")

        if not np.isfinite(dfx):
            return NewtonResult(
                root=x,
                converged=False,
                iterations=k - 1,
                reason="derivative_non_finite",
                trace=trace,
            )
        if abs(dfx) < min_deriv:
            return NewtonResult(
                root=x,
                converged=False,
                iterations=k - 1,
                reason="derivative_too_small",
                trace=trace,
            )

        step = -damping * fx / dfx
        x_next = x + step
        trace.append(NewtonStep(iter_id=k, x=x, fx=fx, dfx=dfx, step=step))

        if not np.isfinite(x_next):
            return NewtonResult(
                root=x,
                converged=False,
                iterations=k,
                reason="state_non_finite",
                trace=trace,
            )

        x = x_next
        if abs(step) <= tol * max(1.0, abs(x)):
            return NewtonResult(
                root=x,
                converged=True,
                iterations=k,
                reason="step_converged",
                trace=trace,
            )

    return NewtonResult(
        root=x,
        converged=False,
        iterations=max_iter,
        reason="max_iter_reached",
        trace=trace,
    )


def print_trace(trace: List[NewtonStep], max_rows: int = 8) -> None:
    """Print a compact trace table."""
    print("trace (iter, x, f(x), f'(x), step):")
    rows = min(len(trace), max_rows)
    for i in range(rows):
        s = trace[i]
        print(
            f"  {s.iter_id:2d}  "
            f"x={s.x: .16e}  "
            f"f={s.fx: .3e}  "
            f"df={s.dfx: .3e}  "
            f"step={s.step: .3e}"
        )
    if len(trace) > max_rows:
        print(f"  ... ({len(trace) - max_rows} more rows)")


def run_case(case: Case, config: dict) -> None:
    """Run one case and print diagnostics."""
    result = newton_solve(
        func=case.func,
        x0=case.x0,
        deriv=case.deriv,
        tol=config["tol"],
        max_iter=config["max_iter"],
        damping=config["damping"],
        diff_eps=config["diff_eps"],
        min_deriv=config["min_deriv"],
    )

    abs_error = abs(result.root - case.expected)
    print("=" * 96)
    print(case.name)
    print(f"initial_x       = {case.x0:.12g}")
    print(f"converged       = {result.converged}")
    print(f"reason          = {result.reason}")
    print(f"iterations      = {result.iterations}")
    print(f"root            = {result.root:.16e}")
    print(f"expected        = {case.expected:.16e}")
    print(f"abs_error       = {abs_error:.3e}")
    print_trace(result.trace)


def main() -> None:
    config = {
        "tol": 1e-12,
        "max_iter": 40,
        "damping": 1.0,
        "diff_eps": 1e-6,
        "min_deriv": 1e-14,
    }

    cases = [
        Case(
            name="Case 1: sqrt(2) from x^2 - 2 = 0",
            func=lambda x: x * x - 2.0,
            deriv=lambda x: 2.0 * x,
            x0=1.0,
            expected=math.sqrt(2.0),
        ),
        Case(
            name="Case 2: fixed-point root of cos(x) - x = 0",
            func=lambda x: math.cos(x) - x,
            deriv=lambda x: -math.sin(x) - 1.0,
            x0=0.5,
            expected=0.7390851332151607,
        ),
        Case(
            name="Case 3: cubic root from x^3 - 7 = 0",
            func=lambda x: x**3 - 7.0,
            deriv=lambda x: 3.0 * x * x,
            x0=2.0,
            expected=7.0 ** (1.0 / 3.0),
        ),
        Case(
            name="Case 4: multiple root (x - 1)^2 = 0",
            func=lambda x: (x - 1.0) ** 2,
            deriv=lambda x: 2.0 * (x - 1.0),
            x0=2.5,
            expected=1.0,
        ),
        Case(
            name="Case 5: exp(x) - 3 = 0 (finite-difference derivative)",
            func=lambda x: math.exp(x) - 3.0,
            deriv=None,
            x0=0.0,
            expected=math.log(3.0),
        ),
    ]

    print("Newton Method MVP (scalar root finding)")
    print(
        "Config: "
        f"tol={config['tol']:.1e}, max_iter={config['max_iter']}, "
        f"damping={config['damping']:.2f}, diff_eps={config['diff_eps']:.1e}"
    )

    for case in cases:
        run_case(case, config)


if __name__ == "__main__":
    main()
