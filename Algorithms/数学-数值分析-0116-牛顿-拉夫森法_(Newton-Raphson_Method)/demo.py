"""Newton-Raphson Method: minimal runnable MVP.

This script solves several scalar nonlinear equations with Newton-Raphson,
prints iteration traces, and reports numerical errors against references.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float]


def check_finite(value: float, name: str) -> float:
    """Validate a scalar is finite and return it as float."""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 30,
    derivative_floor: float = 1e-14,
) -> Tuple[float, List[HistoryItem]]:
    """Solve f(x)=0 with Newton-Raphson.

    Returns:
        root_estimate, history
    history item format:
        (iteration, x_k, f(x_k), |delta_k|)
    """
    x = check_finite(x0, "x0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if derivative_floor <= 0:
        raise ValueError("derivative_floor must be > 0")

    history: List[HistoryItem] = []

    for k in range(1, max_iter + 1):
        fx = float(f(x))
        dfx = float(df(x))
        if not np.isfinite(fx) or not np.isfinite(dfx):
            raise RuntimeError(f"non-finite function value at iter {k}: f={fx}, df={dfx}")
        if abs(dfx) < derivative_floor:
            raise RuntimeError(
                f"derivative too small at iter {k}: |df|={abs(dfx):.3e}, x={x:.16g}"
            )

        delta = fx / dfx
        x_next = x - delta
        if not np.isfinite(x_next):
            raise RuntimeError(f"iteration diverged to non-finite value at iter {k}")

        fx_next = float(f(x_next))
        if not np.isfinite(fx_next):
            raise RuntimeError(f"non-finite residual after update at iter {k}")

        history.append((k, x_next, fx_next, abs(delta)))
        x = x_next

        if abs(fx_next) <= tol:
            break
        if abs(delta) <= tol * (1.0 + abs(x)):
            break

    return x, history


def print_history(history: List[HistoryItem], max_lines: int = 8) -> None:
    """Print first max_lines rows of iteration history."""
    print("  iter | x_k               | f(x_k)            | |delta|")
    print("  -----+-------------------+-------------------+-------------------")
    for it, x_k, fx_k, delta in history[:max_lines]:
        print(f"  {it:>4d} | {x_k:>17.10f} | {fx_k:>17.10e} | {delta:>17.10e}")
    if len(history) > max_lines:
        print(f"  ... ({len(history) - max_lines} more iterations omitted)")


def relative_error(estimate: float, reference: float) -> float:
    denom = max(1.0, abs(reference))
    return abs(estimate - reference) / denom


def run_case(
    name: str,
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    reference: float,
    tol: float = 1e-12,
    max_iter: int = 30,
) -> float:
    """Run one Newton-Raphson case and print report; returns relative error."""
    print(f"\nCase: {name}")
    print(f"  x0={x0}, tol={tol}, max_iter={max_iter}")

    root, history = newton_raphson(f=f, df=df, x0=x0, tol=tol, max_iter=max_iter)
    print_history(history=history, max_lines=8)

    abs_err = abs(root - reference)
    rel_err = relative_error(root, reference)
    final_residual = abs(float(f(root)))

    print(f"  root_estimate = {root:.16f}")
    print(f"  reference     = {reference:.16f}")
    print(f"  abs_error     = {abs_err:.3e}")
    print(f"  rel_error     = {rel_err:.3e}")
    print(f"  final_residual= {final_residual:.3e}")
    print(f"  iterations    = {len(history)}")
    return rel_err


def main() -> None:
    cases = [
        {
            "name": "x^2 - 2 = 0",
            "f": lambda x: x * x - 2.0,
            "df": lambda x: 2.0 * x,
            "x0": 1.5,
            "reference": math.sqrt(2.0),
        },
        {
            "name": "cos(x) - x = 0",
            "f": lambda x: math.cos(x) - x,
            "df": lambda x: -math.sin(x) - 1.0,
            "x0": 0.5,
            "reference": 0.7390851332151607,
        },
        {
            "name": "x^3 - x - 2 = 0",
            "f": lambda x: x * x * x - x - 2.0,
            "df": lambda x: 3.0 * x * x - 1.0,
            "x0": 1.5,
            "reference": 1.5213797068045676,
        },
    ]

    print("Newton-Raphson Method MVP")
    print("=" * 80)

    rel_errors = []
    for case in cases:
        rel_err = run_case(
            name=case["name"],
            f=case["f"],
            df=case["df"],
            x0=case["x0"],
            reference=case["reference"],
            tol=1e-12,
            max_iter=30,
        )
        rel_errors.append(rel_err)

    rel_errors_arr = np.asarray(rel_errors, dtype=float)
    print("\nSummary")
    print("=" * 80)
    print(f"cases={len(cases)}")
    print(f"max_relative_error={rel_errors_arr.max():.3e}")
    print(f"mean_relative_error={rel_errors_arr.mean():.3e}")
    print(f"all_pass={bool(np.all(rel_errors_arr < 1e-10))}")


if __name__ == "__main__":
    main()
