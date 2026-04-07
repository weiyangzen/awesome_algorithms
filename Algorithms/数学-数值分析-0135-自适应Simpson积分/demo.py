"""Minimal runnable MVP for adaptive Simpson integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class AdaptiveSimpsonResult:
    """Container for adaptive Simpson outputs."""

    estimate: float
    abs_error_est: float
    eval_count: int
    accepted_intervals: int
    max_depth_reached: int


def safe_eval(func: Callable[[float], float], x: float) -> float:
    """Evaluate function and ensure finite output."""
    y = float(func(x))
    if not math.isfinite(y):
        raise RuntimeError(f"non-finite function value f({x})={y}")
    return y


def simpson_from_values(a: float, b: float, fa: float, fm: float, fb: float) -> float:
    """Compute one Simpson estimate on [a, b] from endpoint/midpoint values."""
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0


def adaptive_simpson_integral(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 20,
) -> AdaptiveSimpsonResult:
    """Integrate on [a, b] with recursive adaptive Simpson strategy."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f"integration bounds must be finite, got a={a}, b={b}")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if a == b:
        return AdaptiveSimpsonResult(
            estimate=0.0,
            abs_error_est=0.0,
            eval_count=0,
            accepted_intervals=0,
            max_depth_reached=0,
        )

    sign = 1.0
    left, right = a, b
    if b < a:
        sign = -1.0
        left, right = b, a

    fa = safe_eval(func, left)
    fb = safe_eval(func, right)
    mid = 0.5 * (left + right)
    fm = safe_eval(func, mid)

    eval_count = 3
    accepted_intervals = 0
    max_depth_reached = 0

    initial = simpson_from_values(left, right, fa, fm, fb)

    def recurse(
        l: float,
        r: float,
        fl: float,
        fm_local: float,
        fr: float,
        whole: float,
        tol_local: float,
        depth: int,
    ) -> tuple[float, float]:
        nonlocal eval_count, accepted_intervals, max_depth_reached

        m = 0.5 * (l + r)
        lm = 0.5 * (l + m)
        rm = 0.5 * (m + r)

        flm = safe_eval(func, lm)
        frm = safe_eval(func, rm)
        eval_count += 2

        left_est = simpson_from_values(l, m, fl, flm, fm_local)
        right_est = simpson_from_values(m, r, fm_local, frm, fr)

        delta = left_est + right_est - whole
        err_est = abs(delta) / 15.0

        max_depth_reached = max(max_depth_reached, depth)
        if depth >= max_depth or abs(delta) <= 15.0 * tol_local:
            accepted_intervals += 1
            corrected = left_est + right_est + delta / 15.0
            return corrected, err_est

        val_l, err_l = recurse(
            l, m, fl, flm, fm_local, left_est, tol_local * 0.5, depth + 1
        )
        val_r, err_r = recurse(
            m, r, fm_local, frm, fr, right_est, tol_local * 0.5, depth + 1
        )
        return val_l + val_r, err_l + err_r

    estimate, err = recurse(left, right, fa, fm, fb, initial, tol, 0)
    return AdaptiveSimpsonResult(
        estimate=sign * estimate,
        abs_error_est=err,
        eval_count=eval_count,
        accepted_intervals=accepted_intervals,
        max_depth_reached=max_depth_reached,
    )


def relative_error(estimate: float, exact: float) -> float:
    """Relative error with safe zero handling."""
    if exact == 0.0:
        return abs(estimate)
    return abs(estimate - exact) / abs(exact)


def run_case(
    name: str,
    func: Callable[[float], float],
    a: float,
    b: float,
    exact: float,
    tol: float,
    max_depth: int,
) -> None:
    """Run one deterministic integration case and print diagnostics."""
    result = adaptive_simpson_integral(func, a, b, tol=tol, max_depth=max_depth)
    abs_err = abs(result.estimate - exact)
    rel_err = relative_error(result.estimate, exact)

    print("=" * 92)
    print(name)
    print(f"interval          = [{a:.12g}, {b:.12g}]")
    print(f"estimate          = {result.estimate:.16e}")
    print(f"exact             = {exact:.16e}")
    print(f"abs_error         = {abs_err:.3e}")
    print(f"rel_error         = {rel_err:.3e}")
    print(f"error_estimate    = {result.abs_error_est:.3e}")
    print(f"eval_count        = {result.eval_count}")
    print(f"accepted_segments = {result.accepted_intervals}")
    print(f"max_depth_reached = {result.max_depth_reached}")


def main() -> None:
    tol = 1e-10
    max_depth = 20

    cases = [
        ("Case 1: integral of sin(x)", math.sin, 0.0, math.pi, 2.0),
        ("Case 2: integral of exp(x)", math.exp, 0.0, 1.0, math.e - 1.0),
        (
            "Case 3: integral of 1/(1+x^2)",
            lambda x: 1.0 / (1.0 + x * x),
            0.0,
            1.0,
            math.pi / 4.0,
        ),
        ("Case 4: integral of sqrt(x)", math.sqrt, 0.0, 1.0, 2.0 / 3.0),
        (
            "Case 5: integral of cos(20x)",
            lambda x: math.cos(20.0 * x),
            0.0,
            1.0,
            math.sin(20.0) / 20.0,
        ),
        ("Case 6: reversed bounds for sin(x)", math.sin, math.pi, 0.0, -2.0),
    ]

    for case in cases:
        run_case(*case, tol=tol, max_depth=max_depth)


if __name__ == "__main__":
    main()
