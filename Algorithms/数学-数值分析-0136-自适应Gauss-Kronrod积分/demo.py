"""Minimal runnable MVP for adaptive Gauss-Kronrod integration (MATH-0136)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List

import numpy as np


ScalarFunc = Callable[[float], float]


# 15-point Kronrod rule nodes on [0, 1] side (plus center), sorted descending.
_XGK = np.array(
    [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.2077849550078985,
        0.0,
    ],
    dtype=float,
)

# Corresponding Kronrod weights (same order as _XGK).
_WGK = np.array(
    [
        0.02293532201052922,
        0.06309209262997855,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
    ],
    dtype=float,
)

# 7-point Gauss weights for nodes at indices 1,3,5 and center index 7.
_GAUSS_INDEX_TO_WEIGHT = {
    1: 0.1294849661688697,
    3: 0.2797053914892766,
    5: 0.3818300505051189,
    7: 0.4179591836734694,
}


@dataclass
class IntegrationResult:
    name: str
    value: float
    estimated_error: float
    abs_error_vs_exact: float
    evaluations: int
    processed_intervals: int
    max_depth_reached: int
    converged: bool


def qk15_pair(f: ScalarFunc, a: float, b: float) -> tuple[float, float, float, int]:
    """Compute Gauss-7 and Kronrod-15 estimates on [a,b].

    Returns
    -------
    kronrod, gauss, error_estimate, eval_count
    """
    center = 0.5 * (a + b)
    half = 0.5 * (b - a)

    fc = float(f(center))
    kronrod_sum = _WGK[7] * fc
    gauss_sum = _GAUSS_INDEX_TO_WEIGHT[7] * fc
    eval_count = 1

    for i in range(7):
        dx = half * _XGK[i]
        f_left = float(f(center - dx))
        f_right = float(f(center + dx))
        pair = f_left + f_right

        eval_count += 2
        kronrod_sum += _WGK[i] * pair

        if i in _GAUSS_INDEX_TO_WEIGHT:
            gauss_sum += _GAUSS_INDEX_TO_WEIGHT[i] * pair

    kronrod = kronrod_sum * half
    gauss = gauss_sum * half
    err = abs(kronrod - gauss)

    # Numerical guard: avoid returning an unrealistically tiny zero error.
    guard = 50.0 * np.finfo(float).eps * abs(kronrod)
    if err < guard:
        err = guard

    return float(kronrod), float(gauss), float(err), eval_count


def adaptive_gauss_kronrod(
    f: ScalarFunc,
    a: float,
    b: float,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-10,
    max_depth: int = 30,
    max_intervals: int = 200_000,
) -> tuple[float, float, int, int, int, bool]:
    """Adaptive integration on [a,b] using a Gauss-7/Kronrod-15 pair.

    Returns
    -------
    value, estimated_error, evaluations, processed_intervals, max_depth_reached, converged
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("a and b must be finite numbers")
    if abs_tol <= 0.0:
        raise ValueError("abs_tol must be positive")
    if rel_tol < 0.0:
        raise ValueError("rel_tol must be non-negative")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_intervals < 1:
        raise ValueError("max_intervals must be >= 1")
    if a == b:
        return 0.0, 0.0, 0, 0, 0, True

    sign = 1.0
    left, right = float(a), float(b)
    if right < left:
        left, right = right, left
        sign = -1.0

    root_abs_tol = max(abs_tol, 1e-16)
    stack: List[tuple[float, float, float, int]] = [(left, right, root_abs_tol, 0)]

    total_value = 0.0
    total_error = 0.0
    total_evals = 0
    processed = 0
    max_seen_depth = 0
    hit_limit = False

    while stack:
        ai, bi, local_abs_tol, depth = stack.pop()
        if ai == bi:
            continue

        processed += 1
        if processed > max_intervals:
            raise RuntimeError("Exceeded max_intervals before convergence")

        max_seen_depth = max(max_seen_depth, depth)
        k_val, _g_val, err_est, evals = qk15_pair(f, ai, bi)
        total_evals += evals

        local_tol = local_abs_tol + rel_tol * abs(k_val)

        if err_est <= local_tol or depth >= max_depth:
            total_value += k_val
            total_error += err_est
            if depth >= max_depth and err_est > local_tol:
                hit_limit = True
            continue

        mid = 0.5 * (ai + bi)
        if mid == ai or mid == bi:
            total_value += k_val
            total_error += err_est
            hit_limit = True
            continue

        child_abs_tol = 0.5 * local_abs_tol
        stack.append((mid, bi, child_abs_tol, depth + 1))
        stack.append((ai, mid, child_abs_tol, depth + 1))

    signed_value = sign * total_value
    global_tol = abs_tol + rel_tol * abs(signed_value)
    converged = (not hit_limit) and (total_error <= 10.0 * global_tol)

    return signed_value, total_error, total_evals, processed, max_seen_depth, converged


def run_case(
    name: str,
    f: ScalarFunc,
    a: float,
    b: float,
    exact: float,
    abs_tol: float,
    rel_tol: float,
) -> IntegrationResult:
    value, err_est, evals, intervals, depth, converged = adaptive_gauss_kronrod(
        f=f,
        a=a,
        b=b,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        max_depth=40,
    )
    abs_err = abs(value - exact)
    return IntegrationResult(
        name=name,
        value=value,
        estimated_error=err_est,
        abs_error_vs_exact=abs_err,
        evaluations=evals,
        processed_intervals=intervals,
        max_depth_reached=depth,
        converged=converged,
    )


def main() -> None:
    print("Adaptive Gauss-Kronrod (G7-K15) MVP")
    print("=" * 86)

    cases = [
        {
            "name": "sin(x) on [0, pi]",
            "f": lambda x: math.sin(x),
            "a": 0.0,
            "b": math.pi,
            "exact": 2.0,
            "abs_tol": 1e-12,
            "rel_tol": 1e-12,
            "expect_err_lt": 1e-10,
        },
        {
            "name": "exp(-x^2) on [0, 1]",
            "f": lambda x: math.exp(-(x * x)),
            "a": 0.0,
            "b": 1.0,
            "exact": 0.5 * math.sqrt(math.pi) * math.erf(1.0),
            "abs_tol": 1e-12,
            "rel_tol": 1e-12,
            "expect_err_lt": 1e-10,
        },
        {
            "name": "1/(1+10000*(x-0.05)^2) on [0, 1]",
            "f": lambda x: 1.0 / (1.0 + 10_000.0 * (x - 0.05) * (x - 0.05)),
            "a": 0.0,
            "b": 1.0,
            "exact": (math.atan(100.0 * (1.0 - 0.05)) - math.atan(100.0 * (0.0 - 0.05))) / 100.0,
            "abs_tol": 1e-10,
            "rel_tol": 1e-10,
            "expect_err_lt": 1e-8,
        },
    ]

    reports: List[IntegrationResult] = []

    for cfg in cases:
        report = run_case(
            name=cfg["name"],
            f=cfg["f"],
            a=cfg["a"],
            b=cfg["b"],
            exact=cfg["exact"],
            abs_tol=cfg["abs_tol"],
            rel_tol=cfg["rel_tol"],
        )
        reports.append(report)

        print(report.name)
        print(f"  value                 = {report.value:.15f}")
        print(f"  estimated error       = {report.estimated_error:.3e}")
        print(f"  absolute error        = {report.abs_error_vs_exact:.3e}")
        print(f"  evaluations           = {report.evaluations}")
        print(f"  processed intervals   = {report.processed_intervals}")
        print(f"  max depth reached     = {report.max_depth_reached}")
        print(f"  converged             = {report.converged}")
        print("-" * 86)

    # Deterministic sanity checks.
    for cfg, report in zip(cases, reports):
        assert report.converged, f"{report.name} did not converge"
        assert report.abs_error_vs_exact < cfg["expect_err_lt"], (
            f"{report.name} error too large: {report.abs_error_vs_exact:.3e}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
