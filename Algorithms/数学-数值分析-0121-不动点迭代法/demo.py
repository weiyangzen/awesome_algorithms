"""Minimal runnable MVP for fixed-point iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class IterRecord:
    """One iteration snapshot for audit and debugging."""

    k: int
    x_k: float
    x_next: float
    step: float
    residual: float


def ensure_finite_scalar(name: str, value: float) -> float:
    """Validate a scalar can be safely used in numerical iteration."""
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def fixed_point_iteration(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> tuple[float, list[IterRecord], bool]:
    """Run x_{k+1} = g(x_k) until convergence or max_iter."""
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    x = ensure_finite_scalar("x0", x0)
    history: list[IterRecord] = []
    converged = False

    for k in range(1, max_iter + 1):
        x_next = ensure_finite_scalar("x_next", g(x))
        step = abs(x_next - x)

        # Evaluate fixed-point residual at the new point.
        gx_next = ensure_finite_scalar("g(x_next)", g(x_next))
        residual = abs(gx_next - x_next)

        history.append(IterRecord(k=k, x_k=x, x_next=x_next, step=step, residual=residual))
        x = x_next

        if step <= tol and residual <= tol:
            converged = True
            break

    return x, history, converged


def relative_error(estimate: float, reference: float) -> float:
    """Safe relative error with fallback when reference is near zero."""
    denom = max(1.0, abs(reference))
    return abs(estimate - reference) / denom


def estimate_local_slope(g: Callable[[float], float], x_star: float, h: float = 1e-6) -> float:
    """Estimate |g'(x*)| using central difference for diagnostics."""
    xp = ensure_finite_scalar("x_star+h", x_star + h)
    xm = ensure_finite_scalar("x_star-h", x_star - h)
    gp = ensure_finite_scalar("g(x_star+h)", g(xp))
    gm = ensure_finite_scalar("g(x_star-h)", g(xm))
    slope = (gp - gm) / (2.0 * h)
    return abs(slope)


def print_history(history: list[IterRecord], limit: int = 8) -> None:
    """Print a compact iteration trace."""
    print("k | x_k                | x_next             | step               | residual")
    print("--+--------------------+--------------------+--------------------+--------------------")

    if len(history) <= limit:
        rows = history
        for row in rows:
            print(
                f"{row.k:2d} | {row.x_k: .12e} | {row.x_next: .12e} |"
                f" {row.step: .12e} | {row.residual: .12e}"
            )
        return

    head = max(1, limit // 2)
    tail = max(1, limit - head)
    for row in history[:head]:
        print(
            f"{row.k:2d} | {row.x_k: .12e} | {row.x_next: .12e} |"
            f" {row.step: .12e} | {row.residual: .12e}"
        )
    print(".. | ...                | ...                | ...                | ...")
    for row in history[-tail:]:
        print(
            f"{row.k:2d} | {row.x_k: .12e} | {row.x_next: .12e} |"
            f" {row.step: .12e} | {row.residual: .12e}"
        )


def run_case(
    name: str,
    g: Callable[[float], float],
    x0: float,
    reference: Optional[float],
    expect_converged: bool,
    abs_tol: float = 1e-10,
    max_iter: int = 80,
) -> None:
    """Execute one fixed-point case and assert expected behavior."""
    print(f"\n=== {name} ===")
    estimate, history, converged = fixed_point_iteration(g=g, x0=x0, tol=1e-12, max_iter=max_iter)

    print(f"converged: {converged}")
    print(f"iterations: {len(history)}")
    print(f"estimate: {estimate:.16f}")

    if reference is not None:
        abs_error = abs(estimate - reference)
        rel_err = relative_error(estimate, reference)
        print(f"reference: {reference:.16f}")
        print(f"abs_error: {abs_error:.3e}")
        print(f"rel_error: {rel_err:.3e}")

    if converged:
        slope_est = estimate_local_slope(g, estimate)
        print(f"|g'(x*)| estimate: {slope_est:.6f}")
    else:
        print("|g'(x*)| estimate: skipped (case not converged)")

    print_history(history, limit=8)

    if expect_converged and not converged:
        raise AssertionError(f"{name}: expected convergence but did not converge")
    if (not expect_converged) and converged:
        raise AssertionError(f"{name}: expected non-convergence but converged")

    if expect_converged and reference is not None:
        abs_error = abs(estimate - reference)
        if abs_error > abs_tol:
            raise AssertionError(
                f"{name}: abs_error {abs_error:.3e} exceeded tolerance {abs_tol:.3e}"
            )


def main() -> None:
    """Run a few deterministic cases without interactive input."""
    cases = [
        {
            "name": "x = cos(x)",
            "g": lambda x: float(np.cos(x)),
            "x0": 1.0,
            "reference": 0.7390851332151607,
            "expect_converged": True,
            "abs_tol": 1e-10,
            "max_iter": 80,
        },
        {
            "name": "x = exp(-x)",
            "g": lambda x: float(np.exp(-x)),
            "x0": 0.0,
            "reference": 0.5671432904097838,
            "expect_converged": True,
            "abs_tol": 1e-10,
            "max_iter": 80,
        },
        {
            "name": "x = 0.5 * (x + 2/x)",
            "g": lambda x: 0.5 * (x + 2.0 / x),
            "x0": 1.5,
            "reference": float(np.sqrt(2.0)),
            "expect_converged": True,
            "abs_tol": 1e-12,
            "max_iter": 80,
        },
        {
            "name": "x = 1.5x + 0.2 (non-contractive)",
            "g": lambda x: 1.5 * x + 0.2,
            "x0": 0.0,
            "reference": None,
            "expect_converged": False,
            "abs_tol": 0.0,
            "max_iter": 25,
        },
    ]

    for case in cases:
        run_case(
            name=case["name"],
            g=case["g"],
            x0=case["x0"],
            reference=case["reference"],
            expect_converged=case["expect_converged"],
            abs_tol=case["abs_tol"],
            max_iter=case["max_iter"],
        )

    print("\nAll fixed-point MVP checks passed.")


if __name__ == "__main__":
    main()
