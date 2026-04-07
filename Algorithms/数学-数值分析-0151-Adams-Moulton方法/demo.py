"""Minimal runnable MVP: Adams-Moulton (trapezoidal) method for ODE IVP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

Vector = np.ndarray
RhsFunc = Callable[[float, float], float]


@dataclass
class StepDiagnostic:
    """Per-step diagnostics for implicit correction."""

    step: int
    t_n: float
    y_n: float
    y_predict: float
    y_corrected: float
    correction_iters: int
    implicit_residual: float


def check_inputs(h: float, steps: int, tol: float, max_iter: int) -> None:
    """Validate scalar solver controls."""
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive finite, got {h!r}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be positive finite, got {tol!r}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter!r}")


def adams_moulton_trapezoid(
    f: RhsFunc,
    t0: float,
    y0: float,
    h: float,
    steps: int,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> Tuple[Vector, Vector, List[StepDiagnostic]]:
    """Solve y' = f(t, y) with one-step Adams-Moulton (trapezoidal) scheme.

    Formula:
        y_{n+1} = y_n + h/2 * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

    We solve the implicit equation at each step by fixed-point iteration:
        z_{k+1} = y_n + h/2 * [f_n + f(t_{n+1}, z_k)]
    """
    check_inputs(h=h, steps=steps, tol=tol, max_iter=max_iter)
    if not math.isfinite(t0) or not math.isfinite(y0):
        raise ValueError("t0 and y0 must be finite")

    t = np.empty(steps + 1, dtype=float)
    y = np.empty(steps + 1, dtype=float)
    diagnostics: List[StepDiagnostic] = []

    t[0] = t0
    y[0] = y0

    for n in range(steps):
        t_n = t[n]
        y_n = y[n]
        f_n = f(t_n, y_n)
        t_next = t_n + h

        # Explicit Euler predictor.
        y_pred = y_n + h * f_n
        z = y_pred
        converged = False
        iters = 0

        for k in range(1, max_iter + 1):
            z_prev = z
            z = y_n + 0.5 * h * (f_n + f(t_next, z_prev))
            iters = k

            if not math.isfinite(z):
                raise RuntimeError(
                    f"non-finite iterate at step={n}, correction_iter={k}"
                )

            if abs(z - z_prev) <= tol * max(1.0, abs(z)):
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"fixed-point correction did not converge at step={n} within "
                f"{max_iter} iterations"
            )

        residual = abs(z - y_n - 0.5 * h * (f_n + f(t_next, z)))
        t[n + 1] = t_next
        y[n + 1] = z
        diagnostics.append(
            StepDiagnostic(
                step=n,
                t_n=t_n,
                y_n=y_n,
                y_predict=y_pred,
                y_corrected=z,
                correction_iters=iters,
                implicit_residual=residual,
            )
        )

    return t, y, diagnostics


def explicit_euler(
    f: RhsFunc, t0: float, y0: float, h: float, steps: int
) -> Tuple[Vector, Vector]:
    """Baseline method for accuracy comparison."""
    t = np.empty(steps + 1, dtype=float)
    y = np.empty(steps + 1, dtype=float)
    t[0], y[0] = t0, y0
    for n in range(steps):
        t[n + 1] = t[n] + h
        y[n + 1] = y[n] + h * f(t[n], y[n])
    return t, y


def rhs_demo(t: float, y: float) -> float:
    """Test equation: y' = y - t^2 + 1."""
    return y - t * t + 1.0


def exact_solution_demo(t: float) -> float:
    """Exact solution for rhs_demo with y(0)=0.5."""
    return (t + 1.0) ** 2 - 0.5 * math.exp(t)


def safe_order(errors: Sequence[float]) -> List[float]:
    """Estimate observed order p from successive halving of h."""
    rates: List[float] = []
    for i in range(1, len(errors)):
        a = errors[i - 1]
        b = errors[i]
        if a <= 0.0 or b <= 0.0:
            rates.append(float("nan"))
        else:
            rates.append(math.log(a / b, 2.0))
    return rates


def print_diagnostics(diags: List[StepDiagnostic], max_rows: int = 5) -> None:
    """Print a compact prefix of per-step diagnostics."""
    print("  step diagnostics (prefix):")
    for item in diags[:max_rows]:
        print(
            "    step={:2d} t={:.2f} y_pred={:.8f} y_corr={:.8f} "
            "iters={} residual={:.2e}".format(
                item.step,
                item.t_n,
                item.y_predict,
                item.y_corrected,
                item.correction_iters,
                item.implicit_residual,
            )
        )
    if len(diags) > max_rows:
        print(f"    ... ({len(diags) - max_rows} more steps)")


def run_demo() -> None:
    """Run convergence and diagnostic demonstration."""
    t0 = 0.0
    y0 = 0.5
    t_end = 2.0
    hs = [0.2, 0.1, 0.05]
    tol = 1e-12
    max_iter = 20

    print("=" * 88)
    print("Adams-Moulton (trapezoidal) demo for y' = y - t^2 + 1, y(0) = 0.5")
    print("=" * 88)

    am_errors: List[float] = []
    eu_errors: List[float] = []

    for idx, h in enumerate(hs):
        steps = int(round((t_end - t0) / h))

        t_am, y_am, diags = adams_moulton_trapezoid(
            f=rhs_demo,
            t0=t0,
            y0=y0,
            h=h,
            steps=steps,
            tol=tol,
            max_iter=max_iter,
        )
        _, y_eu = explicit_euler(rhs_demo, t0=t0, y0=y0, h=h, steps=steps)

        y_true = exact_solution_demo(t_am[-1])
        am_err = abs(y_am[-1] - y_true)
        eu_err = abs(y_eu[-1] - y_true)
        am_errors.append(am_err)
        eu_errors.append(eu_err)

        mean_iters = float(np.mean([d.correction_iters for d in diags]))
        max_residual = float(np.max([d.implicit_residual for d in diags]))

        print(f"h={h:.3f}, steps={steps}")
        print(f"  y_AM(T)      = {y_am[-1]:.12f}")
        print(f"  y_Euler(T)   = {y_eu[-1]:.12f}")
        print(f"  y_exact(T)   = {y_true:.12f}")
        print(f"  |AM-exact|   = {am_err:.3e}")
        print(f"  |Euler-exact|= {eu_err:.3e}")
        print(f"  mean corr iters = {mean_iters:.2f}")
        print(f"  max implicit residual = {max_residual:.3e}")
        if idx == 0:
            print_diagnostics(diags, max_rows=6)

    am_orders = safe_order(am_errors)
    eu_orders = safe_order(eu_errors)
    print("-" * 88)
    print("Observed order when halving h:")
    for i in range(1, len(hs)):
        print(
            f"  h: {hs[i-1]:.3f} -> {hs[i]:.3f}, "
            f"AM p≈{am_orders[i-1]:.3f}, Euler p≈{eu_orders[i-1]:.3f}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
