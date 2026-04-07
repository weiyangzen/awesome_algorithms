"""Minimal runnable MVP for Backward Differentiation Formulas (BDF1/BDF2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

RhsFunc = Callable[[float, float], float]
JacFunc = Callable[[float, float], float]


@dataclass
class NewtonDiagnostic:
    """Per-step Newton solve diagnostics for implicit BDF updates."""

    method: str
    step: int
    t_next: float
    initial_guess: float
    value: float
    iterations: int
    residual: float


def ensure_finite(name: str, value: float) -> None:
    """Require a scalar to be finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def check_solver_controls(h: float, steps: int, tol: float, max_iter: int) -> None:
    """Validate common controls for fixed-step implicit integration."""
    ensure_finite("h", h)
    ensure_finite("tol", tol)
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")


def require_integer_steps(t0: float, t_end: float, h: float) -> int:
    """Compute number of steps and require an aligned fixed grid."""
    ensure_finite("t0", t0)
    ensure_finite("t_end", t_end)
    ensure_finite("h", h)
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")

    span = t_end - t0
    if span <= 0.0:
        raise ValueError(f"t_end must be greater than t0, got t0={t0}, t_end={t_end}")

    steps_float = span / h
    steps = int(round(steps_float))
    if abs(steps - steps_float) > 1e-12:
        raise ValueError("(t_end - t0) / h must be close to an integer")
    return steps


def stiff_tracking_rhs_factory(lam: float) -> RhsFunc:
    """Return stiff RHS: y' = lam * (y - cos(t)) - sin(t), lam < 0."""
    ensure_finite("lam", lam)
    if lam >= 0.0:
        raise ValueError(f"lam must be negative for this stiff demo, got {lam}")

    def rhs(t: float, y: float) -> float:
        return lam * (y - math.cos(t)) - math.sin(t)

    return rhs


def stiff_tracking_exact(t_values: np.ndarray, t0: float, y0: float, lam: float) -> np.ndarray:
    """Exact solution for y' = lam*(y-cos t)-sin t with initial y(t0)=y0."""
    return np.cos(t_values) + (y0 - math.cos(t0)) * np.exp(lam * (t_values - t0))


def numerical_jacobian_scalar(f: RhsFunc, t: float, y: float, eps: float = 1e-8) -> float:
    """Central finite-difference approximation of df/dy at scalar state."""
    ensure_finite("eps", eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    fp = f(t, y + eps)
    fm = f(t, y - eps)
    if not math.isfinite(fp) or not math.isfinite(fm):
        raise RuntimeError("non-finite function value during Jacobian approximation")
    return (fp - fm) / (2.0 * eps)


def newton_solve_scalar(
    g: Callable[[float], float],
    dg: Callable[[float], float],
    z0: float,
    tol: float,
    max_iter: int,
) -> Tuple[float, int, float]:
    """Solve g(z)=0 by Newton iteration with scalar safeguards."""
    z = z0
    for k in range(1, max_iter + 1):
        gz = g(z)
        dgz = dg(z)

        if not math.isfinite(gz) or not math.isfinite(dgz):
            raise RuntimeError("non-finite Newton state encountered")
        if abs(dgz) < 1e-14:
            raise RuntimeError(f"near-singular Newton derivative: dg={dgz:.3e}")

        z_next = z - gz / dgz
        if not math.isfinite(z_next):
            raise RuntimeError("Newton produced non-finite iterate")

        if abs(z_next - z) <= tol * max(1.0, abs(z_next)):
            residual = abs(g(z_next))
            return z_next, k, residual
        z = z_next

    residual = abs(g(z))
    raise RuntimeError(
        f"Newton did not converge within {max_iter} iterations; last residual={residual:.3e}"
    )


def backward_euler_bdf1(
    f: RhsFunc,
    jac: JacFunc,
    t0: float,
    y0: float,
    h: float,
    steps: int,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[NewtonDiagnostic]]:
    """BDF1 (Backward Euler): y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})."""
    check_solver_controls(h=h, steps=steps, tol=tol, max_iter=max_iter)
    ensure_finite("t0", t0)
    ensure_finite("y0", y0)

    t = np.empty(steps + 1, dtype=float)
    y = np.empty(steps + 1, dtype=float)
    diagnostics: List[NewtonDiagnostic] = []

    t[0] = t0
    y[0] = y0

    for n in range(steps):
        t_next = t[n] + h
        y_prev = y[n]
        guess = y_prev

        def g(z: float) -> float:
            return z - y_prev - h * f(t_next, z)

        def dg(z: float) -> float:
            return 1.0 - h * jac(t_next, z)

        y_next, iters, residual = newton_solve_scalar(g=g, dg=dg, z0=guess, tol=tol, max_iter=max_iter)
        t[n + 1] = t_next
        y[n + 1] = y_next
        diagnostics.append(
            NewtonDiagnostic(
                method="BDF1",
                step=n,
                t_next=t_next,
                initial_guess=guess,
                value=y_next,
                iterations=iters,
                residual=residual,
            )
        )

    return t, y, diagnostics


def bdf2(
    f: RhsFunc,
    jac: JacFunc,
    t0: float,
    y0: float,
    h: float,
    steps: int,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[NewtonDiagnostic]]:
    """BDF2 with an implicit trapezoid startup.

    Formula for n>=1:
        (3 y_{n+1} - 4 y_n + y_{n-1})/(2h) = f(t_{n+1}, y_{n+1})
    """
    check_solver_controls(h=h, steps=steps, tol=tol, max_iter=max_iter)
    ensure_finite("t0", t0)
    ensure_finite("y0", y0)

    t = np.empty(steps + 1, dtype=float)
    y = np.empty(steps + 1, dtype=float)
    diagnostics: List[NewtonDiagnostic] = []

    t[0] = t0
    y[0] = y0

    # Startup step by implicit trapezoid to obtain y_1.
    t1 = t0 + h

    f0 = f(t0, y0)
    guess0 = y0 + h * f0

    def g0(z: float) -> float:
        return z - y0 - 0.5 * h * (f0 + f(t1, z))

    def dg0(z: float) -> float:
        return 1.0 - 0.5 * h * jac(t1, z)

    y1, iters0, residual0 = newton_solve_scalar(
        g=g0, dg=dg0, z0=guess0, tol=tol, max_iter=max_iter
    )
    t[1] = t1
    y[1] = y1
    diagnostics.append(
        NewtonDiagnostic(
            method="BDF2-start(trapezoid)",
            step=0,
            t_next=t1,
            initial_guess=guess0,
            value=y1,
            iterations=iters0,
            residual=residual0,
        )
    )

    if steps == 1:
        return t, y, diagnostics

    for n in range(1, steps):
        t_next = t[n] + h
        # Linear extrapolation predictor for a stronger Newton initial guess.
        guess = y[n] + (y[n] - y[n - 1])

        y_n = y[n]
        y_nm1 = y[n - 1]

        def g(z: float) -> float:
            return 3.0 * z - 4.0 * y_n + y_nm1 - 2.0 * h * f(t_next, z)

        def dg(z: float) -> float:
            return 3.0 - 2.0 * h * jac(t_next, z)

        y_next, iters, residual = newton_solve_scalar(g=g, dg=dg, z0=guess, tol=tol, max_iter=max_iter)
        t[n + 1] = t_next
        y[n + 1] = y_next
        diagnostics.append(
            NewtonDiagnostic(
                method="BDF2",
                step=n,
                t_next=t_next,
                initial_guess=guess,
                value=y_next,
                iterations=iters,
                residual=residual,
            )
        )

    return t, y, diagnostics


def explicit_euler(f: RhsFunc, t0: float, y0: float, h: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit Euler baseline for stiffness comparison."""
    check_solver_controls(h=h, steps=steps, tol=1e-12, max_iter=1)
    t = np.empty(steps + 1, dtype=float)
    y = np.empty(steps + 1, dtype=float)
    t[0], y[0] = t0, y0

    for n in range(steps):
        t[n + 1] = t[n] + h
        y[n + 1] = y[n] + h * f(t[n], y[n])
    return t, y


def error_metrics(y_num: np.ndarray, y_exact: np.ndarray) -> Tuple[float, float]:
    """Return final and max absolute errors."""
    err = np.abs(y_num - y_exact)
    return float(err[-1]), float(np.max(err))


def estimate_orders(results: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Estimate p from pairs (h, error) with successive h-halving."""
    out: List[Tuple[float, float]] = []
    for i in range(len(results) - 1):
        h1, e1 = results[i]
        h2, e2 = results[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        p = math.log(e1 / e2) / math.log(h1 / h2)
        out.append((h2, p))
    return out


def summarize_newton(diags: Sequence[NewtonDiagnostic]) -> Tuple[float, int, float]:
    """Aggregate Newton diagnostics as (mean_iters, max_iters, max_residual)."""
    iters = [d.iterations for d in diags]
    residuals = [d.residual for d in diags]
    return float(np.mean(iters)), int(np.max(iters)), float(np.max(residuals))


def print_diag_prefix(diags: Sequence[NewtonDiagnostic], rows: int = 6) -> None:
    """Print first few implicit-solve diagnostics."""
    print(" step diagnostics (prefix):")
    print(" method            step   t_next      guess          value         iters      residual")
    print("-" * 96)
    show = min(rows, len(diags))
    for d in diags[:show]:
        print(
            f" {d.method:14s}  {d.step:4d}  {d.t_next:7.3f}   {d.initial_guess:12.8f}  "
            f"{d.value:12.8f}  {d.iterations:5d}   {d.residual:10.3e}"
        )
    if len(diags) > show:
        print(f" ... ({len(diags) - show} more steps)")


def run_convergence_experiment(
    f: RhsFunc,
    jac: JacFunc,
    t0: float,
    y0: float,
    t_end: float,
    h_values: Sequence[float],
    lam: float,
) -> None:
    """Run convergence study for BDF1 and BDF2."""
    bdf1_results: List[Tuple[float, float]] = []
    bdf2_results: List[Tuple[float, float]] = []

    print("=" * 108)
    print(f"Convergence study on [0, {t_end}] for stiff tracking equation (lambda={lam})")
    print(" h        steps    BDF1 max_abs_err     BDF2 max_abs_err     BDF1 mean/newton_max_iter   BDF2 mean/newton_max_iter")
    print("-" * 108)

    for h in h_values:
        steps = require_integer_steps(t0=t0, t_end=t_end, h=h)

        t1, y1, d1 = backward_euler_bdf1(
            f=f,
            jac=jac,
            t0=t0,
            y0=y0,
            h=h,
            steps=steps,
        )
        t2, y2, d2 = bdf2(
            f=f,
            jac=jac,
            t0=t0,
            y0=y0,
            h=h,
            steps=steps,
        )

        y_exact_1 = stiff_tracking_exact(t_values=t1, t0=t0, y0=y0, lam=lam)
        y_exact_2 = stiff_tracking_exact(t_values=t2, t0=t0, y0=y0, lam=lam)
        _, bdf1_max = error_metrics(y_num=y1, y_exact=y_exact_1)
        _, bdf2_max = error_metrics(y_num=y2, y_exact=y_exact_2)

        bdf1_results.append((h, bdf1_max))
        bdf2_results.append((h, bdf2_max))

        bdf1_mean_iter, bdf1_max_iter, _ = summarize_newton(d1)
        bdf2_mean_iter, bdf2_max_iter, _ = summarize_newton(d2)

        print(
            f"{h:7.4f}  {steps:6d}    {bdf1_max:14.6e}    {bdf2_max:14.6e}    "
            f"{bdf1_mean_iter:6.2f}/{bdf1_max_iter:2d}                  {bdf2_mean_iter:6.2f}/{bdf2_max_iter:2d}"
        )

    print("=" * 108)
    print("Observed order p (based on max_abs_err)")
    for h, p in estimate_orders(bdf1_results):
        print(f" BDF1: h={h:7.4f} -> p={p:.4f}")
    for h, p in estimate_orders(bdf2_results):
        print(f" BDF2: h={h:7.4f} -> p={p:.4f}")

    _, _, d2_small = bdf2(
        f=f,
        jac=jac,
        t0=t0,
        y0=y0,
        h=h_values[-1],
        steps=require_integer_steps(t0=t0, t_end=t_end, h=h_values[-1]),
    )
    print_diag_prefix(d2_small, rows=6)


def run_stability_comparison(
    f: RhsFunc,
    jac: JacFunc,
    t0: float,
    y0: float,
    t_end: float,
    h: float,
    lam: float,
) -> None:
    """Compare BDF1/BDF2/Euler under a relatively large step on a stiff ODE."""
    steps = require_integer_steps(t0=t0, t_end=t_end, h=h)

    t_b1, y_b1, _ = backward_euler_bdf1(f=f, jac=jac, t0=t0, y0=y0, h=h, steps=steps)
    t_b2, y_b2, _ = bdf2(f=f, jac=jac, t0=t0, y0=y0, h=h, steps=steps)
    t_eu, y_eu = explicit_euler(f=f, t0=t0, y0=y0, h=h, steps=steps)

    ref_b1 = stiff_tracking_exact(t_values=t_b1, t0=t0, y0=y0, lam=lam)
    ref_b2 = stiff_tracking_exact(t_values=t_b2, t0=t0, y0=y0, lam=lam)
    ref_eu = stiff_tracking_exact(t_values=t_eu, t0=t0, y0=y0, lam=lam)

    _, b1_max = error_metrics(y_num=y_b1, y_exact=ref_b1)
    _, b2_max = error_metrics(y_num=y_b2, y_exact=ref_b2)
    _, eu_max = error_metrics(y_num=y_eu, y_exact=ref_eu)

    print("=" * 108)
    print(
        f"Stability comparison on [0, {t_end}] with large step h={h} (lambda={lam}, explicit Euler unstable threshold h<{-2.0 / lam:.3f})"
    )
    print(" method          final_value       exact_final      max_abs_error      max_abs_state")
    print("-" * 108)
    print(
        f" BDF1         {y_b1[-1]:14.8f}   {ref_b1[-1]:14.8f}   {b1_max:14.6e}   {float(np.max(np.abs(y_b1))):14.6e}"
    )
    print(
        f" BDF2         {y_b2[-1]:14.8f}   {ref_b2[-1]:14.8f}   {b2_max:14.6e}   {float(np.max(np.abs(y_b2))):14.6e}"
    )
    print(
        f" Euler        {y_eu[-1]:14.8f}   {ref_eu[-1]:14.8f}   {eu_max:14.6e}   {float(np.max(np.abs(y_eu))):14.6e}"
    )


def main() -> None:
    lam = -20.0
    t0 = 0.0
    y0 = 1.0

    rhs = stiff_tracking_rhs_factory(lam)
    jac = lambda t, y: numerical_jacobian_scalar(rhs, t, y)

    print("BDF demo on stiff scalar ODE: y' = lambda*(y-cos(t)) - sin(t)")
    print(f"Parameters: lambda={lam}, initial condition y(0)={y0}")
    print("Reference exact solution: y(t)=cos(t) (for y0=1)")

    run_convergence_experiment(
        f=rhs,
        jac=jac,
        t0=t0,
        y0=y0,
        t_end=1.0,
        h_values=[0.04, 0.02, 0.01, 0.005],
        lam=lam,
    )

    run_stability_comparison(
        f=rhs,
        jac=jac,
        t0=t0,
        y0=y0,
        t_end=4.0,
        h=0.2,
        lam=lam,
    )


if __name__ == "__main__":
    main()
