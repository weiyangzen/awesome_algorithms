"""Minimal runnable MVP: second-order predictor-corrector method for ODEs.

Method used in this demo:
- Predictor: explicit Euler
- Corrector: trapezoidal rule fixed-point iteration (PE(CE)^m style)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Type alias: right-hand side function y' = f(t, y)
RHS = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class SolverResult:
    """Container for one predictor-corrector solve run."""

    t: np.ndarray
    y: np.ndarray
    y_pred: np.ndarray
    correction_iters: np.ndarray
    step_size: float
    max_corrections: int



def _as_vector(y0: float | np.ndarray) -> np.ndarray:
    """Convert scalar/array-like initial state to 1D float vector."""
    arr = np.atleast_1d(np.array(y0, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"y0 must be scalar or 1D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y0 contains non-finite values")
    return arr



def _validate_grid(t0: float, t_end: float, h: float) -> int:
    """Validate time grid and return integer number of steps."""
    if not (math.isfinite(t0) and math.isfinite(t_end) and math.isfinite(h)):
        raise ValueError("t0, t_end, h must be finite")
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    if t_end <= t0:
        raise ValueError(f"t_end must be > t0, got t0={t0}, t_end={t_end}")

    n_float = (t_end - t0) / h
    n_steps = int(round(n_float))
    if n_steps < 1 or not math.isclose(n_float, n_steps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "(t_end - t0) / h must be an integer within tolerance; "
            f"got {(t_end - t0) / h}"
        )
    return n_steps



def predictor_corrector_heun(
    f: RHS,
    t0: float,
    y0: float | np.ndarray,
    t_end: float,
    h: float,
    *,
    max_corrections: int = 1,
    tol: float = 1e-12,
) -> SolverResult:
    """Solve ODE by second-order predictor-corrector.

    Predictor:
        y_{n+1}^{(0)} = y_n + h f(t_n, y_n)
    Corrector (iterative trapezoidal fixed-point):
        y_{n+1}^{(k+1)} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(k)})]

    Args:
        f: RHS function.
        t0: initial time.
        y0: initial state (scalar or 1D vector).
        t_end: end time.
        h: step size.
        max_corrections: number of corrector updates per step (>=1).
        tol: fixed-point stopping threshold in infinity norm.

    Returns:
        SolverResult with time grid, corrected states, predicted states,
        and used corrector iteration counts.
    """
    if max_corrections < 1:
        raise ValueError(f"max_corrections must be >= 1, got {max_corrections}")
    if tol <= 0.0 or not math.isfinite(tol):
        raise ValueError(f"tol must be finite and > 0, got {tol}")

    n_steps = _validate_grid(t0, t_end, h)
    y0_vec = _as_vector(y0)
    dim = y0_vec.size

    t = np.linspace(t0, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, dim), dtype=float)
    y_pred = np.zeros((n_steps + 1, dim), dtype=float)
    correction_iters = np.zeros(n_steps, dtype=int)

    y[0] = y0_vec
    y_pred[0] = y0_vec

    for n in range(n_steps):
        tn = float(t[n])
        tnp1 = float(t[n + 1])
        yn = y[n]

        fn = np.array(f(tn, yn.copy()), dtype=float)
        if fn.shape != (dim,):
            raise ValueError(
                f"RHS output shape mismatch at step {n}: expected {(dim,)}, got {fn.shape}"
            )
        if not np.all(np.isfinite(fn)):
            raise ValueError(f"RHS produced non-finite values at step {n}")

        # Predictor: explicit Euler.
        y_hat = yn + h * fn
        y_pred[n + 1] = y_hat

        # Corrector: iterative trapezoidal updates.
        y_corr = y_hat.copy()
        used_iters = 0
        for k in range(max_corrections):
            fnp1 = np.array(f(tnp1, y_corr.copy()), dtype=float)
            if fnp1.shape != (dim,):
                raise ValueError(
                    "RHS output shape mismatch in correction at "
                    f"step {n}, iter {k + 1}: expected {(dim,)}, got {fnp1.shape}"
                )
            if not np.all(np.isfinite(fnp1)):
                raise ValueError(
                    f"RHS produced non-finite values in correction at step {n}, iter {k + 1}"
                )

            y_new = yn + 0.5 * h * (fn + fnp1)
            used_iters = k + 1
            if np.linalg.norm(y_new - y_corr, ord=np.inf) < tol:
                y_corr = y_new
                break
            y_corr = y_new

        y[n + 1] = y_corr
        correction_iters[n] = used_iters

    return SolverResult(
        t=t,
        y=y,
        y_pred=y_pred,
        correction_iters=correction_iters,
        step_size=h,
        max_corrections=max_corrections,
    )



def exact_scalar_solution(t: np.ndarray) -> np.ndarray:
    """Exact solution of y' = y - t^2 + 1, y(0)=0.5."""
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)



def scalar_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """RHS for scalar test ODE written in vector form."""
    return np.array([y[0] - t * t + 1.0], dtype=float)



def oscillator_rhs(_: float, y: np.ndarray) -> np.ndarray:
    """Simple harmonic oscillator system.

    y = [x, v], with x' = v, v' = -x.
    Exact for x(0)=0,v(0)=1: x=sin(t), v=cos(t).
    """
    x, v = float(y[0]), float(y[1])
    return np.array([v, -x], dtype=float)



def run_scalar_convergence_demo() -> None:
    """Show second-order convergence behavior for scalar ODE."""
    t0, t_end, y0 = 0.0, 2.0, 0.5
    step_sizes = [0.2, 0.1, 0.05, 0.025]

    print("=" * 88)
    print("Example 1: scalar ODE y' = y - t^2 + 1, y(0)=0.5")
    print("Exact solution: y(t) = (t+1)^2 - 0.5*exp(t)")
    print("Predictor-Corrector: Euler predictor + trapezoidal corrector")
    print("=" * 88)

    prev_error = None
    for h in step_sizes:
        result = predictor_corrector_heun(
            f=scalar_rhs,
            t0=t0,
            y0=y0,
            t_end=t_end,
            h=h,
            max_corrections=1,
            tol=1e-14,
        )
        y_exact = exact_scalar_solution(result.t)
        max_error = float(np.max(np.abs(result.y[:, 0] - y_exact)))

        if prev_error is None:
            ratio_text = "-"
            order_text = "-"
        else:
            ratio = prev_error / max_error
            observed_order = math.log(ratio, 2.0)
            ratio_text = f"{ratio:.4f}"
            order_text = f"{observed_order:.4f}"

        print(
            f"h={h:<7.4f} steps={len(result.t)-1:<4d} "
            f"max_error={max_error:.8e} error_ratio={ratio_text:<8} "
            f"observed_order={order_text}"
        )
        prev_error = max_error

    # Show first few detailed PECE records for one grid.
    h_detail = 0.1
    detail = predictor_corrector_heun(
        f=scalar_rhs,
        t0=t0,
        y0=y0,
        t_end=t_end,
        h=h_detail,
        max_corrections=2,
        tol=1e-14,
    )
    y_exact_detail = exact_scalar_solution(detail.t)

    print("\nDetailed first 5 steps (h=0.1, max_corrections=2):")
    print("step | t_n      | predictor y_hat  | corrected y_{n+1} | exact y_{n+1}    | abs_error")
    for n in range(5):
        tnp1 = detail.t[n + 1]
        y_hat = detail.y_pred[n + 1, 0]
        y_corr = detail.y[n + 1, 0]
        y_ex = y_exact_detail[n + 1]
        err = abs(y_corr - y_ex)
        print(
            f"{n:>4d} | {tnp1:>8.4f} | {y_hat:>15.8f} | {y_corr:>15.8f} "
            f"| {y_ex:>15.8f} | {err:.3e}"
        )



def run_vector_demo() -> None:
    """Solve a 2D harmonic oscillator and report endpoint error."""
    t0, t_end = 0.0, 2.0 * math.pi
    y0 = np.array([0.0, 1.0], dtype=float)
    n_steps = 800
    h = (t_end - t0) / n_steps

    result = predictor_corrector_heun(
        f=oscillator_rhs,
        t0=t0,
        y0=y0,
        t_end=t_end,
        h=h,
        max_corrections=2,
        tol=1e-13,
    )

    x_num, v_num = result.y[-1]
    x_ref, v_ref = math.sin(t_end), math.cos(t_end)
    endpoint_err = float(np.linalg.norm(result.y[-1] - np.array([x_ref, v_ref]), ord=np.inf))

    print("\n" + "=" * 88)
    print("Example 2: harmonic oscillator x'=v, v'=-x")
    print("Initial: x(0)=0, v(0)=1 ; Reference endpoint at 2pi: [0, 1]")
    print("=" * 88)
    print(f"step_size={h}, steps={len(result.t)-1}, max_corrections={result.max_corrections}")
    print(f"numerical endpoint: x={x_num:.10f}, v={v_num:.10f}")
    print(f"reference endpoint: x={x_ref:.10f}, v={v_ref:.10f}")
    print(f"endpoint_inf_error={endpoint_err:.6e}")
    print(
        "avg_correction_iters="
        f"{float(np.mean(result.correction_iters)):.3f}, "
        f"max_correction_iters={int(np.max(result.correction_iters))}"
    )



def main() -> None:
    run_scalar_convergence_demo()
    run_vector_demo()


if __name__ == "__main__":
    main()
