"""Runge-Kutta-Fehlberg (RKF45) minimal runnable MVP.

This script solves an IVP with adaptive step size and prints diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
OdeFunc = Callable[[float, Array], Array]


@dataclass
class RKF45Result:
    ts: Array
    ys: Array
    accepted_steps: int
    rejected_steps: int


def _to_1d_array(y: float | Array) -> Array:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def rkf45_step(f: OdeFunc, t: float, y: Array, h: float) -> tuple[Array, Array, float]:
    """One Fehlberg (4,5) step returning (y4, y5, inf-norm error)."""
    k1 = h * _to_1d_array(f(t, y))
    k2 = h * _to_1d_array(f(t + h / 4.0, y + k1 / 4.0))
    k3 = h * _to_1d_array(f(t + 3.0 * h / 8.0, y + 3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
    k4 = h * _to_1d_array(
        f(
            t + 12.0 * h / 13.0,
            y + 1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0,
        )
    )
    k5 = h * _to_1d_array(
        f(t + h, y + 439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0)
    )
    k6 = h * _to_1d_array(
        f(
            t + h / 2.0,
            y
            - 8.0 * k1 / 27.0
            + 2.0 * k2
            - 3544.0 * k3 / 2565.0
            + 1859.0 * k4 / 4104.0
            - 11.0 * k5 / 40.0,
        )
    )

    y4 = y + 25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0
    y5 = y + 16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0

    err_inf = float(np.linalg.norm(y5 - y4, ord=np.inf))
    return y4, y5, err_inf


def integrate_rkf45(
    f: OdeFunc,
    t0: float,
    y0: float | Array,
    t_end: float,
    *,
    h0: float = 0.2,
    atol: float = 1e-9,
    rtol: float = 1e-6,
    h_min: float = 1e-8,
    h_max: float = 0.5,
    safety: float = 0.9,
    fac_min: float = 0.2,
    fac_max: float = 5.0,
    max_steps: int = 200000,
) -> RKF45Result:
    """Adaptive RKF45 integrator for non-stiff IVPs."""
    if t_end <= t0:
        raise ValueError("t_end must be larger than t0")
    if h0 <= 0.0:
        raise ValueError("h0 must be positive")

    t = float(t0)
    y = _to_1d_array(y0)
    h = min(max(h0, h_min), h_max)

    ts = [t]
    ys = [y.copy()]
    accepted_steps = 0
    rejected_steps = 0

    for _ in range(max_steps):
        if t >= t_end:
            break

        if t + h > t_end:
            h = t_end - t

        _, y5, err = rkf45_step(f, t, y, h)
        scale = atol + rtol * max(float(np.linalg.norm(y, ord=np.inf)), float(np.linalg.norm(y5, ord=np.inf)))
        err_ratio = err / scale if scale > 0.0 else err

        if err_ratio <= 1.0:
            t += h
            y = y5
            accepted_steps += 1
            ts.append(t)
            ys.append(y.copy())
        else:
            rejected_steps += 1

        if err_ratio == 0.0:
            factor = fac_max
        else:
            factor = safety * err_ratio ** (-0.2)  # -1/(p+1), p=4 embedded error control
            factor = float(np.clip(factor, fac_min, fac_max))

        h = float(np.clip(h * factor, h_min, h_max))

        if h <= h_min and err_ratio > 1.0:
            raise RuntimeError("Step size reached h_min while error is still too large")
    else:
        raise RuntimeError("Maximum number of steps reached before hitting t_end")

    return RKF45Result(
        ts=np.asarray(ts, dtype=float),
        ys=np.asarray(ys, dtype=float),
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )


def main() -> None:
    # Test IVP with known analytical solution.
    def f(t: float, y: Array) -> Array:
        return y - t * t + 1.0

    def exact(t: Array) -> Array:
        return (t + 1.0) ** 2 - 0.5 * np.exp(t)

    result = integrate_rkf45(
        f=f,
        t0=0.0,
        y0=np.array([0.5]),
        t_end=2.0,
        h0=0.25,
        atol=1e-9,
        rtol=1e-6,
        h_min=1e-8,
        h_max=0.4,
        safety=0.9,
    )

    ys_exact = exact(result.ts)
    ys_num = result.ys[:, 0]
    abs_errors = np.abs(ys_num - ys_exact)

    print("=== RKF45 Demo ===")
    print(f"accepted_steps: {result.accepted_steps}")
    print(f"rejected_steps: {result.rejected_steps}")
    print(f"final_time:     {result.ts[-1]:.6f}")
    print(f"final_value:    {ys_num[-1]:.12f}")
    print(f"exact_value:    {ys_exact[-1]:.12f}")
    print(f"abs_error:      {abs_errors[-1]:.3e}")
    print(f"max_abs_error_on_grid: {abs_errors.max():.3e}")

    print("\nSample points (t, numerical, exact, abs_error):")
    sample_ids = np.linspace(0, len(result.ts) - 1, num=min(6, len(result.ts)), dtype=int)
    for idx in sample_ids:
        print(
            f"{result.ts[idx]:.6f}, {ys_num[idx]:.12f}, {ys_exact[idx]:.12f}, {abs_errors[idx]:.3e}"
        )


if __name__ == "__main__":
    main()
