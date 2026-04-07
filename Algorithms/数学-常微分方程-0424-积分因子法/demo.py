"""Integrating factor MVP for first-order linear ODE.

Problem form:
    y'(x) + p(x) * y(x) = q(x),  y(x0) = y0

This script runs a deterministic demo without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

VectorFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class IFResult:
    x: np.ndarray
    y_if: np.ndarray
    mu: np.ndarray
    ip: np.ndarray
    iq: np.ndarray


def validate_setup(x0: float, x1: float, num_points: int, y0: float) -> None:
    if not np.isfinite([x0, x1, y0]).all():
        raise ValueError("x0/x1/y0 must be finite.")
    if x1 <= x0:
        raise ValueError(f"Require x1 > x0, got x0={x0}, x1={x1}.")
    if num_points < 2:
        raise ValueError(f"num_points must be >=2, got {num_points}.")


def ensure_vectorized_output(values: np.ndarray, x: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != x.shape:
        raise ValueError(f"{name}(x) shape mismatch: expected {x.shape}, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}(x) contains non-finite values.")
    return values


def solve_with_integrating_factor(
    p: VectorFunc,
    q: VectorFunc,
    x0: float,
    x1: float,
    y0: float,
    num_points: int,
) -> IFResult:
    """Solve y' + p(x) y = q(x) on a fixed grid using integrating factor."""
    validate_setup(x0=x0, x1=x1, num_points=num_points, y0=y0)

    x = np.linspace(x0, x1, num_points, dtype=float)
    p_vals = ensure_vectorized_output(p(x), x=x, name="p")
    q_vals = ensure_vectorized_output(q(x), x=x, name="q")

    ip = cumulative_trapezoid(p_vals, x, initial=0.0)
    mu = np.exp(ip)

    if not np.isfinite(mu).all():
        raise OverflowError("Integrating factor mu has non-finite values; check interval/scaling.")

    iq = cumulative_trapezoid(mu * q_vals, x, initial=0.0)

    c0 = mu[0] * y0
    y_if = (c0 + iq) / mu

    if not np.isfinite(y_if).all():
        raise FloatingPointError("Computed y_if contains non-finite values.")

    return IFResult(x=x, y_if=y_if, mu=mu, ip=ip, iq=iq)


def solve_with_scipy_reference(
    p: VectorFunc,
    q: VectorFunc,
    x: np.ndarray,
    y0: float,
) -> np.ndarray:
    """Reference solver using solve_ivp on the same grid."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # Here p/q are evaluated on scalar t then converted to float.
        t_arr = np.array([t], dtype=float)
        p_t = float(np.asarray(p(t_arr), dtype=float)[0])
        q_t = float(np.asarray(q(t_arr), dtype=float)[0])
        return np.array([q_t - p_t * y[0]], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=(float(x[0]), float(x[-1])),
        y0=np.array([y0], dtype=float),
        t_eval=x,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    y_ref = np.asarray(sol.y[0], dtype=float)
    if y_ref.shape != x.shape:
        raise RuntimeError("Unexpected solve_ivp output shape.")
    return y_ref


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Exact solution for y' + 2xy = x, y(0)=1."""
    return 0.5 + 0.5 * np.exp(-(x**2))


def ode_residual(x: np.ndarray, y: np.ndarray, p: VectorFunc, q: VectorFunc) -> np.ndarray:
    """Discrete residual r = y' + p(x)*y - q(x) using central differences."""
    y_prime = np.gradient(y, x, edge_order=2)
    return y_prime + p(x) * y - q(x)


def print_samples(x: np.ndarray, y_if: np.ndarray, y_exact: np.ndarray, y_ref: np.ndarray) -> None:
    pick = np.array([0, len(x) // 4, len(x) // 2, 3 * len(x) // 4, len(x) - 1], dtype=int)
    print("\nSample points (x, y_if, y_exact, y_scipy):")
    for idx in pick:
        print(
            f"x={x[idx]:.3f} | y_if={y_if[idx]:.10f} | "
            f"y_exact={y_exact[idx]:.10f} | y_scipy={y_ref[idx]:.10f}"
        )


def run_demo() -> None:
    x0, x1 = 0.0, 2.0
    y0 = 1.0
    num_points = 401

    def p(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def q(x: np.ndarray) -> np.ndarray:
        return x

    result = solve_with_integrating_factor(
        p=p,
        q=q,
        x0=x0,
        x1=x1,
        y0=y0,
        num_points=num_points,
    )

    y_exact = exact_solution(result.x)
    y_ref = solve_with_scipy_reference(p=p, q=q, x=result.x, y0=y0)

    abs_err_exact = np.abs(result.y_if - y_exact)
    abs_err_ref = np.abs(result.y_if - y_ref)

    residual = ode_residual(result.x, result.y_if, p=p, q=q)
    abs_residual = np.abs(residual)

    print("=== Integrating Factor Method Demo ===")
    print("ODE: y' + 2xy = x, y(0)=1")
    print(f"grid points: {num_points}, interval: [{x0}, {x1}]")

    print(f"max_abs_err_vs_exact: {np.max(abs_err_exact):.6e}")
    print(f"mean_abs_err_vs_exact: {np.mean(abs_err_exact):.6e}")
    print(f"max_abs_err_vs_scipy: {np.max(abs_err_ref):.6e}")
    print(f"mean_abs_err_vs_scipy: {np.mean(abs_err_ref):.6e}")
    print(f"max_abs_residual: {np.max(abs_residual):.6e}")
    print(f"mean_abs_residual: {np.mean(abs_residual):.6e}")

    print_samples(result.x, result.y_if, y_exact, y_ref)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
