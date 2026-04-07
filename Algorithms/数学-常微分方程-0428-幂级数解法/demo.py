"""Power-series method MVP for second-order linear ODEs.

Target form:
    y''(x) + p(x) y'(x) + q(x) y(x) = 0

Both p(x), q(x) are provided by their Taylor coefficients around x=0:
    p(x) = sum_{k>=0} p_k x^k
    q(x) = sum_{k>=0} q_k x^k
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray


@dataclass(frozen=True)
class ODECase:
    """A deterministic demo case for the power-series solver."""

    name: str
    p_coeffs: Array
    q_coeffs: Array
    y0: float
    dy0: float
    n_terms: int
    x_grid: Array
    reference: Callable[[Array], Array]
    error_tol: float


def _pad_coeffs(coeffs: Sequence[float], n_terms: int) -> Array:
    arr = np.asarray(coeffs, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Coefficient array must be 1D.")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive.")
    if arr.size >= n_terms:
        return arr[:n_terms].copy()
    out = np.zeros(n_terms, dtype=float)
    out[: arr.size] = arr
    return out


def power_series_coefficients(
    p_coeffs: Sequence[float],
    q_coeffs: Sequence[float],
    y0: float,
    dy0: float,
    n_terms: int,
) -> Array:
    """Compute Taylor coefficients a_n of y(x)=sum a_n x^n by recurrence."""
    if n_terms < 2:
        raise ValueError("n_terms must be >= 2.")

    p = _pad_coeffs(p_coeffs, n_terms)
    q = _pad_coeffs(q_coeffs, n_terms)

    a = np.zeros(n_terms, dtype=float)
    a[0] = float(y0)
    a[1] = float(dy0)

    # For n >= 0:
    # (n+2)(n+1)a_{n+2} + sum_{k=0}^n p_k (n-k+1)a_{n-k+1} + sum_{k=0}^n q_k a_{n-k} = 0
    for n in range(n_terms - 2):
        p_conv = 0.0
        q_conv = 0.0
        for k in range(n + 1):
            p_conv += p[k] * (n - k + 1) * a[n - k + 1]
            q_conv += q[k] * a[n - k]
        a[n + 2] = -(p_conv + q_conv) / ((n + 2) * (n + 1))

    return a


def eval_series(coeffs: Sequence[float], x: Array) -> Array:
    coeff_arr = np.asarray(coeffs, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    return np.polynomial.polynomial.polyval(x_arr, coeff_arr)


def derivative_coeffs(coeffs: Sequence[float], order: int) -> Array:
    if order < 0:
        raise ValueError("order must be non-negative.")
    out = np.asarray(coeffs, dtype=float)
    for _ in range(order):
        if out.size <= 1:
            return np.array([0.0], dtype=float)
        idx = np.arange(1, out.size, dtype=float)
        out = out[1:] * idx
    return out


def ode_residual(
    coeffs: Sequence[float],
    p_coeffs: Sequence[float],
    q_coeffs: Sequence[float],
    x: Array,
) -> Array:
    coeff_arr = np.asarray(coeffs, dtype=float)
    p = _pad_coeffs(p_coeffs, coeff_arr.size)
    q = _pad_coeffs(q_coeffs, coeff_arr.size)

    y = eval_series(coeff_arr, x)
    dy = eval_series(derivative_coeffs(coeff_arr, 1), x)
    d2y = eval_series(derivative_coeffs(coeff_arr, 2), x)
    p_x = eval_series(p, x)
    q_x = eval_series(q, x)
    return d2y + p_x * dy + q_x * y


def airy_like_reference(x_eval: Array) -> Array:
    """Reference for y'' - x y = 0, y(0)=1, y'(0)=0 via high-accuracy IVP."""
    x_eval = np.asarray(x_eval, dtype=float)
    if np.any(x_eval < 0):
        raise ValueError("This reference helper expects x_eval >= 0.")

    unique_x, inverse = np.unique(x_eval, return_inverse=True)
    x_max = float(unique_x[-1])
    sol = solve_ivp(
        fun=lambda t, z: np.array([z[1], t * z[0]], dtype=float),
        t_span=(0.0, x_max),
        y0=np.array([1.0, 0.0], dtype=float),
        t_eval=unique_x,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solve_ivp failed: {sol.message}")
    y_unique = sol.y[0]
    return y_unique[inverse]


def run_case(case: ODECase) -> tuple[float, float]:
    coeffs = power_series_coefficients(
        p_coeffs=case.p_coeffs,
        q_coeffs=case.q_coeffs,
        y0=case.y0,
        dy0=case.dy0,
        n_terms=case.n_terms,
    )
    approx = eval_series(coeffs, case.x_grid)
    ref = case.reference(case.x_grid)
    abs_err = np.abs(approx - ref)
    max_err = float(np.max(abs_err))

    residual = ode_residual(coeffs, case.p_coeffs, case.q_coeffs, case.x_grid)
    max_residual = float(np.max(np.abs(residual)))

    print(f"\n=== {case.name} ===")
    print(
        "n_terms={:d}, max|approx-ref|={:.3e}, max|ODE residual|={:.3e}".format(
            case.n_terms, max_err, max_residual
        )
    )
    print("x\t\tapprox\t\t\tref\t\t\tabs_err")
    for x, a, r, e in zip(case.x_grid, approx, ref, abs_err):
        print(f"{x:>5.2f}\t{a: .12e}\t{r: .12e}\t{e: .3e}")

    if max_err > case.error_tol:
        raise RuntimeError(
            f"{case.name} error too large: max_err={max_err:.3e}, tol={case.error_tol:.3e}"
        )
    return max_err, max_residual


def build_cases() -> list[ODECase]:
    case_oscillator = ODECase(
        name="Harmonic oscillator: y'' + y = 0, y(0)=1, y'(0)=0",
        p_coeffs=np.array([0.0], dtype=float),
        q_coeffs=np.array([1.0], dtype=float),
        y0=1.0,
        dy0=0.0,
        n_terms=18,
        x_grid=np.linspace(0.0, np.pi / 2, 8),
        reference=lambda x: np.cos(x),
        error_tol=1e-10,
    )

    case_airy_like = ODECase(
        name="Airy-like: y'' - x y = 0, y(0)=1, y'(0)=0",
        p_coeffs=np.array([0.0], dtype=float),
        q_coeffs=np.array([0.0, -1.0], dtype=float),
        y0=1.0,
        dy0=0.0,
        n_terms=28,
        x_grid=np.linspace(0.0, 1.2, 9),
        reference=airy_like_reference,
        error_tol=2e-10,
    )

    return [case_oscillator, case_airy_like]


def main() -> None:
    cases = build_cases()
    all_max_errors: list[float] = []
    all_max_residuals: list[float] = []

    for case in cases:
        max_err, max_residual = run_case(case)
        all_max_errors.append(max_err)
        all_max_residuals.append(max_residual)

    print("\n=== Summary ===")
    print(f"cases={len(cases)}")
    print(f"worst max|approx-ref| = {max(all_max_errors):.3e}")
    print(f"worst max|ODE residual| = {max(all_max_residuals):.3e}")
    print("All power-series checks passed.")


if __name__ == "__main__":
    main()
