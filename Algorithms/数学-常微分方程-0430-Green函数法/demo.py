"""Green's function method MVP for a second-order Dirichlet boundary problem.

We solve on x in [0, 1]:
    -y''(x) + c * y(x) = f(x),
    y(0) = y(1) = 0,
using the Green representation
    y(x) = integral_0^1 G(x, xi) f(xi) dxi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp

Array = np.ndarray


@dataclass(frozen=True)
class GreenCase:
    """Deterministic test case for the Green-function solver."""

    name: str
    c: float
    forcing: Callable[[Array], Array]
    exact: Callable[[Array], Array] | None
    n_grid: int
    error_tol: float
    residual_tol: float


def _trapezoid_weights(n: int, h: float) -> Array:
    if n < 2:
        raise ValueError("n must be at least 2")
    w = np.full(n, h, dtype=float)
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return w


def green_kernel_matrix(x_eval: Array, xi_grid: Array, c: float) -> Array:
    """Construct G(x, xi) for -y'' + c y = f with y(0)=y(1)=0 and c>=0."""
    if c < 0.0:
        raise ValueError("This MVP supports c >= 0 only.")

    x = np.asarray(x_eval, dtype=float)
    xi = np.asarray(xi_grid, dtype=float)
    x_col = x[:, None]
    xi_row = xi[None, :]

    left = np.minimum(x_col, xi_row)
    right = np.maximum(x_col, xi_row)

    if c < 1e-14:
        # c -> 0 limit: Green kernel of -y'' on [0,1] with Dirichlet BC.
        return left * (1.0 - right)

    k = float(np.sqrt(c))
    denom = k * np.sinh(k)
    return np.sinh(k * left) * np.sinh(k * (1.0 - right)) / denom


def solve_green_dirichlet(x_grid: Array, forcing_values: Array, c: float) -> tuple[Array, Array]:
    """Solve the BVP by quadrature of Green kernel on the same xi-grid."""
    x = np.asarray(x_grid, dtype=float)
    f = np.asarray(forcing_values, dtype=float)

    if x.ndim != 1 or f.ndim != 1 or x.shape != f.shape:
        raise ValueError("x_grid and forcing_values must be 1D arrays of equal length.")
    if x.size < 3:
        raise ValueError("Need at least 3 grid points.")

    h = float(x[1] - x[0])
    if not np.allclose(np.diff(x), h, rtol=0.0, atol=1e-12):
        raise ValueError("This MVP expects a uniform grid.")

    g = green_kernel_matrix(x, x, c)
    weights = _trapezoid_weights(x.size, h)
    y = g @ (weights * f)
    return y, g


def finite_difference_residual(y: Array, f: Array, c: float, h: float) -> Array:
    """Residual of -y'' + c y - f on interior nodes by central differences."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    if y.shape != f.shape or y.size < 3:
        raise ValueError("y and f must have same shape and length >= 3.")

    second = (y[:-2] - 2.0 * y[1:-1] + y[2:]) / (h * h)
    lhs = -second + c * y[1:-1]
    return lhs - f[1:-1]


def reference_solve_bvp(x_grid: Array, forcing_values: Array, c: float) -> Array:
    """High-accuracy reference via scipy.solve_bvp for non-analytic sample."""
    x = np.asarray(x_grid, dtype=float)
    f = np.asarray(forcing_values, dtype=float)

    def rhs(t: Array, z: Array) -> Array:
        f_t = np.interp(t, x, f)
        return np.vstack((z[1], c * z[0] - f_t))

    def bc(ya: Array, yb: Array) -> Array:
        return np.array([ya[0], yb[0]], dtype=float)

    guess = np.zeros((2, x.size), dtype=float)
    guess[0] = 0.1 * x * (1.0 - x)

    sol = solve_bvp(rhs, bc, x, guess, tol=1e-10, max_nodes=20000)
    if not sol.success:
        raise RuntimeError(f"Reference solve_bvp failed: {sol.message}")
    return np.asarray(sol.sol(x)[0], dtype=float)


def run_case(case: GreenCase) -> tuple[float, float]:
    if case.n_grid < 11:
        raise ValueError("n_grid must be >= 11.")
    if case.c < 0.0:
        raise ValueError("c must be >= 0.")

    x = np.linspace(0.0, 1.0, case.n_grid)
    h = float(x[1] - x[0])

    f = np.asarray(case.forcing(x), dtype=float)
    if f.shape != x.shape:
        raise ValueError(f"Forcing shape mismatch in case: {case.name}")
    if not np.all(np.isfinite(f)):
        raise ValueError(f"Forcing contains non-finite values in case: {case.name}")

    y, _g = solve_green_dirichlet(x, f, case.c)

    if case.exact is None:
        ref = reference_solve_bvp(x, f, case.c)
        ref_name = "solve_bvp reference"
    else:
        ref = np.asarray(case.exact(x), dtype=float)
        ref_name = "closed-form reference"

    abs_err = np.abs(y - ref)
    max_err = float(np.max(abs_err))

    residual = finite_difference_residual(y, f, case.c, h)
    max_residual = float(np.max(np.abs(residual)))

    sample_idx = np.linspace(0, case.n_grid - 1, 9, dtype=int)
    df = pd.DataFrame(
        {
            "x": x[sample_idx],
            "green": y[sample_idx],
            "ref": ref[sample_idx],
            "abs_err": abs_err[sample_idx],
        }
    )

    print(f"\n=== {case.name} ===")
    print(f"c={case.c:.4f}, n_grid={case.n_grid}, reference={ref_name}")
    print(f"max|green-ref|={max_err:.3e}, max|FD residual|={max_residual:.3e}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    if max_err > case.error_tol:
        raise RuntimeError(
            f"{case.name} failed error check: {max_err:.3e} > {case.error_tol:.3e}"
        )
    if max_residual > case.residual_tol:
        raise RuntimeError(
            f"{case.name} failed residual check: {max_residual:.3e} > {case.residual_tol:.3e}"
        )
    return max_err, max_residual


def build_cases() -> list[GreenCase]:
    return [
        GreenCase(
            name="Case A: -y'' + 4y = sin(pi x)",
            c=4.0,
            forcing=lambda x: np.sin(np.pi * x),
            exact=lambda x: np.sin(np.pi * x) / (np.pi * np.pi + 4.0),
            n_grid=401,
            error_tol=2e-6,
            residual_tol=2e-4,
        ),
        GreenCase(
            name="Case B: -y'' = 1",
            c=0.0,
            forcing=lambda x: np.ones_like(x),
            exact=lambda x: 0.5 * x * (1.0 - x),
            n_grid=401,
            error_tol=2e-6,
            residual_tol=2e-4,
        ),
        GreenCase(
            name="Case C: -y'' + 1.5y = exp(x)",
            c=1.5,
            forcing=lambda x: np.exp(x),
            exact=None,
            n_grid=401,
            error_tol=3e-6,
            residual_tol=2e-4,
        ),
    ]


def main() -> None:
    cases = build_cases()

    max_errors: list[float] = []
    max_residuals: list[float] = []

    for case in cases:
        err, res = run_case(case)
        max_errors.append(err)
        max_residuals.append(res)

    print("\n=== Summary ===")
    print(f"cases={len(cases)}")
    print(f"worst max|green-ref| = {max(max_errors):.3e}")
    print(f"worst max|FD residual| = {max(max_residuals):.3e}")
    print("All Green-function checks passed.")


if __name__ == "__main__":
    main()
