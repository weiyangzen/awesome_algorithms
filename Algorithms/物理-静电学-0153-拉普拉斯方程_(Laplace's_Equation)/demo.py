"""Minimal runnable MVP for Laplace's equation in electrostatics.

Solve:
    ∇²V = 0,  on (x, y) in [0,1] x [0,1]
with Dirichlet boundary conditions:
    V(x, 0) = 0
    V(0, y) = 0
    V(1, y) = 0
    V(x, 1) = sin(pi x)

The numerical solution is obtained by finite differences + SOR iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class LaplaceResult:
    """Container for numerical and analytic diagnostics."""

    x: np.ndarray
    y: np.ndarray
    potential_num: np.ndarray
    potential_exact: np.ndarray
    iterations: int
    converged: bool
    max_update: float
    residual_inf: float
    max_abs_error: float
    rmse: float


def top_boundary(x: np.ndarray) -> np.ndarray:
    """Top boundary potential V(x,1)=sin(pi*x)."""
    return np.sin(np.pi * x)


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical solution for the selected boundary-value problem."""
    X, Y = np.meshgrid(x, y)
    denom = math.sinh(math.pi)
    return np.sin(np.pi * X) * np.sinh(np.pi * Y) / denom


def initialize_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create uniform grid and apply Dirichlet boundary values."""
    if nx < 5 or ny < 5:
        raise ValueError("nx and ny must be >= 5.")

    x = np.linspace(0.0, 1.0, nx, dtype=float)
    y = np.linspace(0.0, 1.0, ny, dtype=float)

    V = np.zeros((ny, nx), dtype=float)
    V[-1, :] = top_boundary(x)  # y = 1
    # Other three boundaries are already zeros.
    return x, y, V


def sor_solve_laplace(
    V: np.ndarray,
    hx: float,
    hy: float,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, bool, float]:
    """Solve Laplace equation with point-wise SOR updates."""
    if V.ndim != 2:
        raise ValueError("V must be a 2D array.")
    if not (1.0 < omega < 2.0):
        raise ValueError("omega should satisfy 1 < omega < 2 for SOR.")
    if tol <= 0.0 or max_iter < 1:
        raise ValueError("tol must be > 0 and max_iter must be >= 1.")

    ny, nx = V.shape
    hxx = hx * hx
    hyy = hy * hy
    denom = 2.0 * (hxx + hyy)

    # Work on a copy to keep input unchanged from caller perspective.
    U = V.copy()
    max_update = float("inf")
    converged = False

    for it in range(1, max_iter + 1):
        max_update = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                old = U[j, i]
                gs = (
                    hyy * (U[j, i + 1] + U[j, i - 1])
                    + hxx * (U[j + 1, i] + U[j - 1, i])
                ) / denom
                new = (1.0 - omega) * old + omega * gs
                update = abs(new - old)
                if update > max_update:
                    max_update = update
                U[j, i] = new

        if max_update < tol:
            converged = True
            return U, it, converged, max_update

    return U, max_iter, converged, max_update


def residual_inf_norm(U: np.ndarray, hx: float, hy: float) -> float:
    """Compute infinity norm of discrete Laplacian residual on interior nodes."""
    hxx = hx * hx
    hyy = hy * hy
    lap = (
        (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / hxx
        + (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / hyy
    )
    return float(np.max(np.abs(lap)))


def solve_laplace_mvp(
    nx: int = 81,
    ny: int = 81,
    omega: float = 1.92,
    tol: float = 1.0e-10,
    max_iter: int = 20_000,
) -> LaplaceResult:
    """Run a full MVP solve and produce diagnostics."""
    x, y, V0 = initialize_grid(nx=nx, ny=ny)
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])

    V_num, iterations, converged, max_update = sor_solve_laplace(
        V=V0,
        hx=hx,
        hy=hy,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
    )
    V_exact = exact_solution(x=x, y=y)

    interior_num = V_num[1:-1, 1:-1]
    interior_exact = V_exact[1:-1, 1:-1]
    err = interior_num - interior_exact

    max_abs_error = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    residual = residual_inf_norm(U=V_num, hx=hx, hy=hy)

    return LaplaceResult(
        x=x,
        y=y,
        potential_num=V_num,
        potential_exact=V_exact,
        iterations=iterations,
        converged=converged,
        max_update=max_update,
        residual_inf=residual,
        max_abs_error=max_abs_error,
        rmse=rmse,
    )


def run_checks(result: LaplaceResult) -> None:
    """Fail fast if numerical quality is unexpectedly poor."""
    if not result.converged:
        raise AssertionError("SOR solver did not converge within max_iter.")
    if result.residual_inf > 4.0e-5:
        raise AssertionError(f"Residual infinity norm too high: {result.residual_inf:.3e}")
    if result.max_abs_error > 2.0e-4:
        raise AssertionError(f"Max abs error too high: {result.max_abs_error:.3e}")
    if result.rmse > 8.0e-5:
        raise AssertionError(f"RMSE too high: {result.rmse:.3e}")


def sample_profile_table(result: LaplaceResult) -> pd.DataFrame:
    """Sample centerline profile y=0.5 for quick terminal inspection."""
    ny = result.potential_num.shape[0]
    j_mid = ny // 2
    y_mid = result.y[j_mid]

    # Pick evenly spaced x indices across the domain.
    sample_idx = np.linspace(0, len(result.x) - 1, 9, dtype=int)
    rows = {
        "x": result.x[sample_idx],
        "y": np.full(sample_idx.shape, y_mid, dtype=float),
        "V_num": result.potential_num[j_mid, sample_idx],
        "V_exact": result.potential_exact[j_mid, sample_idx],
    }
    table = pd.DataFrame(rows)
    table["abs_error"] = np.abs(table["V_num"] - table["V_exact"])
    return table


def main() -> None:
    result = solve_laplace_mvp()
    run_checks(result)

    print("Laplace Equation MVP report")
    print(f"Grid size                        : {len(result.x)} x {len(result.y)}")
    print(f"Converged                        : {result.converged}")
    print(f"Iterations                       : {result.iterations}")
    print(f"Max update (last iter)           : {result.max_update:.3e}")
    print(f"Residual inf-norm                : {result.residual_inf:.3e}")
    print(f"Max abs error vs exact (interior): {result.max_abs_error:.3e}")
    print(f"RMSE vs exact (interior)         : {result.rmse:.3e}")

    print("\nCenterline profile sample (y=0.5):")
    profile = sample_profile_table(result)
    print(profile.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
