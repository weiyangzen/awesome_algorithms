"""MATH-0160 Fourier spectral method MVP.

This demo covers three classical periodic-domain tasks:
1) spectral first derivative,
2) periodic Poisson solve (-u_xx = f, zero-mean solution),
3) 1D heat equation evolution in Fourier space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralGrid:
    """Uniform periodic grid and matching Fourier wavenumbers."""

    N: int
    L: float
    x: np.ndarray
    k: np.ndarray


def build_grid(N: int, L: float = 2.0 * np.pi) -> SpectralGrid:
    """Build periodic grid [0, L) and FFT-compatible wavenumbers."""
    if N <= 0:
        raise ValueError("N must be positive.")
    if L <= 0:
        raise ValueError("L must be positive.")

    dx = L / N
    x = np.arange(N, dtype=np.float64) * dx
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return SpectralGrid(N=N, L=L, x=x, k=k)


def ensure_grid_compatible(u: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != grid.N:
        raise ValueError(f"Expected 1D array of length {grid.N}, got shape={arr.shape}")
    return arr


def spectral_derivative(u: np.ndarray, grid: SpectralGrid, order: int = 1) -> np.ndarray:
    """Compute periodic derivative using Fourier spectral differentiation."""
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    if order == 0:
        return ensure_grid_compatible(u, grid).copy()

    u_real = ensure_grid_compatible(u, grid)
    u_hat = np.fft.fft(u_real)
    factor = (1j * grid.k) ** order
    du_hat = factor * u_hat
    return np.fft.ifft(du_hat).real


def solve_periodic_poisson(f: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    """Solve -u_xx = f on periodic domain with zero-mean gauge."""
    f_real = ensure_grid_compatible(f, grid)
    f_zero_mean = f_real - np.mean(f_real)

    f_hat = np.fft.fft(f_zero_mean)
    u_hat = np.zeros_like(f_hat, dtype=np.complex128)

    k2 = grid.k * grid.k
    nonzero = k2 > 0.0
    u_hat[nonzero] = f_hat[nonzero] / k2[nonzero]
    u_hat[~nonzero] = 0.0 + 0.0j  # Fix the additive constant (mean mode).

    return np.fft.ifft(u_hat).real


def solve_heat_equation(
    u0: np.ndarray,
    grid: SpectralGrid,
    nu: float,
    dt: float,
    t_end: float,
) -> tuple[np.ndarray, int]:
    """Solve u_t = nu * u_xx by exact modal decay per time step."""
    if nu < 0:
        raise ValueError("nu must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_end < 0:
        raise ValueError("t_end must be non-negative.")

    steps = int(round(t_end / dt))
    if not np.isclose(steps * dt, t_end, rtol=0.0, atol=1e-12):
        raise ValueError("t_end must be an integer multiple of dt for this MVP.")

    u_hat = np.fft.fft(ensure_grid_compatible(u0, grid))
    decay = np.exp(-nu * (grid.k ** 2) * dt)
    for _ in range(steps):
        u_hat *= decay
    u_t = np.fft.ifft(u_hat).real
    return u_t, steps


def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def run_derivative_case(grid: SpectralGrid) -> dict[str, float | int]:
    x = grid.x
    u = np.sin(3.0 * x) + 0.2 * np.cos(5.0 * x)
    exact = 3.0 * np.cos(3.0 * x) - 1.0 * np.sin(5.0 * x)
    num = spectral_derivative(u, grid, order=1)
    return {
        "N": grid.N,
        "l2_error": l2_error(num, exact),
        "linf_error": float(np.max(np.abs(num - exact))),
    }


def run_poisson_case(grid: SpectralGrid) -> dict[str, float | int]:
    x = grid.x
    u_true = np.sin(2.0 * x) + 0.3 * np.cos(4.0 * x)
    f = 4.0 * np.sin(2.0 * x) + 4.8 * np.cos(4.0 * x)  # f = -u_xx
    u_num = solve_periodic_poisson(f, grid)
    return {
        "N": grid.N,
        "l2_error": l2_error(u_num, u_true),
        "linf_error": float(np.max(np.abs(u_num - u_true))),
    }


def run_heat_case(grid: SpectralGrid) -> dict[str, float | int]:
    x = grid.x
    nu = 0.07
    dt = 0.02
    t_end = 1.2

    u0 = np.sin(3.0 * x) + 0.5 * np.cos(6.0 * x)
    u_num, steps = solve_heat_equation(u0, grid, nu=nu, dt=dt, t_end=t_end)
    u_exact = (
        np.exp(-nu * 9.0 * t_end) * np.sin(3.0 * x)
        + 0.5 * np.exp(-nu * 36.0 * t_end) * np.cos(6.0 * x)
    )
    return {
        "N": grid.N,
        "steps": steps,
        "t_end": t_end,
        "l2_error": l2_error(u_num, u_exact),
        "linf_error": float(np.max(np.abs(u_num - u_exact))),
    }


def main() -> None:
    grid = build_grid(N=128, L=2.0 * np.pi)

    derivative_metrics = run_derivative_case(grid)
    poisson_metrics = run_poisson_case(grid)
    heat_metrics = run_heat_case(grid)

    print("Fourier Spectral Method MVP (MATH-0160)")
    print("=" * 72)
    print("Case 1: Spectral derivative")
    print(derivative_metrics)
    print("Case 2: Periodic Poisson")
    print(poisson_metrics)
    print("Case 3: Heat equation evolution")
    print(heat_metrics)

    if derivative_metrics["l2_error"] >= 1e-10 or derivative_metrics["linf_error"] >= 1e-10:
        raise AssertionError("Spectral derivative accuracy check failed.")
    if poisson_metrics["l2_error"] >= 1e-10 or poisson_metrics["linf_error"] >= 1e-10:
        raise AssertionError("Periodic Poisson accuracy check failed.")
    if heat_metrics["l2_error"] >= 1e-10 or heat_metrics["linf_error"] >= 1e-10:
        raise AssertionError("Heat equation accuracy check failed.")

    print("All Fourier spectral checks passed.")


if __name__ == "__main__":
    main()
