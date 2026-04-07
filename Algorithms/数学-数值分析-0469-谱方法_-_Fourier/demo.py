"""Minimal runnable MVP for Fourier spectral method on periodic Poisson equation.

Problem:
    -u''(x) = f(x),  x in [0, 2*pi), periodic boundary.

Method:
    Use FFT to move to Fourier space, solve mode-wise
        u_hat(k) = f_hat(k) / k^2,  k != 0,
    and fix the zero mode by enforcing mean(u)=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralCaseResult:
    nx: int
    l2_error: float
    linf_error: float
    residual_l2: float
    residual_linf: float


def ensure_finite_1d(name: str, arr: np.ndarray) -> np.ndarray:
    """Ensure input is a finite 1D array."""
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def build_periodic_grid(nx: int, length: float = 2.0 * np.pi) -> tuple[np.ndarray, float]:
    """Build periodic grid x_j in [0, length), with spacing dx."""
    if nx < 4:
        raise ValueError("nx must be >= 4")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a positive finite number")
    x = np.linspace(0.0, length, nx, endpoint=False)
    dx = length / nx
    return x, dx


def exact_solution_raw(x: np.ndarray) -> np.ndarray:
    """A smooth periodic exact solution before fixing the mean."""
    exp_part = np.exp(np.sin(x))
    trig_part = 0.3 * np.sin(2.0 * x) - 0.2 * np.cos(3.0 * x)
    return exp_part + trig_part


def forcing_from_exact(x: np.ndarray) -> np.ndarray:
    """Construct f(x) = -u''(x) from exact_solution_raw analytically."""
    exp_part = np.exp(np.sin(x))

    # For u1 = exp(sin x): u1'' = (cos^2 x - sin x) * exp(sin x)
    # Therefore -u1'' = (sin x - cos^2 x) * exp(sin x)
    forcing_exp = (np.sin(x) - np.cos(x) ** 2) * exp_part

    # For u2 = 0.3 sin(2x): -u2'' = 1.2 sin(2x)
    forcing_sin2 = 1.2 * np.sin(2.0 * x)

    # For u3 = -0.2 cos(3x): -u3'' = -1.8 cos(3x)
    forcing_cos3 = -1.8 * np.cos(3.0 * x)

    return forcing_exp + forcing_sin2 + forcing_cos3


def fourier_wavenumbers(nx: int, dx: float) -> np.ndarray:
    """Return angular wavenumbers k for FFT on periodic grid."""
    if nx <= 0 or dx <= 0.0:
        raise ValueError("nx and dx must be positive")
    return 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)


def solve_poisson_fourier(f_values: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    """Solve -u''=f with periodic BC and mean(u)=0 using Fourier spectral method."""
    f = ensure_finite_1d("f_values", np.asarray(f_values, dtype=float))
    nx = f.size
    _x, dx = build_periodic_grid(nx=nx, length=length)

    # Periodic solvability requires mean(f)=0; remove numerical drift explicitly.
    f_zero_mean = f - np.mean(f)

    k = fourier_wavenumbers(nx=nx, dx=dx)
    f_hat = np.fft.fft(f_zero_mean)

    u_hat = np.zeros(nx, dtype=complex)
    nonzero_mode = np.abs(k) > 1e-14
    u_hat[nonzero_mode] = f_hat[nonzero_mode] / (k[nonzero_mode] ** 2)

    # Gauge choice: mean(u)=0.
    u_hat[~nonzero_mode] = 0.0

    u = np.fft.ifft(u_hat).real
    return ensure_finite_1d("u_solution", u)


def poisson_residual(u_values: np.ndarray, f_values: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    """Compute residual r = (-u'') - f in physical space."""
    u = ensure_finite_1d("u_values", np.asarray(u_values, dtype=float))
    f = ensure_finite_1d("f_values", np.asarray(f_values, dtype=float))
    if u.size != f.size:
        raise ValueError("u_values and f_values must have the same length")

    nx = u.size
    _x, dx = build_periodic_grid(nx=nx, length=length)
    k = fourier_wavenumbers(nx=nx, dx=dx)

    # In Fourier space: -u'' <-> k^2 * u_hat.
    lhs = np.fft.ifft((k**2) * np.fft.fft(u)).real

    # Match the solvability projection used by the solver.
    f_zero_mean = f - np.mean(f)
    residual = lhs - f_zero_mean
    return ensure_finite_1d("residual", residual)


def l2_norm(arr: np.ndarray) -> float:
    """Discrete L2 norm with 1/sqrt(N) normalization."""
    return float(np.sqrt(np.mean(np.asarray(arr, dtype=float) ** 2)))


def linf_norm(arr: np.ndarray) -> float:
    """Discrete Linf norm."""
    return float(np.max(np.abs(np.asarray(arr, dtype=float))))


def run_case(nx: int) -> SpectralCaseResult:
    """Run one resolution case and collect error/residual metrics."""
    x, _dx = build_periodic_grid(nx=nx)

    u_exact = exact_solution_raw(x)
    u_exact = u_exact - np.mean(u_exact)  # align with mean(u)=0 gauge

    f = forcing_from_exact(x)
    u_num = solve_poisson_fourier(f_values=f)

    err = u_num - u_exact
    res = poisson_residual(u_values=u_num, f_values=f)

    return SpectralCaseResult(
        nx=nx,
        l2_error=l2_norm(err),
        linf_error=linf_norm(err),
        residual_l2=l2_norm(res),
        residual_linf=linf_norm(res),
    )


def effective_order(err_coarse: float, err_fine: float, n_coarse: int, n_fine: int) -> float:
    """Estimate p from err ~ N^{-p} using two resolutions."""
    if err_coarse <= 0.0 or err_fine <= 0.0:
        raise ValueError("errors must be positive")
    if n_fine <= n_coarse:
        raise ValueError("n_fine must be greater than n_coarse")
    return float(np.log(err_coarse / err_fine) / np.log(n_fine / n_coarse))


def main() -> None:
    resolutions = [8, 12, 16, 20, 24]

    print("=== Fourier Spectral MVP: Periodic Poisson Solver ===")
    print("Equation: -u''(x)=f(x), x in [0,2pi), periodic BC, gauge mean(u)=0")

    results = [run_case(nx) for nx in resolutions]

    print("\nConvergence table")
    print("nx | L2_error      | Linf_error    | residual_L2   | residual_Linf")
    print("---+---------------+---------------+---------------+---------------")
    for r in results:
        print(
            f"{r.nx:2d} | {r.l2_error:13.6e} | {r.linf_error:13.6e} |"
            f" {r.residual_l2:13.6e} | {r.residual_linf:13.6e}"
        )

    print("\nEffective orders p from consecutive resolutions (L2 error)")
    for i in range(len(results) - 1):
        r0 = results[i]
        r1 = results[i + 1]
        p = effective_order(r0.l2_error, r1.l2_error, r0.nx, r1.nx)
        print(f"{r0.nx:2d}->{r1.nx:2d}: p={p:8.3f}")

    first_err = results[0].l2_error
    last_err = results[-1].l2_error
    max_residual = max(r.residual_l2 for r in results)

    if not (last_err < first_err * 1e-5):
        raise AssertionError("Spectral convergence check failed: error did not drop enough")
    if max_residual > 1e-8:
        raise AssertionError(f"Residual too large: {max_residual:.3e}")

    print("\nAll Fourier spectral MVP checks passed.")


if __name__ == "__main__":
    main()
