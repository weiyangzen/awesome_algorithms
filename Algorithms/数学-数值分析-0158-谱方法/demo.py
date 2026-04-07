"""Fourier spectral-method MVP for 1D periodic Poisson equation.

Problem:
    -u''(x) = f(x), x in [0, L), periodic boundary.

This script builds an analytic test case, solves it with FFT-based spectral
inversion, and reports convergence on a sequence of grids.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

TEST_A = 1.1  # Controls smoothness in the analytic benchmark solution.


def periodic_grid(n: int, length: float = 2.0 * math.pi) -> np.ndarray:
    """Create N periodic grid points on [0, L)."""
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be positive finite, got {length}")
    return np.linspace(0.0, length, n, endpoint=False, dtype=float)


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Analytic periodic smooth solution used for verification."""
    d = TEST_A - np.cos(x)
    return 1.0 / d


def forcing_term(x: np.ndarray) -> np.ndarray:
    """Right-hand side f(x) = -u''(x) for u(x) = 1 / (a - cos x)."""
    s = np.sin(x)
    c = np.cos(x)
    d = TEST_A - c
    return c / (d * d) - 2.0 * (s * s) / (d * d * d)


def spectral_poisson_periodic(fx: np.ndarray, length: float = 2.0 * math.pi) -> np.ndarray:
    """Solve -u'' = f on periodic grid with zero-mean gauge using FFT."""
    if fx.ndim != 1:
        raise ValueError(f"fx must be 1D, got ndim={fx.ndim}")
    n = fx.size
    if n < 4:
        raise ValueError(f"fx length must be >= 4, got {n}")
    if not np.all(np.isfinite(fx)):
        raise ValueError("fx contains non-finite values")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be positive finite, got {length}")

    # Frequency grid in angular wavenumbers.
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
    k2 = k * k

    f_hat = np.fft.fft(fx)
    u_hat = np.zeros_like(f_hat, dtype=np.complex128)

    nonzero = k2 > 0.0
    u_hat[nonzero] = f_hat[nonzero] / k2[nonzero]
    u_hat[0] = 0.0  # Fix additive constant by enforcing zero mean.

    u = np.fft.ifft(u_hat).real
    return u


def error_metrics(u_num: np.ndarray, u_ref: np.ndarray) -> tuple[float, float]:
    """Return (L2_error, Linf_error)."""
    if u_num.shape != u_ref.shape:
        raise ValueError("u_num and u_ref must have the same shape")
    diff = u_num - u_ref
    l2_err = float(np.sqrt(np.mean(diff * diff)))
    linf_err = float(np.max(np.abs(diff)))
    return l2_err, linf_err


def safe_log2_rate(prev_err: float, curr_err: float) -> float:
    """Compute log2(prev/curr), handling zero or invalid values."""
    # Near machine precision, rate is dominated by floating-point noise.
    if prev_err <= 1.0e-14 or curr_err <= 1.0e-14:
        return float("nan")
    return float(np.log2(prev_err / curr_err))


def run_convergence_study(
    grid_sizes: Iterable[int],
    length: float = 2.0 * math.pi,
) -> list[dict[str, float]]:
    """Run spectral solve on multiple grid sizes and collect errors."""
    records: list[dict[str, float]] = []

    prev_l2 = float("nan")
    prev_linf = float("nan")

    for n in grid_sizes:
        x = periodic_grid(n, length)
        f = forcing_term(x)

        u_num = spectral_poisson_periodic(f, length)
        u_ref = exact_solution(x)
        u_ref -= float(np.mean(u_ref))  # Match zero-mean gauge in solver.

        l2_err, linf_err = error_metrics(u_num, u_ref)

        record = {
            "N": float(n),
            "L2_error": l2_err,
            "Linf_error": linf_err,
            "rate_L2": safe_log2_rate(prev_l2, l2_err),
            "rate_Linf": safe_log2_rate(prev_linf, linf_err),
        }
        records.append(record)

        prev_l2 = l2_err
        prev_linf = linf_err

    return records


def print_results_table(records: list[dict[str, float]]) -> None:
    """Pretty-print convergence table."""
    header = (
        f"{'N':>6} | {'L2_error':>12} | {'Linf_error':>12} | "
        f"{'rate(L2)':>9} | {'rate(Linf)':>10}"
    )
    print(header)
    print("-" * len(header))

    for rec in records:
        n = int(rec["N"])
        l2e = rec["L2_error"]
        lie = rec["Linf_error"]
        r2 = rec["rate_L2"]
        ri = rec["rate_Linf"]

        r2_str = f"{r2:9.3f}" if np.isfinite(r2) else f"{'-':>9}"
        ri_str = f"{ri:10.3f}" if np.isfinite(ri) else f"{'-':>10}"

        print(f"{n:6d} | {l2e:12.3e} | {lie:12.3e} | {r2_str} | {ri_str}")


def main() -> None:
    grid_sizes = [16, 32, 64, 128, 256]
    pass_threshold = 1.0e-9

    records = run_convergence_study(grid_sizes)
    print("Fourier Spectral Method for 1D Periodic Poisson: -u'' = f")
    print_results_table(records)

    finest = records[-1]
    max_error_all = max(r["Linf_error"] for r in records)
    passed = finest["Linf_error"] < pass_threshold

    print("\nSummary")
    print(f"  max_error_all = {max_error_all:.3e}")
    print(f"  finest_grid_Linf_error = {finest['Linf_error']:.3e}")
    print(f"  pass_threshold = {pass_threshold:.1e}")
    print(f"  pass = {passed}")


if __name__ == "__main__":
    main()
