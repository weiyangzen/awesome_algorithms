"""Laguerre's method MVP.

Non-interactive demo that finds all roots of test polynomials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RootSolveReport:
    name: str
    coefficients: np.ndarray
    laguerre_roots: np.ndarray
    numpy_roots: np.ndarray
    total_iterations: int
    max_residual: float
    max_match_error_vs_numpy: float


def eval_poly_and_derivatives(coeffs: np.ndarray, x: complex) -> tuple[complex, complex, complex]:
    """Evaluate P(x), P'(x), P''(x) in one Horner-style scan.

    coeffs are in descending power order.
    """
    p = complex(coeffs[0])
    dp = 0j
    ddp = 0j

    for c in coeffs[1:]:
        ddp = ddp * x + 2.0 * dp
        dp = dp * x + p
        p = p * x + c

    return p, dp, ddp


def laguerre_single_root(
    coeffs: np.ndarray,
    x0: complex,
    tol: float = 1e-12,
    max_iter: int = 120,
) -> tuple[complex, int, bool]:
    """Find one root of polynomial defined by coeffs using Laguerre iterations."""
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("Polynomial degree must be at least 1.")

    x = complex(x0)
    tiny = 1e-18

    for it in range(1, max_iter + 1):
        p, dp, ddp = eval_poly_and_derivatives(coeffs, x)

        if abs(p) < tol:
            return x, it, True

        if abs(p) < tiny:
            x += complex(tol, tol)
            continue

        g = dp / p
        h = g * g - ddp / p
        rad = np.sqrt((n - 1) * (n * h - g * g))

        denom_plus = g + rad
        denom_minus = g - rad
        denom = denom_plus if abs(denom_plus) >= abs(denom_minus) else denom_minus

        if abs(denom) < tiny:
            x += complex(1e-6, -1e-6)
            continue

        step = n / denom
        x_next = x - step

        if abs(x_next - x) < tol:
            return x_next, it, True

        x = x_next

    return x, max_iter, False


def deflate_polynomial(coeffs: np.ndarray, root: complex) -> tuple[np.ndarray, complex]:
    """Synthetic division by (x - root)."""
    degree = len(coeffs) - 1
    if degree < 1:
        raise ValueError("Cannot deflate a constant polynomial.")

    new_coeffs = np.empty(degree, dtype=np.complex128)
    b = coeffs[0]
    new_coeffs[0] = b

    for i in range(1, degree):
        b = coeffs[i] + b * root
        new_coeffs[i] = b

    remainder = coeffs[-1] + b * root
    return new_coeffs, remainder


def _sort_roots(roots: Sequence[complex]) -> np.ndarray:
    return np.array(
        sorted(roots, key=lambda z: (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))),
        dtype=np.complex128,
    )


def _max_matching_error(a: np.ndarray, b: np.ndarray) -> float:
    """Greedy nearest-neighbor matching error; adequate for small demo sets."""
    remaining = list(b)
    worst = 0.0
    for root in a:
        idx = int(np.argmin([abs(root - r) for r in remaining]))
        err = abs(root - remaining[idx])
        worst = max(worst, float(err))
        remaining.pop(idx)
    return worst


def find_all_roots_laguerre(
    coeffs: Sequence[complex],
    tol: float = 1e-12,
    max_iter: int = 120,
) -> tuple[np.ndarray, int]:
    """Find all polynomial roots by repeated Laguerre solve + deflation."""
    work = np.array(coeffs, dtype=np.complex128)
    if len(work) < 2:
        raise ValueError("Need at least degree-1 polynomial.")
    if abs(work[0]) == 0:
        raise ValueError("Leading coefficient must be non-zero.")

    roots: list[complex] = []
    total_iterations = 0
    rng = np.random.default_rng(2026)

    while len(work) > 2:
        degree = len(work) - 1
        cauchy_radius = 1.0 + float(np.max(np.abs(work[1:] / work[0])))

        converged = False
        best_candidate = 0j
        best_value = float("inf")

        for attempt in range(16):
            angle = 2.0 * np.pi * attempt / 16.0
            jitter = complex(rng.normal(scale=0.05), rng.normal(scale=0.05))
            x0 = cauchy_radius * np.exp(1j * angle) + jitter

            candidate, iters, ok = laguerre_single_root(work, x0, tol=tol, max_iter=max_iter)
            total_iterations += iters

            value = abs(np.polyval(work, candidate))
            if value < best_value:
                best_value = float(value)
                best_candidate = candidate

            if ok:
                converged = True
                best_candidate = candidate
                break

        if not converged:
            raise RuntimeError(
                f"Laguerre iteration failed to converge for degree-{degree} polynomial; "
                f"best residual={best_value:.3e}."
            )

        # One polishing pass on the current polynomial often improves deflation quality.
        polished, iters, _ = laguerre_single_root(work, best_candidate, tol=tol * 0.1, max_iter=max_iter)
        total_iterations += iters

        roots.append(polished)
        work, remainder = deflate_polynomial(work, polished)

        if abs(remainder) > 1e-6:
            # Keep going for MVP, but surface if the deflation residual is unexpectedly large.
            print(f"[warn] deflation remainder is {abs(remainder):.3e}")

    # Final linear root.
    last_root = -work[1] / work[0]
    roots.append(last_root)

    return np.array(roots, dtype=np.complex128), total_iterations


def run_case(name: str, coeffs: Sequence[complex]) -> RootSolveReport:
    coeff_arr = np.array(coeffs, dtype=np.complex128)
    laguerre_roots, total_iters = find_all_roots_laguerre(coeff_arr)
    numpy_roots = np.roots(coeff_arr)

    laguerre_roots = _sort_roots(laguerre_roots)
    numpy_roots = _sort_roots(numpy_roots)

    residuals = [abs(np.polyval(coeff_arr, root)) for root in laguerre_roots]
    max_residual = float(max(residuals))
    max_match_error_vs_numpy = _max_matching_error(laguerre_roots, numpy_roots)

    return RootSolveReport(
        name=name,
        coefficients=coeff_arr,
        laguerre_roots=laguerre_roots,
        numpy_roots=numpy_roots,
        total_iterations=total_iters,
        max_residual=max_residual,
        max_match_error_vs_numpy=max_match_error_vs_numpy,
    )


def format_complex(z: complex) -> str:
    real = float(np.real(z))
    imag = float(np.imag(z))
    if abs(real) < 1e-13:
        real = 0.0
    if abs(imag) < 1e-13:
        imag = 0.0
    return f"{real:+.10f}{imag:+.10f}j"


def main() -> None:
    cases = [
        {
            "name": "Case-1: (x-1)(x+2)(x^2+1)",
            "coeffs": np.array([1.0, 1.0, -1.0, 1.0, -2.0], dtype=np.complex128),
        },
        {
            "name": "Case-2: roots=[0.5, -1.2, 2+1j, 2-1j]",
            "coeffs": np.poly(np.array([0.5, -1.2, 2.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex128)),
        },
    ]

    print("Laguerre's Method Demo")
    print("=" * 78)

    for case in cases:
        report = run_case(case["name"], case["coeffs"])
        print(report.name)
        print(f"  degree: {len(report.coefficients) - 1}")
        print(f"  total iterations: {report.total_iterations}")
        print(f"  max |P(r)|: {report.max_residual:.3e}")
        print(f"  max match error vs numpy.roots: {report.max_match_error_vs_numpy:.3e}")

        print("  roots (Laguerre):")
        for root in report.laguerre_roots:
            print(f"    {format_complex(root)}")

        print("  roots (NumPy):")
        for root in report.numpy_roots:
            print(f"    {format_complex(root)}")

        print("-" * 78)


if __name__ == "__main__":
    main()
