"""Durand-Kerner method MVP.

Non-interactive demo that computes all polynomial roots using
Weierstrass-Durand-Kerner simultaneous iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DKReport:
    name: str
    coefficients: np.ndarray
    dk_roots: np.ndarray
    numpy_roots: np.ndarray
    iterations: int
    converged: bool
    max_step: float
    max_residual: float
    max_error_vs_numpy: float


def normalize_polynomial(coeffs: Sequence[complex], zero_tol: float = 1e-15) -> np.ndarray:
    """Trim leading zeros and normalize to monic form."""
    arr = np.array(coeffs, dtype=np.complex128).copy()
    if arr.ndim != 1:
        arr = arr.ravel()

    nz = np.where(np.abs(arr) > zero_tol)[0]
    if len(nz) == 0:
        raise ValueError("All coefficients are zero.")

    arr = arr[nz[0] :]
    if len(arr) < 2:
        raise ValueError("Need at least degree-1 polynomial.")

    lead = arr[0]
    if abs(lead) <= zero_tol:
        raise ValueError("Leading coefficient must be non-zero.")

    return arr / lead


def cauchy_radius(monic_coeffs: np.ndarray) -> float:
    """Cauchy bound radius for roots of a monic polynomial."""
    if len(monic_coeffs) < 2:
        return 1.0
    return 1.0 + float(np.max(np.abs(monic_coeffs[1:])))


def initial_guesses(monic_coeffs: np.ndarray) -> np.ndarray:
    """Initialize roots on a circle with a fixed phase shift."""
    n = len(monic_coeffs) - 1
    radius = cauchy_radius(monic_coeffs)
    angles = (2.0 * np.pi * np.arange(n) / n) + (np.pi / (2.0 * n))
    guesses = radius * np.exp(1j * angles)

    # Small deterministic shift helps avoid exact symmetry-induced collisions.
    guesses += complex(0.137, -0.071)
    return guesses.astype(np.complex128)


def _pairwise_denominators(roots: np.ndarray) -> np.ndarray:
    diff = roots[:, None] - roots[None, :]
    np.fill_diagonal(diff, 1.0 + 0.0j)
    return np.prod(diff, axis=1)


def durand_kerner(
    coeffs: Sequence[complex],
    tol_step: float = 1e-12,
    tol_residual: float = 1e-12,
    max_iter: int = 300,
    collision_eps: float = 1e-14,
) -> tuple[np.ndarray, int, bool, float, float]:
    """Find all roots via synchronous Durand-Kerner iteration."""
    poly = normalize_polynomial(coeffs)
    roots = initial_guesses(poly)

    max_step = float("inf")
    max_residual = float("inf")

    for it in range(1, max_iter + 1):
        pvals = np.polyval(poly, roots)
        denoms = _pairwise_denominators(roots)

        bad = np.abs(denoms) < collision_eps
        if np.any(bad):
            # Deterministic tiny perturbation to separate collided estimates.
            idx = np.where(bad)[0]
            for rank, i in enumerate(idx, start=1):
                roots[i] += (1e-8 * rank) + 1j * (5e-9 * rank)

            pvals = np.polyval(poly, roots)
            denoms = _pairwise_denominators(roots)
            if np.any(np.abs(denoms) < collision_eps):
                max_step = float("inf")
                max_residual = float(np.max(np.abs(pvals)))
                return roots, it, False, max_step, max_residual

        delta = pvals / denoms
        roots_next = roots - delta

        max_step = float(np.max(np.abs(delta)))
        max_residual = float(np.max(np.abs(np.polyval(poly, roots_next))))

        roots = roots_next

        if max_step < tol_step and max_residual < tol_residual:
            return roots, it, True, max_step, max_residual

    return roots, max_iter, False, max_step, max_residual


def _sort_roots(roots: Sequence[complex]) -> np.ndarray:
    return np.array(
        sorted(roots, key=lambda z: (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))),
        dtype=np.complex128,
    )


def _max_matching_error(a: np.ndarray, b: np.ndarray) -> float:
    """Greedy nearest-neighbor matching for small demo cases."""
    remaining = list(b)
    worst = 0.0
    for root in a:
        idx = int(np.argmin([abs(root - r) for r in remaining]))
        err = float(abs(root - remaining[idx]))
        if err > worst:
            worst = err
        remaining.pop(idx)
    return worst


def run_case(name: str, coeffs: Sequence[complex]) -> DKReport:
    coeff_arr = np.array(coeffs, dtype=np.complex128)
    dk_roots, iters, converged, max_step, max_residual = durand_kerner(coeff_arr)

    numpy_roots = np.roots(coeff_arr)
    dk_sorted = _sort_roots(dk_roots)
    np_sorted = _sort_roots(numpy_roots)

    max_error_vs_numpy = _max_matching_error(dk_sorted, np_sorted)

    return DKReport(
        name=name,
        coefficients=coeff_arr,
        dk_roots=dk_sorted,
        numpy_roots=np_sorted,
        iterations=iters,
        converged=converged,
        max_step=max_step,
        max_residual=max_residual,
        max_error_vs_numpy=max_error_vs_numpy,
    )


def fmt_complex(z: complex) -> str:
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
            "name": "Case-2: roots=[0.5, -1.2, 2±1j, -0.25±0.75j]",
            "coeffs": np.poly(
                np.array(
                    [0.5, -1.2, 2.0 + 1.0j, 2.0 - 1.0j, -0.25 + 0.75j, -0.25 - 0.75j],
                    dtype=np.complex128,
                )
            ),
        },
    ]

    print("Durand-Kerner Method Demo")
    print("=" * 80)

    for case in cases:
        report = run_case(case["name"], case["coeffs"])
        print(report.name)
        print(f"  degree: {len(report.coefficients) - 1}")
        print(f"  converged: {report.converged}")
        print(f"  iterations: {report.iterations}")
        print(f"  max step: {report.max_step:.3e}")
        print(f"  max |P(r)|: {report.max_residual:.3e}")
        print(f"  max match error vs numpy.roots: {report.max_error_vs_numpy:.3e}")

        print("  roots (Durand-Kerner):")
        for root in report.dk_roots:
            print(f"    {fmt_complex(root)}")

        print("  roots (NumPy):")
        for root in report.numpy_roots:
            print(f"    {fmt_complex(root)}")

        print("-" * 80)


if __name__ == "__main__":
    main()
