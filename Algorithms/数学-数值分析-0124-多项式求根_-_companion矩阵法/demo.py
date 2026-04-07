"""Companion-matrix polynomial root finder (minimal runnable MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Case:
    """Single demo case."""

    name: str
    coeffs: np.ndarray
    expected_roots: np.ndarray | None = None


def normalize_to_monic(coeffs: Iterable[complex], tol: float = 0.0) -> np.ndarray:
    """Trim leading zeros and normalize polynomial to monic form."""
    arr = np.asarray(list(coeffs), dtype=np.complex128).flatten()
    if arr.size == 0:
        raise ValueError("Coefficient array must not be empty.")

    idx = 0
    while idx < arr.size and abs(arr[idx]) <= tol:
        idx += 1

    if idx == arr.size:
        raise ValueError("All coefficients are zero; polynomial is undefined.")

    trimmed = arr[idx:]
    if trimmed.size < 2:
        raise ValueError("Constant polynomial has no finite roots.")

    lead = trimmed[0]
    if abs(lead) <= tol:
        raise ValueError("Leading coefficient must be non-zero.")

    return trimmed / lead


def build_companion_matrix(coeffs: Iterable[complex]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build companion matrix for p(x)=a0*x^n + ... + an.
    Returns (companion_matrix, monic_coeffs).
    """
    monic = normalize_to_monic(coeffs)
    n = monic.size - 1

    companion = np.zeros((n, n), dtype=np.complex128)
    companion[0, :] = -monic[1:]
    if n > 1:
        companion[1:, :-1] = np.eye(n - 1, dtype=np.complex128)
    return companion, monic


def roots_via_companion(coeffs: Iterable[complex]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute roots using companion matrix eigenvalues."""
    companion, monic = build_companion_matrix(coeffs)
    roots = np.linalg.eigvals(companion)
    return roots, companion, monic


def evaluate_polynomial_horner(coeffs: Iterable[complex], x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial at vector x via Horner's method."""
    c = np.asarray(list(coeffs), dtype=np.complex128)
    y = np.zeros_like(x, dtype=np.complex128)
    for a in c:
        y = y * x + a
    return y


def sort_complex(values: np.ndarray) -> np.ndarray:
    """Deterministic sort by (real, imag) for presentation/comparison."""
    return np.asarray(
        sorted(np.asarray(values, dtype=np.complex128), key=lambda z: (float(np.real(z)), float(np.imag(z)))),
        dtype=np.complex128,
    )


def max_residual(coeffs: np.ndarray, roots: np.ndarray) -> float:
    vals = evaluate_polynomial_horner(coeffs, roots)
    return float(np.max(np.abs(vals)))


def max_root_error(found: np.ndarray, expected: np.ndarray) -> float:
    found_s = sort_complex(found)
    expected_s = sort_complex(expected)
    if found_s.size != expected_s.size:
        raise ValueError("found and expected roots have different lengths.")
    return float(np.max(np.abs(found_s - expected_s)))


def format_complex(z: complex) -> str:
    """Compact complex number formatter."""
    real = float(np.real(z))
    imag = float(np.imag(z))
    if abs(real) < 5e-13:
        real = 0.0
    if abs(imag) < 5e-13:
        imag = 0.0
    sign = "+" if imag >= 0 else "-"
    return f"{real: .10f}{sign}{abs(imag):.10f}j"


def build_demo_cases() -> list[Case]:
    """Construct deterministic demo cases."""
    roots1 = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    roots2 = np.array([1.0 + 2.0j, 1.0 - 2.0j, -0.5], dtype=np.complex128)
    roots3 = np.array([1.0, 1.0, 1.0], dtype=np.complex128)

    rng = np.random.default_rng(2026)
    z = rng.uniform(-1.8, 1.8) + 1j * rng.uniform(0.3, 1.5)
    roots4 = np.array([z, np.conjugate(z), rng.uniform(-2.0, 2.0), -1.0, 0.25], dtype=np.complex128)

    return [
        Case("distinct_real_degree3", np.poly(roots1).astype(np.complex128), roots1),
        Case("complex_pair_degree3", np.poly(roots2).astype(np.complex128), roots2),
        Case("triple_root_degree3", np.poly(roots3).astype(np.complex128), roots3),
        Case("mixed_degree5", np.poly(roots4).astype(np.complex128), roots4),
    ]


def run_case(case: Case) -> tuple[float, float | None]:
    roots, companion, monic = roots_via_companion(case.coeffs)
    residual = max_residual(case.coeffs, roots)
    tol = 1e-8 * (1.0 + float(np.max(np.abs(case.coeffs))))
    status = "PASS" if residual <= tol else "FAIL"

    print(f"=== {case.name} ===")
    print(f"degree: {companion.shape[0]}")
    print(f"monic coeffs: {np.array2string(monic, precision=6, suppress_small=False)}")
    print(f"max_residual: {residual:.3e}  (tol={tol:.3e})  status={status}")

    sorted_roots = sort_complex(roots)
    print("roots via companion:")
    for i, root in enumerate(sorted_roots, start=1):
        print(f"  r{i}: {format_complex(root)}")

    root_err = None
    if case.expected_roots is not None:
        root_err = max_root_error(roots, case.expected_roots)
        print(f"max_root_error_vs_expected: {root_err:.3e}")

    print()
    if residual > tol:
        raise AssertionError(f"{case.name}: residual {residual:.3e} exceeded tolerance {tol:.3e}")
    return residual, root_err


def main() -> None:
    print("Companion matrix method demo")
    print("Polynomial roots computed as eigenvalues of companion matrices.\n")

    all_residuals: list[float] = []
    all_root_errors: list[float] = []

    for case in build_demo_cases():
        residual, root_err = run_case(case)
        all_residuals.append(residual)
        if root_err is not None:
            all_root_errors.append(root_err)

    print("=== summary ===")
    print(f"cases: {len(all_residuals)}")
    print(f"worst residual: {max(all_residuals):.3e}")
    if all_root_errors:
        print(f"worst root error (against known roots): {max(all_root_errors):.3e}")
    print("done.")


if __name__ == "__main__":
    main()
