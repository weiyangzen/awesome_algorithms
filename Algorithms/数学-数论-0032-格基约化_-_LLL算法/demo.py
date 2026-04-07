"""LLL lattice basis reduction minimal runnable MVP.

This demo implements LLL directly (not via third-party black-box APIs):
- Gram-Schmidt orthogonalization
- Size reduction
- Lovasz condition checks and swaps

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

IntMatrix = NDArray[np.int64]
FloatMatrix = NDArray[np.float64]


@dataclass
class LLLStats:
    iterations: int = 0
    size_reductions: int = 0
    swaps: int = 0
    lovasz_checks: int = 0


def nearest_integer(x: float) -> int:
    """Round to nearest integer with symmetric tie handling."""
    if x >= 0:
        return int(np.floor(x + 0.5))
    return int(np.ceil(x - 0.5))


def gram_schmidt(basis: IntMatrix) -> Tuple[FloatMatrix, FloatMatrix, NDArray[np.float64]]:
    """Compute Gram-Schmidt data for row-basis vectors.

    Returns:
        mu: Coefficients matrix where mu[i, j] (j < i) is projection factor.
        b_star: Orthogonalized vectors.
        b_star_norm2: Squared norms of b_star rows.
    """
    n, m = basis.shape
    mu = np.zeros((n, n), dtype=np.float64)
    b_star = np.zeros((n, m), dtype=np.float64)
    b_star_norm2 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        v = basis[i].astype(np.float64).copy()
        for j in range(i):
            if b_star_norm2[j] <= 1e-18:
                mu[i, j] = 0.0
                continue
            mu[i, j] = float(np.dot(basis[i], b_star[j]) / b_star_norm2[j])
            v -= mu[i, j] * b_star[j]
        b_star[i] = v
        b_star_norm2[i] = float(np.dot(v, v))

    return mu, b_star, b_star_norm2


def check_lll_conditions(basis: IntMatrix, delta: float, tol: float = 1e-10) -> tuple[bool, bool]:
    """Check size-reduction and Lovasz conditions for a basis."""
    mu, _, b_star_norm2 = gram_schmidt(basis)
    n = basis.shape[0]

    size_ok = True
    for i in range(n):
        for j in range(i):
            if abs(mu[i, j]) > 0.5 + tol:
                size_ok = False
                break
        if not size_ok:
            break

    lovasz_ok = True
    for k in range(1, n):
        lhs = b_star_norm2[k]
        rhs = (delta - mu[k, k - 1] ** 2) * b_star_norm2[k - 1]
        if lhs + tol < rhs:
            lovasz_ok = False
            break

    return size_ok, lovasz_ok


def hadamard_ratio(basis: IntMatrix) -> float:
    """Return Hadamard ratio in (0, 1] for full-rank row basis."""
    n = basis.shape[0]
    basis_f = basis.astype(np.float64)
    gram = basis_f @ basis_f.T
    det_gram = float(np.linalg.det(gram))

    if det_gram < 0 and abs(det_gram) < 1e-10:
        det_gram = 0.0
    if det_gram <= 0:
        return 0.0

    volume = np.sqrt(det_gram)
    norms = np.linalg.norm(basis_f, axis=1)
    denom = float(np.prod(norms))
    if denom <= 0:
        return 0.0

    return float((volume / denom) ** (1.0 / n))


def shortest_vector_norm(basis: IntMatrix) -> float:
    """Return shortest row vector Euclidean norm."""
    norms = np.linalg.norm(basis.astype(np.float64), axis=1)
    return float(np.min(norms))


def lll_reduce(basis: IntMatrix, delta: float = 0.75, max_iterations: int = 20_000) -> tuple[IntMatrix, LLLStats]:
    """Reduce integer row-basis using a simple LLL loop."""
    if basis.ndim != 2:
        raise ValueError("basis must be a 2D array")
    if not (0.25 < delta < 1.0):
        raise ValueError("delta must satisfy 1/4 < delta < 1")

    b = np.array(basis, dtype=np.int64, copy=True)
    n, m = b.shape

    if n == 0:
        return b, LLLStats()
    if n > m:
        raise ValueError("row basis cannot be full-rank when n > m")

    rank = np.linalg.matrix_rank(b.astype(np.float64))
    if rank < n:
        raise ValueError("basis rows must be linearly independent (full row rank)")

    stats = LLLStats()
    k = 1
    mu, _, b_star_norm2 = gram_schmidt(b)

    while k < n:
        if stats.iterations >= max_iterations:
            raise RuntimeError("max_iterations exceeded; possible numerical instability")

        # Size reduction: push mu[k, j] into [-1/2, 1/2].
        for j in range(k - 1, -1, -1):
            q = nearest_integer(float(mu[k, j]))
            if q != 0:
                b[k] = b[k] - q * b[j]
                stats.size_reductions += 1

        mu, _, b_star_norm2 = gram_schmidt(b)

        stats.lovasz_checks += 1
        lhs = b_star_norm2[k]
        rhs = (delta - mu[k, k - 1] ** 2) * b_star_norm2[k - 1]

        if lhs + 1e-12 >= rhs:
            k += 1
        else:
            b[[k, k - 1]] = b[[k - 1, k]]
            stats.swaps += 1
            mu, _, b_star_norm2 = gram_schmidt(b)
            k = max(1, k - 1)

        stats.iterations += 1

    return b, stats


def run_case(name: str, basis: IntMatrix, delta: float) -> None:
    """Run and print one deterministic LLL demo case."""
    print(f"\n=== {name} ===")
    print("Input basis:")
    print(basis)

    before_short = shortest_vector_norm(basis)
    before_hadamard = hadamard_ratio(basis)

    t0 = time.perf_counter()
    reduced, stats = lll_reduce(basis, delta=delta)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    after_short = shortest_vector_norm(reduced)
    after_hadamard = hadamard_ratio(reduced)
    size_ok, lovasz_ok = check_lll_conditions(reduced, delta=delta)

    print("Reduced basis:")
    print(reduced)
    print(
        "Metrics: "
        f"shortest_norm {before_short:.6f} -> {after_short:.6f}, "
        f"hadamard_ratio {before_hadamard:.6f} -> {after_hadamard:.6f}"
    )
    print(
        "Checks: "
        f"size_reduction_ok={size_ok}, lovasz_ok={lovasz_ok}, "
        f"iterations={stats.iterations}, size_reductions={stats.size_reductions}, "
        f"swaps={stats.swaps}, lovasz_checks={stats.lovasz_checks}, "
        f"time_ms={elapsed_ms:.3f}"
    )


def main() -> None:
    """Run deterministic demo cases without interactive input."""
    delta = 0.75
    cases: list[tuple[str, IntMatrix]] = [
        (
            "Case 1: 2D small basis",
            np.array(
                [
                    [105, 821],
                    [37, 287],
                ],
                dtype=np.int64,
            ),
        ),
        (
            "Case 2: 3D textbook-style basis",
            np.array(
                [
                    [1, 1, 1],
                    [-1, 0, 2],
                    [3, 5, 6],
                ],
                dtype=np.int64,
            ),
        ),
        (
            "Case 3: 4D mixed-sign basis",
            np.array(
                [
                    [4, 1, 3, -1],
                    [2, 1, -3, 4],
                    [1, 0, -2, 7],
                    [6, 2, 9, -5],
                ],
                dtype=np.int64,
            ),
        ),
    ]

    print("=== LLL Lattice Basis Reduction Demo ===")
    print(f"delta = {delta}")

    for name, basis in cases:
        run_case(name, basis, delta)


if __name__ == "__main__":
    main()
