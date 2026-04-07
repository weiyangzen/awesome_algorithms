"""Smith normal form invariants for a small computational topology MVP.

This script demonstrates how Smith-invariant factors of boundary matrices
support homology-group computation on simple CW complexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
from typing import Iterable, Sequence

import numpy as np


Matrix = list[list[int]]


@dataclass(frozen=True)
class SurfaceExample:
    """A tiny CW-complex surface model with one 0-cell and one 2-cell."""

    name: str
    boundary2: Matrix  # Matrix of d2: C2 -> C1, shape (rank(C1), rank(C2))


@dataclass(frozen=True)
class HomologySummary:
    """Computed homology information for H0, H1, H2."""

    name: str
    smith_diag: list[int]
    h0: str
    h1: str
    h2: str


def to_int_matrix(matrix: Sequence[Sequence[int]]) -> Matrix:
    """Validate rectangularity and cast all entries to Python ints."""
    if not matrix:
        return []
    width = len(matrix[0])
    if width == 0:
        return [list(map(int, row)) for row in matrix]
    out: Matrix = []
    for row in matrix:
        if len(row) != width:
            raise ValueError("Matrix rows must have equal length.")
        out.append([int(v) for v in row])
    return out


def matrix_shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def rank_over_rationals(matrix: Matrix) -> int:
    """Compute rank over Q by Gaussian elimination with Fraction arithmetic."""
    m, n = matrix_shape(matrix)
    if m == 0 or n == 0:
        return 0

    a = [[Fraction(v) for v in row] for row in matrix]
    rank = 0
    pivot_row = 0

    for col in range(n):
        candidate = None
        for r in range(pivot_row, m):
            if a[r][col] != 0:
                candidate = r
                break
        if candidate is None:
            continue

        if candidate != pivot_row:
            a[pivot_row], a[candidate] = a[candidate], a[pivot_row]

        pivot = a[pivot_row][col]
        for c in range(col, n):
            a[pivot_row][c] /= pivot

        for r in range(m):
            if r == pivot_row:
                continue
            factor = a[r][col]
            if factor == 0:
                continue
            for c in range(col, n):
                a[r][c] -= factor * a[pivot_row][c]

        rank += 1
        pivot_row += 1
        if pivot_row == m:
            break

    return rank


def det_bareiss(square: Matrix) -> int:
    """Exact determinant using Bareiss fraction-free elimination."""
    n = len(square)
    if n == 0:
        return 1
    if any(len(row) != n for row in square):
        raise ValueError("det_bareiss expects a square matrix.")

    a = [row[:] for row in square]
    sign = 1
    prev = 1

    for k in range(n - 1):
        if a[k][k] == 0:
            swap_row = None
            for r in range(k + 1, n):
                if a[r][k] != 0:
                    swap_row = r
                    break
            if swap_row is None:
                return 0
            a[k], a[swap_row] = a[swap_row], a[k]
            sign *= -1

        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                a[i][j] = numerator // prev

        prev = pivot
        for i in range(k + 1, n):
            a[i][k] = 0

    return sign * a[n - 1][n - 1]


def k_by_k_submatrix(matrix: Matrix, row_ids: Iterable[int], col_ids: Iterable[int]) -> Matrix:
    rows = list(row_ids)
    cols = list(col_ids)
    return [[matrix[r][c] for c in cols] for r in rows]


def determinantal_divisor(matrix: Matrix, k: int) -> int:
    """Compute Delta_k = gcd of all k x k minors' determinants."""
    if k == 0:
        return 1

    m, n = matrix_shape(matrix)
    if k > m or k > n:
        return 0

    g = 0
    for row_ids in itertools.combinations(range(m), k):
        for col_ids in itertools.combinations(range(n), k):
            minor = k_by_k_submatrix(matrix, row_ids, col_ids)
            det = abs(det_bareiss(minor))
            g = math.gcd(g, det)
            if g == 1:
                return 1
    return g


def smith_invariant_factors(matrix: Matrix) -> list[int]:
    """Return non-zero Smith invariant factors d1|d2|...|dr for an integer matrix."""
    matrix = to_int_matrix(matrix)
    rank = rank_over_rationals(matrix)
    if rank == 0:
        return []

    deltas = [1]
    for k in range(1, rank + 1):
        delta_k = determinantal_divisor(matrix, k)
        if delta_k == 0:
            raise ValueError("Unexpected zero determinantal divisor for non-zero rank.")
        if delta_k % deltas[-1] != 0:
            raise ValueError("Determinantal divisors do not satisfy divisibility chain.")
        deltas.append(delta_k)

    factors = [deltas[i] // deltas[i - 1] for i in range(1, len(deltas))]
    for i in range(len(factors) - 1):
        if factors[i + 1] % factors[i] != 0:
            raise ValueError("Invariant factors must satisfy d_i | d_{i+1}.")
    return factors


def cokernel_decomposition(matrix: Matrix) -> tuple[int, list[int], list[int]]:
    """For Z^m / im(A), return (free_rank, torsion_factors, full_smith_diag)."""
    m, _ = matrix_shape(matrix)
    diag = smith_invariant_factors(matrix)
    rank = len(diag)
    free_rank = m - rank
    torsion = [d for d in diag if d > 1]
    return free_rank, torsion, diag


def group_to_string(free_rank: int, torsion: Sequence[int]) -> str:
    parts: list[str] = []
    if free_rank == 1:
        parts.append("Z")
    elif free_rank > 1:
        parts.append(f"Z^{free_rank}")

    for d in torsion:
        parts.append(f"Z/{d}Z")

    if not parts:
        return "0"
    return " ⊕ ".join(parts)


def analyze_surface(surface: SurfaceExample) -> HomologySummary:
    """Analyze H0, H1, H2 for examples with one 0-cell and d1 = 0."""
    b2 = to_int_matrix(surface.boundary2)
    n1, n2 = matrix_shape(b2)

    # For one-vertex CW surfaces here: d1 = 0, so H1 = coker(d2), H0 = Z.
    h1_free_rank, h1_torsion, smith_diag = cokernel_decomposition(b2)
    h1 = group_to_string(h1_free_rank, h1_torsion)

    rank_d2 = len(smith_diag)
    h2_free_rank = n2 - rank_d2  # H2 = ker(d2) because C3 = 0.
    h2 = group_to_string(h2_free_rank, [])

    h0 = "Z"
    return HomologySummary(name=surface.name, smith_diag=smith_diag, h0=h0, h1=h1, h2=h2)


def run_self_checks(results: Sequence[HomologySummary]) -> None:
    expected = {
        "Torus T^2": ("Z", "Z^2", "Z"),
        "Real Projective Plane RP^2": ("Z", "Z/2Z", "0"),
        "Klein Bottle K": ("Z", "Z ⊕ Z/2Z", "0"),
    }

    for item in results:
        exp = expected[item.name]
        got = (item.h0, item.h1, item.h2)
        if got != exp:
            raise AssertionError(f"{item.name} homology mismatch: expected {exp}, got {got}")


def main() -> None:
    examples = [
        SurfaceExample(name="Torus T^2", boundary2=[[0], [0]]),
        SurfaceExample(name="Real Projective Plane RP^2", boundary2=[[2]]),
        SurfaceExample(name="Klein Bottle K", boundary2=[[2], [0]]),
    ]

    results = [analyze_surface(surface) for surface in examples]
    run_self_checks(results)

    print("Smith invariants for d2 and resulting homology groups:\n")
    for surface, summary in zip(examples, results):
        b2_array = np.array(surface.boundary2, dtype=int)
        diag_str = summary.smith_diag if summary.smith_diag else [0]
        print(f"[{summary.name}]")
        print(f"d2 matrix:\n{b2_array}")
        print(f"non-zero Smith factors of d2: {diag_str}")
        print(f"H0 = {summary.h0}")
        print(f"H1 = {summary.h1}")
        print(f"H2 = {summary.h2}")
        print()

    print("All checks passed.")


if __name__ == "__main__":
    main()
