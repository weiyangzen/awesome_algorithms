"""Minimal runnable MVP for Smith normal form over integers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SmithResult:
    """Container for Smith normal form outputs."""

    D: np.ndarray
    U: np.ndarray
    V: np.ndarray


def to_object_int_matrix(data: Iterable[Iterable[int]]) -> np.ndarray:
    """Convert input into a 2D integer matrix with Python-int precision."""
    arr = np.asarray(data, dtype=object)
    if arr.ndim != 2:
        raise ValueError("input must be a 2D matrix")

    out = np.empty(arr.shape, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if isinstance(value, (np.integer, int)):
                out[i, j] = int(value)
                continue
            if isinstance(value, (np.floating, float)) and float(value).is_integer():
                out[i, j] = int(value)
                continue
            raise ValueError(f"non-integer entry at ({i}, {j}): {value!r}")
    return out


def identity_object(n: int) -> np.ndarray:
    """Build an n x n identity matrix with dtype=object."""
    mat = np.zeros((n, n), dtype=object)
    for i in range(n):
        mat[i, i] = 1
    return mat


def swap_rows(mat: np.ndarray, i: int, j: int) -> None:
    if i == j:
        return
    tmp = mat[i, :].copy()
    mat[i, :] = mat[j, :]
    mat[j, :] = tmp


def swap_cols(mat: np.ndarray, i: int, j: int) -> None:
    if i == j:
        return
    tmp = mat[:, i].copy()
    mat[:, i] = mat[:, j]
    mat[:, j] = tmp


def add_row_multiple(mat: np.ndarray, src: int, dst: int, k: int) -> None:
    if k == 0:
        return
    cols = mat.shape[1]
    for c in range(cols):
        mat[dst, c] = int(mat[dst, c]) + int(k) * int(mat[src, c])


def add_col_multiple(mat: np.ndarray, src: int, dst: int, k: int) -> None:
    if k == 0:
        return
    rows = mat.shape[0]
    for r in range(rows):
        mat[r, dst] = int(mat[r, dst]) + int(k) * int(mat[r, src])


def scale_row(mat: np.ndarray, row: int, k: int) -> None:
    cols = mat.shape[1]
    for c in range(cols):
        mat[row, c] = int(k) * int(mat[row, c])


def op_swap_rows(A: np.ndarray, U: np.ndarray, i: int, j: int) -> None:
    swap_rows(A, i, j)
    swap_rows(U, i, j)


def op_swap_cols(A: np.ndarray, V: np.ndarray, i: int, j: int) -> None:
    swap_cols(A, i, j)
    swap_cols(V, i, j)


def op_add_row(A: np.ndarray, U: np.ndarray, src: int, dst: int, k: int) -> None:
    add_row_multiple(A, src, dst, k)
    add_row_multiple(U, src, dst, k)


def op_add_col(A: np.ndarray, V: np.ndarray, src: int, dst: int, k: int) -> None:
    add_col_multiple(A, src, dst, k)
    add_col_multiple(V, src, dst, k)


def op_scale_row(A: np.ndarray, U: np.ndarray, row: int, k: int) -> None:
    scale_row(A, row, k)
    scale_row(U, row, k)


def find_min_abs_nonzero(A: np.ndarray, start: int) -> tuple[int, int] | None:
    """Find non-zero entry with minimal absolute value in A[start:, start:]."""
    m, n = A.shape
    best: tuple[int, int] | None = None
    best_abs: int | None = None
    for i in range(start, m):
        for j in range(start, n):
            v = int(A[i, j])
            if v == 0:
                continue
            av = abs(v)
            if best_abs is None or av < best_abs:
                best_abs = av
                best = (i, j)
    return best


def reduce_pivot_block(A: np.ndarray, U: np.ndarray, V: np.ndarray, k: int) -> None:
    """Reduce around pivot (k, k) until row/col are clear and divisibility holds."""
    m, n = A.shape

    while True:
        if int(A[k, k]) < 0:
            op_scale_row(A, U, k, -1)

        # Euclidean elimination for pivot column.
        for i in range(k + 1, m):
            while int(A[i, k]) != 0:
                pivot = int(A[k, k])
                q = int(A[i, k]) // pivot
                op_add_row(A, U, k, i, -q)
                if int(A[i, k]) != 0 and abs(int(A[i, k])) < abs(int(A[k, k])):
                    op_swap_rows(A, U, i, k)
                    if int(A[k, k]) < 0:
                        op_scale_row(A, U, k, -1)

        # Euclidean elimination for pivot row.
        for j in range(k + 1, n):
            while int(A[k, j]) != 0:
                pivot = int(A[k, k])
                q = int(A[k, j]) // pivot
                op_add_col(A, V, k, j, -q)
                if int(A[k, j]) != 0 and abs(int(A[k, j])) < abs(int(A[k, k])):
                    op_swap_cols(A, V, j, k)
                    if int(A[k, k]) < 0:
                        op_scale_row(A, U, k, -1)

        pivot = int(A[k, k])
        offender: tuple[int, int] | None = None
        for i in range(k + 1, m):
            for j in range(k + 1, n):
                if int(A[i, j]) % pivot != 0:
                    offender = (i, j)
                    break
            if offender is not None:
                break

        if offender is None:
            return

        # Inject a non-divisible trailing entry into pivot column,
        # then loop again to re-run Euclidean reduction.
        _, j_bad = offender
        op_add_col(A, V, j_bad, k, 1)


def smith_normal_form_int(data: Iterable[Iterable[int]]) -> SmithResult:
    """Compute Smith normal form D with unimodular U, V such that U*A*V = D."""
    A = to_object_int_matrix(data)
    A0_shape = A.shape
    m, n = A0_shape

    U = identity_object(m)
    V = identity_object(n)

    k = 0
    while k < m and k < n:
        pos = find_min_abs_nonzero(A, k)
        if pos is None:
            break

        i, j = pos
        op_swap_rows(A, U, k, i)
        op_swap_cols(A, V, k, j)

        if int(A[k, k]) == 0:
            # Defensive guard; with min-abs nonzero pivot this should not happen.
            k += 1
            continue

        reduce_pivot_block(A, U, V, k)

        if int(A[k, k]) < 0:
            op_scale_row(A, U, k, -1)

        k += 1

    return SmithResult(D=A, U=U, V=V)


def bareiss_determinant(mat: np.ndarray) -> int:
    """Exact determinant for square integer matrix via Bareiss algorithm."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("determinant is defined only for square matrices")

    n = mat.shape[0]
    if n == 0:
        return 1

    A = [[int(mat[i, j]) for j in range(n)] for i in range(n)]
    sign = 1
    denom = 1

    for k in range(n - 1):
        if A[k][k] == 0:
            pivot_row = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    pivot_row = i
                    break
            if pivot_row is None:
                return 0
            A[k], A[pivot_row] = A[pivot_row], A[k]
            sign *= -1

        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (pivot * A[i][j] - A[i][k] * A[k][j]) // denom

        for i in range(k + 1, n):
            A[i][k] = 0
        denom = pivot

    return sign * A[n - 1][n - 1]


def check_diagonal_and_divisibility(D: np.ndarray) -> tuple[bool, list[int]]:
    """Check Smith-form diagonal shape and divisibility chain."""
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            if i != j and int(D[i, j]) != 0:
                return False, []

    diag_len = min(m, n)
    diag_values = [abs(int(D[t, t])) for t in range(diag_len)]

    nonzero = [d for d in diag_values if d != 0]
    if any(d <= 0 for d in nonzero):
        return False, nonzero

    for i in range(len(nonzero) - 1):
        if nonzero[i + 1] % nonzero[i] != 0:
            return False, nonzero

    # Zero diagonals must be trailing.
    seen_zero = False
    for d in diag_values:
        if d == 0:
            seen_zero = True
        elif seen_zero:
            return False, nonzero

    return True, nonzero


def matrices_equal(A: np.ndarray, B: np.ndarray) -> bool:
    if A.shape != B.shape:
        return False
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            if int(A[i, j]) != int(B[i, j]):
                return False
    return True


def matrix_to_pretty_str(mat: np.ndarray) -> str:
    """Render small matrix in aligned integer format."""
    rows, cols = mat.shape
    if rows == 0 or cols == 0:
        return "[]"
    str_rows = [[str(int(mat[i, j])) for j in range(cols)] for i in range(rows)]
    widths = [max(len(str_rows[i][j]) for i in range(rows)) for j in range(cols)]
    lines: list[str] = []
    for i in range(rows):
        items = [str_rows[i][j].rjust(widths[j]) for j in range(cols)]
        lines.append("[ " + " ".join(items) + " ]")
    return "\n".join(lines)


def run_case(name: str, data: Iterable[Iterable[int]]) -> None:
    A0 = to_object_int_matrix(data)
    result = smith_normal_form_int(A0)

    reconstructed = result.U @ A0 @ result.V
    if not matrices_equal(reconstructed, result.D):
        raise AssertionError("U @ A @ V != D")

    ok_smith, factors = check_diagonal_and_divisibility(result.D)
    if not ok_smith:
        raise AssertionError("D is not in Smith normal form")

    det_u = bareiss_determinant(result.U)
    det_v = bareiss_determinant(result.V)
    if abs(det_u) != 1:
        raise AssertionError(f"U is not unimodular, det(U)={det_u}")
    if abs(det_v) != 1:
        raise AssertionError(f"V is not unimodular, det(V)={det_v}")

    print("=" * 92)
    print(name)
    print("Input A:")
    print(matrix_to_pretty_str(A0))
    print("\nSmith D:")
    print(matrix_to_pretty_str(result.D))
    print(f"Invariant factors (non-zero diagonal): {factors}")
    print(f"det(U)={det_u}, det(V)={det_v}")
    print("Verification: U*A*V == D and divisibility chain holds.")


def main() -> None:
    cases = [
        (
            "Case 1: 2x2 full-rank matrix",
            [
                [4, 6],
                [3, 9],
            ],
        ),
        (
            "Case 2: 2x3 rectangular matrix",
            [
                [2, 4, 4],
                [6, 6, 12],
            ],
        ),
        (
            "Case 3: rank-deficient 3x3 matrix",
            [
                [0, 0, 0],
                [0, 5, 10],
                [0, 15, 20],
            ],
        ),
    ]

    for name, mat in cases:
        run_case(name=name, data=mat)

    print("\nAll Smith normal form checks passed.")


if __name__ == "__main__":
    main()
