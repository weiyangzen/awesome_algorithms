"""Minimal runnable MVP for Remez algorithm (polynomial minimax approximation).

This script builds a degree-n minimax polynomial approximation on [a, b]
for a target scalar function using a classic exchange-style Remez iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class RemezResult:
    coeffs_asc: Array
    ripple: float
    reference_points: Array
    history: list[tuple[int, float, float, float, float]]
    max_abs_error: float


def validate_inputs(
    degree: int,
    a: float,
    b: float,
    max_iter: int,
    grid_size: int,
    tol: float,
) -> None:
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if not np.isfinite([a, b]).all() or b <= a:
        raise ValueError("interval must satisfy finite a < b")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if grid_size < max(1000, 50 * (degree + 2)):
        raise ValueError("grid_size is too small for stable extrema detection")
    if tol <= 0:
        raise ValueError("tol must be > 0")


def chebyshev_reference_points(a: float, b: float, m: int) -> Array:
    k = np.arange(m, dtype=float)
    x = np.cos(np.pi * k / (m - 1))
    mapped = 0.5 * (a + b) + 0.5 * (b - a) * x
    return np.sort(mapped)


def eval_poly_asc(coeffs_asc: Array, x: Array) -> Array:
    y = np.zeros_like(x, dtype=float)
    for c in coeffs_asc[::-1]:
        y = y * x + c
    return y


def build_remez_system(x_ref: Array) -> Array:
    m = x_ref.size
    degree = m - 2
    a_mat = np.empty((m, m), dtype=float)
    for k in range(degree + 1):
        a_mat[:, k] = x_ref ** k
    a_mat[:, degree + 1] = (-1.0) ** np.arange(m)
    return a_mat


def solve_remez_step(f: Callable[[Array], Array], x_ref: Array) -> tuple[Array, float]:
    a_mat = build_remez_system(x_ref)
    rhs = np.asarray(f(x_ref), dtype=float)
    try:
        sol = np.linalg.solve(a_mat, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Remez linear system is singular/ill-conditioned") from exc
    coeffs = sol[:-1]
    ripple = float(sol[-1])
    return coeffs, ripple


def local_abs_extrema_indices(err: Array) -> Array:
    n = err.size
    idx: list[int] = [0]
    abs_err = np.abs(err)
    for i in range(1, n - 1):
        if abs_err[i] >= abs_err[i - 1] and abs_err[i] >= abs_err[i + 1]:
            idx.append(i)
    idx.append(n - 1)
    return np.asarray(sorted(set(idx)), dtype=int)


def signed_bit(v: float) -> int:
    return 1 if v >= 0.0 else -1


def enforce_alternation(candidates: Array, err: Array) -> Array:
    if candidates.size == 0:
        return candidates
    selected: list[int] = [int(candidates[0])]
    for idx in candidates[1:]:
        idx_int = int(idx)
        prev = selected[-1]
        if signed_bit(err[idx_int]) != signed_bit(err[prev]):
            selected.append(idx_int)
        else:
            if abs(err[idx_int]) > abs(err[prev]):
                selected[-1] = idx_int
    return np.asarray(selected, dtype=int)


def best_alternating_window(indices: Array, err: Array, target_count: int) -> Array:
    if indices.size <= target_count:
        return indices
    best_start = 0
    best_score = -np.inf
    abs_err = np.abs(err)
    for start in range(indices.size - target_count + 1):
        window = indices[start : start + target_count]
        score = float(np.min(abs_err[window]))
        if score > best_score:
            best_score = score
            best_start = start
    return indices[best_start : best_start + target_count]


def select_reference_points(grid: Array, err: Array, count: int) -> Array:
    candidates = local_abs_extrema_indices(err)
    alternating = enforce_alternation(candidates, err)

    if alternating.size >= count:
        chosen = best_alternating_window(alternating, err, count)
        return np.sort(grid[chosen])

    # Fallback: add largest absolute-error points when extrema are insufficient.
    order = np.argsort(np.abs(err))[::-1]
    chosen_set = {int(i) for i in alternating.tolist()}
    for idx in order:
        chosen_set.add(int(idx))
        if len(chosen_set) >= count:
            break
    chosen = np.asarray(sorted(chosen_set), dtype=int)
    if chosen.size > count:
        chosen = chosen[:count]
    return np.sort(grid[chosen])


def remez_minimax(
    f: Callable[[Array], Array],
    degree: int,
    a: float,
    b: float,
    *,
    max_iter: int = 25,
    grid_size: int = 20001,
    tol: float = 1e-10,
) -> RemezResult:
    validate_inputs(degree, a, b, max_iter, grid_size, tol)

    m = degree + 2
    x_ref = chebyshev_reference_points(a, b, m)
    grid = np.linspace(a, b, grid_size)
    history: list[tuple[int, float, float, float, float]] = []

    for it in range(1, max_iter + 1):
        coeffs, ripple = solve_remez_step(f, x_ref)
        err = np.asarray(f(grid), dtype=float) - eval_poly_asc(coeffs, grid)
        max_abs = float(np.max(np.abs(err)))
        ripple_abs = abs(ripple)
        ripple_gap = abs(max_abs - ripple_abs) / max(1.0, max_abs)

        new_ref = select_reference_points(grid, err, m)
        move = float(np.max(np.abs(new_ref - x_ref)))
        history.append((it, max_abs, ripple_abs, ripple_gap, move))

        x_ref = new_ref
        if ripple_gap < tol and move < 1e-12 * (b - a):
            break

    final_coeffs, final_ripple = solve_remez_step(f, x_ref)
    final_err = np.asarray(f(grid), dtype=float) - eval_poly_asc(final_coeffs, grid)
    final_max_abs = float(np.max(np.abs(final_err)))

    return RemezResult(
        coeffs_asc=final_coeffs,
        ripple=float(final_ripple),
        reference_points=x_ref,
        history=history,
        max_abs_error=final_max_abs,
    )


def least_squares_poly(
    f: Callable[[Array], Array],
    degree: int,
    a: float,
    b: float,
    sample_size: int = 4097,
) -> Array:
    x = np.linspace(a, b, sample_size)
    y = np.asarray(f(x), dtype=float)
    coeffs_desc = np.polyfit(x, y, degree)
    return coeffs_desc[::-1]


def main() -> None:
    np.set_printoptions(precision=12, suppress=True)

    degree = 5
    a, b = -1.0, 1.0

    def f(x: Array) -> Array:
        return np.exp(x)

    remez = remez_minimax(f, degree, a, b, max_iter=30, grid_size=30001, tol=1e-9)

    ls_coeffs = least_squares_poly(f, degree, a, b)
    dense = np.linspace(a, b, 100001)

    remez_err = np.asarray(f(dense), dtype=float) - eval_poly_asc(remez.coeffs_asc, dense)
    ls_err = np.asarray(f(dense), dtype=float) - eval_poly_asc(ls_coeffs, dense)

    remez_sup = float(np.max(np.abs(remez_err)))
    ls_sup = float(np.max(np.abs(ls_err)))

    ref_err = np.asarray(f(remez.reference_points), dtype=float) - eval_poly_asc(
        remez.coeffs_asc, remez.reference_points
    )

    print("Remez algorithm MVP: minimax polynomial approximation")
    print(f"target function: exp(x), interval=[{a}, {b}], degree={degree}")
    print()

    print("[1] Iteration history (iter, max|err|, |ripple|, gap, max_ref_move)")
    for row in remez.history:
        print(
            f"  iter={row[0]:2d}, max|err|={row[1]:.6e}, |ripple|={row[2]:.6e}, "
            f"gap={row[3]:.3e}, move={row[4]:.3e}"
        )
    print()

    print("[2] Final polynomial coefficients (ascending powers)")
    print("  p(x) = c0 + c1*x + ... + c5*x^5")
    print(" ", remez.coeffs_asc)
    print(f"  ripple E (signed): {remez.ripple:.12e}")
    print(f"  max|f-p| on dense grid: {remez_sup:.12e}")
    print()

    print("[3] Equioscillation check at final reference points")
    print("  x_ref:", remez.reference_points)
    print("  err(x_ref):", ref_err)
    print("  sign(err):", np.sign(ref_err).astype(int))
    print(
        "  |err| range at ref points: "
        f"[{np.min(np.abs(ref_err)):.6e}, {np.max(np.abs(ref_err)):.6e}]"
    )
    print()

    print("[4] Comparison with degree-5 least-squares polynomial (L2 fit)")
    print(f"  Remez sup-norm error      : {remez_sup:.12e}")
    print(f"  Least-squares sup-norm err: {ls_sup:.12e}")


if __name__ == "__main__":
    main()
