"""Minimal runnable MVP for Bessel functions in mathematical physics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import special


EPS = 1e-12


@dataclass
class BesselMVPResult:
    x_grid: np.ndarray
    max_order: int
    j_values: np.ndarray
    ref_values: np.ndarray
    abs_error: np.ndarray
    recurrence_residual_max: np.ndarray
    ode_residual_max: dict[int, float]
    orthogonality_matrix: np.ndarray
    orthogonality_diag_ref: np.ndarray


def bessel_j_series_seed(
    order: int,
    x: np.ndarray,
    tol: float = 1e-15,
    max_terms: int = 800,
) -> np.ndarray:
    """Compute J_order(x) for integer order >= 0 using power-series recurrence.

    J_n(x) = sum_{m=0}^inf (-1)^m / (m! (m+n)!) * (x/2)^(2m+n)

    The implementation updates each term from the previous one:
      term_{m+1} = term_m * (-(x/2)^2) / ((m+1)(m+1+n))
    """
    if order < 0:
        raise ValueError("order must be >= 0")

    x = np.asarray(x, dtype=float)
    half_sq = (x * 0.5) ** 2

    term = (x * 0.5) ** order / math.factorial(order)
    out = term.copy()

    for m in range(max_terms - 1):
        term = term * (-half_sq) / ((m + 1) * (m + 1 + order))
        out += term
        if float(np.max(np.abs(term))) < tol:
            break

    return out


def bessel_j_upward(max_order: int, x: np.ndarray) -> np.ndarray:
    """Compute J_0..J_max_order on x using series seeds + upward recurrence."""
    if max_order < 0:
        raise ValueError("max_order must be >= 0")

    x = np.asarray(x, dtype=float)
    j = np.zeros((max_order + 1, x.size), dtype=float)

    j[0, :] = bessel_j_series_seed(0, x)
    if max_order >= 1:
        j[1, :] = bessel_j_series_seed(1, x)

    nonzero = np.abs(x) > EPS
    near_zero = ~nonzero

    # Enforce exact limits at x=0 for integer orders.
    if np.any(near_zero):
        j[0, near_zero] = 1.0
        if max_order >= 1:
            j[1:, near_zero] = 0.0

    for n in range(1, max_order):
        j[n + 1, nonzero] = (2.0 * n / x[nonzero]) * j[n, nonzero] - j[n - 1, nonzero]
        if np.any(near_zero):
            j[n + 1, near_zero] = 0.0

    # Forward recurrence loses accuracy for high order at small x;
    # patch that regime with direct series evaluation.
    small_x = np.abs(x) < max(8.0, float(max_order) + 2.0)
    if np.any(small_x):
        for n in range(max_order + 1):
            j[n, small_x] = bessel_j_series_seed(n, x[small_x])

    return j


def recurrence_residuals(j_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return max residual of J_{n+1} - 2n/x J_n + J_{n-1} for n=1..N-1."""
    max_order = j_values.shape[0] - 1
    residuals = np.zeros(max_order + 1, dtype=float)
    nonzero = np.abs(x) > 1e-8

    for n in range(1, max_order):
        lhs = j_values[n + 1, nonzero]
        rhs = (2.0 * n / x[nonzero]) * j_values[n, nonzero] - j_values[n - 1, nonzero]
        residuals[n] = float(np.max(np.abs(lhs - rhs)))

    return residuals


def ode_residual(x: np.ndarray, y: np.ndarray, order: int) -> float:
    """Max absolute residual of Bessel ODE on a safe interior interval.

    ODE: x^2 y'' + x y' + (x^2 - n^2) y = 0
    """
    dy = np.gradient(y, x, edge_order=2)
    ddy = np.gradient(dy, x, edge_order=2)
    res = x**2 * ddy + x * dy + (x**2 - order**2) * y

    # Avoid singular point and boundary finite-difference noise.
    mask = (x >= 0.25) & (x <= 19.75)
    return float(np.max(np.abs(res[mask])))


def orthogonality_check(order: int, mode_count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Check integral orthogonality on [0,1] using zeros of J_order.

    I_mk = integral_0^1 r J_n(alpha_m r) J_n(alpha_k r) dr
    where alpha_m are zeros of J_n.
    """
    alphas = special.jn_zeros(order, mode_count)
    r = np.linspace(0.0, 1.0, 5001)

    basis = []
    for alpha in alphas:
        values = bessel_j_series_seed(order, alpha * r)
        basis.append(values)
    basis = np.array(basis)

    gram = np.zeros((mode_count, mode_count), dtype=float)
    for i in range(mode_count):
        for k in range(mode_count):
            integrand = r * basis[i] * basis[k]
            gram[i, k] = float(np.trapezoid(integrand, r))

    diag_ref = 0.5 * special.jv(order + 1, alphas) ** 2
    return gram, diag_ref


def build_error_table(abs_error: np.ndarray, recurrence_residual_max: np.ndarray) -> pd.DataFrame:
    """Construct per-order quality table."""
    max_order = abs_error.shape[0] - 1
    rows = []
    for n in range(max_order + 1):
        row = {
            "order": n,
            "max_abs_error_vs_scipy": float(np.max(abs_error[n])),
            "rms_error_vs_scipy": float(np.sqrt(np.mean(abs_error[n] ** 2))),
            "max_recurrence_residual": float(recurrence_residual_max[n]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def run_mvp(max_order: int = 6) -> BesselMVPResult:
    """Run the complete Bessel-function MVP workflow."""
    x_grid = np.linspace(0.0, 20.0, 2001)

    j_values = bessel_j_upward(max_order=max_order, x=x_grid)
    ref_values = np.vstack([special.jv(n, x_grid) for n in range(max_order + 1)])

    abs_error = np.abs(j_values - ref_values)
    rec_residual = recurrence_residuals(j_values, x_grid)

    selected_orders = [0, 1, 3, 6]
    ode_residual_max = {n: ode_residual(x_grid, j_values[n], n) for n in selected_orders if n <= max_order}

    orth_mat, orth_diag_ref = orthogonality_check(order=0, mode_count=4)

    return BesselMVPResult(
        x_grid=x_grid,
        max_order=max_order,
        j_values=j_values,
        ref_values=ref_values,
        abs_error=abs_error,
        recurrence_residual_max=rec_residual,
        ode_residual_max=ode_residual_max,
        orthogonality_matrix=orth_mat,
        orthogonality_diag_ref=orth_diag_ref,
    )


def main() -> None:
    result = run_mvp(max_order=6)
    table = build_error_table(result.abs_error, result.recurrence_residual_max)

    offdiag = result.orthogonality_matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    orth_offdiag_max = float(np.max(np.abs(offdiag)))

    diag_err = np.abs(np.diag(result.orthogonality_matrix) - result.orthogonality_diag_ref)
    diag_rel_err_max = float(np.max(diag_err / np.maximum(np.abs(result.orthogonality_diag_ref), EPS)))

    max_abs_error = float(np.max(result.abs_error))
    max_recur_res = float(np.max(result.recurrence_residual_max[1:result.max_order]))
    max_ode_res = float(max(result.ode_residual_max.values()))

    checks = {
        "max abs error vs scipy < 2e-8": max_abs_error < 2e-8,
        "max recurrence residual < 1e-11": max_recur_res < 1e-11,
        "max ODE residual (FD) < 5e-3": max_ode_res < 5e-3,
        "orthogonality offdiag max < 1e-5": orth_offdiag_max < 1e-5,
        "orthogonality diag relative error < 5e-4": diag_rel_err_max < 5e-4,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Bessel Functions MVP (PHYS-0144) ===")
    print("Method: series seeds (J0, J1) + upward recurrence")
    print(f"x-grid: [{result.x_grid[0]:.1f}, {result.x_grid[-1]:.1f}], samples={result.x_grid.size}")
    print(f"max order: {result.max_order}")

    print("\nPer-order error summary:")
    print(table.to_string(index=False))

    print("\nODE residual summary (finite-difference check):")
    for n, val in result.ode_residual_max.items():
        print(f"- n={n}: max |x^2 y'' + x y' + (x^2-n^2) y| = {val:.3e}")

    print("\nOrthogonality check (n=0, first 4 zeros on [0,1]):")
    print("Gram matrix I_mk = integral_0^1 r J_0(alpha_m r) J_0(alpha_k r) dr")
    print(pd.DataFrame(result.orthogonality_matrix).to_string(index=False, header=False))
    print(f"max |offdiag(I)| = {orth_offdiag_max:.3e}")
    print(f"max relative diag error = {diag_rel_err_max:.3e}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
