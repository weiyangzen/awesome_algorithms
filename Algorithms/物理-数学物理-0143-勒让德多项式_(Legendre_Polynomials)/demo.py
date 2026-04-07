"""Minimal runnable MVP for Legendre polynomials."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import eval_legendre, roots_legendre


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the reproducible MVP run."""

    max_degree: int = 8
    grid_size: int = 401
    orth_quad_order: int = 80
    projection_degree: int = 10
    projection_quad_order: int = 160


def legendre_values_recurrence(max_degree: int, x: np.ndarray) -> np.ndarray:
    """Evaluate P_0..P_max_degree on x via the three-term recurrence."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    x = np.asarray(x, dtype=np.float64)
    values = np.zeros((max_degree + 1, x.size), dtype=np.float64)

    values[0] = 1.0
    if max_degree >= 1:
        values[1] = x

    for n in range(2, max_degree + 1):
        values[n] = ((2 * n - 1) * x * values[n - 1] - (n - 1) * values[n - 2]) / n

    return values


def legendre_poly_rodrigues(degree: int) -> np.poly1d:
    """Build P_n(x) in power basis using Rodrigues' formula."""
    if degree < 0:
        raise ValueError("degree must be non-negative")

    base = np.poly1d([1.0, 0.0, -1.0]) ** degree  # (x^2 - 1)^n
    deriv = np.polyder(base, m=degree)
    return deriv / (2**degree * math.factorial(degree))


def legendre_ode_residual(degree: int, x: np.ndarray) -> np.ndarray:
    """Residual of (1-x^2) y'' - 2x y' + n(n+1) y = 0 for y=P_n."""
    poly = legendre_poly_rodrigues(degree)
    first = np.polyder(poly, m=1)
    second = np.polyder(poly, m=2)

    x = np.asarray(x, dtype=np.float64)
    y = np.polyval(poly, x)
    y1 = np.polyval(first, x)
    y2 = np.polyval(second, x)
    return (1.0 - x * x) * y2 - 2.0 * x * y1 + degree * (degree + 1.0) * y


def orthogonality_errors(max_degree: int, quad_order: int) -> tuple[float, float]:
    """Return off-diagonal and diagonal errors for integral PmPn over [-1,1]."""
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    values = legendre_values_recurrence(max_degree=max_degree, x=nodes)

    gram = (values * weights[None, :]) @ values.T
    target_diag = np.array([2.0 / (2 * n + 1) for n in range(max_degree + 1)], dtype=np.float64)

    diag_error = float(np.max(np.abs(np.diag(gram) - target_diag)))
    offdiag = gram - np.diag(np.diag(gram))
    offdiag_error = float(np.max(np.abs(offdiag)))
    return offdiag_error, diag_error


def legendre_projection_coefficients(
    function, degree: int, quad_order: int
) -> np.ndarray:
    """Compute coefficients a_n of truncated Legendre series for f(x)."""
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    values = legendre_values_recurrence(max_degree=degree, x=nodes)
    fvals = function(nodes)

    coeffs = np.zeros(degree + 1, dtype=np.float64)
    for n in range(degree + 1):
        coeffs[n] = 0.5 * (2 * n + 1) * np.sum(weights * fvals * values[n])
    return coeffs


def evaluate_legendre_series(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum_n a_n P_n(x) on x."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    values = legendre_values_recurrence(max_degree=coefficients.size - 1, x=np.asarray(x, dtype=np.float64))
    return coefficients @ values


def main() -> None:
    cfg = DemoConfig()

    x = np.linspace(-1.0, 1.0, cfg.grid_size)
    x_inner = np.linspace(-0.98, 0.98, cfg.grid_size)

    recurrence_values = legendre_values_recurrence(max_degree=cfg.max_degree, x=x)

    rows = []
    for n in range(cfg.max_degree + 1):
        scipy_vals = eval_legendre(n, x)
        rod_vals = np.polyval(legendre_poly_rodrigues(n), x)

        rec_vs_scipy = float(np.max(np.abs(recurrence_values[n] - scipy_vals)))
        rod_vs_scipy = float(np.max(np.abs(rod_vals - scipy_vals)))

        ode_res = legendre_ode_residual(n, x_inner)
        ode_residual_max = float(np.max(np.abs(ode_res)))

        roots, _weights = roots_legendre(max(1, n))
        root_check_values = legendre_values_recurrence(max_degree=max(1, n), x=roots)
        root_residual_max = 0.0 if n == 0 else float(np.max(np.abs(root_check_values[n])))

        rows.append(
            {
                "n": n,
                "max|rec-scipy|": rec_vs_scipy,
                "max|rodrigues-scipy|": rod_vs_scipy,
                "max|ODE residual|": ode_residual_max,
                "max|P_n(root)|": root_residual_max,
            }
        )

    comparison_df = pd.DataFrame(rows)

    orth_offdiag, orth_diag = orthogonality_errors(
        max_degree=cfg.max_degree,
        quad_order=cfg.orth_quad_order,
    )

    coeffs = legendre_projection_coefficients(
        function=np.exp,
        degree=cfg.projection_degree,
        quad_order=cfg.projection_quad_order,
    )

    x_eval = np.linspace(-1.0, 1.0, 1201)
    approx = evaluate_legendre_series(coeffs, x_eval)
    truth = np.exp(x_eval)
    max_abs_projection_error = float(np.max(np.abs(approx - truth)))

    qx, qw = np.polynomial.legendre.leggauss(cfg.projection_quad_order)
    qerr = evaluate_legendre_series(coeffs, qx) - np.exp(qx)
    l2_projection_error = float(np.sqrt(np.sum(qw * qerr * qerr)))

    print("=== Legendre Polynomial Consistency Checks ===")
    print(comparison_df.to_string(index=False))
    print()
    print("=== Orthogonality Check ===")
    print(f"max offdiag integral error: {orth_offdiag:.3e}")
    print(f"max diag integral error:    {orth_diag:.3e}")
    print()
    print("=== Truncated Legendre Expansion for exp(x) ===")
    coeff_preview = pd.DataFrame(
        {
            "n": np.arange(coeffs.size),
            "a_n": coeffs,
        }
    )
    print(coeff_preview.head(8).to_string(index=False))
    print(f"max abs reconstruction error on dense grid: {max_abs_projection_error:.3e}")
    print(f"L2 reconstruction error on [-1,1]:         {l2_projection_error:.3e}")

    assert float(comparison_df["max|rec-scipy|"].max()) < 1e-11
    assert float(comparison_df["max|rodrigues-scipy|"].max()) < 1e-10
    assert float(comparison_df["max|ODE residual|"].max()) < 1e-9
    assert float(comparison_df.loc[comparison_df["n"] >= 1, "max|P_n(root)|"].max()) < 1e-12
    assert orth_offdiag < 1e-12
    assert orth_diag < 1e-12
    assert max_abs_projection_error < 2e-9

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
