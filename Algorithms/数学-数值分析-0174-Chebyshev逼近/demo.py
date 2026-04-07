"""Minimal runnable MVP: Chebyshev approximation on [-1, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class ApproximationReport:
    """Summary metrics for one approximation run."""

    name: str
    degree: int
    sample_count: int
    max_abs_error: float
    rmse: float
    coeffs: np.ndarray


def validate_hyperparams(degree: int, sample_count: int) -> None:
    """Validate approximation hyperparameters."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if sample_count < degree + 1:
        raise ValueError(
            "sample_count must be at least degree+1, "
            f"got degree={degree}, sample_count={sample_count}"
        )


def chebyshev_gauss_nodes(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Chebyshev-Gauss nodes x_k and angles theta_k."""
    k = np.arange(sample_count, dtype=float)
    theta = (k + 0.5) * np.pi / float(sample_count)
    x = np.cos(theta)
    return x, theta


def fit_chebyshev_by_discrete_orthogonality(
    func: ArrayFunc,
    degree: int,
    sample_count: int,
) -> np.ndarray:
    """Fit Chebyshev coefficients c_j using discrete orthogonality on Gauss nodes.

    We approximate f(x) by
        p_n(x) = sum_{j=0}^{degree} c_j T_j(x).

    On theta_k=(k+1/2)pi/N and x_k=cos(theta_k), coefficients are computed via
        c_j = (2/N) * sum_k f(x_k) cos(j*theta_k),
    with c_0 halved at the end.
    """
    validate_hyperparams(degree, sample_count)

    x_nodes, theta = chebyshev_gauss_nodes(sample_count)
    y_nodes = np.asarray(func(x_nodes), dtype=float)

    if y_nodes.shape != x_nodes.shape:
        raise ValueError(
            "func must return an array with the same shape as input nodes, "
            f"got y shape {y_nodes.shape}, x shape {x_nodes.shape}"
        )
    if not np.all(np.isfinite(y_nodes)):
        raise ValueError("function values contain non-finite numbers")

    coeffs = np.empty(degree + 1, dtype=float)
    scale = 2.0 / float(sample_count)
    for j in range(degree + 1):
        cos_row = np.cos(float(j) * theta)
        coeffs[j] = scale * float(np.dot(cos_row, y_nodes))
    coeffs[0] *= 0.5
    return coeffs


def eval_chebyshev_clenshaw(x: np.ndarray | float, coeffs: np.ndarray) -> np.ndarray | float:
    """Evaluate sum_j coeffs[j] * T_j(x) via Clenshaw recurrence."""
    x_arr = np.asarray(x, dtype=float)
    c = np.asarray(coeffs, dtype=float)

    if c.ndim != 1 or c.size == 0:
        raise ValueError("coeffs must be a non-empty 1D array")

    b_kplus1 = np.zeros_like(x_arr, dtype=float)
    b_kplus2 = np.zeros_like(x_arr, dtype=float)

    for ck in c[:0:-1]:
        b_k = 2.0 * x_arr * b_kplus1 - b_kplus2 + ck
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    y = x_arr * b_kplus1 - b_kplus2 + c[0]
    if np.isscalar(x):
        return float(y)
    return y


def evaluate_report(
    name: str,
    func: ArrayFunc,
    degree: int,
    sample_count: int,
    grid_size: int = 4001,
) -> ApproximationReport:
    """Fit coefficients and compute dense-grid approximation errors."""
    coeffs = fit_chebyshev_by_discrete_orthogonality(
        func=func,
        degree=degree,
        sample_count=sample_count,
    )

    x_grid = np.linspace(-1.0, 1.0, grid_size)
    y_true = np.asarray(func(x_grid), dtype=float)
    y_pred = np.asarray(eval_chebyshev_clenshaw(x_grid, coeffs), dtype=float)

    if not np.all(np.isfinite(y_true)):
        raise ValueError("true function returned non-finite values on evaluation grid")

    err = y_pred - y_true
    max_abs_error = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    return ApproximationReport(
        name=name,
        degree=degree,
        sample_count=sample_count,
        max_abs_error=max_abs_error,
        rmse=rmse,
        coeffs=coeffs,
    )


def run_polynomial_exactness_check() -> None:
    """Self-check: known Chebyshev-series polynomial should be recovered exactly."""
    true_coeffs = np.array([0.5, 1.2, -0.7, 0.0, 0.0, 0.3], dtype=float)

    def poly_from_cheb(x: np.ndarray) -> np.ndarray:
        return np.asarray(eval_chebyshev_clenshaw(x, true_coeffs), dtype=float)

    est_coeffs = fit_chebyshev_by_discrete_orthogonality(
        func=poly_from_cheb,
        degree=5,
        sample_count=64,
    )

    coeff_error = float(np.max(np.abs(est_coeffs - true_coeffs)))

    x_test = np.linspace(-1.0, 1.0, 2049)
    y_true = poly_from_cheb(x_test)
    y_est = np.asarray(eval_chebyshev_clenshaw(x_test, est_coeffs), dtype=float)
    grid_error = float(np.max(np.abs(y_est - y_true)))

    print("Polynomial exactness check:")
    print(f"  max_coeff_error = {coeff_error:.3e}")
    print(f"  max_grid_error  = {grid_error:.3e}")

    assert coeff_error < 1e-12
    assert grid_error < 1e-12


def run_examples() -> None:
    """Run deterministic examples without any interactive input."""
    examples = [
        {
            "name": "exp(x)",
            "func": lambda x: np.exp(x),
            "degrees": [4, 8, 12, 16],
            "sample_factor": 6,
        },
        {
            "name": "Runge 1/(1+25x^2)",
            "func": lambda x: 1.0 / (1.0 + 25.0 * x * x),
            "degrees": [4, 8, 12, 16, 24],
            "sample_factor": 8,
        },
        {
            "name": "abs(x) (nonsmooth)",
            "func": lambda x: np.abs(x),
            "degrees": [4, 8, 12, 16, 24],
            "sample_factor": 8,
        },
    ]

    print("=" * 88)
    print("Chebyshev Approximation MVP (discrete orthogonality + Clenshaw)")
    print("=" * 88)

    run_polynomial_exactness_check()
    print("-" * 88)

    for item in examples:
        name = str(item["name"])
        func = item["func"]
        degrees = item["degrees"]
        sample_factor = int(item["sample_factor"])

        print(f"Example: {name}")
        print("degree  sample_count  max_abs_error      rmse")

        for degree in degrees:
            sample_count = max(32, sample_factor * (degree + 1))
            report = evaluate_report(
                name=name,
                func=func,
                degree=int(degree),
                sample_count=sample_count,
            )
            print(
                f"{report.degree:>6d}"
                f"  {report.sample_count:>12d}"
                f"  {report.max_abs_error:>14.6e}"
                f"  {report.rmse:>10.6e}"
            )

        last_report = evaluate_report(
            name=name,
            func=func,
            degree=int(degrees[-1]),
            sample_count=max(32, sample_factor * (int(degrees[-1]) + 1)),
        )
        preview = ", ".join(f"{v:+.3e}" for v in last_report.coeffs[:6])
        print(f"coeff preview (first 6) for degree {degrees[-1]}: [{preview}]")
        print("-" * 88)


def main() -> None:
    run_examples()


if __name__ == "__main__":
    main()
