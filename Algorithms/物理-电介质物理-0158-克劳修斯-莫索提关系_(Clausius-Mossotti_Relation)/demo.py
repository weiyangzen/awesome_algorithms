"""Minimal runnable MVP for Clausius-Mossotti relation.

Core physics relation for isotropic dielectric media:
    (epsilon_r - 1) / (epsilon_r + 2) = N * alpha / (3 * epsilon_0)

This script demonstrates three practical operations:
1) Forward model: predict epsilon_r from number density N and molecular polarizability alpha.
2) Inverse point estimate: recover alpha from measured epsilon_r and N.
3) Global parameter fit: estimate a single alpha from multiple noisy measurements
   using linearized least squares and nonlinear least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

EPSILON_0 = 8.8541878128e-12  # F/m (vacuum permittivity)


@dataclass(frozen=True)
class CMObservation:
    """Measurement dataset for Clausius-Mossotti inversion."""

    number_density: np.ndarray  # N, unit: 1/m^3
    epsilon_r_measured: np.ndarray  # relative permittivity, dimensionless

    def validate(self) -> None:
        n = np.asarray(self.number_density, dtype=np.float64)
        eps = np.asarray(self.epsilon_r_measured, dtype=np.float64)

        if n.ndim != 1 or eps.ndim != 1:
            raise ValueError("number_density and epsilon_r_measured must be 1-D arrays")
        if n.size != eps.size:
            raise ValueError("number_density and epsilon_r_measured must have equal length")
        if n.size < 3:
            raise ValueError("At least 3 samples are required for a meaningful fit")
        if np.any(n <= 0.0):
            raise ValueError("number_density values must be strictly positive")
        if np.any(eps <= 0.0):
            raise ValueError("epsilon_r values must be strictly positive")
        if np.any(np.isclose(eps + 2.0, 0.0, atol=1e-14, rtol=0.0)):
            raise ValueError("epsilon_r too close to -2 causes singular CM transform")


def cm_ratio_from_epsilon_r(epsilon_r: np.ndarray | float) -> np.ndarray:
    """Compute Clausius-Mossotti left-hand ratio (epsilon_r - 1)/(epsilon_r + 2)."""
    eps = np.asarray(epsilon_r, dtype=np.float64)
    if np.any(eps <= 0.0):
        raise ValueError("epsilon_r must be > 0")

    denom = eps + 2.0
    if np.any(np.isclose(denom, 0.0, atol=1e-14, rtol=0.0)):
        raise ValueError("epsilon_r + 2 is too close to zero")
    return (eps - 1.0) / denom


def epsilon_r_from_cm_ratio(cm_ratio: np.ndarray | float) -> np.ndarray:
    """Recover epsilon_r from Clausius-Mossotti ratio via algebraic inversion."""
    ratio = np.asarray(cm_ratio, dtype=np.float64)
    if np.any(ratio >= 1.0):
        raise ValueError("CM ratio must be < 1 to avoid divergence in epsilon_r")

    denom = 1.0 - ratio
    if np.any(np.isclose(denom, 0.0, atol=1e-14, rtol=0.0)):
        raise ValueError("1 - cm_ratio is too close to zero")
    return (1.0 + 2.0 * ratio) / denom


def epsilon_r_from_number_density_alpha(
    number_density: np.ndarray | float,
    alpha: float,
) -> np.ndarray:
    """Forward model epsilon_r(N, alpha) from Clausius-Mossotti relation."""
    n = np.asarray(number_density, dtype=np.float64)
    if np.any(n <= 0.0):
        raise ValueError("number_density must be > 0")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    ratio = n * alpha / (3.0 * EPSILON_0)
    return epsilon_r_from_cm_ratio(ratio)


def alpha_from_epsilon_r_number_density(
    epsilon_r: np.ndarray | float,
    number_density: np.ndarray | float,
) -> np.ndarray:
    """Pointwise inverse estimate alpha = 3*epsilon_0/N * ((epsilon_r-1)/(epsilon_r+2))."""
    eps = np.asarray(epsilon_r, dtype=np.float64)
    n = np.asarray(number_density, dtype=np.float64)
    if np.any(n <= 0.0):
        raise ValueError("number_density must be > 0")

    ratio = cm_ratio_from_epsilon_r(eps)
    return 3.0 * EPSILON_0 * ratio / n


def fit_alpha_linearized(data: CMObservation) -> float:
    """Estimate alpha using linearized least squares in CM-ratio space.

    Let y_i = (epsilon_i - 1)/(epsilon_i + 2), x_i = N_i/(3*epsilon_0),
    then y_i ≈ x_i * alpha. Solve one-parameter LS through origin:
        alpha_hat = (x^T y) / (x^T x).
    """
    data.validate()
    n = np.asarray(data.number_density, dtype=np.float64)
    eps = np.asarray(data.epsilon_r_measured, dtype=np.float64)

    x = n / (3.0 * EPSILON_0)
    y = cm_ratio_from_epsilon_r(eps)

    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("Degenerate design matrix: x^T x <= 0")
    return float(np.dot(x, y) / denominator)


def fit_alpha_nonlinear(data: CMObservation, alpha_init: float) -> tuple[float, float]:
    """Estimate alpha by minimizing epsilon_r residuals with scipy.least_squares."""
    data.validate()
    if alpha_init <= 0.0:
        raise ValueError("alpha_init must be > 0")

    n = np.asarray(data.number_density, dtype=np.float64)
    eps_meas = np.asarray(data.epsilon_r_measured, dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        alpha_val = float(params[0])
        eps_pred = epsilon_r_from_number_density_alpha(n, alpha_val)
        return eps_pred - eps_meas

    # Keep solver strictly inside the physically valid CM region: N*alpha/(3*epsilon_0) < 1.
    alpha_upper = 0.99 * (3.0 * EPSILON_0 / float(np.max(n)))
    if alpha_init >= alpha_upper:
        alpha_init = 0.5 * alpha_upper

    result = least_squares(
        residual,
        x0=np.array([alpha_init], dtype=np.float64),
        bounds=(1e-45, alpha_upper),
        method="dogbox",
    )
    if not result.success:
        raise RuntimeError(f"Nonlinear fit failed: {result.message}")

    alpha_hat = float(result.x[0])
    residual_norm = float(np.linalg.norm(result.fun))
    return alpha_hat, residual_norm


def _build_synthetic_dataset(alpha_true: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct deterministic synthetic measurements for reproducible demo."""
    number_density = np.array([2.0, 3.0, 4.5, 6.0, 7.5], dtype=np.float64) * 1e27
    epsilon_r_true = epsilon_r_from_number_density_alpha(number_density, alpha_true)

    # Deterministic pseudo-measurement noise (no randomness).
    perturbation = np.array([0.0007, -0.0005, 0.0010, -0.0008, 0.0006], dtype=np.float64)
    epsilon_r_measured = epsilon_r_true + perturbation
    return number_density, epsilon_r_true, epsilon_r_measured


def _relative_error(estimate: float, truth: float) -> float:
    return float(abs(estimate - truth) / abs(truth))


def main() -> None:
    alpha_true = 1.65e-40  # C*m^2/V

    n, eps_true, eps_measured = _build_synthetic_dataset(alpha_true)
    data = CMObservation(number_density=n, epsilon_r_measured=eps_measured)
    data.validate()

    alpha_pointwise = alpha_from_epsilon_r_number_density(eps_measured, n)
    alpha_linear = fit_alpha_linearized(data)
    alpha_nonlinear, residual_norm = fit_alpha_nonlinear(data, alpha_init=alpha_linear)

    eps_pred_linear = epsilon_r_from_number_density_alpha(n, alpha_linear)
    eps_pred_nonlinear = epsilon_r_from_number_density_alpha(n, alpha_nonlinear)

    table = pd.DataFrame(
        {
            "N_1_per_m3": n,
            "epsilon_r_true": eps_true,
            "epsilon_r_measured": eps_measured,
            "epsilon_r_pred_linear": eps_pred_linear,
            "epsilon_r_pred_nonlinear": eps_pred_nonlinear,
            "alpha_pointwise_est": alpha_pointwise,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "alpha_true": alpha_true,
                "alpha_linear_fit": alpha_linear,
                "alpha_nonlinear_fit": alpha_nonlinear,
                "linear_rel_error": _relative_error(alpha_linear, alpha_true),
                "nonlinear_rel_error": _relative_error(alpha_nonlinear, alpha_true),
                "nonlinear_residual_l2": residual_norm,
            }
        ]
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Clausius-Mossotti Relation MVP")
    print("(epsilon_r - 1)/(epsilon_r + 2) = N*alpha/(3*epsilon_0)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.10e}"))
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.10e}"))

    # Self-check 1: noise-free inverse should recover alpha_true.
    alpha_clean = alpha_from_epsilon_r_number_density(eps_true, n)
    assert np.allclose(alpha_clean, alpha_true, rtol=1e-12, atol=0.0)

    # Self-check 2: fitted alpha should stay near truth under mild noise.
    assert _relative_error(alpha_linear, alpha_true) < 0.03
    assert _relative_error(alpha_nonlinear, alpha_true) < 0.03

    # Self-check 3: all predicted permittivity values must remain physical (>0).
    assert np.all(eps_pred_linear > 0.0)
    assert np.all(eps_pred_nonlinear > 0.0)

    print("All checks passed.")


if __name__ == "__main__":
    main()
