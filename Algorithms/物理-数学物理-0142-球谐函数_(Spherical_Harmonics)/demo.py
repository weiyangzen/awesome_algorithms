"""Minimal runnable MVP for spherical harmonics (PHYS-0142).

This script intentionally keeps the implementation explicit:
- Builds complex spherical harmonics from associated Legendre polynomials.
- Validates against SciPy's `sph_harm_y` API.
- Verifies approximate orthonormality on a sphere grid.
- Demonstrates coefficient projection and reconstruction.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import special
from sklearn.metrics import mean_squared_error


LM = Tuple[int, int]


def _check_lm(l: int, m: int) -> None:
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}.")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got l={l}, m={m}.")


def normalization_factor(l: int, m_abs: int) -> float:
    """Normalization factor of complex spherical harmonics."""
    numerator = (2 * l + 1) * math.factorial(l - m_abs)
    denominator = 4.0 * math.pi * math.factorial(l + m_abs)
    return math.sqrt(numerator / denominator)


def spherical_harmonic_manual(
    l: int,
    m: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Compute Y_l^m(theta, phi) using explicit formula.

    Convention:
    - theta in [0, pi] is polar (colatitude)
    - phi in [0, 2*pi) is azimuth
    """
    _check_lm(l, m)

    theta_arr = np.asarray(theta, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)

    if m < 0:
        m_abs = -m
        y_pos = spherical_harmonic_manual(l, m_abs, theta_arr, phi_arr)
        return ((-1) ** m_abs) * np.conjugate(y_pos)

    m_abs = m
    x = np.cos(theta_arr)
    # scipy.special.lpmv includes the Condon-Shortley phase (-1)^m.
    p_lm = special.lpmv(m_abs, l, x)
    norm = normalization_factor(l, m_abs)
    return norm * p_lm * np.exp(1j * m_abs * phi_arr)


def build_sphere_grid(n_theta: int = 96, n_phi: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular spherical grid with integration weights dOmega."""
    theta = np.linspace(0.0, math.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    dtheta = float(theta[1] - theta[0])
    dphi = float(phi[1] - phi[0])
    weights = np.sin(theta_grid) * dtheta * dphi
    return theta_grid, phi_grid, weights


def inner_product(f: np.ndarray, g: np.ndarray, weights: np.ndarray) -> complex:
    """Approximate <f, g> = integral f * conj(g) dOmega."""
    return np.sum(f * np.conjugate(g) * weights)


def compare_with_scipy(theta: np.ndarray, phi: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Compare manual implementation with scipy.special.sph_harm_y."""
    probe_terms: list[LM] = [(0, 0), (1, -1), (2, 0), (3, 2), (4, -3)]
    rows = []
    max_abs_error = 0.0

    for l, m in probe_terms:
        y_manual = spherical_harmonic_manual(l, m, theta, phi)
        y_scipy = special.sph_harm_y(l, m, theta, phi)
        err = np.max(np.abs(y_manual - y_scipy))
        max_abs_error = max(max_abs_error, float(err))
        rows.append({"l": l, "m": m, "max_abs_error": float(err)})

    return pd.DataFrame(rows), max_abs_error


def orthogonality_metrics(
    basis: Dict[LM, np.ndarray],
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return max diagonal deviation and max off-diagonal magnitude."""
    keys = list(basis.keys())
    max_diag_dev = 0.0
    max_offdiag = 0.0

    for i, k1 in enumerate(keys):
        y1 = basis[k1]
        for j, k2 in enumerate(keys):
            y2 = basis[k2]
            val = inner_product(y1, y2, weights)
            if i == j:
                max_diag_dev = max(max_diag_dev, float(abs(val - 1.0)))
            else:
                max_offdiag = max(max_offdiag, float(abs(val)))

    return max_diag_dev, max_offdiag


def project_coefficients(
    signal: np.ndarray,
    basis: Dict[LM, np.ndarray],
    weights: np.ndarray,
) -> Dict[LM, complex]:
    """Project signal onto basis using numerical inner products."""
    return {k: inner_product(signal, y, weights) for k, y in basis.items()}


def reconstruct_signal(coeffs: Dict[LM, complex], basis: Dict[LM, np.ndarray]) -> np.ndarray:
    """Reconstruct signal from basis and coefficients."""
    shape = next(iter(basis.values())).shape
    out = np.zeros(shape, dtype=np.complex128)
    for k, c in coeffs.items():
        out += c * basis[k]
    return out


def rmse_complex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE for complex arrays."""
    return float(np.sqrt(np.mean(np.abs(y_true - y_pred) ** 2)))


def main() -> None:
    theta, phi, weights = build_sphere_grid(n_theta=96, n_phi=192)

    # 1) Cross-check explicit implementation against SciPy API.
    scipy_df, max_api_error = compare_with_scipy(theta, phi)

    # 2) Build basis up to L=4 for orthogonality and projection demos.
    l_max = 4
    basis: Dict[LM, np.ndarray] = {
        (l, m): spherical_harmonic_manual(l, m, theta, phi)
        for l in range(l_max + 1)
        for m in range(-l, l + 1)
    }

    max_diag_dev, max_offdiag = orthogonality_metrics(basis, weights)

    # 3) Synthesize a signal from a sparse set of true coefficients.
    true_coeffs: Dict[LM, complex] = {
        (0, 0): 0.70 + 0.00j,
        (2, 1): 1.10 - 0.30j,
        (3, -2): -0.45 + 0.80j,
    }
    signal = reconstruct_signal(true_coeffs, basis)

    # 4) Recover coefficients and reconstruct signal.
    estimated_coeffs = project_coefficients(signal, basis, weights)
    reconstructed = reconstruct_signal(estimated_coeffs, basis)

    true_coef_error = max(abs(estimated_coeffs[k] - v) for k, v in true_coeffs.items())
    leakage = max(abs(v) for k, v in estimated_coeffs.items() if k not in true_coeffs)

    # RMSE in three ways: numpy (complex), sklearn (2-channel), torch (vector norm).
    rmse_np = rmse_complex(signal, reconstructed)

    y_true_2ch = np.column_stack([signal.real.ravel(), signal.imag.ravel()])
    y_pred_2ch = np.column_stack([reconstructed.real.ravel(), reconstructed.imag.ravel()])
    rmse_sklearn = float(np.sqrt(mean_squared_error(y_true_2ch, y_pred_2ch)))

    err_mag = np.abs(signal - reconstructed).ravel()
    err_tensor = torch.from_numpy(err_mag)
    rmse_torch = float(torch.linalg.vector_norm(err_tensor) / math.sqrt(err_tensor.numel()))

    # Summarize recovered coefficients.
    coeff_rows = [
        {
            "l": l,
            "m": m,
            "real": float(c.real),
            "imag": float(c.imag),
            "abs": float(abs(c)),
        }
        for (l, m), c in estimated_coeffs.items()
    ]
    coeff_df = (
        pd.DataFrame(coeff_rows)
        .sort_values("abs", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )

    print("=== SciPy API Consistency (manual vs sph_harm_y) ===")
    print(scipy_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print(f"max_api_error = {max_api_error:.3e}")

    print("\n=== Orthonormality Check (l<=4 grid integral) ===")
    print(f"max_diag_dev = {max_diag_dev:.3e}")
    print(f"max_offdiag = {max_offdiag:.3e}")

    print("\n=== Top Recovered Coefficients ===")
    print(coeff_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\n=== Reconstruction Errors ===")
    print(f"true_coef_error = {true_coef_error:.3e}")
    print(f"leakage        = {leakage:.3e}")
    print(f"rmse_np        = {rmse_np:.3e}")
    print(f"rmse_sklearn   = {rmse_sklearn:.3e}")
    print(f"rmse_torch     = {rmse_torch:.3e}")

    # Tight enough for this grid while robust across environments.
    assert max_api_error < 5e-13
    assert max_diag_dev < 4e-3
    assert max_offdiag < 4e-3
    assert true_coef_error < 4e-3
    assert leakage < 4e-3
    assert rmse_np < 2e-3
    assert abs(rmse_np - rmse_torch) < 1e-12

    print("\nAll spherical harmonics MVP checks passed.")


if __name__ == "__main__":
    main()
