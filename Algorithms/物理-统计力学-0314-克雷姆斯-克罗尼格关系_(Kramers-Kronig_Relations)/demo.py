"""MVP for Kramers-Kronig relations using explicit principal-value quadrature.

The script is deterministic and non-interactive:
1) build a causal Lorentz susceptibility spectrum,
2) reconstruct real/imag parts via discrete principal-value Kramers-Kronig integrals,
3) report reconstruction errors and run acceptance checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KKConfig:
    amplitude: float = 1.0
    omega_0: float = 2.5
    gamma: float = 0.7
    omega_max: float = 25.0
    n_points: int = 3001
    eval_omega_limit: float = 17.0
    rel_l2_tol_real: float = 0.02
    rel_l2_tol_imag: float = 0.02
    max_abs_tol_real: float = 0.01
    max_abs_tol_imag: float = 0.015

    def validate(self) -> None:
        if self.amplitude == 0.0:
            raise ValueError("amplitude must be non-zero")
        if self.omega_0 <= 0.0:
            raise ValueError("omega_0 must be > 0")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be > 0")
        if self.n_points < 1001:
            raise ValueError("n_points must be >= 1001 for stable principal-value integration")
        if self.n_points % 2 == 0:
            raise ValueError("n_points must be odd to keep omega=0 on the grid")
        if not (0.0 < self.eval_omega_limit < self.omega_max):
            raise ValueError("eval_omega_limit must satisfy 0 < eval_omega_limit < omega_max")


def build_frequency_grid(cfg: KKConfig) -> np.ndarray:
    return np.linspace(-cfg.omega_max, cfg.omega_max, cfg.n_points, dtype=np.float64)


def lorentz_susceptibility(omega: np.ndarray, cfg: KKConfig) -> np.ndarray:
    denominator = (cfg.omega_0**2 - omega**2) - 1j * cfg.gamma * omega
    return cfg.amplitude / denominator


def kk_principal_value_transform(omega: np.ndarray, values: np.ndarray, sign: float) -> np.ndarray:
    """Compute discrete principal-value transform:

    result(omega_i) = sign/pi * PV ∫ values(omega')/(omega' - omega_i) d omega'

    The integral is approximated on a uniform grid with trapezoidal endpoint weights.
    """

    if omega.ndim != 1 or values.ndim != 1:
        raise ValueError("omega and values must be 1D arrays")
    if omega.shape[0] != values.shape[0]:
        raise ValueError("omega and values must have the same length")

    steps = np.diff(omega)
    if np.any(steps <= 0.0):
        raise ValueError("omega grid must be strictly increasing")
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        raise ValueError("omega grid must be uniformly spaced")

    delta = float(steps[0])
    n = omega.size

    weights = np.ones(n, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    weighted_values = values * weights

    transformed = np.empty(n, dtype=np.float64)
    for i, omega_i in enumerate(omega):
        denominator = omega - omega_i
        # Principal value: explicitly remove singular point denominator=0 at i.
        contribution = np.divide(
            weighted_values,
            denominator,
            out=np.zeros_like(weighted_values),
            where=denominator != 0.0,
        )
        transformed[i] = float(np.sum(contribution))

    return sign * delta * transformed / np.pi


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    diff_norm = float(np.sqrt(np.mean((reference[mask] - estimate[mask]) ** 2)))
    ref_norm = float(np.sqrt(np.mean(reference[mask] ** 2)))
    return diff_norm / max(ref_norm, 1e-12)


def build_sample_table(
    omega: np.ndarray,
    real_true: np.ndarray,
    imag_true: np.ndarray,
    real_kk: np.ndarray,
    imag_kk: np.ndarray,
) -> pd.DataFrame:
    checkpoints = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
    rows = []
    for point in checkpoints:
        idx = int(np.argmin(np.abs(omega - point)))
        rows.append(
            {
                "omega": float(omega[idx]),
                "real_true": float(real_true[idx]),
                "real_from_imag_KK": float(real_kk[idx]),
                "imag_true": float(imag_true[idx]),
                "imag_from_real_KK": float(imag_kk[idx]),
            }
        )
    return pd.DataFrame(rows)


def run_mvp(cfg: KKConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    omega = build_frequency_grid(cfg)
    chi = lorentz_susceptibility(omega, cfg)

    real_true = np.real(chi)
    imag_true = np.imag(chi)

    # K-K pair on the full symmetric frequency axis.
    # Re(chi) = + 1/pi PV∫ Im(chi)/(w'-w) dw'
    # Im(chi) = - 1/pi PV∫ Re(chi)/(w'-w) dw'
    real_from_imag = kk_principal_value_transform(omega, imag_true, sign=+1.0)
    imag_from_real = kk_principal_value_transform(omega, real_true, sign=-1.0)

    mask = np.abs(omega) <= cfg.eval_omega_limit

    metrics = {
        "rel_l2_real": relative_l2_error(real_true, real_from_imag, mask),
        "rel_l2_imag": relative_l2_error(imag_true, imag_from_real, mask),
        "max_abs_real": float(np.max(np.abs(real_true[mask] - real_from_imag[mask]))),
        "max_abs_imag": float(np.max(np.abs(imag_true[mask] - imag_from_real[mask]))),
        "even_symmetry_real": float(np.max(np.abs(real_true - real_true[::-1]))),
        "odd_symmetry_imag": float(np.max(np.abs(imag_true + imag_true[::-1]))),
    }

    metrics_df = pd.DataFrame(
        [
            {"metric": "relative_L2_error_Re(chi)", "value": metrics["rel_l2_real"]},
            {"metric": "relative_L2_error_Im(chi)", "value": metrics["rel_l2_imag"]},
            {"metric": "max_abs_error_Re(chi)", "value": metrics["max_abs_real"]},
            {"metric": "max_abs_error_Im(chi)", "value": metrics["max_abs_imag"]},
            {"metric": "symmetry_even_error_Re(chi)", "value": metrics["even_symmetry_real"]},
            {"metric": "symmetry_odd_error_Im(chi)", "value": metrics["odd_symmetry_imag"]},
        ]
    )

    sample_df = build_sample_table(omega, real_true, imag_true, real_from_imag, imag_from_real)
    return metrics_df, sample_df, metrics


def main() -> None:
    cfg = KKConfig()
    cfg.validate()

    metrics_df, sample_df, metrics = run_mvp(cfg)

    print("=== Kramers-Kronig Relations MVP ===")
    print(
        "config:",
        f"A={cfg.amplitude}, omega_0={cfg.omega_0}, gamma={cfg.gamma},",
        f"omega_max={cfg.omega_max}, n_points={cfg.n_points}, eval_omega_limit={cfg.eval_omega_limit}",
    )
    print()
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print(sample_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    if metrics["rel_l2_real"] > cfg.rel_l2_tol_real:
        raise AssertionError(
            f"Relative L2 error for Re(chi) too large: {metrics['rel_l2_real']:.4f} > {cfg.rel_l2_tol_real:.4f}"
        )
    if metrics["rel_l2_imag"] > cfg.rel_l2_tol_imag:
        raise AssertionError(
            f"Relative L2 error for Im(chi) too large: {metrics['rel_l2_imag']:.4f} > {cfg.rel_l2_tol_imag:.4f}"
        )
    if metrics["max_abs_real"] > cfg.max_abs_tol_real:
        raise AssertionError(
            f"Max abs error for Re(chi) too large: {metrics['max_abs_real']:.4f} > {cfg.max_abs_tol_real:.4f}"
        )
    if metrics["max_abs_imag"] > cfg.max_abs_tol_imag:
        raise AssertionError(
            f"Max abs error for Im(chi) too large: {metrics['max_abs_imag']:.4f} > {cfg.max_abs_tol_imag:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
