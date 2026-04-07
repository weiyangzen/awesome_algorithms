"""Minimal runnable MVP for magnetization (PHYS-0159)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class LinearMaterial:
    """Linear isotropic material with M = chi * H."""

    name: str
    chi: float


@dataclass(frozen=True)
class SaturatingMaterial:
    """Simple anhysteretic saturation model M = Ms * tanh(H/H0)."""

    name: str
    ms: float
    h0: float


def magnetization_linear(h: np.ndarray, chi: float) -> np.ndarray:
    """Return magnetization M for linear model (A/m)."""
    h_arr = np.asarray(h, dtype=float)
    return chi * h_arr


def magnetization_saturating_tanh(h: np.ndarray, ms: float, h0: float) -> np.ndarray:
    """Return magnetization M for a saturation model (A/m)."""
    if h0 <= 0:
        raise ValueError("h0 must be positive")
    h_arr = np.asarray(h, dtype=float)
    return ms * np.tanh(h_arr / h0)


def magnetic_flux_density(h: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Compute B = mu0 * (H + M), SI units (tesla)."""
    h_arr = np.asarray(h, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    return mu_0 * (h_arr + m_arr)


def estimate_susceptibility_linear_regression(
    h_samples: np.ndarray,
    m_samples: np.ndarray,
) -> tuple[float, float]:
    """Estimate chi from noisy low-field data using OLS without intercept."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(h_samples.reshape(-1, 1), m_samples)
    chi_hat = float(reg.coef_[0])
    r2 = float(reg.score(h_samples.reshape(-1, 1), m_samples))
    return chi_hat, r2


def _sat_model_for_fit(h: np.ndarray, ms: float, h0: float) -> np.ndarray:
    return magnetization_saturating_tanh(h, ms, h0)


def fit_saturation_parameters(
    h_samples: np.ndarray,
    m_samples: np.ndarray,
    p0: tuple[float, float] = (6.0e5, 1.0e5),
) -> tuple[float, float]:
    """Fit (Ms, H0) from synthetic nonlinear data."""
    params, _ = curve_fit(
        _sat_model_for_fit,
        h_samples,
        m_samples,
        p0=p0,
        bounds=([1.0e5, 1.0e4], [2.0e6, 1.0e6]),
        maxfev=20000,
    )
    return float(params[0]), float(params[1])


def torch_consistency_check(h: np.ndarray, ms: float, h0: float) -> float:
    """Compute max absolute |B_torch - B_numpy| over a field grid."""
    h_np = np.asarray(h, dtype=np.float64)
    m_np = magnetization_saturating_tanh(h_np, ms, h0)
    b_np = magnetic_flux_density(h_np, m_np)

    h_t = torch.tensor(h_np, dtype=torch.float64)
    m_t = ms * torch.tanh(h_t / h0)
    b_t = mu_0 * (h_t + m_t)

    return float(np.max(np.abs(b_t.numpy() - b_np)))


def build_material_table(
    h_probe: np.ndarray,
    dia: LinearMaterial,
    para: LinearMaterial,
    ferro: SaturatingMaterial,
) -> pd.DataFrame:
    """Create a compact table comparing M and B for three materials."""
    m_dia = magnetization_linear(h_probe, dia.chi)
    m_para = magnetization_linear(h_probe, para.chi)
    m_ferro = magnetization_saturating_tanh(h_probe, ferro.ms, ferro.h0)

    b_dia = magnetic_flux_density(h_probe, m_dia)
    b_para = magnetic_flux_density(h_probe, m_para)
    b_ferro = magnetic_flux_density(h_probe, m_ferro)

    return pd.DataFrame(
        {
            "H_A_per_m": h_probe,
            "M_dia_A_per_m": m_dia,
            "M_para_A_per_m": m_para,
            "M_ferro_A_per_m": m_ferro,
            "B_dia_T": b_dia,
            "B_para_T": b_para,
            "B_ferro_T": b_ferro,
        }
    )


def main() -> None:
    rng = np.random.default_rng(20260407)

    # 1) Low-field linear susceptibility estimation.
    chi_true = 8.0e-3
    h_low = np.linspace(-3.0e4, 3.0e4, 121)
    m_low_true = magnetization_linear(h_low, chi_true)
    m_low_obs = m_low_true + rng.normal(0.0, 30.0, size=h_low.shape)

    chi_hat, low_field_r2 = estimate_susceptibility_linear_regression(h_low, m_low_obs)

    # 2) Nonlinear saturation parameter fitting.
    ms_true = 8.0e5
    h0_true = 1.2e5
    h_sat = np.linspace(-6.0e5, 6.0e5, 241)
    m_sat_true = magnetization_saturating_tanh(h_sat, ms_true, h0_true)
    m_sat_obs = m_sat_true + rng.normal(0.0, 1.5e4, size=h_sat.shape)

    ms_hat, h0_hat = fit_saturation_parameters(h_sat, m_sat_obs)

    # 3) Cross-framework consistency check (NumPy vs PyTorch).
    h_check = np.linspace(-8.0e5, 8.0e5, 2001)
    max_np_torch_diff = torch_consistency_check(h_check, ms_hat, h0_hat)

    dia = LinearMaterial(name="Diamagnet", chi=-1.0e-4)
    para = LinearMaterial(name="Paramagnet", chi=chi_hat)
    ferro = SaturatingMaterial(name="Ferromagnet(tanh)", ms=ms_hat, h0=h0_hat)

    h_probe = np.array([-3.0e5, -1.0e5, 0.0, 1.0e5, 3.0e5], dtype=float)
    table = build_material_table(h_probe, dia, para, ferro)

    rel_err_chi = abs(chi_hat - chi_true) / abs(chi_true)
    rel_err_ms = abs(ms_hat - ms_true) / ms_true
    rel_err_h0 = abs(h0_hat - h0_true) / h0_true

    print("=== Magnetization MVP (PHYS-0159) ===")
    print(f"chi_true={chi_true:.6e}, chi_hat={chi_hat:.6e}, low_field_R2={low_field_r2:.6f}")
    print(
        f"Ms_true={ms_true:.6e}, Ms_hat={ms_hat:.6e}, rel_err_ms={rel_err_ms:.3%}; "
        f"H0_true={h0_true:.6e}, H0_hat={h0_hat:.6e}, rel_err_h0={rel_err_h0:.3%}"
    )
    print(f"max_abs(B_torch - B_numpy)={max_np_torch_diff:.3e} T")

    print("\n=== Material Comparison Table ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    checks = {
        "susceptibility_relative_error < 8%": rel_err_chi < 0.08,
        "low_field_R2 > 0.94": low_field_r2 > 0.94,
        "Ms_fit_relative_error < 5%": rel_err_ms < 0.05,
        "H0_fit_relative_error < 7%": rel_err_h0 < 0.07,
        "numpy_torch_consistency < 1e-12 T": max_np_torch_diff < 1e-12,
    }

    print("\n=== Threshold Checks ===")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
