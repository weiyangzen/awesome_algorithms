"""Minimal runnable MVP for Einstein Field Equations (PHYS-0045).

This demo uses a spatially-flat FRW cosmology and verifies three equations
that come directly from Einstein Field Equations (with cosmological constant):

1) H^2 = (8*pi*G/3) * rho + (Lambda*c^2/3)
2) dot(H) = -4*pi*G * (rho + p/c^2)
3) ddot(a)/a = -(4*pi*G/3) * (rho + 3p/c^2) + (Lambda*c^2/3)

We model matter (w=0) and radiation (w=1/3) explicitly in T_{mu nu}, and
include dark-energy as Lambda on the geometry side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import G, c, parsec

EPS = 1.0e-30
MPC_M = 1.0e6 * parsec


@dataclass(frozen=True)
class FRWConfig:
    """Configuration for the flat-FRW Einstein-equation consistency check."""

    h0_km_s_mpc: float = 67.66
    omega_m: float = 0.3111
    omega_r: float = 9.2e-5
    omega_lambda: float = 1.0 - 0.3111 - 9.2e-5
    a_min: float = 0.02
    a_max: float = 1.5
    n_points: int = 320
    flatness_tol: float = 2.0e-4


@dataclass(frozen=True)
class Diagnostics:
    """Residual diagnostics for Einstein-equation checks."""

    eq1_rel_max: float
    dot_h_rel_max: float
    accel_rel_max: float
    h_model_rel_diff: float


@dataclass(frozen=True)
class CosmologyBackground:
    """Derived background constants from user-friendly cosmology parameters."""

    h0_si: float
    rho_crit0: float
    lambda_si: float


def to_h0_si(h0_km_s_mpc: float) -> float:
    """Convert H0 from km/s/Mpc to s^-1."""
    return float(h0_km_s_mpc * 1000.0 / MPC_M)


def validate_config(cfg: FRWConfig) -> None:
    """Validate basic physical and numerical constraints."""
    if cfg.h0_km_s_mpc <= 0.0:
        raise ValueError("h0_km_s_mpc must be positive.")
    if cfg.a_min <= 0.0 or cfg.a_max <= 0.0 or cfg.a_max <= cfg.a_min:
        raise ValueError("Scale-factor bounds must satisfy 0 < a_min < a_max.")
    if cfg.n_points < 32:
        raise ValueError("n_points must be >= 32.")
    if cfg.omega_m < 0.0 or cfg.omega_r < 0.0 or cfg.omega_lambda < 0.0:
        raise ValueError("Density parameters must be non-negative for this MVP.")

    omega_sum = cfg.omega_m + cfg.omega_r + cfg.omega_lambda
    if abs(omega_sum - 1.0) > cfg.flatness_tol:
        raise ValueError(
            "This MVP assumes flat FRW (Omega_m + Omega_r + Omega_lambda ~= 1). "
            f"Got sum={omega_sum:.8f}."
        )


def build_background(cfg: FRWConfig) -> CosmologyBackground:
    """Build SI constants implied by FRW parameters."""
    h0_si = to_h0_si(cfg.h0_km_s_mpc)
    rho_crit0 = 3.0 * h0_si * h0_si / (8.0 * np.pi * G)
    lambda_si = 3.0 * cfg.omega_lambda * h0_si * h0_si / (c * c)
    return CosmologyBackground(
        h0_si=float(h0_si),
        rho_crit0=float(rho_crit0),
        lambda_si=float(lambda_si),
    )


def matter_radiation_fields(
    a: np.ndarray,
    cfg: FRWConfig,
    bg: CosmologyBackground,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rho_m(a), rho_r(a), rho_total(a), p_total(a)."""
    rho_m = bg.rho_crit0 * cfg.omega_m * a ** (-3.0)
    rho_r = bg.rho_crit0 * cfg.omega_r * a ** (-4.0)
    rho_total = rho_m + rho_r
    pressure_total = (rho_r * c * c) / 3.0
    return rho_m, rho_r, rho_total, pressure_total


def hubble_from_einstein(
    rho_total: np.ndarray,
    lambda_si: float,
) -> np.ndarray:
    """Compute H(a) from the first Friedmann equation."""
    h2 = (8.0 * np.pi * G / 3.0) * rho_total + (lambda_si * c * c / 3.0)
    return np.sqrt(h2)


def hubble_from_omegas(a: np.ndarray, cfg: FRWConfig, bg: CosmologyBackground) -> np.ndarray:
    """Compute H(a) from the standard dimensionless E(a)^2 model."""
    e2 = cfg.omega_m * a ** (-3.0) + cfg.omega_r * a ** (-4.0) + cfg.omega_lambda
    return bg.h0_si * np.sqrt(e2)


def dot_h_from_a_derivative(
    a: np.ndarray,
    h: np.ndarray,
    cfg: FRWConfig,
    bg: CosmologyBackground,
) -> np.ndarray:
    """Compute dot(H) via chain rule dot(H)=a*H*dH/da from E(a)."""
    e2 = cfg.omega_m * a ** (-3.0) + cfg.omega_r * a ** (-4.0) + cfg.omega_lambda
    de2_da = -3.0 * cfg.omega_m * a ** (-4.0) - 4.0 * cfg.omega_r * a ** (-5.0)
    dh_da = bg.h0_si * 0.5 * de2_da / np.sqrt(e2)
    return a * h * dh_da


def run_einstein_friedmann_mvp(
    cfg: FRWConfig,
) -> tuple[pd.DataFrame, Diagnostics, CosmologyBackground]:
    """Run consistency checks for FRW equations derived from Einstein equations."""
    validate_config(cfg)
    bg = build_background(cfg)

    a = np.geomspace(cfg.a_min, cfg.a_max, cfg.n_points)

    rho_m, rho_r, rho_total, pressure_total = matter_radiation_fields(a=a, cfg=cfg, bg=bg)

    h_einstein = hubble_from_einstein(rho_total=rho_total, lambda_si=bg.lambda_si)
    h_omega = hubble_from_omegas(a=a, cfg=cfg, bg=bg)

    dot_h_lhs = dot_h_from_a_derivative(a=a, h=h_omega, cfg=cfg, bg=bg)
    dot_h_rhs = -4.0 * np.pi * G * (rho_total + pressure_total / (c * c))

    accel_lhs = dot_h_lhs + h_omega * h_omega
    accel_rhs = (
        -(4.0 * np.pi * G / 3.0) * (rho_total + 3.0 * pressure_total / (c * c))
        + (bg.lambda_si * c * c / 3.0)
    )

    eq1_res = h_einstein * h_einstein - ((8.0 * np.pi * G / 3.0) * rho_total + (bg.lambda_si * c * c / 3.0))
    dot_h_res = dot_h_lhs - dot_h_rhs
    accel_res = accel_lhs - accel_rhs

    eq1_rel_max = float(np.max(np.abs(eq1_res)) / (np.max(h_einstein * h_einstein) + EPS))
    dot_h_rel_max = float(np.max(np.abs(dot_h_res)) / (np.max(np.abs(dot_h_rhs)) + EPS))
    accel_rel_max = float(np.max(np.abs(accel_res)) / (np.max(np.abs(accel_rhs)) + EPS))
    h_model_rel_diff = float(np.max(np.abs(h_einstein - h_omega)) / (np.max(np.abs(h_omega)) + EPS))

    q = -accel_lhs / (h_omega * h_omega)

    table = pd.DataFrame(
        {
            "a": a,
            "rho_m_kg_m3": rho_m,
            "rho_r_kg_m3": rho_r,
            "H_s_inv": h_omega,
            "dot_H_s_inv2": dot_h_lhs,
            "q_deceleration": q,
            "eq1_residual": eq1_res,
            "dotH_residual": dot_h_res,
            "accel_residual": accel_res,
        }
    )

    diagnostics = Diagnostics(
        eq1_rel_max=eq1_rel_max,
        dot_h_rel_max=dot_h_rel_max,
        accel_rel_max=accel_rel_max,
        h_model_rel_diff=h_model_rel_diff,
    )
    return table, diagnostics, bg


def main() -> None:
    cfg = FRWConfig()
    table, diag, bg = run_einstein_friedmann_mvp(cfg)

    # Non-interactive correctness checks.
    assert diag.eq1_rel_max < 1e-13, "First Friedmann equation residual too large."
    assert diag.dot_h_rel_max < 3e-12, "dot(H) consistency residual too large."
    assert diag.accel_rel_max < 3e-12, "Acceleration equation residual too large."
    assert diag.h_model_rel_diff < 1e-13, "H(a) models disagree unexpectedly."

    # Compact output at representative scale factors.
    sample_idx = np.unique(np.linspace(0, len(table) - 1, 9, dtype=int))
    sample = table.iloc[sample_idx].copy()

    print("Einstein Field Equations MVP (flat FRW, SI units)")
    print(f"H0 = {cfg.h0_km_s_mpc:.3f} km/s/Mpc = {bg.h0_si:.6e} s^-1")
    print(
        "Omega_m={:.6f}, Omega_r={:.6e}, Omega_lambda={:.6f}".format(
            cfg.omega_m,
            cfg.omega_r,
            cfg.omega_lambda,
        )
    )
    print(f"Lambda = {bg.lambda_si:.6e} m^-2")
    print()
    print("Residual diagnostics (relative max):")
    print(f"  Eq1 (H^2):     {diag.eq1_rel_max:.3e}")
    print(f"  dot(H):        {diag.dot_h_rel_max:.3e}")
    print(f"  acceleration:  {diag.accel_rel_max:.3e}")
    print(f"  H model diff:  {diag.h_model_rel_diff:.3e}")
    print()
    print("Sample trajectory over scale factor a:")
    print(sample.to_string(index=False, justify="right", float_format=lambda x: f"{x:.6e}"))


if __name__ == "__main__":
    main()
