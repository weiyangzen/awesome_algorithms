"""Electroweak theory MVP.

This script builds a minimal, auditable numerical demonstration for the
SU(2)_L x U(1)_Y gauge sector of the Standard Model:
1) one-loop running of g1 (hypercharge) and g2 (weak isospin),
2) consistency check between analytic and numerical RG solutions,
3) derived observables: sin^2(theta_W), alpha_em, m_W, m_Z, rho,
4) neutral-current couplings (vector/axial) for representative fermions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ElectroweakConfig:
    """Configuration for the electroweak one-loop running demo."""

    mu_ref_gev: float = 91.1876  # Z-pole reference scale
    alpha_em_ref: float = 1.0 / 127.95
    sin2_theta_w_ref: float = 0.23122
    higgs_vev_gev: float = 246.22

    # One-loop beta coefficients for g1 (hypercharge) and g2 (weak isospin).
    # Conventions: d g_i / d ln(mu) = (b_i / (16*pi^2)) * g_i^3
    b1: float = 41.0 / 6.0
    b2: float = -19.0 / 6.0


def initial_couplings(cfg: ElectroweakConfig) -> tuple[float, float, float]:
    """Infer (g1, g2, e) from alpha_em and sin^2(theta_W) at the reference scale."""
    if not (0.0 < cfg.alpha_em_ref < 1.0):
        raise ValueError("alpha_em_ref must be in (0, 1).")
    if not (0.0 < cfg.sin2_theta_w_ref < 1.0):
        raise ValueError("sin2_theta_w_ref must be in (0, 1).")

    e = float(np.sqrt(4.0 * np.pi * cfg.alpha_em_ref))
    s = float(np.sqrt(cfg.sin2_theta_w_ref))
    c = float(np.sqrt(1.0 - cfg.sin2_theta_w_ref))

    g2 = e / s
    g1 = e / c
    return g1, g2, e


def one_loop_beta(g: float, b: float) -> float:
    """One-loop gauge beta function: d g / d ln(mu)."""
    if g <= 0.0:
        raise ValueError("Gauge coupling g must be positive.")
    return (b / (16.0 * np.pi**2)) * g**3


def integrate_gauge_couplings(
    mu_grid: np.ndarray,
    mu_ref: float,
    g1_ref: float,
    g2_ref: float,
    cfg: ElectroweakConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate one-loop RG equations for (g1, g2)."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if mu_grid.ndim != 1 or mu_grid.size < 2:
        raise ValueError("mu_grid must be 1D with at least two points.")
    if np.any(mu_grid <= 0.0):
        raise ValueError("mu_grid must contain positive values.")
    if not np.all(np.diff(mu_grid) > 0.0):
        raise ValueError("mu_grid must be strictly ascending.")
    if mu_ref <= 0.0:
        raise ValueError("mu_ref must be positive.")
    if not np.isclose(mu_grid[0], mu_ref):
        raise ValueError("mu_grid[0] must match mu_ref.")

    t_eval = np.log(mu_grid)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        g1 = max(float(y[0]), 1e-14)
        g2 = max(float(y[1]), 1e-14)
        return np.array(
            [
                one_loop_beta(g1, cfg.b1),
                one_loop_beta(g2, cfg.b2),
            ],
            dtype=float,
        )

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=np.array([g1_ref, g2_ref], dtype=float),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    g1_vals = sol.y[0]
    g2_vals = sol.y[1]
    if np.any(~np.isfinite(g1_vals)) or np.any(~np.isfinite(g2_vals)):
        raise RuntimeError("Encountered non-finite couplings.")
    if np.any(g1_vals <= 0.0) or np.any(g2_vals <= 0.0):
        raise RuntimeError("Encountered non-positive couplings.")

    return g1_vals, g2_vals


def analytic_running(mu_grid: np.ndarray, mu_ref: float, g_ref: float, b: float) -> np.ndarray:
    """Closed-form one-loop running for d g / d ln(mu) = (b/(16*pi^2)) g^3."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if np.any(mu_grid <= 0.0):
        raise ValueError("mu_grid must be positive.")
    if mu_ref <= 0.0 or g_ref <= 0.0:
        raise ValueError("mu_ref and g_ref must be positive.")

    log_ratio = np.log(mu_grid / mu_ref)
    denom = (1.0 / g_ref**2) - (b / (8.0 * np.pi**2)) * log_ratio
    if np.any(denom <= 0.0):
        raise ValueError("Analytic denominator became non-positive (Landau-pole region).")
    return 1.0 / np.sqrt(denom)


def electroweak_observables(g1: np.ndarray, g2: np.ndarray, higgs_vev_gev: float) -> pd.DataFrame:
    """Compute derived electroweak observables from (g1, g2)."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    if g1.shape != g2.shape:
        raise ValueError("g1 and g2 must share shape.")
    if np.any(g1 <= 0.0) or np.any(g2 <= 0.0):
        raise ValueError("g1 and g2 must be positive.")
    if higgs_vev_gev <= 0.0:
        raise ValueError("higgs_vev_gev must be positive.")

    denom = g1**2 + g2**2
    sin2_theta_w = g1**2 / denom
    cos2_theta_w = g2**2 / denom

    e_vals = g1 * g2 / np.sqrt(denom)
    alpha_em = e_vals**2 / (4.0 * np.pi)

    m_w = 0.5 * higgs_vev_gev * g2
    m_z = 0.5 * higgs_vev_gev * np.sqrt(denom)

    rho = m_w**2 / (m_z**2 * np.maximum(cos2_theta_w, 1e-14))

    return pd.DataFrame(
        {
            "sin2_theta_w": sin2_theta_w,
            "cos2_theta_w": cos2_theta_w,
            "e": e_vals,
            "alpha_em": alpha_em,
            "mW_GeV": m_w,
            "mZ_GeV": m_z,
            "rho_tree": rho,
            "mW_over_mZ": m_w / m_z,
            "cos_theta_w": np.sqrt(cos2_theta_w),
        }
    )


def neutral_current_couplings(sin2_theta_w: float) -> pd.DataFrame:
    """Return Z neutral-current couplings in (gV, gA, gL, gR) convention."""
    if not (0.0 < sin2_theta_w < 1.0):
        raise ValueError("sin2_theta_w must be in (0, 1).")

    # (name, T3, Q)
    fermions = [
        ("nu_e", +0.5, 0.0),
        ("e", -0.5, -1.0),
        ("u", +0.5, 2.0 / 3.0),
        ("d", -0.5, -1.0 / 3.0),
    ]

    rows: list[dict[str, float | str]] = []
    for name, t3, charge in fermions:
        g_v = t3 - 2.0 * charge * sin2_theta_w
        g_a = t3
        g_l = t3 - charge * sin2_theta_w
        g_r = -charge * sin2_theta_w
        rows.append(
            {
                "fermion": name,
                "T3": t3,
                "Q": charge,
                "gV": g_v,
                "gA": g_a,
                "gL": g_l,
                "gR": g_r,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = ElectroweakConfig()

    g1_ref, g2_ref, e_ref = initial_couplings(cfg)
    mu_grid = cfg.mu_ref_gev * np.geomspace(1.0, 100.0, 16)

    g1_numeric, g2_numeric = integrate_gauge_couplings(
        mu_grid=mu_grid,
        mu_ref=cfg.mu_ref_gev,
        g1_ref=g1_ref,
        g2_ref=g2_ref,
        cfg=cfg,
    )

    g1_analytic = analytic_running(mu_grid, cfg.mu_ref_gev, g1_ref, cfg.b1)
    g2_analytic = analytic_running(mu_grid, cfg.mu_ref_gev, g2_ref, cfg.b2)

    running_df = pd.DataFrame(
        {
            "mu_GeV": mu_grid,
            "g1_numeric": g1_numeric,
            "g2_numeric": g2_numeric,
            "g1_analytic": g1_analytic,
            "g2_analytic": g2_analytic,
        }
    )
    running_df["rel_err_g1"] = np.abs(
        (running_df["g1_numeric"] - running_df["g1_analytic"]) / np.maximum(running_df["g1_analytic"], 1e-14)
    )
    running_df["rel_err_g2"] = np.abs(
        (running_df["g2_numeric"] - running_df["g2_analytic"]) / np.maximum(running_df["g2_analytic"], 1e-14)
    )

    obs_df = electroweak_observables(g1_numeric, g2_numeric, cfg.higgs_vev_gev)
    running_df = pd.concat([running_df, obs_df], axis=1)

    sin2_at_mz = float(running_df.loc[0, "sin2_theta_w"])
    sin2_at_high = float(running_df.loc[running_df.index[-1], "sin2_theta_w"])

    nc_mz = neutral_current_couplings(sin2_at_mz)
    nc_mz.insert(0, "scale", "mu=MZ")

    nc_high = neutral_current_couplings(sin2_at_high)
    nc_high.insert(0, "scale", "mu=100*MZ")

    nc_df = pd.concat([nc_mz, nc_high], ignore_index=True)

    print("=== Electroweak Input Summary ===")
    print(f"mu_ref = {cfg.mu_ref_gev:.4f} GeV")
    print(f"alpha_em(mu_ref) = {cfg.alpha_em_ref:.8f}")
    print(f"sin^2(theta_W)(mu_ref) = {cfg.sin2_theta_w_ref:.8f}")
    print(f"g1(mu_ref) = {g1_ref:.8f}, g2(mu_ref) = {g2_ref:.8f}, e(mu_ref) = {e_ref:.8f}")
    print(f"beta coefficients: b1={cfg.b1:.8f}, b2={cfg.b2:.8f}")
    print()

    print("=== Running Couplings and Derived Observables ===")
    print(running_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Neutral Current Couplings (Z-fermion) ===")
    print(nc_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    max_rel_err_g1 = float(running_df["rel_err_g1"].max())
    max_rel_err_g2 = float(running_df["rel_err_g2"].max())
    max_rho_dev = float(np.max(np.abs(running_df["rho_tree"] - 1.0)))
    max_mass_relation_dev = float(np.max(np.abs(running_df["mW_over_mZ"] - running_df["cos_theta_w"])))

    # Deterministic sanity checks for this MVP.
    assert max_rel_err_g1 < 5e-7, f"g1 numeric/analytic mismatch too large: {max_rel_err_g1:.3e}"
    assert max_rel_err_g2 < 5e-7, f"g2 numeric/analytic mismatch too large: {max_rel_err_g2:.3e}"
    assert np.all(np.diff(running_df["g1_numeric"]) > 0.0), "g1 should increase with mu at one-loop."
    assert np.all(np.diff(running_df["g2_numeric"]) < 0.0), "g2 should decrease with mu at one-loop."
    assert np.all(np.diff(running_df["sin2_theta_w"]) > 0.0), "sin^2(theta_W) should increase in this range."
    assert max_rho_dev < 2e-12, f"Tree-level rho deviates too much: {max_rho_dev:.3e}"
    assert max_mass_relation_dev < 2e-12, f"mW/mZ vs cos(theta_W) mismatch: {max_mass_relation_dev:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
