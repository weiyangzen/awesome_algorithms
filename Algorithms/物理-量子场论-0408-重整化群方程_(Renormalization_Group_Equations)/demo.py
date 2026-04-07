"""Minimal MVP for Renormalization Group Equations (RGE) in QFT.

This script demonstrates a unified one-coupling RG flow framework with:
1) QCD running coupling (one-loop and two-loop),
2) QED one-loop running coupling,
3) fixed-point analysis, including a Banks-Zaks point for QCD at N_f=16 (two-loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class BetaModel:
    """One-coupling beta polynomial: beta(g) = c1*g + c2*g^2 + c3*g^3."""

    name: str
    c1: float
    c2: float
    c3: float


def qcd_coefficients(n_f: int) -> tuple[float, float, float, float]:
    """Return (beta0, beta1, c2, c3) for d alpha / d ln(mu) = c2*alpha^2 + c3*alpha^3."""
    if n_f < 0:
        raise ValueError("n_f must be non-negative.")

    beta0 = 11.0 - (2.0 / 3.0) * n_f
    beta1 = 102.0 - (38.0 / 3.0) * n_f

    c2 = -beta0 / (2.0 * np.pi)
    c3 = -beta1 / (4.0 * np.pi**2)
    return beta0, beta1, c2, c3


def qed_one_loop_coefficient(n_f: int) -> float:
    """Return c2 for QED one-loop: d alpha / d ln(mu) = c2 * alpha^2."""
    if n_f <= 0:
        raise ValueError("QED n_f must be positive.")
    return (2.0 * n_f) / (3.0 * np.pi)


def beta_polynomial(g: float, model: BetaModel) -> float:
    """Compute beta(g) = c1*g + c2*g^2 + c3*g^3."""
    if g <= 0.0:
        raise ValueError("Coupling g must be positive.")
    return model.c1 * g + model.c2 * g**2 + model.c3 * g**3


def beta_derivative(g: float, model: BetaModel) -> float:
    """Compute beta'(g)."""
    return model.c1 + 2.0 * model.c2 * g + 3.0 * model.c3 * g**2


def integrate_rge(mu_grid: np.ndarray, mu_ref: float, g_ref: float, model: BetaModel) -> np.ndarray:
    """Integrate d g / d ln(mu) = beta(g) on a strictly ascending mu grid."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if mu_grid.ndim != 1 or mu_grid.size < 2:
        raise ValueError("mu_grid must be a 1D array with at least two points.")
    if np.any(mu_grid <= 0.0):
        raise ValueError("mu_grid must contain strictly positive values.")
    if not np.all(np.diff(mu_grid) > 0.0):
        raise ValueError("mu_grid must be strictly ascending.")
    if mu_ref <= 0.0 or g_ref <= 0.0:
        raise ValueError("mu_ref and g_ref must be positive.")
    if not np.isclose(mu_grid[0], mu_ref):
        raise ValueError("mu_grid[0] must equal mu_ref.")

    t_eval = np.log(mu_grid)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        g = max(float(y[0]), 1e-14)
        val = model.c1 * g + model.c2 * g**2 + model.c3 * g**3
        return np.array([val], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=np.array([g_ref], dtype=float),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed for {model.name}: {sol.message}")

    g_values = sol.y[0]
    if np.any(~np.isfinite(g_values)) or np.any(g_values <= 0.0):
        raise RuntimeError(f"Non-physical coupling encountered for {model.name}.")

    return g_values


def one_loop_closed_form(mu_grid: np.ndarray, mu_ref: float, g_ref: float, c2: float) -> np.ndarray:
    """Closed form for d g / d ln(mu) = c2 * g^2."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    log_ratio = np.log(mu_grid / mu_ref)
    denom = 1.0 - c2 * g_ref * log_ratio
    if np.any(denom <= 0.0):
        raise ValueError("Closed-form denominator reached non-positive value (Landau-pole region).")
    return g_ref / denom


def fixed_points(model: BetaModel, atol: float = 1e-12) -> list[float]:
    """Return all non-negative real fixed points of beta(g)=0."""
    # beta(g) = g * (c1 + c2*g + c3*g^2)
    roots: list[float] = [0.0]

    if abs(model.c3) <= atol:
        if abs(model.c2) > atol:
            candidate = -model.c1 / model.c2
            if candidate >= -atol:
                roots.append(max(0.0, candidate))
    else:
        disc = model.c2**2 - 4.0 * model.c3 * model.c1
        if disc >= -atol:
            disc = max(disc, 0.0)
            sqrt_disc = float(np.sqrt(disc))
            cand1 = (-model.c2 + sqrt_disc) / (2.0 * model.c3)
            cand2 = (-model.c2 - sqrt_disc) / (2.0 * model.c3)
            for candidate in (cand1, cand2):
                if candidate >= -atol:
                    roots.append(max(0.0, candidate))

    unique_roots = sorted(set(round(x, 12) for x in roots))
    return [float(x) for x in unique_roots]


def classify_fixed_point(g_star: float, model: BetaModel, tol: float = 1e-10) -> str:
    """Classify fixed point stability for increasing mu (UV direction)."""
    slope = beta_derivative(g_star, model)
    if slope < -tol:
        return "UV-attractive"
    if slope > tol:
        return "UV-repulsive"
    return "marginal"


def make_coefficient_table() -> pd.DataFrame:
    """Build a compact coefficient summary table."""
    beta0_5, beta1_5, c2_qcd_5, c3_qcd_5 = qcd_coefficients(5)
    beta0_16, beta1_16, c2_qcd_16, c3_qcd_16 = qcd_coefficients(16)
    c2_qed_1 = qed_one_loop_coefficient(1)

    rows = [
        {
            "model": "QCD N_f=5 (1-loop)",
            "beta0": beta0_5,
            "beta1": beta1_5,
            "c1": 0.0,
            "c2": c2_qcd_5,
            "c3": 0.0,
        },
        {
            "model": "QCD N_f=5 (2-loop)",
            "beta0": beta0_5,
            "beta1": beta1_5,
            "c1": 0.0,
            "c2": c2_qcd_5,
            "c3": c3_qcd_5,
        },
        {
            "model": "QED N_f=1 (1-loop)",
            "beta0": np.nan,
            "beta1": np.nan,
            "c1": 0.0,
            "c2": c2_qed_1,
            "c3": 0.0,
        },
        {
            "model": "QCD N_f=16 (2-loop)",
            "beta0": beta0_16,
            "beta1": beta1_16,
            "c1": 0.0,
            "c2": c2_qcd_16,
            "c3": c3_qcd_16,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    mu_ref = 2.0
    alpha_qcd_ref = 0.30
    alpha_qed_ref = 1.0 / 137.0
    mu_grid = mu_ref * np.geomspace(1.0, 1.0e4, 18)

    _, _, c2_qcd_5, c3_qcd_5 = qcd_coefficients(5)
    c2_qed_1 = qed_one_loop_coefficient(1)

    model_qcd_1l = BetaModel(name="QCD N_f=5 (1-loop)", c1=0.0, c2=c2_qcd_5, c3=0.0)
    model_qcd_2l = BetaModel(name="QCD N_f=5 (2-loop)", c1=0.0, c2=c2_qcd_5, c3=c3_qcd_5)
    model_qed_1l = BetaModel(name="QED N_f=1 (1-loop)", c1=0.0, c2=c2_qed_1, c3=0.0)

    alpha_qcd_1l_numeric = integrate_rge(mu_grid, mu_ref, alpha_qcd_ref, model_qcd_1l)
    alpha_qcd_2l_numeric = integrate_rge(mu_grid, mu_ref, alpha_qcd_ref, model_qcd_2l)
    alpha_qed_1l_numeric = integrate_rge(mu_grid, mu_ref, alpha_qed_ref, model_qed_1l)

    alpha_qcd_1l_analytic = one_loop_closed_form(mu_grid, mu_ref, alpha_qcd_ref, c2_qcd_5)
    alpha_qed_1l_analytic = one_loop_closed_form(mu_grid, mu_ref, alpha_qed_ref, c2_qed_1)

    report_df = pd.DataFrame(
        {
            "mu_GeV": mu_grid,
            "alpha_qcd_1l_analytic": alpha_qcd_1l_analytic,
            "alpha_qcd_1l_numeric": alpha_qcd_1l_numeric,
            "alpha_qcd_2l_numeric": alpha_qcd_2l_numeric,
            "alpha_qed_1l_analytic": alpha_qed_1l_analytic,
            "alpha_qed_1l_numeric": alpha_qed_1l_numeric,
        }
    )
    report_df["rel_err_qcd_1l"] = np.abs(
        (report_df["alpha_qcd_1l_numeric"] - report_df["alpha_qcd_1l_analytic"])
        / np.maximum(report_df["alpha_qcd_1l_analytic"], 1e-14)
    )
    report_df["rel_err_qed_1l"] = np.abs(
        (report_df["alpha_qed_1l_numeric"] - report_df["alpha_qed_1l_analytic"])
        / np.maximum(report_df["alpha_qed_1l_analytic"], 1e-14)
    )

    # Banks-Zaks fixed point candidate for QCD N_f=16 at two-loop.
    _, _, c2_qcd_16, c3_qcd_16 = qcd_coefficients(16)
    model_qcd_16_2l = BetaModel(name="QCD N_f=16 (2-loop)", c1=0.0, c2=c2_qcd_16, c3=c3_qcd_16)
    fp_values = fixed_points(model_qcd_16_2l)
    fp_table = pd.DataFrame(
        {
            "g_star": fp_values,
            "beta(g_star)": [
                model_qcd_16_2l.c1 * g + model_qcd_16_2l.c2 * g**2 + model_qcd_16_2l.c3 * g**3
                for g in fp_values
            ],
            "stability": [classify_fixed_point(g, model_qcd_16_2l) for g in fp_values],
        }
    )

    print("=== Coefficient Summary ===")
    print(make_coefficient_table().to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Running Couplings ===")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== QCD N_f=16 (two-loop) Fixed Points ===")
    print(fp_table.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    max_rel_err_qcd = float(report_df["rel_err_qcd_1l"].max())
    max_rel_err_qed = float(report_df["rel_err_qed_1l"].max())

    positive_nonzero_fp = [g for g in fp_values if g > 1e-8]

    # Deterministic sanity checks for this MVP.
    assert max_rel_err_qcd < 5e-6, f"QCD one-loop mismatch too large: {max_rel_err_qcd:.3e}"
    assert max_rel_err_qed < 5e-6, f"QED one-loop mismatch too large: {max_rel_err_qed:.3e}"
    assert np.all(np.diff(alpha_qcd_1l_numeric) < 0.0), "QCD one-loop alpha should decrease with mu."
    assert np.all(np.diff(alpha_qcd_2l_numeric) < 0.0), "QCD two-loop alpha should decrease with mu."
    assert np.all(np.diff(alpha_qed_1l_numeric) > 0.0), "QED one-loop alpha should increase with mu."
    assert len(positive_nonzero_fp) >= 1, "Expected a positive non-trivial fixed point for QCD N_f=16 at two-loop."

    fp_nz = positive_nonzero_fp[0]
    beta_at_fp = model_qcd_16_2l.c1 * fp_nz + model_qcd_16_2l.c2 * fp_nz**2 + model_qcd_16_2l.c3 * fp_nz**3
    assert abs(beta_at_fp) < 1e-10, f"beta(g*) should be ~0, got {beta_at_fp:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
