"""QED one-loop running coupling MVP.

This script demonstrates:
1) one-loop QED beta-function driven running of alpha(mu),
2) numerical ODE solution vs one-loop closed form,
3) Landau pole estimation in a toy fixed-flavor setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class QEDScenario:
    name: str
    charge_squares: tuple[float, ...]
    mu_ref_gev: float
    alpha_ref: float
    mu_max_factor: float


def qed_one_loop_beta_coefficient(charge_squares: tuple[float, ...]) -> float:
    """Return b1 in d alpha / d ln(mu) = b1 * alpha^2."""
    if len(charge_squares) == 0:
        raise ValueError("charge_squares must contain at least one active fermion.")
    charge_squares_arr = np.asarray(charge_squares, dtype=float)
    if np.any(charge_squares_arr <= 0.0):
        raise ValueError("All charge_squares entries must be positive.")
    sum_q2 = float(np.sum(charge_squares_arr))
    return (2.0 / (3.0 * np.pi)) * sum_q2


def beta_alpha_one_loop(alpha: float, b1: float) -> float:
    """One-loop QED beta function in terms of alpha(mu)."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if b1 <= 0.0:
        raise ValueError("b1 must be positive for QED with charged fermions.")
    return b1 * alpha**2


def alpha_one_loop_closed_form(mu: np.ndarray, mu_ref: float, alpha_ref: float, b1: float) -> np.ndarray:
    """Closed-form one-loop running alpha(mu)."""
    mu = np.asarray(mu, dtype=float)
    if np.any(mu <= 0.0):
        raise ValueError("All mu values must be positive.")
    if mu_ref <= 0.0:
        raise ValueError("mu_ref must be positive.")
    if alpha_ref <= 0.0:
        raise ValueError("alpha_ref must be positive.")

    denom = 1.0 - alpha_ref * b1 * np.log(mu / mu_ref)
    if np.any(denom <= 0.0):
        raise ValueError("Encountered Landau-pole crossing in requested mu range.")
    return alpha_ref / denom


def landau_pole_scale(mu_ref: float, alpha_ref: float, b1: float) -> float:
    """Estimate one-loop Landau pole mu_pole = mu_ref * exp(1/(alpha_ref*b1))."""
    if mu_ref <= 0.0 or alpha_ref <= 0.0 or b1 <= 0.0:
        raise ValueError("mu_ref, alpha_ref, b1 must all be positive.")
    exponent = 1.0 / (alpha_ref * b1)
    return float(mu_ref * np.exp(exponent))


def integrate_running_alpha(
    mu_grid: np.ndarray,
    mu_ref: float,
    alpha_ref: float,
    b1: float,
) -> np.ndarray:
    """Integrate d alpha / d ln(mu) = b1 * alpha^2 over an ascending mu grid."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if mu_grid.ndim != 1 or mu_grid.size < 2:
        raise ValueError("mu_grid must be 1D with at least two points.")
    if np.any(mu_grid <= 0.0):
        raise ValueError("mu_grid must be positive.")
    if not np.all(np.diff(mu_grid) > 0.0):
        raise ValueError("mu_grid must be strictly ascending.")
    if not np.isclose(mu_grid[0], mu_ref):
        raise ValueError("mu_grid[0] must be equal to mu_ref.")

    t_eval = np.log(mu_grid)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        alpha = max(float(y[0]), 1e-14)
        return np.array([beta_alpha_one_loop(alpha=alpha, b1=b1)], dtype=float)

    solution = solve_ivp(
        rhs,
        t_span=t_span,
        y0=np.array([alpha_ref], dtype=float),
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    alpha_values = solution.y[0]
    if np.any(~np.isfinite(alpha_values)) or np.any(alpha_values <= 0.0):
        raise RuntimeError("Non-finite or non-positive alpha encountered in integration.")
    return alpha_values


def analyze_scenario(scenario: QEDScenario, n_grid: int = 16) -> tuple[pd.DataFrame, dict[str, float]]:
    b1 = qed_one_loop_beta_coefficient(scenario.charge_squares)
    mu_grid = scenario.mu_ref_gev * np.geomspace(1.0, scenario.mu_max_factor, n_grid)

    alpha_closed = alpha_one_loop_closed_form(
        mu=mu_grid,
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        b1=b1,
    )
    alpha_numeric = integrate_running_alpha(
        mu_grid=mu_grid,
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        b1=b1,
    )
    mu_pole = landau_pole_scale(
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        b1=b1,
    )

    report_df = pd.DataFrame(
        {
            "mu_GeV": mu_grid,
            "alpha_closed_form": alpha_closed,
            "alpha_numeric": alpha_numeric,
        }
    )
    report_df["abs_err"] = np.abs(report_df["alpha_numeric"] - report_df["alpha_closed_form"])

    max_rel_err = float(
        np.max(
            np.abs(
                (alpha_numeric - alpha_closed)
                / np.maximum(alpha_closed, 1e-14)
            )
        )
    )

    summary = {
        "sum_q2": float(np.sum(np.asarray(scenario.charge_squares, dtype=float))),
        "b1": b1,
        "mu_pole_gev": mu_pole,
        "max_rel_err": max_rel_err,
        "alpha_start": float(alpha_numeric[0]),
        "alpha_end": float(alpha_numeric[-1]),
    }
    return report_df, summary


def main() -> None:
    alpha_0 = 1.0 / 137.035999084
    electron_mass_gev = 0.00051099895

    baseline = QEDScenario(
        name="e-only toy model",
        charge_squares=(1.0,),
        mu_ref_gev=electron_mass_gev,
        alpha_ref=alpha_0,
        mu_max_factor=1.0e6,
    )
    comparison = QEDScenario(
        name="3-lepton toy model",
        charge_squares=(1.0, 1.0, 1.0),
        mu_ref_gev=electron_mass_gev,
        alpha_ref=alpha_0,
        mu_max_factor=1.0e6,
    )

    baseline_df, baseline_summary = analyze_scenario(baseline)
    _, comparison_summary = analyze_scenario(comparison)

    coeff_df = pd.DataFrame(
        [
            {
                "scenario": baseline.name,
                "sum_q2": baseline_summary["sum_q2"],
                "b1": baseline_summary["b1"],
                "log10(mu_pole/GeV)": np.log10(baseline_summary["mu_pole_gev"]),
            },
            {
                "scenario": comparison.name,
                "sum_q2": comparison_summary["sum_q2"],
                "b1": comparison_summary["b1"],
                "log10(mu_pole/GeV)": np.log10(comparison_summary["mu_pole_gev"]),
            },
        ]
    )

    print("=== QED One-Loop Coefficient Summary ===")
    print(coeff_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(
        "Baseline scenario: "
        f"{baseline.name}, mu_ref={baseline.mu_ref_gev:.9f} GeV, alpha_ref={baseline.alpha_ref:.10f}"
    )
    print("=== Running Coupling Table (Baseline) ===")
    print(baseline_df.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Assertions for physics trend and numerical consistency.
    assert baseline_summary["max_rel_err"] < 1e-6, (
        f"Numeric vs analytic mismatch too large: {baseline_summary['max_rel_err']:.3e}"
    )
    assert baseline_summary["alpha_end"] > baseline_summary["alpha_start"], (
        "QED alpha should increase with energy scale in one-loop running."
    )
    assert baseline_summary["mu_pole_gev"] > float(baseline_df["mu_GeV"].iloc[-1]), (
        "Configured mu range should stay below Landau pole."
    )
    assert comparison_summary["alpha_end"] > baseline_summary["alpha_end"], (
        "Larger sum_q2 should yield faster UV growth of alpha."
    )
    assert comparison_summary["mu_pole_gev"] < baseline_summary["mu_pole_gev"], (
        "Larger sum_q2 should lower the Landau pole scale."
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
