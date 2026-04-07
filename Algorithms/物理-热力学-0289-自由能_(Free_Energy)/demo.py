"""Minimal runnable MVP for free-energy minimization.

We model an ideal binary mixture A/B at fixed (T, P):

    g(x) = (1-x) * mu_A0 + x * mu_B0
           + R*T * [(1-x) ln(1-x) + x ln(x)]

where x is the mole fraction of component B and g is molar Gibbs free energy.
At equilibrium, g is minimized with respect to x in (0, 1).

This script demonstrates:
1) Numerical minimization (scipy.optimize.minimize_scalar),
2) Analytic minimizer from d g / d x = 0,
3) Consistency checks (gradient near zero, positive curvature, lower than pure states).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)
X_EPS = 1e-12


@dataclass(frozen=True)
class BinaryFreeEnergyModel:
    """Model parameters for an ideal binary mixture."""

    temperature: float  # K
    mu_a0: float  # J/mol
    mu_b0: float  # J/mol

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0 K, got {self.temperature}")


def _check_open_fraction(x_b: float) -> None:
    if not (0.0 < x_b < 1.0):
        raise ValueError(f"x_b must satisfy 0 < x_b < 1, got {x_b}")


def gibbs_molar(model: BinaryFreeEnergyModel, x_b: float) -> float:
    """Molar Gibbs free energy g(x_b) for ideal binary mixture."""
    model.validate()
    _check_open_fraction(x_b)

    x_a = 1.0 - x_b
    energetic = x_a * model.mu_a0 + x_b * model.mu_b0
    mixing_entropy_term = R_UNIVERSAL * model.temperature * (
        x_a * np.log(x_a) + x_b * np.log(x_b)
    )
    return float(energetic + mixing_entropy_term)


def gibbs_gradient(model: BinaryFreeEnergyModel, x_b: float) -> float:
    """First derivative d g / d x_b."""
    model.validate()
    _check_open_fraction(x_b)

    delta_mu = model.mu_b0 - model.mu_a0
    return float(delta_mu + R_UNIVERSAL * model.temperature * np.log(x_b / (1.0 - x_b)))


def gibbs_hessian(model: BinaryFreeEnergyModel, x_b: float) -> float:
    """Second derivative d^2 g / d x_b^2."""
    model.validate()
    _check_open_fraction(x_b)

    return float(
        R_UNIVERSAL * model.temperature * (1.0 / x_b + 1.0 / (1.0 - x_b))
    )


def analytic_equilibrium_fraction(model: BinaryFreeEnergyModel) -> float:
    """Closed-form equilibrium x_b from d g / d x_b = 0."""
    model.validate()
    z = (model.mu_b0 - model.mu_a0) / (R_UNIVERSAL * model.temperature)
    z = float(np.clip(z, -700.0, 700.0))
    x_b = 1.0 / (1.0 + np.exp(z))
    return float(x_b)


def numerical_equilibrium_fraction(
    model: BinaryFreeEnergyModel,
    xatol: float = 1e-12,
) -> dict[str, float]:
    """Numerically minimize g(x_b) on (0, 1)."""
    model.validate()

    result = minimize_scalar(
        lambda x: gibbs_molar(model, float(x)),
        bounds=(X_EPS, 1.0 - X_EPS),
        method="bounded",
        options={"xatol": xatol, "maxiter": 500},
    )

    if not result.success:
        raise RuntimeError(f"Minimization failed: {result.message}")

    x_star = float(result.x)
    g_star = float(result.fun)
    return {
        "x_b_eq_numeric": x_star,
        "g_eq_numeric": g_star,
        "nfev": float(result.nfev),
    }


def pure_state_gibbs(model: BinaryFreeEnergyModel) -> tuple[float, float]:
    """Pure-state molar Gibbs energies for A and B."""
    model.validate()
    return float(model.mu_a0), float(model.mu_b0)


def composition_regime(x_b: float) -> str:
    if x_b < 0.05:
        return "A-rich"
    if x_b > 0.95:
        return "B-rich"
    return "mixed"


def build_equilibrium_scan(
    temperatures: np.ndarray,
    mu_a0: float,
    mu_b0: float,
) -> pd.DataFrame:
    """Scan equilibrium composition over a temperature grid."""
    rows: list[dict[str, float | str]] = []

    for temp in temperatures:
        model = BinaryFreeEnergyModel(temperature=float(temp), mu_a0=mu_a0, mu_b0=mu_b0)

        numeric = numerical_equilibrium_fraction(model)
        x_num = float(numeric["x_b_eq_numeric"])
        g_num = float(numeric["g_eq_numeric"])

        x_ana = analytic_equilibrium_fraction(model)
        grad = gibbs_gradient(model, x_num)
        curvature = gibbs_hessian(model, x_num)
        g_pure_a, g_pure_b = pure_state_gibbs(model)

        rows.append(
            {
                "T_K": float(temp),
                "x_b_eq_numeric": x_num,
                "x_b_eq_analytic": x_ana,
                "abs_x_error": abs(x_num - x_ana),
                "g_eq_J_per_mol": g_num,
                "g_pure_A_J_per_mol": g_pure_a,
                "g_pure_B_J_per_mol": g_pure_b,
                "g_margin_to_best_pure": min(g_pure_a, g_pure_b) - g_num,
                "dgdx_at_numeric": grad,
                "d2gdx2_at_numeric": curvature,
                "nfev": float(numeric["nfev"]),
                "regime": composition_regime(x_num),
            }
        )

    return pd.DataFrame(rows)


def build_free_energy_profile(
    model: BinaryFreeEnergyModel,
    num_points: int = 21,
) -> pd.DataFrame:
    """Build g(x_b) profile around the equilibrium point."""
    if num_points < 5:
        raise ValueError("num_points must be at least 5")

    x_grid = np.linspace(0.01, 0.99, num_points)
    g_values = np.array([gibbs_molar(model, float(x)) for x in x_grid], dtype=np.float64)

    x_eq = analytic_equilibrium_fraction(model)
    g_eq = gibbs_molar(model, x_eq)

    return pd.DataFrame(
        {
            "x_b": x_grid,
            "g_J_per_mol": g_values,
            "delta_g_to_eq": g_values - g_eq,
        }
    )


def main() -> None:
    mu_a0 = -10_000.0  # J/mol
    mu_b0 = -9_600.0  # J/mol (B has higher standard chemical potential)
    temperatures = np.array([250.0, 300.0, 400.0, 600.0, 900.0], dtype=np.float64)

    scan_df = build_equilibrium_scan(temperatures=temperatures, mu_a0=mu_a0, mu_b0=mu_b0)

    profile_model = BinaryFreeEnergyModel(temperature=300.0, mu_a0=mu_a0, mu_b0=mu_b0)
    profile_df = build_free_energy_profile(profile_model, num_points=15)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Free Energy MVP: ideal binary mixture Gibbs minimization")
    print("g(x) = x*mu_B0 + (1-x)*mu_A0 + R*T[(1-x)ln(1-x)+xlnx]")
    print()
    print("Equilibrium scan over temperature:")
    print(scan_df.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print()
    print("Free-energy profile at T=300 K (subset):")
    print(profile_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Deterministic correctness checks.
    max_x_error = float(scan_df["abs_x_error"].max())
    max_abs_grad = float(np.max(np.abs(scan_df["dgdx_at_numeric"].to_numpy(dtype=np.float64))))
    min_curvature = float(scan_df["d2gdx2_at_numeric"].min())
    min_margin = float(scan_df["g_margin_to_best_pure"].min())

    assert max_x_error < 5e-8, f"analytic/numeric x mismatch too large: {max_x_error}"
    assert max_abs_grad < 5e-4, f"gradient at numeric optimum too large: {max_abs_grad}"
    assert min_curvature > 0.0, f"minimum curvature must be positive, got {min_curvature}"
    assert min_margin > 0.0, f"equilibrium g should be below pure states, got margin {min_margin}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
