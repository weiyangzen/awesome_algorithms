"""Chemical potential: minimal runnable numerical MVP.

This script demonstrates two core computational tasks around chemical potential:
1) Verify definition μ_i = (∂G/∂n_i)_{T,P,n_j} via finite differences.
2) Solve a two-phase partition equilibrium by enforcing μ_A^alpha = μ_A^beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


R_GAS = 8.314462618  # J/(mol*K)
P0 = 1.0e5  # Pa


@dataclass(frozen=True)
class ChemicalPotentialConfig:
    """Configuration for the numerical checks."""

    temperature: float = 298.15  # K
    pressure: float = 1.5e5  # Pa
    mu_ref_a: float = -8200.0  # J/mol
    mu_ref_b: float = -9100.0  # J/mol
    fd_step: float = 1e-6  # mol
    tolerance: float = 2e-5  # J/mol
    grid_points: int = 9


@dataclass(frozen=True)
class PartitionConfig:
    """Toy two-phase partition setup for component A."""

    total_a: float = 1.4  # mol
    solvent_alpha: float = 1.2  # mol (component B, fixed in alpha)
    solvent_beta: float = 2.0  # mol (component B, fixed in beta)
    mu_shift_alpha: float = -250.0  # J/mol
    mu_shift_beta: float = 350.0  # J/mol
    root_tol: float = 1e-10
    max_iter: int = 120


def standard_chemical_potential(mu_ref: float, temperature: float, pressure: float) -> float:
    """Return μ_i*(T, P) modeled as μ_ref + RT ln(P/P0)."""
    return mu_ref + R_GAS * temperature * np.log(pressure / P0)


def ideal_mixture_gibbs(n_a: float, n_b: float, cfg: ChemicalPotentialConfig) -> float:
    """Total Gibbs free energy for a binary ideal mixture at fixed T, P."""
    if n_a <= 0.0 or n_b <= 0.0:
        raise ValueError("n_a and n_b must be strictly positive.")

    n_total = n_a + n_b
    x_a = n_a / n_total
    x_b = n_b / n_total

    mu_a_star = standard_chemical_potential(cfg.mu_ref_a, cfg.temperature, cfg.pressure)
    mu_b_star = standard_chemical_potential(cfg.mu_ref_b, cfg.temperature, cfg.pressure)

    mixing_term = R_GAS * cfg.temperature * (n_a * np.log(x_a) + n_b * np.log(x_b))
    return n_a * mu_a_star + n_b * mu_b_star + mixing_term


def analytic_mu_a(n_a: float, n_b: float, cfg: ChemicalPotentialConfig) -> float:
    """Analytical chemical potential of A for the model in ideal_mixture_gibbs."""
    x_a = n_a / (n_a + n_b)
    return standard_chemical_potential(cfg.mu_ref_a, cfg.temperature, cfg.pressure) + R_GAS * cfg.temperature * np.log(x_a)


def analytic_mu_b(n_a: float, n_b: float, cfg: ChemicalPotentialConfig) -> float:
    """Analytical chemical potential of B for the model in ideal_mixture_gibbs."""
    x_b = n_b / (n_a + n_b)
    return standard_chemical_potential(cfg.mu_ref_b, cfg.temperature, cfg.pressure) + R_GAS * cfg.temperature * np.log(x_b)


def finite_difference_mu_a(n_a: float, n_b: float, cfg: ChemicalPotentialConfig) -> float:
    """Numerical μ_A = (∂G/∂n_A) with central finite difference."""
    h = cfg.fd_step
    g_plus = ideal_mixture_gibbs(n_a + h, n_b, cfg)
    g_minus = ideal_mixture_gibbs(n_a - h, n_b, cfg)
    return (g_plus - g_minus) / (2.0 * h)


def finite_difference_mu_b(n_a: float, n_b: float, cfg: ChemicalPotentialConfig) -> float:
    """Numerical μ_B = (∂G/∂n_B) with central finite difference."""
    h = cfg.fd_step
    g_plus = ideal_mixture_gibbs(n_a, n_b + h, cfg)
    g_minus = ideal_mixture_gibbs(n_a, n_b - h, cfg)
    return (g_plus - g_minus) / (2.0 * h)


def build_verification_table(cfg: ChemicalPotentialConfig) -> pd.DataFrame:
    """Generate pointwise analytical-vs-numerical μ comparison over a grid."""
    n_a_values = np.linspace(0.2, 2.0, cfg.grid_points)
    n_b_values = np.linspace(0.3, 2.2, cfg.grid_points)

    rows: list[dict[str, float]] = []
    for n_a in n_a_values:
        for n_b in n_b_values:
            mu_a_num = finite_difference_mu_a(float(n_a), float(n_b), cfg)
            mu_b_num = finite_difference_mu_b(float(n_a), float(n_b), cfg)
            mu_a_ana = analytic_mu_a(float(n_a), float(n_b), cfg)
            mu_b_ana = analytic_mu_b(float(n_a), float(n_b), cfg)

            rows.append(
                {
                    "n_a": float(n_a),
                    "n_b": float(n_b),
                    "mu_a_analytic": mu_a_ana,
                    "mu_a_numeric": mu_a_num,
                    "mu_a_abs_error": abs(mu_a_num - mu_a_ana),
                    "mu_b_analytic": mu_b_ana,
                    "mu_b_numeric": mu_b_num,
                    "mu_b_abs_error": abs(mu_b_num - mu_b_ana),
                }
            )

    return pd.DataFrame(rows)


def bisection_root(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float, int]:
    """Deterministic bisection solver for scalar root finding."""
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo, f_lo, 0
    if f_hi == 0.0:
        return hi, f_hi, 0
    if f_lo * f_hi > 0.0:
        raise ValueError("Bisection requires opposite signs at interval ends.")

    mid = 0.5 * (lo + hi)
    f_mid = func(mid)
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)

        if abs(f_mid) <= tol or 0.5 * (hi - lo) <= tol:
            return mid, f_mid, iteration

        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return mid, f_mid, max_iter


def solve_partition_equilibrium(
    cfg: ChemicalPotentialConfig,
    partition_cfg: PartitionConfig,
) -> dict[str, float]:
    """Solve μ_A^alpha = μ_A^beta for a two-phase partition toy model."""

    def mu_a_phase_a(n_a_alpha: float) -> float:
        x_a_alpha = n_a_alpha / (n_a_alpha + partition_cfg.solvent_alpha)
        base = standard_chemical_potential(cfg.mu_ref_a, cfg.temperature, cfg.pressure)
        return base + partition_cfg.mu_shift_alpha + R_GAS * cfg.temperature * np.log(x_a_alpha)

    def mu_a_phase_b(n_a_alpha: float) -> float:
        n_a_beta = partition_cfg.total_a - n_a_alpha
        x_a_beta = n_a_beta / (n_a_beta + partition_cfg.solvent_beta)
        base = standard_chemical_potential(cfg.mu_ref_a, cfg.temperature, cfg.pressure)
        return base + partition_cfg.mu_shift_beta + R_GAS * cfg.temperature * np.log(x_a_beta)

    def objective(n_a_alpha: float) -> float:
        return mu_a_phase_a(n_a_alpha) - mu_a_phase_b(n_a_alpha)

    eps = 1e-8
    lo = eps
    hi = partition_cfg.total_a - eps
    root, residual, iterations = bisection_root(
        objective,
        lo,
        hi,
        tol=partition_cfg.root_tol,
        max_iter=partition_cfg.max_iter,
    )

    n_a_alpha = root
    n_a_beta = partition_cfg.total_a - n_a_alpha
    mu_alpha = mu_a_phase_a(n_a_alpha)
    mu_beta = mu_a_phase_b(n_a_alpha)

    return {
        "n_a_alpha": float(n_a_alpha),
        "n_a_beta": float(n_a_beta),
        "mu_a_alpha": float(mu_alpha),
        "mu_a_beta": float(mu_beta),
        "mu_gap": float(mu_alpha - mu_beta),
        "root_residual": float(residual),
        "iterations": float(iterations),
    }


def main() -> None:
    cfg = ChemicalPotentialConfig()
    partition_cfg = PartitionConfig()

    verification_df = build_verification_table(cfg)
    max_err_a = float(verification_df["mu_a_abs_error"].max())
    max_err_b = float(verification_df["mu_b_abs_error"].max())
    mean_err_a = float(verification_df["mu_a_abs_error"].mean())
    mean_err_b = float(verification_df["mu_b_abs_error"].mean())

    summary_df = pd.DataFrame(
        [
            {
                "metric": "mu_a",
                "max_abs_error": max_err_a,
                "mean_abs_error": mean_err_a,
                "tolerance": cfg.tolerance,
                "passed": max_err_a <= cfg.tolerance,
            },
            {
                "metric": "mu_b",
                "max_abs_error": max_err_b,
                "mean_abs_error": mean_err_b,
                "tolerance": cfg.tolerance,
                "passed": max_err_b <= cfg.tolerance,
            },
        ]
    )

    partition_result = solve_partition_equilibrium(cfg, partition_cfg)
    partition_df = pd.DataFrame([partition_result])

    print("Chemical potential derivative check summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\nSample grid rows (first 10):")
    print(verification_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\nTwo-phase partition equilibrium (mu_A^alpha = mu_A^beta):")
    print(partition_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    assert bool(summary_df["passed"].all()), "Finite-difference chemical potential check failed."
    assert abs(partition_result["mu_gap"]) <= 1e-6, "Partition equilibrium root is not accurate enough."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
