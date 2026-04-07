"""Minimal renormalization MVP in scalar phi^4 theory.

Model choices:
- Euclidean momentum cutoff regularization with cutoff Lambda.
- One-loop tadpole and bubble contributions.
- Renormalization conditions imposed at subtraction momentum mu_sub.

Goal:
- Show how bare parameters (m_B^2, lambda_B) absorb cutoff dependence,
  while renormalized predictions become nearly cutoff independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, newton


@dataclass(frozen=True)
class Phi4Setup:
    """Configuration for one-loop renormalization in phi^4."""

    m_ren: float
    lambda_ren: float
    mu_sub: float
    cutoffs: tuple[float, ...]
    probe_momenta: tuple[float, ...]


def tadpole_integral_cutoff(cutoff: float, mass: float) -> float:
    """One-loop tadpole integral with sharp 4D Euclidean cutoff.

    I_tad(Λ,m) = 1/(16π²) * [Λ² - m² ln(1 + Λ²/m²)]
    """
    if cutoff <= 0.0 or mass <= 0.0:
        raise ValueError("cutoff and mass must be positive.")

    ratio = (cutoff * cutoff) / (mass * mass)
    return (cutoff * cutoff - mass * mass * np.log1p(ratio)) / (16.0 * np.pi**2)


def bubble_integral_cutoff(cutoff: float, mass: float, momentum: float) -> float:
    """One-loop bubble integral with Feynman parameter x in [0,1].

    B(Λ,m,p) = 1/(16π²) * ∫_0^1 dx [ ln(1 + Λ²/M²) - Λ²/(Λ² + M²) ]
    where M² = m² + x(1-x)p².

    Here p is Euclidean momentum magnitude.
    """
    if cutoff <= 0.0 or mass <= 0.0 or momentum < 0.0:
        raise ValueError("cutoff/mass must be positive, momentum must be non-negative.")

    cutoff_sq = cutoff * cutoff
    mass_sq = mass * mass
    momentum_sq = momentum * momentum

    def integrand(x: float) -> float:
        m2_eff = mass_sq + x * (1.0 - x) * momentum_sq
        return np.log1p(cutoff_sq / m2_eff) - cutoff_sq / (cutoff_sq + m2_eff)

    value, _ = quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    return value / (16.0 * np.pi**2)


def solve_bare_lambda(lambda_ren: float, bubble_at_subtraction: float) -> float:
    """Solve lambda_ren = lambda_B - 3*lambda_B^2*B_sub for lambda_B."""
    if lambda_ren <= 0.0:
        raise ValueError("lambda_ren must be positive.")
    if bubble_at_subtraction <= 0.0:
        raise ValueError("bubble_at_subtraction must be positive.")

    # Reality check of the exact quadratic discriminant.
    discriminant = 1.0 - 12.0 * bubble_at_subtraction * lambda_ren
    if discriminant <= 0.0:
        raise ValueError(
            "No real perturbative bare coupling: 1 - 12*B*lambda_ren <= 0. "
            "Decrease lambda_ren or the subtraction-point bubble integral."
        )

    def f(lambda_bare: float) -> float:
        return lambda_bare - 3.0 * bubble_at_subtraction * lambda_bare**2 - lambda_ren

    def fp(lambda_bare: float) -> float:
        return 1.0 - 6.0 * bubble_at_subtraction * lambda_bare

    try:
        root = float(newton(func=f, x0=lambda_ren, fprime=fp, tol=1e-14, maxiter=100))
    except RuntimeError:
        left = 1e-12
        right = max(1.0, 4.0 * lambda_ren)
        f_left = f(left)
        f_right = f(right)
        while f_left * f_right > 0.0 and right < 1000.0:
            right *= 2.0
            f_right = f(right)
        if f_left * f_right > 0.0:
            raise RuntimeError("Failed to bracket lambda_B root.")
        root = float(brentq(f, left, right, xtol=1e-14, rtol=1e-12, maxiter=200))

    if root <= 0.0:
        raise RuntimeError("Solved bare coupling is non-positive, outside perturbative setup.")
    return root


def compute_bare_parameters(cutoff: float, setup: Phi4Setup) -> tuple[float, float, float]:
    """Compute (lambda_B, m_B^2, B_sub) for one cutoff."""
    b_sub = bubble_integral_cutoff(cutoff=cutoff, mass=setup.m_ren, momentum=setup.mu_sub)
    lambda_bare = solve_bare_lambda(lambda_ren=setup.lambda_ren, bubble_at_subtraction=b_sub)

    # One-loop renormalization condition for the 2-point function at p=0.
    # m_R^2 = m_B^2 + (lambda_B/2) * I_tad  =>  m_B^2 = m_R^2 - (lambda_B/2) * I_tad
    i_tad = tadpole_integral_cutoff(cutoff=cutoff, mass=setup.m_ren)
    mass_bare_sq = setup.m_ren**2 - 0.5 * lambda_bare * i_tad

    return lambda_bare, mass_bare_sq, b_sub


def effective_renormalized_coupling(cutoff: float, mass: float, lambda_bare: float, momentum: float) -> float:
    """One-loop renormalized 4-point coupling at momentum p."""
    b_p = bubble_integral_cutoff(cutoff=cutoff, mass=mass, momentum=momentum)
    return lambda_bare - 3.0 * lambda_bare**2 * b_p


def build_report(setup: Phi4Setup) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all report tables for renormalized and naive comparisons."""
    if not setup.cutoffs:
        raise ValueError("At least one cutoff is required.")

    cutoff_rows: list[dict[str, float]] = []
    renorm_rows: list[dict[str, float]] = []

    for cutoff in setup.cutoffs:
        lambda_bare, mass_bare_sq, b_sub = compute_bare_parameters(cutoff=cutoff, setup=setup)
        lambda_at_sub = effective_renormalized_coupling(
            cutoff=cutoff,
            mass=setup.m_ren,
            lambda_bare=lambda_bare,
            momentum=setup.mu_sub,
        )

        cutoff_rows.append(
            {
                "cutoff": cutoff,
                "bubble_sub": b_sub,
                "lambda_bare": lambda_bare,
                "mass_bare_sq": mass_bare_sq,
                "lambda_sub_reconstructed": lambda_at_sub,
                "sub_abs_error": abs(lambda_at_sub - setup.lambda_ren),
            }
        )

        for p in setup.probe_momenta:
            lam_eff = effective_renormalized_coupling(
                cutoff=cutoff,
                mass=setup.m_ren,
                lambda_bare=lambda_bare,
                momentum=p,
            )
            renorm_rows.append({"cutoff": cutoff, "p": p, "lambda_eff": lam_eff})

    cutoff_df = pd.DataFrame(cutoff_rows)
    renorm_df = pd.DataFrame(renorm_rows)

    # Naive comparison: keep bare coupling fixed at first cutoff and vary cutoff anyway.
    baseline_cutoff = setup.cutoffs[0]
    baseline_row = cutoff_df.loc[cutoff_df["cutoff"] == baseline_cutoff].iloc[0]
    baseline_lambda_bare = float(baseline_row["lambda_bare"])

    naive_rows: list[dict[str, float]] = []
    for cutoff in setup.cutoffs:
        for p in setup.probe_momenta:
            lam_eff = effective_renormalized_coupling(
                cutoff=cutoff,
                mass=setup.m_ren,
                lambda_bare=baseline_lambda_bare,
                momentum=p,
            )
            naive_rows.append({"cutoff": cutoff, "p": p, "lambda_eff": lam_eff})
    naive_df = pd.DataFrame(naive_rows)

    def spread_table(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        out = (
            df.groupby("p", as_index=False)["lambda_eff"]
            .agg(lambda_min="min", lambda_max="max")
            .sort_values("p")
        )
        out["spread"] = out["lambda_max"] - out["lambda_min"]
        out["mode"] = tag
        return out

    spread_df = pd.concat(
        [spread_table(renorm_df, "renormalized"), spread_table(naive_df, "naive_fixed_bare")],
        ignore_index=True,
    )

    return cutoff_df, renorm_df, naive_df, spread_df


def main() -> None:
    setup = Phi4Setup(
        m_ren=1.0,
        lambda_ren=0.20,
        mu_sub=2.0,
        cutoffs=(20.0, 40.0, 80.0, 160.0, 320.0, 640.0),
        probe_momenta=(0.0, 2.0, 5.0, 10.0),
    )

    cutoff_df, renorm_df, naive_df, spread_df = build_report(setup)

    renorm_pivot = renorm_df.pivot(index="p", columns="cutoff", values="lambda_eff").sort_index()
    naive_pivot = naive_df.pivot(index="p", columns="cutoff", values="lambda_eff").sort_index()

    print("=== Setup ===")
    print(
        f"m_R={setup.m_ren:.4f}, lambda_R={setup.lambda_ren:.4f}, "
        f"mu_sub={setup.mu_sub:.4f}, cutoffs={list(setup.cutoffs)}"
    )
    print()

    print("=== Bare Parameters Determined by Renormalization Conditions ===")
    print(cutoff_df.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print()

    print("=== Renormalized Coupling lambda_eff(p) With Per-Cutoff Counterterms ===")
    print(renorm_pivot.to_string(float_format=lambda x: f"{x:.10f}"))
    print()

    print("=== Naive Comparison (Fixed Bare Coupling From Smallest Cutoff) ===")
    print(naive_pivot.to_string(float_format=lambda x: f"{x:.10f}"))
    print()

    print("=== Cutoff Sensitivity Summary ===")
    print(spread_df.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Deterministic sanity checks.
    max_sub_error = float(cutoff_df["sub_abs_error"].max())
    lambda_bare_diff = np.diff(cutoff_df["lambda_bare"].to_numpy())
    mass_bare_diff = np.diff(cutoff_df["mass_bare_sq"].to_numpy())

    renorm_spread = spread_df.loc[spread_df["mode"] == "renormalized", "spread"].to_numpy()
    naive_spread = spread_df.loc[spread_df["mode"] == "naive_fixed_bare", "spread"].to_numpy()

    assert max_sub_error < 1e-10, f"Subtraction-point condition failed: {max_sub_error:.3e}"
    assert np.all(lambda_bare_diff > 0.0), "lambda_B should increase with larger cutoff in this setup."
    assert np.all(mass_bare_diff < 0.0), "m_B^2 should decrease with larger cutoff in this setup."
    assert np.all(renorm_spread < 5e-3), "Renormalized prediction still too cutoff-sensitive."
    assert np.all(naive_spread > renorm_spread), "Naive fixed-bare scan should be more cutoff-sensitive."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
