"""Minimal runnable MVP for Standard Model consistency checks.

The script performs three deterministic checks:
1) Electroweak charge relation Q = T3 + Y/2
2) Gauge and gravitational anomaly cancellation per generation
3) One-loop running of SM gauge couplings (alpha1, alpha2, alpha3)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWComponent:
    """Electroweak component with (T3, Y) quantum numbers."""

    name: str
    t3: float
    hypercharge: float
    expected_q: float


@dataclass(frozen=True)
class ChiralField:
    """Left-chiral Weyl field used for anomaly calculations."""

    name: str
    su3_dim: int
    su2_dim: int
    hypercharge: Fraction
    su3_dynkin_index: Fraction
    su2_dynkin_index: Fraction


@dataclass(frozen=True)
class RunningInputs:
    """Reference inputs at the Z pole for one-loop running."""

    mz_gev: float = 91.1876
    alpha_em_inv_mz: float = 127.955
    sin2_theta_w_mz: float = 0.23122
    alpha_s_mz: float = 0.1179


def standard_model_components() -> list[EWComponent]:
    """Return a minimal set of SM components for Q = T3 + Y/2 validation."""
    return [
        EWComponent("u_L", +0.5, +1.0 / 3.0, +2.0 / 3.0),
        EWComponent("d_L", -0.5, +1.0 / 3.0, -1.0 / 3.0),
        EWComponent("u_R", 0.0, +4.0 / 3.0, +2.0 / 3.0),
        EWComponent("d_R", 0.0, -2.0 / 3.0, -1.0 / 3.0),
        EWComponent("nu_L", +0.5, -1.0, 0.0),
        EWComponent("e_L", -0.5, -1.0, -1.0),
        EWComponent("e_R", 0.0, -2.0, -1.0),
        EWComponent("H+", +0.5, +1.0, +1.0),
        EWComponent("H0", -0.5, +1.0, 0.0),
    ]


def electric_charge(t3: float, hypercharge: float) -> float:
    """Compute electric charge from electroweak quantum numbers."""
    return t3 + 0.5 * hypercharge


def charge_consistency_table(components: Iterable[EWComponent]) -> tuple[pd.DataFrame, float]:
    """Build a table comparing predicted and expected electric charges."""
    rows: list[dict[str, float | str]] = []
    max_abs_error = 0.0

    for comp in components:
        predicted_q = electric_charge(comp.t3, comp.hypercharge)
        error = predicted_q - comp.expected_q
        max_abs_error = max(max_abs_error, abs(error))
        rows.append(
            {
                "component": comp.name,
                "T3": comp.t3,
                "Y": comp.hypercharge,
                "Q_pred": predicted_q,
                "Q_expected": comp.expected_q,
                "error": error,
            }
        )

    frame = pd.DataFrame(rows)
    return frame, max_abs_error


def standard_model_chiral_fields() -> list[ChiralField]:
    """Minimal fermion content for one SM generation in left-chiral basis."""
    half = Fraction(1, 2)
    return [
        # Q_L = (u_L, d_L): (3, 2, +1/3)
        ChiralField("Q_L", 3, 2, Fraction(1, 3), half, half),
        # L_L = (nu_L, e_L): (1, 2, -1)
        ChiralField("L_L", 1, 2, Fraction(-1, 1), Fraction(0, 1), half),
        # Right-handed fields represented as left-chiral conjugates:
        # u_R^c: (bar{3}, 1, -4/3), d_R^c: (bar{3}, 1, +2/3), e_R^c: (1, 1, +2)
        ChiralField("u_R^c", 3, 1, Fraction(-4, 3), half, Fraction(0, 1)),
        ChiralField("d_R^c", 3, 1, Fraction(2, 3), half, Fraction(0, 1)),
        ChiralField("e_R^c", 1, 1, Fraction(2, 1), Fraction(0, 1), Fraction(0, 1)),
    ]


def anomaly_coefficients(
    fields: Iterable[ChiralField],
    generations: int = 3,
) -> dict[str, Fraction]:
    """Compute anomaly coefficients in exact rational arithmetic."""
    g = Fraction(generations, 1)

    su3_su3_u1 = Fraction(0, 1)
    su2_su2_u1 = Fraction(0, 1)
    u1_cubed = Fraction(0, 1)
    grav_grav_u1 = Fraction(0, 1)

    for field in fields:
        y = field.hypercharge
        color_mult = Fraction(field.su3_dim, 1)
        weak_mult = Fraction(field.su2_dim, 1)

        su3_su3_u1 += g * y * field.su3_dynkin_index * weak_mult
        su2_su2_u1 += g * y * field.su2_dynkin_index * color_mult
        u1_cubed += g * y**3 * color_mult * weak_mult
        grav_grav_u1 += g * y * color_mult * weak_mult

    return {
        "[SU(3)]^2 U(1)_Y": su3_su3_u1,
        "[SU(2)]^2 U(1)_Y": su2_su2_u1,
        "[U(1)_Y]^3": u1_cubed,
        "[Gravity]^2 U(1)_Y": grav_grav_u1,
    }


def initial_inverse_alphas(inputs: RunningInputs) -> np.ndarray:
    """Compute (1/alpha1, 1/alpha2, 1/alpha3) at MZ."""
    alpha_em = 1.0 / inputs.alpha_em_inv_mz
    sin2 = inputs.sin2_theta_w_mz
    cos2 = 1.0 - sin2

    alpha1 = (5.0 / 3.0) * alpha_em / cos2  # GUT normalization
    alpha2 = alpha_em / sin2
    alpha3 = inputs.alpha_s_mz

    return np.array([1.0 / alpha1, 1.0 / alpha2, 1.0 / alpha3], dtype=float)


def one_loop_inverse_alpha(
    mu_gev: np.ndarray,
    inv_alpha_at_mz: np.ndarray,
    beta_coeff: np.ndarray,
    mz_gev: float,
) -> np.ndarray:
    """One-loop RGE: alpha_i^{-1}(mu) = alpha_i^{-1}(MZ) - b_i/(2pi) ln(mu/MZ)."""
    if np.any(mu_gev <= 0.0):
        raise ValueError("All scales mu must be positive.")

    log_ratio = np.log(mu_gev / mz_gev)[:, None]
    slope = beta_coeff[None, :] / (2.0 * np.pi)
    return inv_alpha_at_mz[None, :] - slope * log_ratio


def alpha12_crossing_scale(inv_alpha_at_mz: np.ndarray, beta_coeff: np.ndarray, mz_gev: float) -> float:
    """Analytical crossing scale where alpha1^{-1}(mu) = alpha2^{-1}(mu)."""
    delta_inv = inv_alpha_at_mz[0] - inv_alpha_at_mz[1]
    delta_b = beta_coeff[0] - beta_coeff[1]
    log_ratio = delta_inv * 2.0 * np.pi / delta_b
    return float(mz_gev * np.exp(log_ratio))


def running_summary(inputs: RunningInputs) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scan one-loop running and return a compact table plus key diagnostics."""
    beta_coeff = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0], dtype=float)
    labels = ["alpha1_inv", "alpha2_inv", "alpha3_inv"]

    inv0 = initial_inverse_alphas(inputs)
    mu_grid = np.logspace(np.log10(inputs.mz_gev), 17.0, 2400)
    inv_grid = one_loop_inverse_alpha(mu_grid, inv0, beta_coeff, inputs.mz_gev)

    spread = np.max(inv_grid, axis=1) - np.min(inv_grid, axis=1)
    closest_idx = int(np.argmin(spread))

    sample_scales = np.array([1.0e3, 1.0e6, 1.0e10, 1.0e14, 1.0e16], dtype=float)
    sample_inv = one_loop_inverse_alpha(sample_scales, inv0, beta_coeff, inputs.mz_gev)

    frame = pd.DataFrame(sample_inv, columns=labels)
    frame.insert(0, "mu_GeV", sample_scales)
    frame.insert(1, "log10_mu", np.log10(sample_scales))

    mu12 = alpha12_crossing_scale(inv0, beta_coeff, inputs.mz_gev)
    inv_at_mu12 = one_loop_inverse_alpha(np.array([mu12]), inv0, beta_coeff, inputs.mz_gev)[0]

    summary = {
        "closest_mu_gev": float(mu_grid[closest_idx]),
        "closest_spread": float(spread[closest_idx]),
        "mu12_crossing_gev": mu12,
        "alpha3_mismatch_at_mu12": float(abs(inv_at_mu12[2] - inv_at_mu12[0])),
        "inv_alpha1_at_closest": float(inv_grid[closest_idx, 0]),
        "inv_alpha2_at_closest": float(inv_grid[closest_idx, 1]),
        "inv_alpha3_at_closest": float(inv_grid[closest_idx, 2]),
    }
    return frame, summary


def main() -> None:
    # 1) Charge relation check.
    charge_frame, max_charge_error = charge_consistency_table(standard_model_components())

    # 2) Anomaly cancellation check.
    anomaly = anomaly_coefficients(standard_model_chiral_fields(), generations=3)
    anomaly_rows = [
        {
            "anomaly": name,
            "value_fraction": str(value),
            "value_float": float(value),
        }
        for name, value in anomaly.items()
    ]
    anomaly_frame = pd.DataFrame(anomaly_rows)

    # 3) One-loop gauge running.
    running_frame, running_diag = running_summary(RunningInputs())

    # Deterministic acceptance checks.
    if max_charge_error > 1e-12:
        raise RuntimeError(f"Charge relation check failed: max error = {max_charge_error:.3e}")

    if any(value != 0 for value in anomaly.values()):
        raise RuntimeError("Anomaly cancellation failed: at least one coefficient is non-zero.")

    if running_diag["closest_spread"] < 1.0:
        raise RuntimeError("Unexpected near-exact gauge coupling unification in minimal SM one-loop running.")

    print("Standard Model MVP: charge consistency, anomaly cancellation, one-loop running")
    print()

    print("[1] Electroweak charge relation Q = T3 + Y/2")
    print(charge_frame.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print(f"max |Q_pred - Q_expected| = {max_charge_error:.3e}")
    print()

    print("[2] Gauge/gravitational anomaly coefficients (3 generations)")
    print(anomaly_frame.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[3] One-loop running snapshots (SM beta: b1=41/10, b2=-19/6, b3=-7)")
    print(running_frame.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()

    print("Diagnostics")
    print(f"closest approach scale mu* [GeV]         = {running_diag['closest_mu_gev']:.6e}")
    print(f"minimal spread max(inv_a)-min(inv_a)    = {running_diag['closest_spread']:.6f}")
    print(f"alpha1-alpha2 crossing scale mu12 [GeV] = {running_diag['mu12_crossing_gev']:.6e}")
    print(f"|inv_alpha3 - inv_alpha1| at mu12       = {running_diag['alpha3_mismatch_at_mu12']:.6f}")
    print()

    print("MVP checks passed: SM charge/anomaly consistency validated; exact gauge unification absent at one-loop.")


if __name__ == "__main__":
    main()
