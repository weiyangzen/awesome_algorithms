"""Minimal runnable MVP for Alpha Decay Theory.

This script implements a transparent Gamow-WKB alpha-decay estimator:
1) validate alpha-decay inputs,
2) build Coulomb barrier geometry,
3) integrate WKB action through the forbidden region,
4) estimate penetrability, decay constant, and half-life.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import integrate

# Physical constants in convenient nuclear units.
E2_COULOMB_MEV_FM = 1.43996448  # e^2 / (4*pi*epsilon0) in MeV*fm
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm
AMU_MEV_C2 = 931.49410242  # atomic mass unit in MeV/c^2
C_LIGHT_FM_S = 2.99792458e23  # speed of light in fm/s
LN2 = math.log(2.0)
R0_FM = 1.20  # radius coefficient for touching radius


@dataclass(frozen=True)
class AlphaDecayCase:
    """One alpha-decay candidate for the MVP estimator."""

    nuclide: str
    A_parent: int
    Z_parent: int
    Q_MeV: float
    preformation: float = 0.20
    observed_t1_2_s: float | None = None


def validate_case(case: AlphaDecayCase) -> tuple[bool, str]:
    """Basic physical sanity checks for this simplified model."""

    if case.A_parent <= 4:
        return False, "A_parent must be > 4"
    if case.Z_parent <= 2:
        return False, "Z_parent must be > 2"
    if case.Q_MeV <= 0.0:
        return False, "Q <= 0, decay closed"
    if not (0.0 < case.preformation <= 1.0):
        return False, "preformation must be in (0, 1]"
    return True, "ok"


def daughter_numbers(case: AlphaDecayCase) -> tuple[int, int]:
    """Return daughter mass and charge numbers after alpha emission."""

    return case.A_parent - 4, case.Z_parent - 2


def alpha_kinetic_energy_mev(case: AlphaDecayCase) -> float:
    """Recoil-corrected alpha kinetic energy from Q-value."""

    A_d, _Z_d = daughter_numbers(case)
    return case.Q_MeV * A_d / case.A_parent


def touching_radius_fm(case: AlphaDecayCase) -> float:
    """Nuclear touching radius R = r0*(A_d^(1/3) + A_alpha^(1/3))."""

    A_d, _Z_d = daughter_numbers(case)
    return R0_FM * (A_d ** (1.0 / 3.0) + 4.0 ** (1.0 / 3.0))


def reduced_mass_mev_c2(case: AlphaDecayCase) -> float:
    """Reduced mass of daughter-alpha two-body system."""

    A_d, _Z_d = daughter_numbers(case)
    return AMU_MEV_C2 * (4.0 * A_d) / (4.0 + A_d)


def coulomb_potential_mev(Z_d: int, r_fm: float) -> float:
    """Outside touching radius, Coulomb barrier V(r)=2*Z_d*e^2/r."""

    return (2.0 * Z_d * E2_COULOMB_MEV_FM) / r_fm


def outer_turning_point_fm(Z_d: int, E_alpha_mev: float) -> float:
    """Classical outer turning point b where V(b)=E_alpha."""

    return (2.0 * Z_d * E2_COULOMB_MEV_FM) / E_alpha_mev


def gamow_action(case: AlphaDecayCase) -> tuple[float, float, float, float, float]:
    """Compute WKB action G through the Coulomb barrier.

    Returns:
      G, R, b, E_alpha, mu
    """

    A_d, Z_d = daughter_numbers(case)
    if A_d <= 0 or Z_d <= 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    E_alpha = alpha_kinetic_energy_mev(case)
    if E_alpha <= 0.0:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    R = touching_radius_fm(case)
    b = outer_turning_point_fm(Z_d, E_alpha)
    if b <= R:
        return math.nan, R, b, E_alpha, math.nan

    mu = reduced_mass_mev_c2(case)
    prefactor = math.sqrt(2.0 * mu) / HBAR_C_MEV_FM

    def integrand(r_fm: float) -> float:
        barrier_gap = coulomb_potential_mev(Z_d, r_fm) - E_alpha
        return math.sqrt(max(barrier_gap, 0.0))

    integral_value, _ = integrate.quad(
        integrand,
        R,
        b,
        epsabs=1e-10,
        epsrel=1e-8,
        limit=200,
    )
    G = prefactor * integral_value
    return G, R, b, E_alpha, mu


def estimate_half_life(case: AlphaDecayCase) -> dict[str, float | str | bool]:
    """Estimate alpha-decay half-life with a minimal Gamow-WKB model."""

    valid, reason = validate_case(case)
    out: dict[str, float | str | bool] = {
        "nuclide": case.nuclide,
        "valid": valid,
        "reason": reason,
        "A_parent": case.A_parent,
        "Z_parent": case.Z_parent,
        "Q_MeV": case.Q_MeV,
        "preformation": case.preformation,
    }

    if not valid:
        out.update({
            "G": np.nan,
            "penetrability": np.nan,
            "assault_freq_s^-1": np.nan,
            "lambda_s^-1": np.nan,
            "t1_2_s": np.nan,
            "log10_t1_2_s": np.nan,
        })
        return out

    G, R, b, E_alpha, mu = gamow_action(case)
    if not np.isfinite(G):
        out.update({
            "reason": "invalid turning points for WKB barrier",
            "G": np.nan,
            "penetrability": np.nan,
            "assault_freq_s^-1": np.nan,
            "lambda_s^-1": np.nan,
            "t1_2_s": np.nan,
            "log10_t1_2_s": np.nan,
        })
        return out

    beta = math.sqrt(max(2.0 * E_alpha / mu, 0.0))
    if beta <= 0.0 or beta >= 1.0:
        out.update({
            "reason": "non-physical alpha velocity",
            "G": G,
            "penetrability": np.nan,
            "assault_freq_s^-1": np.nan,
            "lambda_s^-1": np.nan,
            "t1_2_s": np.nan,
            "log10_t1_2_s": np.nan,
        })
        return out

    assault_freq = beta * C_LIGHT_FM_S / (2.0 * R)
    log_penetrability = -2.0 * G
    penetrability = math.exp(log_penetrability) if log_penetrability > -700 else 0.0

    log_lambda = math.log(case.preformation) + math.log(assault_freq) + log_penetrability
    lambda_val = math.exp(log_lambda) if log_lambda > -700 else 0.0

    log_t12 = math.log(LN2) - log_lambda
    t12_s = math.exp(log_t12) if log_t12 < 700 else math.inf
    log10_t12 = log_t12 / math.log(10.0)

    out.update({
        "R_fm": R,
        "b_fm": b,
        "E_alpha_MeV": E_alpha,
        "G": G,
        "penetrability": penetrability,
        "assault_freq_s^-1": assault_freq,
        "lambda_s^-1": lambda_val,
        "t1_2_s": t12_s,
        "log10_t1_2_s": log10_t12,
        "geiger_nuttall_x": (case.Z_parent - 2) / math.sqrt(E_alpha),
    })

    if case.observed_t1_2_s is not None and case.observed_t1_2_s > 0.0:
        observed_log10 = math.log10(case.observed_t1_2_s)
        out["observed_t1_2_s"] = case.observed_t1_2_s
        out["observed_log10_t1_2_s"] = observed_log10
        out["abs_log10_error"] = abs(log10_t12 - observed_log10)
    else:
        out["observed_t1_2_s"] = np.nan
        out["observed_log10_t1_2_s"] = np.nan
        out["abs_log10_error"] = np.nan

    return out


def main() -> None:
    # Representative alpha emitters across very different half-life scales.
    cases = [
        AlphaDecayCase("212Po -> 208Pb + alpha", A_parent=212, Z_parent=84, Q_MeV=8.95, preformation=0.30, observed_t1_2_s=2.99e-7),
        AlphaDecayCase("222Rn -> 218Po + alpha", A_parent=222, Z_parent=86, Q_MeV=5.59, preformation=0.20, observed_t1_2_s=3.31e5),
        AlphaDecayCase("238U -> 234Th + alpha", A_parent=238, Z_parent=92, Q_MeV=4.27, preformation=0.10, observed_t1_2_s=1.41e17),
        AlphaDecayCase("232Th -> 228Ra + alpha", A_parent=232, Z_parent=90, Q_MeV=4.08, preformation=0.10, observed_t1_2_s=4.43e17),
        AlphaDecayCase("239Pu -> 235U + alpha", A_parent=239, Z_parent=94, Q_MeV=5.24, preformation=0.08, observed_t1_2_s=7.61e11),
    ]

    rows = [estimate_half_life(case) for case in cases]
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x: .4e}")

    show_cols = [
        "nuclide",
        "Q_MeV",
        "preformation",
        "G",
        "penetrability",
        "lambda_s^-1",
        "t1_2_s",
        "log10_t1_2_s",
        "observed_log10_t1_2_s",
        "abs_log10_error",
    ]

    print("Alpha Decay Theory MVP (Gamow-WKB, l=0 simplified model):")
    print(df[show_cols])


if __name__ == "__main__":
    main()
