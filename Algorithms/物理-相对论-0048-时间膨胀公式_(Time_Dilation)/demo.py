"""Minimal runnable MVP for Time Dilation.

This script provides:
1) Lorentz factor / time-dilation calculations.
2) Minkowski interval consistency checks.
3) A muon-survival scenario comparing classical vs relativistic prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c


@dataclass(frozen=True)
class MuonScenario:
    """Input/output bundle for the muon survival demonstration."""

    altitude_m: float
    beta: float
    proper_lifetime_s: float
    initial_count: int


def _validate_beta(beta: np.ndarray) -> None:
    if np.any(beta < 0.0) or np.any(beta >= 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1.")


def lorentz_gamma(beta: np.ndarray | float) -> np.ndarray:
    """Return Lorentz factor gamma = 1/sqrt(1-beta^2)."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    _validate_beta(beta_arr)
    return 1.0 / np.sqrt(1.0 - beta_arr * beta_arr)


def time_dilation(proper_time_s: np.ndarray | float, beta: np.ndarray | float) -> np.ndarray:
    """Return coordinate-time interval dt from proper-time interval d_tau."""
    tau = np.asarray(proper_time_s, dtype=np.float64)
    gamma = lorentz_gamma(beta)
    return gamma * tau


def beta_from_gamma(gamma: np.ndarray | float) -> np.ndarray:
    """Inverse map for gamma >= 1: beta = sqrt(1 - 1/gamma^2)."""
    g = np.asarray(gamma, dtype=np.float64)
    if np.any(g < 1.0):
        raise ValueError("gamma must satisfy gamma >= 1.")
    return np.sqrt(1.0 - 1.0 / (g * g))


def make_time_dilation_table(proper_time_us: float = 1.0) -> pd.DataFrame:
    """Build a small benchmark table over representative beta values."""
    beta = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.99, 0.999], dtype=np.float64)
    gamma = lorentz_gamma(beta)
    dt_us = time_dilation(proper_time_us, beta)
    velocity = beta * c
    return pd.DataFrame(
        {
            "beta_v_over_c": beta,
            "gamma": gamma,
            "velocity_m_per_s": velocity,
            "proper_time_us": np.full_like(beta, proper_time_us),
            "dilated_time_us": dt_us,
        }
    )


def verify_interval_invariance(beta: float, proper_time_s: float) -> dict[str, float]:
    """Check c^2 dt^2 - dx^2 == c^2 d_tau^2 for colinear motion."""
    gamma = float(lorentz_gamma(beta))
    dt = gamma * proper_time_s
    v = beta * c
    dx = v * dt
    left = (c * dt) ** 2 - dx**2
    right = (c * proper_time_s) ** 2
    rel_err = abs(left - right) / max(right, 1e-30)

    # Strict enough for float64 and simple arithmetic path.
    assert rel_err < 1e-12, "Interval invariance check failed."
    return {
        "beta": beta,
        "gamma": gamma,
        "left_c2dt2_minus_dx2": left,
        "right_c2dtau2": right,
        "relative_error": rel_err,
    }


def muon_survival_report(cfg: MuonScenario) -> dict[str, float]:
    """Compare classical vs relativistic atmospheric muon survival."""
    beta = cfg.beta
    gamma = float(lorentz_gamma(beta))
    velocity = beta * c
    travel_time_s = cfg.altitude_m / velocity

    # Classical (incorrect at relativistic speed): no lifetime dilation.
    p_classical = float(np.exp(-travel_time_s / cfg.proper_lifetime_s))

    # Relativistic: proper time experienced by muon shrinks by gamma.
    proper_elapsed_s = travel_time_s / gamma
    p_relativistic = float(np.exp(-proper_elapsed_s / cfg.proper_lifetime_s))

    expected_classical = cfg.initial_count * p_classical
    expected_relativistic = cfg.initial_count * p_relativistic

    # Relativistic prediction must be higher whenever gamma > 1.
    assert p_relativistic > p_classical

    return {
        "altitude_m": cfg.altitude_m,
        "beta": beta,
        "gamma": gamma,
        "velocity_m_per_s": velocity,
        "travel_time_us": travel_time_s * 1e6,
        "proper_lifetime_us": cfg.proper_lifetime_s * 1e6,
        "classical_survival_prob": p_classical,
        "relativistic_survival_prob": p_relativistic,
        "expected_classical_count": expected_classical,
        "expected_relativistic_count": expected_relativistic,
        "survival_gain_factor": p_relativistic / max(p_classical, 1e-300),
    }


def run_assertions() -> None:
    """Numerical sanity checks for core formulas."""
    # beta=0 => gamma=1
    assert np.isclose(float(lorentz_gamma(0.0)), 1.0, atol=1e-15)

    # beta=0.8 => gamma=1/sqrt(1-0.64)=1/0.6=1.666...
    assert np.isclose(float(lorentz_gamma(0.8)), 5.0 / 3.0, atol=1e-15)

    # inverse consistency
    g = float(lorentz_gamma(0.91))
    beta_recovered = float(beta_from_gamma(g))
    assert abs(beta_recovered - 0.91) < 1e-12


def main() -> None:
    run_assertions()

    print("=== Time Dilation Table (proper time = 1.0 us) ===")
    table = make_time_dilation_table(proper_time_us=1.0)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\n=== Minkowski Interval Check ===")
    interval_report = verify_interval_invariance(beta=0.92, proper_time_s=2.5e-6)
    for key, value in interval_report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")

    print("\n=== Muon Survival Demo (10 km altitude) ===")
    scenario = MuonScenario(
        altitude_m=10_000.0,
        beta=0.998,
        proper_lifetime_s=2.1969811e-6,
        initial_count=100_000,
    )
    report = muon_survival_report(scenario)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
