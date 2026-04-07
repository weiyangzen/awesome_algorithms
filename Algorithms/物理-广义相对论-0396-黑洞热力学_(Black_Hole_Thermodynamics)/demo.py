"""Minimal MVP for Black Hole Thermodynamics (Schwarzschild case).

The script computes thermodynamic quantities, verifies the first-law relation,
and compares numerical vs analytic Hawking evaporation trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import constants
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class EvaporationScenario:
    m0_kg: float
    time_fraction_end: float
    num_time_points: int


def schwarzschild_radius(mass_kg: np.ndarray) -> np.ndarray:
    mass_kg = np.asarray(mass_kg, dtype=float)
    if np.any(mass_kg <= 0.0):
        raise ValueError("Mass must be positive.")
    return 2.0 * constants.G * mass_kg / constants.c**2


def horizon_area(mass_kg: np.ndarray) -> np.ndarray:
    radius = schwarzschild_radius(mass_kg)
    return 4.0 * np.pi * radius**2


def hawking_temperature(mass_kg: np.ndarray) -> np.ndarray:
    mass_kg = np.asarray(mass_kg, dtype=float)
    if np.any(mass_kg <= 0.0):
        raise ValueError("Mass must be positive.")
    numerator = constants.hbar * constants.c**3
    denominator = 8.0 * np.pi * constants.G * constants.k * mass_kg
    return numerator / denominator


def bekenstein_hawking_entropy(mass_kg: np.ndarray) -> np.ndarray:
    area = horizon_area(mass_kg)
    prefactor = constants.k * constants.c**3 / (4.0 * constants.G * constants.hbar)
    return prefactor * area


def schwarzschild_heat_capacity(mass_kg: np.ndarray) -> np.ndarray:
    mass_kg = np.asarray(mass_kg, dtype=float)
    if np.any(mass_kg <= 0.0):
        raise ValueError("Mass must be positive.")
    return -8.0 * np.pi * constants.k * constants.G * mass_kg**2 / (constants.hbar * constants.c)


def first_law_relative_error(mass_kg: float, delta_frac: float = 1e-6) -> float:
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")
    if not (0.0 < delta_frac < 1.0):
        raise ValueError("delta_frac must be in (0, 1).")

    delta_m = mass_kg * delta_frac
    m_plus = mass_kg + delta_m
    m_minus = mass_kg - delta_m
    if m_minus <= 0.0:
        raise ValueError("delta_frac too large for the provided mass.")

    s_plus = float(bekenstein_hawking_entropy(np.array([m_plus]))[0])
    s_minus = float(bekenstein_hawking_entropy(np.array([m_minus]))[0])
    ds_dm_numeric = (s_plus - s_minus) / (2.0 * delta_m)

    t_here = float(hawking_temperature(np.array([mass_kg]))[0])
    lhs = constants.c**2
    rhs = t_here * ds_dm_numeric
    return abs(lhs - rhs) / lhs


def evaporation_alpha() -> float:
    return constants.hbar * constants.c**4 / (15360.0 * np.pi * constants.G**2)


def evaporation_lifetime(m0_kg: float) -> float:
    if m0_kg <= 0.0:
        raise ValueError("m0_kg must be positive.")
    alpha = evaporation_alpha()
    return m0_kg**3 / (3.0 * alpha)


def evaporation_mass_analytic(t_s: np.ndarray, m0_kg: float) -> np.ndarray:
    t_s = np.asarray(t_s, dtype=float)
    if np.any(t_s < 0.0):
        raise ValueError("Time values must be non-negative.")
    alpha = evaporation_alpha()
    cube_term = m0_kg**3 - 3.0 * alpha * t_s
    if np.any(cube_term <= 0.0):
        raise ValueError("Analytic mass formula requires t < lifetime.")
    return np.cbrt(cube_term)


def integrate_evaporation(m0_kg: float, t_eval_s: np.ndarray) -> np.ndarray:
    if m0_kg <= 0.0:
        raise ValueError("m0_kg must be positive.")
    t_eval_s = np.asarray(t_eval_s, dtype=float)
    if t_eval_s.ndim != 1 or t_eval_s.size < 2:
        raise ValueError("t_eval_s must be a 1D array with at least two points.")
    if np.any(t_eval_s < 0.0):
        raise ValueError("t_eval_s must be non-negative.")
    if not np.all(np.diff(t_eval_s) > 0.0):
        raise ValueError("t_eval_s must be strictly increasing.")

    alpha = evaporation_alpha()

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        m = max(float(y[0]), 1e-18)
        return np.array([-alpha / (m**2)], dtype=float)

    solution = solve_ivp(
        rhs,
        t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
        y0=np.array([m0_kg], dtype=float),
        t_eval=t_eval_s,
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Evaporation ODE failed: {solution.message}")

    masses = solution.y[0]
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise RuntimeError("Non-finite or non-positive mass encountered during integration.")
    return masses


def build_thermo_table(masses_kg: np.ndarray) -> pd.DataFrame:
    masses_kg = np.asarray(masses_kg, dtype=float)
    if np.any(masses_kg <= 0.0):
        raise ValueError("All masses must be positive.")
    if not np.all(np.diff(masses_kg) > 0.0):
        raise ValueError("masses_kg must be strictly increasing for monotonic checks.")

    return pd.DataFrame(
        {
            "mass_kg": masses_kg,
            "radius_m": schwarzschild_radius(masses_kg),
            "area_m2": horizon_area(masses_kg),
            "temperature_K": hawking_temperature(masses_kg),
            "entropy_J_per_K": bekenstein_hawking_entropy(masses_kg),
            "heat_capacity_J_per_K": schwarzschild_heat_capacity(masses_kg),
        }
    )


def main() -> None:
    mass_samples_kg = np.array([1.0e8, 1.0e9, 1.0e10, 1.0e12], dtype=float)
    thermo_df = build_thermo_table(mass_samples_kg)

    first_law_df = pd.DataFrame(
        {
            "mass_kg": mass_samples_kg,
            "relative_error_dMc2_vs_TdS": [first_law_relative_error(m) for m in mass_samples_kg],
        }
    )

    scenario = EvaporationScenario(m0_kg=1.0e8, time_fraction_end=0.9, num_time_points=12)
    tau_s = evaporation_lifetime(scenario.m0_kg)
    t_eval_s = np.linspace(0.0, scenario.time_fraction_end * tau_s, scenario.num_time_points)

    m_numeric = integrate_evaporation(m0_kg=scenario.m0_kg, t_eval_s=t_eval_s)
    m_analytic = evaporation_mass_analytic(t_s=t_eval_s, m0_kg=scenario.m0_kg)

    evap_df = pd.DataFrame(
        {
            "time_s": t_eval_s,
            "time_over_tau": t_eval_s / tau_s,
            "mass_numeric_kg": m_numeric,
            "mass_analytic_kg": m_analytic,
        }
    )
    evap_df["rel_err"] = np.abs(
        evap_df["mass_numeric_kg"] - evap_df["mass_analytic_kg"]
    ) / np.maximum(evap_df["mass_analytic_kg"], 1e-30)

    tau_ratio_numeric = evaporation_lifetime(2.0 * scenario.m0_kg) / evaporation_lifetime(scenario.m0_kg)
    tau_ratio_expected = 8.0

    print("=== Schwarzschild Black Hole Thermodynamics ===")
    print(thermo_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print("=== First-Law Consistency Check ===")
    print(first_law_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print("=== Hawking Evaporation: Numeric vs Analytic ===")
    print(evap_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print(f"Evaporation lifetime tau(M0={scenario.m0_kg:.3e} kg): {tau_s:.8e} s")
    print(f"Lifetime scaling check tau(2M)/tau(M): numeric={tau_ratio_numeric:.8f}, expected={tau_ratio_expected:.8f}")

    temperatures = thermo_df["temperature_K"].to_numpy()
    entropies = thermo_df["entropy_J_per_K"].to_numpy()
    heat_caps = thermo_df["heat_capacity_J_per_K"].to_numpy()
    first_law_max_err = float(first_law_df["relative_error_dMc2_vs_TdS"].max())
    evaporation_max_rel_err = float(evap_df["rel_err"].max())

    assert np.all(np.diff(temperatures) < 0.0), "Temperature should decrease with mass."
    assert np.all(np.diff(entropies) > 0.0), "Entropy should increase with mass."
    assert np.all(heat_caps < 0.0), "Schwarzschild heat capacity should be negative."
    assert first_law_max_err < 1e-8, f"First-law relative error too large: {first_law_max_err:.3e}"
    assert np.all(np.diff(m_numeric) < 0.0), "Mass should monotonically decrease during evaporation."
    assert evaporation_max_rel_err < 2e-6, (
        f"Evaporation numeric/analytic mismatch too large: {evaporation_max_rel_err:.3e}"
    )
    assert np.isclose(tau_ratio_numeric, tau_ratio_expected, rtol=1e-12), "Lifetime should scale as M^3."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
