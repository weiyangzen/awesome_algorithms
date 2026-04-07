"""Minimal runnable MVP for gravitational redshift in Schwarzschild spacetime.

The script compares:
- exact general-relativistic redshift for static observers
- weak-field approximation z ~ DeltaPhi / c^2

It prints a scenario table, an Earth radial profile, and runs self-checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import G, c


@dataclass(frozen=True)
class RedshiftScenario:
    name: str
    mass_kg: float
    r_emit_m: float
    r_obs_m: float


def schwarzschild_radius(mass_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2."""
    return 2.0 * G * mass_kg / (c * c)


def _validate_radius(label: str, radius_m: float, r_s: float) -> None:
    """Ensure radius is outside the event horizon for static-observer formula."""
    if np.isinf(radius_m):
        return
    if radius_m <= r_s:
        raise ValueError(
            f"{label}={radius_m:.6e} m must be greater than Schwarzschild radius {r_s:.6e} m"
        )


def redshift_exact(r_s: float, r_emit_m: float, r_obs_m: float) -> float:
    """Exact gravitational redshift for static observers in Schwarzschild metric.

    z = sqrt((1-r_s/r_obs)/(1-r_s/r_emit)) - 1
    with r_obs=inf treated as limit (1-r_s/r_obs)=1.
    """
    _validate_radius("r_emit_m", r_emit_m, r_s)
    _validate_radius("r_obs_m", r_obs_m, r_s)

    f_emit = 1.0 - r_s / r_emit_m
    f_obs = 1.0 if np.isinf(r_obs_m) else 1.0 - r_s / r_obs_m

    if f_emit <= 0.0 or f_obs <= 0.0:
        raise ValueError("Metric factor became non-positive; check radii.")

    return math.sqrt(f_obs / f_emit) - 1.0


def redshift_weak_field(mass_kg: float, r_emit_m: float, r_obs_m: float) -> float:
    """Weak-field approximation z ~ (Phi_obs - Phi_emit)/c^2."""
    inv_obs = 0.0 if np.isinf(r_obs_m) else 1.0 / r_obs_m
    return (G * mass_kg / (c * c)) * (1.0 / r_emit_m - inv_obs)


def relative_error(reference: float, estimate: float) -> float:
    """Return |estimate-reference|/|reference| with safe zero handling."""
    denom = abs(reference)
    if denom == 0.0:
        return float("nan")
    return abs(estimate - reference) / denom


def build_scenario_table(
    scenarios: list[RedshiftScenario],
    lambda_emit_nm: float,
) -> pd.DataFrame:
    """Compute exact/approx redshift metrics for all scenarios."""
    rows: list[dict[str, float | str]] = []

    for s in scenarios:
        r_s = schwarzschild_radius(s.mass_kg)
        z_exact = redshift_exact(r_s, s.r_emit_m, s.r_obs_m)
        z_weak = redshift_weak_field(s.mass_kg, s.r_emit_m, s.r_obs_m)
        nu_ratio = 1.0 / (1.0 + z_exact)

        rows.append(
            {
                "scenario": s.name,
                "mass_kg": s.mass_kg,
                "r_s_m": r_s,
                "r_emit_m": s.r_emit_m,
                "r_obs_m": s.r_obs_m,
                "z_exact": z_exact,
                "z_weak": z_weak,
                "relative_error_weak": relative_error(z_exact, z_weak),
                "nu_obs_over_nu_emit": nu_ratio,
                "lambda_emit_nm": lambda_emit_nm,
                "lambda_obs_nm": lambda_emit_nm * (1.0 + z_exact),
            }
        )

    return pd.DataFrame(rows)


def build_earth_profile(
    mass_kg: float,
    r_surface_m: float,
    r_obs_m: float,
    n_points: int = 8,
) -> pd.DataFrame:
    """Create a radius-redshift profile for monotonic trend checking."""
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    radii = np.linspace(r_surface_m, 8.0 * r_surface_m, n_points)
    r_s = schwarzschild_radius(mass_kg)
    z_values = np.array([redshift_exact(r_s, r, r_obs_m) for r in radii], dtype=float)

    return pd.DataFrame(
        {
            "r_emit_m": radii,
            "altitude_km": (radii - r_surface_m) / 1e3,
            "z_exact": z_values,
        }
    )


def run_consistency_checks(table: pd.DataFrame, earth_profile: pd.DataFrame) -> None:
    """Basic numerical/physical sanity checks for this MVP."""
    z = table["z_exact"].to_numpy(dtype=float)
    nu_ratio = table["nu_obs_over_nu_emit"].to_numpy(dtype=float)

    assert np.all(z > 0.0), "Outward propagation should produce positive gravitational redshift."
    assert np.all(nu_ratio < 1.0), "Observed frequency should be lower than emitted frequency."

    earth_inf_err = float(
        table.loc[
            table["scenario"] == "Earth surface -> infinity", "relative_error_weak"
        ].iloc[0]
    )
    ns_err = float(
        table.loc[
            table["scenario"] == "Neutron star surface -> infinity", "relative_error_weak"
        ].iloc[0]
    )

    # Weak-field approximation should be excellent on Earth but noticeably worse near neutron star.
    assert earth_inf_err < 1e-6
    assert ns_err > 5e-2

    z_profile = earth_profile["z_exact"].to_numpy(dtype=float)
    assert np.all(np.diff(z_profile) < 0.0), "Larger emission radius should yield smaller redshift."


def main() -> None:
    m_sun = 1.98847e30
    r_sun = 6.957e8
    m_earth = 5.9722e24
    r_earth = 6.371e6

    scenarios = [
        RedshiftScenario(
            name="Earth surface -> GPS orbit",
            mass_kg=m_earth,
            r_emit_m=r_earth,
            r_obs_m=r_earth + 2.02e7,
        ),
        RedshiftScenario(
            name="Earth surface -> infinity",
            mass_kg=m_earth,
            r_emit_m=r_earth,
            r_obs_m=float("inf"),
        ),
        RedshiftScenario(
            name="Sun photosphere -> infinity",
            mass_kg=m_sun,
            r_emit_m=r_sun,
            r_obs_m=float("inf"),
        ),
        RedshiftScenario(
            name="White dwarf surface -> infinity",
            mass_kg=0.60 * m_sun,
            r_emit_m=0.012 * r_sun,
            r_obs_m=float("inf"),
        ),
        RedshiftScenario(
            name="Neutron star surface -> infinity",
            mass_kg=1.40 * m_sun,
            r_emit_m=12_000.0,
            r_obs_m=float("inf"),
        ),
    ]

    lambda_emit_nm = 500.0
    table = build_scenario_table(scenarios, lambda_emit_nm=lambda_emit_nm)
    earth_profile = build_earth_profile(
        mass_kg=m_earth,
        r_surface_m=r_earth,
        r_obs_m=float("inf"),
        n_points=8,
    )

    print("Gravitational Redshift MVP (Schwarzschild static observers)")
    print("=" * 72)
    with pd.option_context("display.precision", 10, "display.width", 180):
        print(table.to_string(index=False))

    print("\nEarth radial profile (observer at infinity)")
    print("=" * 72)
    with pd.option_context("display.precision", 12, "display.width", 120):
        print(earth_profile.to_string(index=False))

    run_consistency_checks(table, earth_profile)
    print("\nChecks passed: sign, monotonicity, and weak-field validity regime are consistent.")


if __name__ == "__main__":
    main()
