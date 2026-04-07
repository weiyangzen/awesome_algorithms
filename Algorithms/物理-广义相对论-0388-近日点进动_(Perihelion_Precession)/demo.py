"""Minimal runnable MVP for perihelion precession in General Relativity.

This script models bound orbits with a first post-Newtonian correction in
Schwarzschild spacetime using the standard orbit equation in terms of
w = p/r and azimuth angle phi:

    d2w/dphi2 + w = 1 + epsilon * w^2,
    epsilon = 3GM/(c^2 p),
    p = a(1-e^2).

Perihelion events (w' = 0, negative crossing) provide the perihelion-to-
perihelion azimuth period, so the relativistic precession per orbit is:

    Delta_phi = <phi_{k+1} - phi_k> - 2*pi.

The script compares this numeric estimate against the textbook weak-field GR
formula:

    Delta_phi_GR = 6*pi*GM / (a(1-e^2)c^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# SI constants
G_NEWTON = 6.67430e-11
C_LIGHT = 299_792_458.0
M_SUN_KG = 1.98847e30
AU_M = 1.495978707e11
DAY_S = 86_400.0
RAD_TO_ARCSEC = 206_264.80624709636
CENTURY_DAYS = 36_525.0


@dataclass(frozen=True)
class PlanetOrbit:
    name: str
    semimajor_axis_au: float
    eccentricity: float


def _validate_orbit(orbit: PlanetOrbit) -> None:
    if orbit.semimajor_axis_au <= 0.0 or not np.isfinite(orbit.semimajor_axis_au):
        raise ValueError(f"invalid semimajor_axis_au: {orbit.semimajor_axis_au}")
    if orbit.eccentricity < 0.0 or orbit.eccentricity >= 1.0 or not np.isfinite(orbit.eccentricity):
        raise ValueError(f"eccentricity must satisfy 0 <= e < 1, got {orbit.eccentricity}")


def semimajor_axis_m(orbit: PlanetOrbit) -> float:
    return orbit.semimajor_axis_au * AU_M


def semi_latus_rectum_m(orbit: PlanetOrbit) -> float:
    a = semimajor_axis_m(orbit)
    return a * (1.0 - orbit.eccentricity * orbit.eccentricity)


def gr_epsilon(mu: float, p: float) -> float:
    if mu <= 0.0 or p <= 0.0:
        raise ValueError("mu and p must be positive")
    return 3.0 * mu / (C_LIGHT * C_LIGHT * p)


def precession_formula_rad_per_orbit(mu: float, orbit: PlanetOrbit) -> float:
    a = semimajor_axis_m(orbit)
    e = orbit.eccentricity
    return 6.0 * np.pi * mu / (a * (1.0 - e * e) * C_LIGHT * C_LIGHT)


def orbital_period_days(mu: float, orbit: PlanetOrbit) -> float:
    a = semimajor_axis_m(orbit)
    period_s = 2.0 * np.pi * np.sqrt(a**3 / mu)
    return period_s / DAY_S


def _gr_orbit_rhs(phi: float, state: np.ndarray, epsilon: float) -> np.ndarray:
    del phi
    w, w_prime = state
    return np.array([w_prime, 1.0 + epsilon * (w * w) - w], dtype=float)


def _build_perihelion_event():
    # Perihelion corresponds to local maxima of w(phi), i.e. w' crossing + -> -.
    def event(phi: float, state: np.ndarray) -> float:
        del phi
        return float(state[1])

    event.terminal = False
    event.direction = -1.0
    return event


def _invariant_energy_like(w: np.ndarray, w_prime: np.ndarray, epsilon: float) -> np.ndarray:
    # First integral of the ODE: 0.5 w'^2 + 0.5 w^2 - w - (epsilon/3) w^3 = const.
    return 0.5 * (w_prime * w_prime) + 0.5 * (w * w) - w - (epsilon / 3.0) * (w**3)


def integrate_precession_numeric(
    mu: float,
    orbit: PlanetOrbit,
    n_orbits: int = 320,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    max_step: float = 0.10,
) -> dict[str, float]:
    _validate_orbit(orbit)
    if n_orbits < 20:
        raise ValueError("n_orbits must be >= 20 for a stable average")

    e = orbit.eccentricity
    p = semi_latus_rectum_m(orbit)
    epsilon = gr_epsilon(mu, p)

    # Perihelion initial condition at phi=0.
    w0 = 1.0 + e
    state0 = np.array([w0, 0.0], dtype=float)
    phi_max = 2.0 * np.pi * (n_orbits + 5)

    sol = solve_ivp(
        fun=lambda phi, y: _gr_orbit_rhs(phi, y, epsilon),
        t_span=(0.0, phi_max),
        y0=state0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
        events=_build_perihelion_event(),
    )

    if not sol.success:
        raise RuntimeError(f"integration failed for {orbit.name}: {sol.message}")

    perihelion_phi = np.asarray(sol.t_events[0], dtype=float)
    # Drop the initial root at phi=0.
    perihelion_phi = perihelion_phi[perihelion_phi > 1e-10]
    if perihelion_phi.size < n_orbits:
        raise RuntimeError(
            f"insufficient perihelion events for {orbit.name}: got {perihelion_phi.size}, need >= {n_orbits}"
        )

    gaps = np.diff(perihelion_phi)
    delta_phi_samples = gaps - 2.0 * np.pi
    delta_phi_mean = float(np.mean(delta_phi_samples))
    delta_phi_std = float(np.std(delta_phi_samples, ddof=1)) if delta_phi_samples.size > 1 else 0.0

    # Numerical diagnostics from dense sampling.
    sample_count = 4000
    phi_grid = np.linspace(0.0, perihelion_phi[-1], sample_count)
    sampled = sol.sol(phi_grid)
    w = sampled[0]
    w_prime = sampled[1]

    invariant = _invariant_energy_like(w, w_prime, epsilon)
    invariant_drift = float(np.max(np.abs(invariant - invariant[0])))

    # Central-difference derivative on dense-output w' for a stable ODE residual.
    diff_h = 1e-4
    phi_mid = phi_grid[(phi_grid > diff_h) & (phi_grid < perihelion_phi[-1] - diff_h)]
    wp_plus = sol.sol(phi_mid + diff_h)[1]
    wp_minus = sol.sol(phi_mid - diff_h)[1]
    w_double_prime_fd = (wp_plus - wp_minus) / (2.0 * diff_h)
    w_mid = sol.sol(phi_mid)[0]
    ode_residual = w_double_prime_fd - (1.0 + epsilon * (w_mid * w_mid) - w_mid)
    ode_residual_rms = float(np.sqrt(np.mean(ode_residual * ode_residual)))

    return {
        "epsilon": epsilon,
        "event_count": float(perihelion_phi.size),
        "delta_phi_numeric_rad_per_orbit": delta_phi_mean,
        "delta_phi_numeric_std_rad": delta_phi_std,
        "invariant_drift": invariant_drift,
        "ode_residual_rms": ode_residual_rms,
    }


def run_planet_suite(mu: float, planets: list[PlanetOrbit], n_orbits: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, float | str]] = []
    diagnostics: dict[str, dict[str, float]] = {}

    for planet in planets:
        _validate_orbit(planet)
        formula_rad = precession_formula_rad_per_orbit(mu, planet)
        period_days = orbital_period_days(mu, planet)
        orbits_per_century = CENTURY_DAYS / period_days

        numeric = integrate_precession_numeric(mu=mu, orbit=planet, n_orbits=n_orbits)
        diagnostics[planet.name] = numeric

        formula_arcsec_per_orbit = formula_rad * RAD_TO_ARCSEC
        numeric_arcsec_per_orbit = numeric["delta_phi_numeric_rad_per_orbit"] * RAD_TO_ARCSEC
        rel_error = abs(numeric_arcsec_per_orbit - formula_arcsec_per_orbit) / formula_arcsec_per_orbit

        rows.append(
            {
                "planet": planet.name,
                "a_AU": planet.semimajor_axis_au,
                "e": planet.eccentricity,
                "epsilon": numeric["epsilon"],
                "period_days": period_days,
                "precession_formula_arcsec_per_orbit": formula_arcsec_per_orbit,
                "precession_numeric_arcsec_per_orbit": numeric_arcsec_per_orbit,
                "relative_error_formula_vs_numeric": rel_error,
                "precession_formula_arcsec_per_century": formula_arcsec_per_orbit * orbits_per_century,
                "precession_numeric_arcsec_per_century": numeric_arcsec_per_orbit * orbits_per_century,
                "perihelion_events_used": numeric["event_count"],
            }
        )

    return pd.DataFrame(rows), diagnostics


def main() -> None:
    mu_sun = G_NEWTON * M_SUN_KG
    n_orbits = 320

    planets = [
        PlanetOrbit("Mercury", semimajor_axis_au=0.38709893, eccentricity=0.205630),
        PlanetOrbit("Venus", semimajor_axis_au=0.72333199, eccentricity=0.006772),
        PlanetOrbit("Earth", semimajor_axis_au=1.00000011, eccentricity=0.016710),
        PlanetOrbit("Mars", semimajor_axis_au=1.52366231, eccentricity=0.093412),
    ]

    df, diagnostics = run_planet_suite(mu=mu_sun, planets=planets, n_orbits=n_orbits)

    print("=== Perihelion Precession MVP (PHYS-0369) ===")
    print("Model: Schwarzschild weak-field orbit equation with perihelion event detection")
    print(f"Solar mu = GM_sun = {mu_sun:.9e} m^3/s^2")
    print(f"Integrated perihelion cycles per planet: {n_orbits}")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.9e}"))

    print("\nDiagnostics (numerical quality):")
    for name in [p.name for p in planets]:
        d = diagnostics[name]
        print(
            f"- {name}: invariant_drift={d['invariant_drift']:.3e}, "
            f"ode_residual_rms={d['ode_residual_rms']:.3e}, "
            f"delta_phi_std_rad={d['delta_phi_numeric_std_rad']:.3e}"
        )

    mercury_century = float(
        df.loc[df["planet"] == "Mercury", "precession_numeric_arcsec_per_century"].iloc[0]
    )
    max_rel_error = float(df["relative_error_formula_vs_numeric"].max())
    max_invariant_drift = max(float(v["invariant_drift"]) for v in diagnostics.values())
    max_residual = max(float(v["ode_residual_rms"]) for v in diagnostics.values())

    # Basic self-checks for this MVP.
    if not (40.0 <= mercury_century <= 46.0):
        raise RuntimeError(
            f"unexpected Mercury precession per century: {mercury_century:.6f} arcsec/century"
        )
    if max_rel_error > 2e-4:
        raise RuntimeError(f"formula-vs-numeric relative error too large: {max_rel_error:.3e}")
    if max_invariant_drift > 2e-8:
        raise RuntimeError(f"invariant drift too large: {max_invariant_drift:.3e}")
    if max_residual > 1e-8:
        raise RuntimeError(f"ODE residual RMS too large: {max_residual:.3e}")

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
