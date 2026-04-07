"""Minimal runnable MVP for radiation pressure (PHYS-0172).

This script demonstrates radiation pressure on an opaque surface:
1) Continuum momentum-flux formula.
2) Independent photon-momentum derivation.
3) Deterministic checks for absorber/reflector and oblique incidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, h


@dataclass(frozen=True)
class SailCase:
    """Simple solar-sail-like scenario."""

    intensity_w_m2: float = 1361.0
    reflectivity: float = 0.88
    incidence_deg: float = 25.0
    area_m2: float = 120.0
    mass_kg: float = 18.0


def _validate_common_inputs(intensity_w_m2: np.ndarray | float, reflectivity: float, incidence_deg: float) -> None:
    intensity = np.asarray(intensity_w_m2, dtype=float)
    if np.any(intensity < 0.0):
        raise ValueError("intensity_w_m2 must be non-negative")
    if not (0.0 <= reflectivity <= 1.0):
        raise ValueError("reflectivity must be within [0, 1]")
    if not (0.0 <= incidence_deg <= 90.0):
        raise ValueError("incidence_deg must be within [0, 90]")


def radiation_pressure(intensity_w_m2: np.ndarray | float, reflectivity: float, incidence_deg: float) -> np.ndarray:
    """Compute normal radiation pressure for an opaque surface.

    Formula:
        p = (I / c) * (1 + R) * cos(theta)^2

    Parameters
    ----------
    intensity_w_m2:
        Incident intensity along beam direction (W/m^2), scalar or array.
    reflectivity:
        Surface reflectivity R in [0, 1]. Opaque-surface assumption implies
        absorptivity = 1 - R and transmissivity = 0.
    incidence_deg:
        Angle between beam direction and surface normal in degrees.

    Returns
    -------
    np.ndarray
        Pressure in Pa with same broadcasted shape as intensity input.
    """
    _validate_common_inputs(intensity_w_m2, reflectivity, incidence_deg)
    intensity = np.asarray(intensity_w_m2, dtype=float)
    theta = np.deg2rad(float(incidence_deg))
    return (intensity / c) * (1.0 + reflectivity) * np.cos(theta) ** 2


def radiation_force(
    intensity_w_m2: np.ndarray | float,
    reflectivity: float,
    incidence_deg: float,
    area_m2: float,
) -> np.ndarray:
    """Compute normal force from pressure times illuminated area."""
    if area_m2 < 0.0:
        raise ValueError("area_m2 must be non-negative")
    return radiation_pressure(intensity_w_m2, reflectivity, incidence_deg) * area_m2


def photon_energy(wavelength_m: float) -> float:
    """Photon energy E = h*c/lambda."""
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    return float(h * c / wavelength_m)


def photon_flux_on_surface(intensity_w_m2: float, wavelength_m: float, incidence_deg: float) -> float:
    """Photons hitting unit surface area per second (includes projection cos(theta))."""
    if intensity_w_m2 < 0.0:
        raise ValueError("intensity_w_m2 must be non-negative")
    if not (0.0 <= incidence_deg <= 90.0):
        raise ValueError("incidence_deg must be within [0, 90]")
    theta = np.deg2rad(incidence_deg)
    return float(intensity_w_m2 * np.cos(theta) / photon_energy(wavelength_m))


def photon_based_pressure(intensity_w_m2: float, wavelength_m: float, reflectivity: float, incidence_deg: float) -> float:
    """Compute pressure from photon counting and per-photon momentum change.

    For one photon, normal momentum change is:
        Delta p_n = (1 + R) * (h/lambda) * cos(theta)
    Pressure is then flux * Delta p_n.
    """
    _validate_common_inputs(float(intensity_w_m2), reflectivity, incidence_deg)
    theta = np.deg2rad(incidence_deg)
    photon_flux = photon_flux_on_surface(intensity_w_m2, wavelength_m, incidence_deg)
    photon_momentum = h / wavelength_m
    delta_p_normal = (1.0 + reflectivity) * photon_momentum * np.cos(theta)
    return float(photon_flux * delta_p_normal)


def build_pressure_table(intensities: np.ndarray, reflectivities: np.ndarray, incidence_deg: float) -> pd.DataFrame:
    """Build a tidy table of pressure values for multiple intensities and reflectivities."""
    rows: list[dict[str, float]] = []
    for refl in reflectivities:
        pressure_values = radiation_pressure(intensities, float(refl), incidence_deg)
        for intensity, pressure in zip(intensities, pressure_values, strict=True):
            rows.append(
                {
                    "intensity_W_m2": float(intensity),
                    "reflectivity": float(refl),
                    "incidence_deg": float(incidence_deg),
                    "pressure_Pa": float(pressure),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    intensities = np.array([250.0, 750.0, 1361.0, 5000.0], dtype=float)
    reflectivities = np.array([0.0, 0.5, 1.0], dtype=float)

    pressure_table = build_pressure_table(intensities, reflectivities, incidence_deg=0.0)

    # Check 1: perfect reflector has twice pressure of perfect absorber at normal incidence.
    p_abs = float(radiation_pressure(1000.0, reflectivity=0.0, incidence_deg=0.0))
    p_ref = float(radiation_pressure(1000.0, reflectivity=1.0, incidence_deg=0.0))
    np.testing.assert_allclose(p_ref, 2.0 * p_abs, rtol=1e-12, atol=0.0)

    # Check 2: oblique incidence scales with cos^2(theta).
    theta_test = 60.0
    p_theta0 = float(radiation_pressure(1000.0, reflectivity=0.3, incidence_deg=0.0))
    p_theta = float(radiation_pressure(1000.0, reflectivity=0.3, incidence_deg=theta_test))
    np.testing.assert_allclose(p_theta, p_theta0 * np.cos(np.deg2rad(theta_test)) ** 2, rtol=1e-12, atol=0.0)

    # Check 3: continuum formula and photon-momentum derivation are consistent.
    intensity_test = 8.0e4
    reflectivity_test = 0.85
    incidence_test = 37.0
    wavelength_test = 532e-9
    p_continuum = float(radiation_pressure(intensity_test, reflectivity_test, incidence_test))
    p_photon = photon_based_pressure(intensity_test, wavelength_test, reflectivity_test, incidence_test)
    np.testing.assert_allclose(p_continuum, p_photon, rtol=1e-12, atol=0.0)

    # Check 4: for absorber and normal incidence, pressure equals energy density u = I/c.
    np.testing.assert_allclose(
        float(radiation_pressure(intensity_test, reflectivity=0.0, incidence_deg=0.0)),
        intensity_test / c,
        rtol=1e-12,
        atol=0.0,
    )

    sail = SailCase()
    sail_pressure = float(radiation_pressure(sail.intensity_w_m2, sail.reflectivity, sail.incidence_deg))
    sail_force = float(radiation_force(sail.intensity_w_m2, sail.reflectivity, sail.incidence_deg, sail.area_m2))
    sail_acceleration = sail_force / sail.mass_kg

    check_rows = [
        {
            "check": "reflector pressure = 2 * absorber pressure (theta=0)",
            "status": "OK",
        },
        {
            "check": "pressure scales with cos^2(theta)",
            "status": "OK",
        },
        {
            "check": "continuum pressure == photon-based pressure",
            "status": "OK",
        },
        {
            "check": "absorber normal-incidence pressure = I/c",
            "status": "OK",
        },
    ]

    print("=== Radiation Pressure MVP (PHYS-0172) ===")
    print(f"c = {c:.9e} m/s")
    print("\n[Pressure table at normal incidence]")
    print(pressure_table.to_string(index=False))

    print("\n[Cross-check values]")
    print(f"Continuum pressure : {p_continuum:.9e} Pa")
    print(f"Photon-based value : {p_photon:.9e} Pa")
    print(f"Absolute difference: {abs(p_continuum - p_photon):.3e} Pa")

    print("\n[Solar sail-like scenario]")
    print(
        "Inputs: I={I:.1f} W/m^2, R={R:.2f}, theta={th:.1f} deg, A={A:.1f} m^2, m={m:.1f} kg".format(
            I=sail.intensity_w_m2,
            R=sail.reflectivity,
            th=sail.incidence_deg,
            A=sail.area_m2,
            m=sail.mass_kg,
        )
    )
    print(f"Pressure      : {sail_pressure:.9e} Pa")
    print(f"Force         : {sail_force:.9e} N")
    print(f"Acceleration  : {sail_acceleration:.9e} m/s^2")

    print("\n[Checks]")
    print(pd.DataFrame(check_rows).to_string(index=False))
    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
