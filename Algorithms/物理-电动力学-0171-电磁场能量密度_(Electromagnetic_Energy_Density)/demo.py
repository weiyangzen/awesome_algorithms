"""Electromagnetic energy density MVP.

This script demonstrates:
1) A monochromatic plane wave in vacuum.
2) A static capacitor-like electric field.

Both are evaluated with:
    u = 1/2 * (epsilon * |E|^2 + |B|^2 / mu)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.854_187_812_8e-12  # vacuum permittivity (F/m)
MU0 = 1.256_637_062_12e-6  # vacuum permeability (H/m)
C0 = 1.0 / np.sqrt(EPS0 * MU0)  # speed of light (m/s)


@dataclass(frozen=True)
class LinearIsotropicMedium:
    """Simple linear isotropic medium defined by epsilon and mu."""

    epsilon: float = EPS0
    mu: float = MU0


def electromagnetic_energy_density(
    e_field: np.ndarray, b_field: np.ndarray, medium: LinearIsotropicMedium
) -> np.ndarray:
    """Compute electromagnetic energy density for vector fields.

    Parameters
    ----------
    e_field, b_field:
        Arrays with shape (..., 3), representing E and B vector fields.
    medium:
        Material parameters (epsilon, mu).

    Returns
    -------
    np.ndarray
        Energy density array with shape (...) in J/m^3.
    """
    e_field = np.asarray(e_field, dtype=float)
    b_field = np.asarray(b_field, dtype=float)
    if e_field.shape != b_field.shape or e_field.shape[-1] != 3:
        raise ValueError("e_field and b_field must share shape (..., 3)")

    e_sq = np.sum(e_field * e_field, axis=-1)
    b_sq = np.sum(b_field * b_field, axis=-1)
    return 0.5 * (medium.epsilon * e_sq + b_sq / medium.mu)


def build_plane_wave_fields(
    x: np.ndarray, t: float, e0: float, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 1D plane wave propagating along +x in vacuum."""
    omega = 2.0 * np.pi * frequency_hz
    k = omega / C0
    phase = k * x - omega * t

    e_field = np.zeros((x.size, 3), dtype=float)
    b_field = np.zeros((x.size, 3), dtype=float)
    e_field[:, 1] = e0 * np.sin(phase)  # Ey
    b_field[:, 2] = (e0 / C0) * np.sin(phase)  # Bz
    return e_field, b_field


def build_capacitor_like_fields(x: np.ndarray, e0: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an idealized static field: uniform E, zero B."""
    e_field = np.zeros((x.size, 3), dtype=float)
    b_field = np.zeros((x.size, 3), dtype=float)
    e_field[:, 0] = e0
    return e_field, b_field


def main() -> None:
    medium = LinearIsotropicMedium()
    x = np.linspace(0.0, 1.0, 2000, endpoint=False)

    # Case A: plane wave
    e0_wave = 120.0  # V/m
    wavelength = 0.1  # m (10 full wavelengths over the 1 m domain)
    frequency = C0 / wavelength  # Hz
    t = 3.3e-9  # s
    e_wave, b_wave = build_plane_wave_fields(x, t, e0_wave, frequency)
    u_wave = electromagnetic_energy_density(e_wave, b_wave, medium)

    u_e_wave = 0.5 * medium.epsilon * np.sum(e_wave * e_wave, axis=-1)
    u_b_wave = 0.5 * np.sum(b_wave * b_wave, axis=-1) / medium.mu
    max_term_balance_error = np.max(np.abs(u_e_wave - u_b_wave))

    u_wave_mean = float(np.mean(u_wave))
    u_wave_mean_theory = 0.5 * medium.epsilon * e0_wave**2

    s_vec = np.cross(e_wave, b_wave) / medium.mu
    s_x = s_vec[:, 0]
    max_poynting_relation_error = np.max(np.abs(s_x - C0 * u_wave))

    # Case B: static capacitor-like field
    e0_static = 5.0e3  # V/m
    e_static, b_static = build_capacitor_like_fields(x, e0_static)
    u_static = electromagnetic_energy_density(e_static, b_static, medium)
    u_static_theory = 0.5 * medium.epsilon * e0_static**2

    # Numerical checks
    np.testing.assert_allclose(u_wave_mean, u_wave_mean_theory, rtol=2e-3, atol=0.0)
    np.testing.assert_allclose(np.mean(u_static), u_static_theory, rtol=1e-12, atol=0.0)

    dx = x[1] - x[0]
    total_energy_line_density = float(np.sum(u_wave) * dx)  # J/m^2 for unit cross section

    print("Electromagnetic Energy Density MVP")
    print(f"Grid points: {x.size}, dx = {dx:.6e} m")
    print("--- Plane wave (vacuum) ---")
    print(f"Mean u (numeric) : {u_wave_mean:.6e} J/m^3")
    print(f"Mean u (theory)  : {u_wave_mean_theory:.6e} J/m^3")
    print(f"Max |u_E-u_B|    : {max_term_balance_error:.6e} J/m^3")
    print(f"Max |Sx-c*u|     : {max_poynting_relation_error:.6e} W/m^2")
    print(f"Integral u dx    : {total_energy_line_density:.6e} J/m^2")
    print("--- Static field ---")
    print(f"u (numeric mean) : {float(np.mean(u_static)):.6e} J/m^3")
    print(f"u (theory)       : {u_static_theory:.6e} J/m^3")


if __name__ == "__main__":
    main()
