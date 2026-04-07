"""Electromagnetic momentum MVP.

This script demonstrates vacuum electromagnetic momentum for Gaussian wave packets.
It validates the core relations

1) g = epsilon0 * (E x B)                   (momentum density)
2) g = S / c^2                               (Poynting-momentum relation)
3) P_em = U / c for a unidirectional packet   (integrated relation)

and shows momentum cancellation for symmetric counter-propagating packets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, epsilon_0, mu_0

C0 = float(c)
EPS0 = float(epsilon_0)
MU0 = float(mu_0)


@dataclass(frozen=True)
class VacuumMedium:
    """Vacuum constitutive parameters."""

    epsilon: float = EPS0
    mu: float = MU0


def electromagnetic_energy_density(
    e_field: np.ndarray,
    b_field: np.ndarray,
    medium: VacuumMedium,
) -> np.ndarray:
    """Compute u = 1/2 (epsilon|E|^2 + |B|^2/mu)."""
    e_field = np.asarray(e_field, dtype=float)
    b_field = np.asarray(b_field, dtype=float)
    if e_field.shape != b_field.shape or e_field.shape[-1] != 3:
        raise ValueError("e_field and b_field must share shape (..., 3)")

    e_sq = np.sum(e_field * e_field, axis=-1)
    b_sq = np.sum(b_field * b_field, axis=-1)
    return 0.5 * (medium.epsilon * e_sq + b_sq / medium.mu)


def poynting_vector(e_field: np.ndarray, b_field: np.ndarray, medium: VacuumMedium) -> np.ndarray:
    """Compute S = (E x B) / mu."""
    e_field = np.asarray(e_field, dtype=float)
    b_field = np.asarray(b_field, dtype=float)
    if e_field.shape != b_field.shape or e_field.shape[-1] != 3:
        raise ValueError("e_field and b_field must share shape (..., 3)")
    return np.cross(e_field, b_field) / medium.mu


def electromagnetic_momentum_density(
    e_field: np.ndarray,
    b_field: np.ndarray,
    medium: VacuumMedium,
) -> np.ndarray:
    """Compute vacuum momentum density g = epsilon * (E x B)."""
    e_field = np.asarray(e_field, dtype=float)
    b_field = np.asarray(b_field, dtype=float)
    if e_field.shape != b_field.shape or e_field.shape[-1] != 3:
        raise ValueError("e_field and b_field must share shape (..., 3)")
    return medium.epsilon * np.cross(e_field, b_field)


def build_gaussian_plane_wave(
    x: np.ndarray,
    t: float,
    e0: float,
    wavelength: float,
    sigma: float,
    x0: float,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a y-polarized Gaussian wave packet in vacuum.

    Parameters
    ----------
    direction:
        +1 for +x propagation, -1 for -x propagation.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    k = 2.0 * np.pi / wavelength
    xi = x - x0 - direction * C0 * t
    envelope = np.exp(-(xi**2) / (2.0 * sigma**2))
    carrier = np.cos(k * xi)
    ey = e0 * envelope * carrier

    e_field = np.zeros((x.size, 3), dtype=float)
    b_field = np.zeros((x.size, 3), dtype=float)
    e_field[:, 1] = ey
    b_field[:, 2] = direction * ey / C0
    return e_field, b_field


def integrate_scalar_over_x(values: np.ndarray, x: np.ndarray, area: float = 1.0) -> float:
    """Integrate scalar density over x with a fixed cross-section area."""
    values = np.asarray(values, dtype=float)
    return float(np.trapezoid(values, x) * area)


def integrate_vector_over_x(vectors: np.ndarray, x: np.ndarray, area: float = 1.0) -> np.ndarray:
    """Integrate vector density over x with a fixed cross-section area."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    integrated = np.trapezoid(vectors, x, axis=0) * area
    return np.asarray(integrated, dtype=float)


def summarize_case(case_name: str, x: np.ndarray, e_field: np.ndarray, b_field: np.ndarray, medium: VacuumMedium) -> dict[str, float]:
    """Compute integrated EM energy/momentum diagnostics for one field configuration."""
    u = electromagnetic_energy_density(e_field, b_field, medium)
    s = poynting_vector(e_field, b_field, medium)
    g = electromagnetic_momentum_density(e_field, b_field, medium)

    u_total = integrate_scalar_over_x(u, x)
    p_total = integrate_vector_over_x(g, x)

    max_gs_consistency = float(np.max(np.abs(g - s / (C0**2))))
    max_transverse_momentum_density = float(np.max(np.abs(g[:, 1:])))
    return {
        "case": case_name,
        "U_total_J_per_m2": u_total,
        "P_x_Ns_per_m2": float(p_total[0]),
        "U_over_c": u_total / C0,
        "P_over_Uc": float(p_total[0] / (u_total / C0)) if u_total > 0.0 else np.nan,
        "max|g-S/c^2|": max_gs_consistency,
        "max|g_transverse|": max_transverse_momentum_density,
    }


def main() -> None:
    medium = VacuumMedium()
    x = np.linspace(-5.0, 5.0, 20000)
    t = 0.0

    e0 = 180.0  # V/m
    wavelength = 0.25  # m
    sigma = 0.50  # m

    # Case A: one forward packet (+x)
    e_fwd, b_fwd = build_gaussian_plane_wave(
        x=x,
        t=t,
        e0=e0,
        wavelength=wavelength,
        sigma=sigma,
        x0=-1.0,
        direction=+1,
    )

    # Case B: symmetric forward + backward packets, net momentum should cancel.
    e_bwd, b_bwd = build_gaussian_plane_wave(
        x=x,
        t=t,
        e0=e0,
        wavelength=wavelength,
        sigma=sigma,
        x0=+1.0,
        direction=-1,
    )
    e_pair = e_fwd + e_bwd
    b_pair = b_fwd + b_bwd

    record_forward = summarize_case("forward_packet", x, e_fwd, b_fwd, medium)
    record_pair = summarize_case("counter_propagating_pair", x, e_pair, b_pair, medium)
    summary_df = pd.DataFrame([record_forward, record_pair])

    # Physics checks
    np.testing.assert_allclose(
        record_forward["P_x_Ns_per_m2"],
        record_forward["U_over_c"],
        rtol=8e-4,
        atol=0.0,
    )

    np.testing.assert_allclose(
        record_forward["max|g-S/c^2|"],
        0.0,
        atol=2e-12,
        rtol=0.0,
    )

    # Symmetric pair keeps substantial energy but near-zero net momentum.
    np.testing.assert_allclose(
        record_pair["P_x_Ns_per_m2"],
        0.0,
        atol=2e-16,
        rtol=0.0,
    )
    assert record_pair["U_total_J_per_m2"] > 0.0

    print("Electromagnetic Momentum MVP")
    print(f"Grid points: {x.size}, x-range: [{x[0]:.2f}, {x[-1]:.2f}] m")
    print()
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))


if __name__ == "__main__":
    main()
