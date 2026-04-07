"""Electric quadrupole radiation MVP.

This script implements a minimal, auditable model for a harmonic electric
quadrupole mode:
    Q(t) = Q0 * cos(omega t) * diag(-1/2, -1/2, 1)

It validates consistency among:
1) Closed-form total radiated power for the selected mode,
2) Integrated analytic angular distribution,
3) Integrated tensor-kernel angular distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.854_187_812_8e-12  # vacuum permittivity (F/m)
MU0 = 1.256_637_062_12e-6  # vacuum permeability (H/m)
C0 = 1.0 / np.sqrt(EPS0 * MU0)  # speed of light (m/s)


@dataclass(frozen=True)
class QuadrupoleConfig:
    """Configuration for harmonic electric quadrupole radiation."""

    q0_coulomb_m2: float = 1.0e-14
    frequency_hz: float = 5.0e8
    n_theta: int = 720
    n_phi: int = 720
    n_time: int = 720


def quadrupole_shape_tensor() -> np.ndarray:
    """Return the traceless axisymmetric mode shape S = diag(-1/2,-1/2,1)."""
    return np.diag(np.array([-0.5, -0.5, 1.0], dtype=np.float64))


def build_angular_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create midpoint theta-phi grid and sphere-element weights dOmega."""
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi

    theta = (np.arange(n_theta, dtype=np.float64) + 0.5) * dtheta
    phi = (np.arange(n_phi, dtype=np.float64) + 0.5) * dphi

    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing="ij")
    domega_2d = np.sin(theta_2d) * dtheta * dphi
    return theta_2d, phi_2d, domega_2d


def total_power_theory_axisymmetric(cfg: QuadrupoleConfig) -> float:
    """Closed-form average power for Q=Q0 cos(wt) diag(-1/2,-1/2,1).

    <P> = (3 w^6 Q0^2) / (80 pi eps0 c^5)
    """
    omega = 2.0 * np.pi * cfg.frequency_hz
    return 3.0 * omega**6 * cfg.q0_coulomb_m2**2 / (80.0 * np.pi * EPS0 * C0**5)


def angular_power_density_axisymmetric(theta_2d: np.ndarray, cfg: QuadrupoleConfig) -> np.ndarray:
    """Analytic time-averaged dP/dOmega for the axisymmetric quadrupole mode.

    <dP/dOmega> = (9 w^6 Q0^2 / (128 pi^2 eps0 c^5)) * sin^2(theta) * cos^2(theta)
    """
    omega = 2.0 * np.pi * cfg.frequency_hz
    prefactor = 9.0 * omega**6 * cfg.q0_coulomb_m2**2 / (128.0 * np.pi**2 * EPS0 * C0**5)
    s = np.sin(theta_2d)
    c = np.cos(theta_2d)
    return prefactor * (s**2) * (c**2)


def angular_power_density_from_tensor_kernel(
    theta_2d: np.ndarray,
    phi_2d: np.ndarray,
    cfg: QuadrupoleConfig,
) -> np.ndarray:
    """Compute <dP/dOmega> via tensor-kernel formula.

    Formula used:
      <dP/dOmega> = (1/(16*pi^2*eps0*c^5)) * <| n x (n x (Q''' . n)) |^2>

    For Q(t) = Q0 cos(wt) S, Q'''(t) = w^3 Q0 sin(wt) S.
    Time average is estimated numerically from n_time samples of sin^2(wt).
    """
    omega = 2.0 * np.pi * cfg.frequency_hz
    shape = quadrupole_shape_tensor()

    # Direction vectors n(theta, phi).
    sin_t = np.sin(theta_2d)
    nx = sin_t * np.cos(phi_2d)
    ny = sin_t * np.sin(phi_2d)
    nz = np.cos(theta_2d)
    n = np.stack([nx, ny, nz], axis=-1)

    # u = S . n
    u = np.einsum("ij,...j->...i", shape, n)

    # u_perp = n x (n x u) = u - n (n·u)
    nu = np.einsum("...i,...i->...", n, u)
    u_perp = u - nu[..., None] * n
    kernel = np.einsum("...i,...i->...", u_perp, u_perp)

    # Numerical time average of sin^2(wt) over one period.
    period = 1.0 / cfg.frequency_hz
    t = np.linspace(0.0, period, cfg.n_time, endpoint=False)
    avg_phase2 = float(np.mean(np.sin(omega * t) ** 2))

    prefactor = (omega**6 * cfg.q0_coulomb_m2**2 * avg_phase2) / (16.0 * np.pi**2 * EPS0 * C0**5)
    return prefactor * kernel


def integrate_over_sphere(dpdomega_2d: np.ndarray, domega_2d: np.ndarray) -> float:
    """Integrate dP/dOmega over sphere to get total power."""
    return float(np.sum(dpdomega_2d * domega_2d))


def main() -> None:
    cfg = QuadrupoleConfig()

    theta_2d, phi_2d, domega_2d = build_angular_grid(cfg.n_theta, cfg.n_phi)

    dp_analytic = angular_power_density_axisymmetric(theta_2d, cfg)
    dp_tensor = angular_power_density_from_tensor_kernel(theta_2d, phi_2d, cfg)

    p_theory = total_power_theory_axisymmetric(cfg)
    p_num_analytic = integrate_over_sphere(dp_analytic, domega_2d)
    p_num_tensor = integrate_over_sphere(dp_tensor, domega_2d)

    # Symmetry check: axisymmetric mode should be nearly phi-invariant.
    phi_std_over_mean = float(np.mean(np.std(dp_tensor, axis=1) / np.maximum(np.mean(dp_tensor, axis=1), 1e-300)))

    # Consistency checks.
    np.testing.assert_allclose(p_num_analytic, p_theory, rtol=5e-5, atol=0.0)
    np.testing.assert_allclose(p_num_tensor, p_theory, rtol=5e-5, atol=0.0)
    np.testing.assert_allclose(dp_tensor, dp_analytic, rtol=5e-5, atol=0.0)
    np.testing.assert_allclose(phi_std_over_mean, 0.0, atol=1e-12, rtol=0.0)

    # Angular samples from analytic expression (phi-independent for this mode).
    deg = np.array([5.0, 20.0, 35.0, 45.0, 60.0, 75.0, 85.0], dtype=np.float64)
    theta_samples = np.deg2rad(deg)
    theta_samples_2d = theta_samples[:, None]
    dp_samples = angular_power_density_axisymmetric(theta_samples_2d, cfg)[:, 0]

    print("Electric Quadrupole Radiation MVP")
    print(
        f"Config: Q0={cfg.q0_coulomb_m2:.3e} C*m^2, f={cfg.frequency_hz:.3e} Hz, "
        f"n_theta={cfg.n_theta}, n_phi={cfg.n_phi}, n_time={cfg.n_time}"
    )
    print("--- Total power checks ---")
    print(f"P_theory                 : {p_theory:.6e} W")
    print(f"P_numeric (analytic dΩ)  : {p_num_analytic:.6e} W")
    print(f"P_numeric (tensor kernel): {p_num_tensor:.6e} W")
    print(f"RelErr analytic/theory   : {abs(p_num_analytic - p_theory) / p_theory:.3e}")
    print(f"RelErr tensor/theory     : {abs(p_num_tensor - p_theory) / p_theory:.3e}")
    print("--- Angular samples dP/dΩ (W/sr) ---")
    for d, v in zip(deg, dp_samples):
        print(f"theta={d:5.1f} deg -> {v:.6e}")


if __name__ == "__main__":
    main()
