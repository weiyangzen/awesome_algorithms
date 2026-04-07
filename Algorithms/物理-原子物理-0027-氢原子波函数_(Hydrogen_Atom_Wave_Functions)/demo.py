"""Minimal runnable MVP for Hydrogen atom wave functions.

This script implements hydrogenic bound-state wave functions in spherical coordinates,
then validates normalization, orthogonality, expectation value <r>, and energy levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import eval_genlaguerre, factorial, lpmv


BOHR_RADIUS = 1.0  # atomic units
RYDBERG_ENERGY_EV = 13.605693122994


@dataclass
class GridSpec:
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray


def validate_quantum_numbers(n: int, l: int, m: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    if abs(m) > l:
        raise ValueError("m must satisfy |m| <= l")


def radial_wavefunction(n: int, l: int, r: np.ndarray, a0: float = BOHR_RADIUS) -> np.ndarray:
    """Return normalized radial wavefunction R_{n,l}(r) for hydrogen."""
    validate_quantum_numbers(n=n, l=l, m=0)

    rho = 2.0 * r / (float(n) * float(a0))
    k = n - l - 1
    alpha = 2 * l + 1

    prefactor = np.sqrt(
        (2.0 / (n * a0)) ** 3
        * float(factorial(k, exact=False))
        / (2.0 * n * float(factorial(n + l, exact=False)))
    )

    laguerre = eval_genlaguerre(k, alpha, rho)
    return prefactor * np.exp(-rho / 2.0) * np.power(rho, l) * laguerre


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return normalized spherical harmonic Y_l^m(theta, phi)."""
    if l < 0:
        raise ValueError("l must be >= 0")
    if abs(m) > l:
        raise ValueError("m must satisfy |m| <= l")

    if m < 0:
        mp = -m
        y_pos = spherical_harmonic(l=l, m=mp, theta=theta, phi=phi)
        return ((-1) ** mp) * np.conjugate(y_pos)

    x = np.cos(theta)
    plm = lpmv(m, l, x)
    norm = np.sqrt(
        (2.0 * l + 1.0)
        / (4.0 * np.pi)
        * float(factorial(l - m, exact=False))
        / float(factorial(l + m, exact=False))
    )
    return norm * plm * np.exp(1j * m * phi)


def hydrogen_wavefunction_on_grid(
    n: int,
    l: int,
    m: int,
    grid: GridSpec,
    a0: float = BOHR_RADIUS,
) -> np.ndarray:
    """Return psi_{n,l,m}(r,theta,phi) sampled on a tensor-product spherical grid."""
    validate_quantum_numbers(n=n, l=l, m=m)

    rr = grid.r[:, None, None]
    tt = grid.theta[None, :, None]
    pp = grid.phi[None, None, :]

    radial = radial_wavefunction(n=n, l=l, r=rr, a0=a0)
    angular = spherical_harmonic(l=l, m=m, theta=tt, phi=pp)
    return radial * angular


def integrate_spherical(values: np.ndarray, grid: GridSpec) -> complex:
    """Integrate values(r,theta,phi) over dr dtheta dphi using trapezoidal rules."""
    step_phi = np.trapezoid(values, grid.phi, axis=2)
    step_theta = np.trapezoid(step_phi, grid.theta, axis=1)
    step_r = np.trapezoid(step_theta, grid.r, axis=0)
    return step_r


def probability_density(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def check_normalization(psi: np.ndarray, grid: GridSpec) -> float:
    rr = grid.r[:, None, None]
    sin_theta = np.sin(grid.theta)[None, :, None]
    integrand = probability_density(psi) * rr * rr * sin_theta
    norm = integrate_spherical(integrand, grid)
    return float(np.real(norm))


def check_overlap(psi_a: np.ndarray, psi_b: np.ndarray, grid: GridSpec) -> complex:
    rr = grid.r[:, None, None]
    sin_theta = np.sin(grid.theta)[None, :, None]
    integrand = np.conjugate(psi_a) * psi_b * rr * rr * sin_theta
    return integrate_spherical(integrand, grid)


def expectation_r(psi: np.ndarray, grid: GridSpec) -> float:
    rr = grid.r[:, None, None]
    sin_theta = np.sin(grid.theta)[None, :, None]
    prob = probability_density(psi)

    numerator = integrate_spherical(prob * rr * rr * rr * sin_theta, grid)
    denominator = integrate_spherical(prob * rr * rr * sin_theta, grid)
    return float(np.real(numerator / denominator))


def analytic_expectation_r(n: int, l: int, a0: float = BOHR_RADIUS) -> float:
    validate_quantum_numbers(n=n, l=l, m=0)
    return 0.5 * a0 * (3.0 * n * n - l * (l + 1))


def hydrogen_energy_ev(n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    return -RYDBERG_ENERGY_EV / float(n * n)


def build_grid(
    r_max: float = 30.0,
    nr: int = 220,
    ntheta: int = 90,
    nphi: int = 120,
) -> GridSpec:
    r = np.linspace(1e-6, r_max, nr, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, ntheta, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, dtype=np.float64)
    return GridSpec(r=r, theta=theta, phi=phi)


def main() -> None:
    grid = build_grid()

    psi_100 = hydrogen_wavefunction_on_grid(1, 0, 0, grid)
    psi_200 = hydrogen_wavefunction_on_grid(2, 0, 0, grid)
    psi_211 = hydrogen_wavefunction_on_grid(2, 1, 1, grid)

    norm_100 = check_normalization(psi_100, grid)
    norm_211 = check_normalization(psi_211, grid)

    overlap_100_200 = check_overlap(psi_100, psi_200, grid)
    overlap_abs = float(np.abs(overlap_100_200))

    exp_r_100 = expectation_r(psi_100, grid)
    exp_r_100_ref = analytic_expectation_r(1, 0)
    exp_r_err = abs(exp_r_100 - exp_r_100_ref)

    energies = [(n, hydrogen_energy_ev(n)) for n in range(1, 5)]

    print("Hydrogen atom wavefunction MVP")
    print(f"grid=(nr={grid.r.size}, ntheta={grid.theta.size}, nphi={grid.phi.size})")
    print(f"norm_100={norm_100:.8f}")
    print(f"norm_211={norm_211:.8f}")
    print(f"|<100|200>|={overlap_abs:.3e}")
    print(f"<r>_100={exp_r_100:.8f} a0")
    print(f"<r>_100_ref={exp_r_100_ref:.8f} a0")
    print(f"<r>_abs_err={exp_r_err:.3e} a0")
    print("energy_levels_ev=")
    for n, e in energies:
        print(f"  n={n}: {e:.8f} eV")

    assert abs(norm_100 - 1.0) < 2.0e-2, f"1s normalization failed: {norm_100}"
    assert abs(norm_211 - 1.0) < 2.0e-2, f"2p(m=1) normalization failed: {norm_211}"
    assert overlap_abs < 2.5e-2, f"Orthogonality check failed: |<100|200>|={overlap_abs}"
    assert exp_r_err < 2.5e-2, f"<r> check failed: err={exp_r_err}"
    assert np.isclose(energies[1][1] / energies[0][1], 0.25, atol=1e-12), "Energy ratio E2/E1 mismatch"

    print("All checks passed.")


if __name__ == "__main__":
    main()
