"""Minimal Partial Wave Analysis MVP for quantum scattering.

Model:
- 3D central potential scattering with a spherical square well
  V(r) = -V0 (r < a), 0 (r >= a)
- Elastic scattering only.

We compute phase shifts delta_l by matching logarithmic derivatives at r=a,
then evaluate:
- partial cross sections sigma_l
- total cross section sigma_tot
- differential cross section d sigma / d Omega
- optical theorem consistency check
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import special


def wave_numbers(E: float, V0: float, mass: float = 1.0, hbar: float = 1.0) -> tuple[float, float]:
    """Return external and internal wave numbers (k, q)."""
    if E <= 0.0:
        raise ValueError("E must be positive for scattering states.")
    if E + V0 <= 0.0:
        raise ValueError("E + V0 must be positive so internal momentum is real.")

    k = math.sqrt(2.0 * mass * E) / hbar
    q = math.sqrt(2.0 * mass * (E + V0)) / hbar
    return k, q


def phase_shift_square_well(
    l: int,
    k: float,
    q: float,
    a: float,
) -> float:
    """Compute phase shift delta_l for spherical square-well potential.

    Using boundary matching at r=a for reduced radial wave u_l(r):
      inside:  u = A * j_l(qr)
      outside: u = B * [cos(delta) j_l(kr) - sin(delta) y_l(kr)]

    Solving matching equations yields:
      tan(delta) = [k j'_l(ka) - beta j_l(ka)] / [k y'_l(ka) - beta y_l(ka)]
      beta = q j'_l(qa) / j_l(qa)
    """
    if l < 0:
        raise ValueError("l must be non-negative.")
    if a <= 0.0:
        raise ValueError("a must be positive.")

    ka = k * a
    qa = q * a

    j_ka = special.spherical_jn(l, ka)
    y_ka = special.spherical_yn(l, ka)
    j_ka_p = special.spherical_jn(l, ka, derivative=True)
    y_ka_p = special.spherical_yn(l, ka, derivative=True)

    j_qa = special.spherical_jn(l, qa)
    j_qa_p = special.spherical_jn(l, qa, derivative=True)

    if abs(j_qa) < 1e-12:
        raise RuntimeError(
            f"Encountered near-node at l={l}: j_l(qa)≈0, unstable log-derivative matching."
        )

    beta = q * j_qa_p / j_qa

    num = k * j_ka_p - beta * j_ka
    den = k * y_ka_p - beta * y_ka

    delta = math.atan2(num, den)
    return float(delta)


def compute_phase_shifts(
    l_max: int,
    E: float,
    V0: float,
    a: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute delta_l for l=0..l_max and return (k, deltas)."""
    if l_max < 0:
        raise ValueError("l_max must be non-negative.")

    k, q = wave_numbers(E=E, V0=V0, mass=mass, hbar=hbar)
    deltas = np.array([phase_shift_square_well(l, k, q, a) for l in range(l_max + 1)], dtype=float)
    return k, deltas


def scattering_amplitude(theta: float, k: float, deltas: np.ndarray) -> complex:
    """Partial-wave scattering amplitude f(theta)."""
    mu = math.cos(theta)
    total = 0.0j
    for l, delta in enumerate(deltas):
        p_l = special.eval_legendre(l, mu)
        total += (2 * l + 1) * np.exp(1j * delta) * math.sin(delta) * p_l
    return total / k


def differential_cross_section(thetas: Iterable[float], k: float, deltas: np.ndarray) -> np.ndarray:
    """Return d sigma / d Omega over angle grid."""
    values = []
    for theta in thetas:
        amp = scattering_amplitude(theta, k, deltas)
        values.append(abs(amp) ** 2)
    return np.asarray(values, dtype=float)


def partial_cross_sections(k: float, deltas: np.ndarray) -> np.ndarray:
    """Return sigma_l = (4*pi/k^2) (2l+1) sin^2(delta_l)."""
    l_vals = np.arange(deltas.size)
    return (4.0 * math.pi / (k**2)) * (2 * l_vals + 1) * np.sin(deltas) ** 2


def total_cross_section(k: float, deltas: np.ndarray) -> float:
    """Return sigma_tot via partial-wave sum."""
    return float(np.sum(partial_cross_sections(k, deltas)))


def optical_theorem_residual(k: float, deltas: np.ndarray, sigma_tot: float) -> float:
    """Check Im f(0) = k sigma_tot / (4*pi), return absolute residual."""
    f0 = scattering_amplitude(theta=0.0, k=k, deltas=deltas)
    rhs = k * sigma_tot / (4.0 * math.pi)
    return abs(f0.imag - rhs)


def main() -> None:
    # Fixed, non-interactive configuration for reproducible MVP output.
    E = 5.0
    V0 = 12.0
    a = 1.0
    l_max = 8
    mass = 1.0
    hbar = 1.0

    k, deltas = compute_phase_shifts(l_max=l_max, E=E, V0=V0, a=a, mass=mass, hbar=hbar)
    sigma_l = partial_cross_sections(k, deltas)
    sigma_tot = total_cross_section(k, deltas)
    ot_res = optical_theorem_residual(k, deltas, sigma_tot)

    angle_deg = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0], dtype=float)
    angle_rad = np.deg2rad(angle_deg)
    dsdo = differential_cross_section(angle_rad, k, deltas)

    if not np.all(sigma_l >= 0.0):
        raise RuntimeError("Physical sanity check failed: partial cross section must be non-negative.")

    print("Partial Wave Analysis Demo: Spherical Square-Well Scattering")
    print(f"Parameters: E={E:.3f}, V0={V0:.3f}, a={a:.3f}, l_max={l_max}, m={mass:.1f}, hbar={hbar:.1f}")
    print(f"k = {k:.6f}")
    print("-" * 78)
    print(f"{'l':>2} {'delta_l(rad)':>14} {'delta_l(deg)':>14} {'sigma_l':>14}")
    print("-" * 78)
    for l, delta, sig in zip(range(deltas.size), deltas, sigma_l):
        print(f"{l:2d} {delta:14.8f} {np.rad2deg(delta):14.6f} {sig:14.8f}")
    print("-" * 78)
    print(f"sigma_tot = {sigma_tot:.8f}")
    print(f"optical_theorem_abs_residual = {ot_res:.3e}")
    print("-" * 78)
    print(f"{'theta(deg)':>10} {'dσ/dΩ':>16}")
    print("-" * 78)
    for deg, val in zip(angle_deg, dsdo):
        print(f"{deg:10.1f} {val:16.8f}")


if __name__ == "__main__":
    main()
