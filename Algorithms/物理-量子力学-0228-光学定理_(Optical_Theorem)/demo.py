"""Optical theorem MVP using hard-sphere quantum scattering.

We verify, for purely elastic scattering, that
    sigma_tot = (4*pi/k) * Im f(theta=0)
using partial-wave expansion with hard-sphere phase shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn


@dataclass(frozen=True)
class OpticalTheoremConfig:
    """Configuration for the optical-theorem numerical experiment."""

    sphere_radius: float = 1.0
    k_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    l_buffer: int = 18
    min_lmax: int = 12
    quad_order: int = 480
    optical_rtol: float = 1e-12
    elastic_rtol: float = 2e-3


def choose_lmax(k: float, radius: float, l_buffer: int, min_lmax: int) -> int:
    """Heuristic truncation for partial-wave expansion."""
    ka = k * radius
    return max(min_lmax, int(np.ceil(ka + l_buffer)))


def hard_sphere_phase_shifts(ka: float, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute hard-sphere phase shifts from boundary condition u_l(r=a)=0.

    For hard-sphere scattering:
        tan(delta_l) = j_l(ka) / y_l(ka)
    where j_l and y_l are spherical Bessel functions of first/second kind.
    """
    ell = np.arange(lmax + 1, dtype=np.int64)
    jl = spherical_jn(ell, ka)
    yl = spherical_yn(ell, ka)
    delta = np.arctan2(jl, yl)
    return ell, delta


def partial_wave_coefficients(ell: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Return coefficients c_l = (2l+1) e^{i delta_l} sin(delta_l)."""
    return (2.0 * ell + 1.0) * np.exp(1j * delta) * np.sin(delta)


def forward_scattering_amplitude(k: float, coeffs: np.ndarray) -> complex:
    """Compute f(0) from partial waves using P_l(1)=1."""
    return complex(np.sum(coeffs) / k)


def scattering_amplitude_on_mu_grid(k: float, coeffs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Compute f(theta) on grid mu=cos(theta) with partial-wave sum."""
    amp = np.zeros_like(mu, dtype=np.complex128)
    for ell, coeff in enumerate(coeffs):
        amp += coeff * eval_legendre(ell, mu)
    return amp / k


def sigma_total_from_phase_shifts(k: float, ell: np.ndarray, delta: np.ndarray) -> float:
    """Total cross section from phase shifts.

    sigma_tot = (4*pi/k^2) * sum_l (2l+1) sin^2(delta_l)
    """
    weights = (2.0 * ell + 1.0) * np.sin(delta) ** 2
    return float((4.0 * np.pi / (k * k)) * np.sum(weights))


def sigma_optical_theorem(k: float, f0: complex) -> float:
    """Total cross section via the optical theorem."""
    return float((4.0 * np.pi / k) * np.imag(f0))


def sigma_elastic_angle_integral(k: float, coeffs: np.ndarray, quad_order: int) -> float:
    """Compute sigma_el = 2*pi * integral_{-1}^{1} |f(mu)|^2 dmu."""
    mu, w = np.polynomial.legendre.leggauss(quad_order)
    amp = scattering_amplitude_on_mu_grid(k, coeffs, mu)
    return float(2.0 * np.pi * np.sum(w * np.abs(amp) ** 2))


def run_case(k: float, cfg: OpticalTheoremConfig) -> dict[str, float]:
    """Run one momentum point and return diagnostics."""
    lmax = choose_lmax(k, cfg.sphere_radius, cfg.l_buffer, cfg.min_lmax)
    ka = k * cfg.sphere_radius
    ell, delta = hard_sphere_phase_shifts(ka=ka, lmax=lmax)
    coeffs = partial_wave_coefficients(ell, delta)

    f0 = forward_scattering_amplitude(k, coeffs)
    sigma_tot = sigma_total_from_phase_shifts(k, ell, delta)
    sigma_opt = sigma_optical_theorem(k, f0)
    sigma_el = sigma_elastic_angle_integral(k, coeffs, cfg.quad_order)

    rel_err_optical = abs(sigma_tot - sigma_opt) / max(sigma_tot, 1e-15)
    rel_err_elastic = abs(sigma_tot - sigma_el) / max(sigma_tot, 1e-15)

    s_matrix = np.exp(2j * delta)
    unitarity_deviation = float(np.max(np.abs(np.abs(s_matrix) - 1.0)))

    return {
        "k": float(k),
        "ka": float(ka),
        "lmax": float(lmax),
        "sigma_tot_phase": float(sigma_tot),
        "sigma_tot_optical": float(sigma_opt),
        "sigma_elastic_int": float(sigma_el),
        "rel_err_optical": float(rel_err_optical),
        "rel_err_elastic": float(rel_err_elastic),
        "unitarity_dev": float(unitarity_deviation),
    }


def main() -> None:
    cfg = OpticalTheoremConfig()

    print("Optical theorem MVP: hard-sphere elastic quantum scattering")
    print(
        "k  | ka  | lmax | sigma_phase | sigma_optical | sigma_elastic | "
        "rel_err_opt | rel_err_el | max||S_l|-1|"
    )

    results: list[dict[str, float]] = []
    for k in cfg.k_values:
        row = run_case(float(k), cfg)
        results.append(row)
        print(
            f"{row['k']:>3.1f} | "
            f"{row['ka']:>3.1f} | "
            f"{int(row['lmax']):>4d} | "
            f"{row['sigma_tot_phase']:>11.6f} | "
            f"{row['sigma_tot_optical']:>13.6f} | "
            f"{row['sigma_elastic_int']:>13.6f} | "
            f"{row['rel_err_optical']:>11.3e} | "
            f"{row['rel_err_elastic']:>10.3e} | "
            f"{row['unitarity_dev']:>10.3e}"
        )

    for row in results:
        assert row["rel_err_optical"] < cfg.optical_rtol, "Optical theorem mismatch is too large."
        assert row["rel_err_elastic"] < cfg.elastic_rtol, "Elastic angular integral mismatch is too large."
        assert row["unitarity_dev"] < 1e-12, "Hard-sphere S-matrix should be unitary."

    low_energy = min(results, key=lambda r: r["ka"])
    sigma_classical_geometric = np.pi * cfg.sphere_radius**2
    assert low_energy["sigma_tot_phase"] > 2.0 * sigma_classical_geometric, (
        "Low-energy hard-sphere quantum cross section should exceed classical geometric area."
    )

    print("All checks passed.")


if __name__ == "__main__":
    main()
