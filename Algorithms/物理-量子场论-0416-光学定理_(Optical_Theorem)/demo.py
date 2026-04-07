"""Minimal runnable MVP for the optical theorem (partial-wave form)."""

from __future__ import annotations

import numpy as np


def build_s_matrix(delta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Build partial-wave S-matrix elements S_l = eta_l * exp(2 i delta_l)."""
    eta_clipped = np.clip(eta, 0.0, 1.0)
    return eta_clipped * np.exp(2j * delta)


def forward_amplitude(s_l: np.ndarray, k: float) -> complex:
    """Forward scattering amplitude f(0) from partial waves."""
    l = np.arange(s_l.size)
    return np.sum((2 * l + 1) * (s_l - 1.0)) / (2j * k)


def cross_sections_from_partial_waves(s_l: np.ndarray, k: float) -> tuple[float, float, float]:
    """Return (sigma_tot, sigma_el, sigma_reac) from partial-wave sums."""
    l = np.arange(s_l.size)
    degeneracy = 2 * l + 1

    sigma_tot = (2.0 * np.pi / (k * k)) * np.sum(degeneracy * (1.0 - np.real(s_l)))
    sigma_el = (np.pi / (k * k)) * np.sum(degeneracy * np.abs(1.0 - s_l) ** 2)
    sigma_reac = (np.pi / (k * k)) * np.sum(degeneracy * (1.0 - np.abs(s_l) ** 2))
    return float(sigma_tot), float(sigma_el), float(sigma_reac)


def legendre_table(mu: np.ndarray, l_max: int) -> np.ndarray:
    """Compute P_l(mu) table for l=0..l_max using three-term recurrence."""
    p = np.zeros((mu.size, l_max + 1), dtype=np.float64)
    p[:, 0] = 1.0
    if l_max >= 1:
        p[:, 1] = mu
    for l in range(1, l_max):
        p[:, l + 1] = ((2 * l + 1) * mu * p[:, l] - l * p[:, l - 1]) / (l + 1)
    return p


def scattering_amplitude(theta: np.ndarray, s_l: np.ndarray, k: float) -> np.ndarray:
    """Compute f(theta) from partial-wave expansion."""
    mu = np.cos(theta)
    l_max = s_l.size - 1
    p = legendre_table(mu, l_max)
    l = np.arange(l_max + 1)
    weights = (2 * l + 1) * (s_l - 1.0)
    return (p @ weights) / (2j * k)


def elastic_cross_section_from_angles(theta: np.ndarray, f_theta: np.ndarray) -> float:
    """Numerically integrate sigma_el = 2pi * integral(|f(theta)|^2 sin(theta) dtheta)."""
    integrand = np.abs(f_theta) ** 2 * np.sin(theta)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(integrand, theta)
    else:
        area = np.trapz(integrand, theta)
    return float(2.0 * np.pi * area)


def generate_profile(l_max: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth synthetic (delta_l, eta_l) profiles for demo cases."""
    l = np.arange(l_max + 1, dtype=np.float64)

    # Smooth phase profile; decays with l but keeps nontrivial structure.
    delta = 0.80 * np.exp(-l / 4.0) + 0.10 * np.cos(0.9 * l) / (1.0 + l)

    if mode == "elastic_only":
        eta = np.ones_like(delta)
    elif mode == "inelastic":
        # Add moderate absorption centered around mid-l partial waves.
        eta = 1.0 - 0.35 * np.exp(-((l - 3.0) / 2.2) ** 2)
        eta = np.clip(eta, 0.55, 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return delta, eta


def run_case(name: str, *, k: float, l_max: int, n_theta: int = 2001) -> dict[str, float | complex]:
    """Run one self-contained optical-theorem consistency check."""
    delta, eta = generate_profile(l_max, name)
    s_l = build_s_matrix(delta, eta)

    f0 = forward_amplitude(s_l, k)
    sigma_tot_optical = (4.0 * np.pi / k) * np.imag(f0)

    sigma_tot_pw, sigma_el_pw, sigma_reac_pw = cross_sections_from_partial_waves(s_l, k)

    theta = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
    f_theta = scattering_amplitude(theta, s_l, k)
    sigma_el_num = elastic_cross_section_from_angles(theta, f_theta)

    return {
        "k": k,
        "l_max": float(l_max),
        "f0": f0,
        "sigma_tot_optical": sigma_tot_optical,
        "sigma_tot_pw": sigma_tot_pw,
        "sigma_el_pw": sigma_el_pw,
        "sigma_reac_pw": sigma_reac_pw,
        "sigma_el_num": sigma_el_num,
        "err_optical": abs(sigma_tot_optical - sigma_tot_pw),
        "err_balance": abs(sigma_tot_pw - (sigma_el_pw + sigma_reac_pw)),
        "err_elastic_num": abs(sigma_el_pw - sigma_el_num),
    }


def fmt_complex(z: complex) -> str:
    return f"{z.real:+.10f}{z.imag:+.10f}j"


def print_case(name: str, result: dict[str, float | complex]) -> None:
    print(f"\n=== Case: {name} ===")
    print(f"k                   = {result['k']:.6f}")
    print(f"l_max               = {int(result['l_max'])}")
    print(f"f(0)                = {fmt_complex(result['f0'])}")
    print(f"sigma_tot(optical)  = {result['sigma_tot_optical']:.12f}")
    print(f"sigma_tot(partial)  = {result['sigma_tot_pw']:.12f}")
    print(f"sigma_el(partial)   = {result['sigma_el_pw']:.12f}")
    print(f"sigma_reac(partial) = {result['sigma_reac_pw']:.12f}")
    print(f"sigma_el(numeric)   = {result['sigma_el_num']:.12f}")
    print(f"|optical gap|       = {result['err_optical']:.3e}")
    print(f"|balance gap|       = {result['err_balance']:.3e}")
    print(f"|elastic int gap|   = {result['err_elastic_num']:.3e}")


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # Two representative scenarios: purely elastic and with absorption.
    common = {"k": 1.40, "l_max": 10, "n_theta": 2401}
    cases = ["elastic_only", "inelastic"]

    print("Optical theorem MVP via partial-wave S-matrix")
    print("Formula checked: sigma_tot = (4*pi/k) * Im f(0)")

    for name in cases:
        result = run_case(name, **common)
        print_case(name, result)


if __name__ == "__main__":
    main()
