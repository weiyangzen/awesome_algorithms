"""S-Matrix Theory MVP for quantum scattering.

This script builds partial-wave S-matrix elements from square-well phase shifts,
then evaluates scattering observables and consistency checks:
- partial-wave unitarity (elastic case)
- differential cross section
- total/elastic/reaction cross sections
- optical theorem consistency

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import special


@dataclass(frozen=True)
class SMatrixConfig:
    mass: float = 1.0
    hbar: float = 1.0
    energy: float = 4.0
    well_depth: float = 10.0
    well_radius: float = 1.0
    l_max: int = 8
    sample_angles_deg: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 90.0, 120.0, 150.0, 180.0)
    node_tol: float = 1e-12


def wave_numbers(energy: float, well_depth: float, mass: float, hbar: float) -> tuple[float, float]:
    """Return outside/inside wave numbers (k, q) for a square-well model."""
    if energy <= 0.0:
        raise ValueError("energy must be positive for scattering states.")
    if mass <= 0.0 or hbar <= 0.0:
        raise ValueError("mass and hbar must be positive.")
    if energy + well_depth <= 0.0:
        raise ValueError("energy + well_depth must be positive.")

    k = math.sqrt(2.0 * mass * energy) / hbar
    q = math.sqrt(2.0 * mass * (energy + well_depth)) / hbar
    return float(k), float(q)


def phase_shift_square_well(l: int, k: float, q: float, a: float, node_tol: float = 1e-12) -> float:
    """Compute partial-wave phase shift delta_l by boundary matching at r=a.

    Matching formula:
      tan(delta_l) = [k j'_l(ka) - beta j_l(ka)] / [k y'_l(ka) - beta y_l(ka)]
      beta = q j'_l(qa) / j_l(qa)
    """
    if l < 0:
        raise ValueError("l must be non-negative.")
    if a <= 0.0:
        raise ValueError("well_radius must be positive.")

    ka = k * a
    qa = q * a

    j_ka = special.spherical_jn(l, ka)
    y_ka = special.spherical_yn(l, ka)
    j_ka_p = special.spherical_jn(l, ka, derivative=True)
    y_ka_p = special.spherical_yn(l, ka, derivative=True)

    j_qa = special.spherical_jn(l, qa)
    j_qa_p = special.spherical_jn(l, qa, derivative=True)

    if abs(j_qa) < node_tol:
        raise RuntimeError(
            f"Unstable logarithmic derivative at l={l}: j_l(q*a) is too close to zero."
        )

    beta = q * j_qa_p / j_qa
    num = k * j_ka_p - beta * j_ka
    den = k * y_ka_p - beta * y_ka

    return float(math.atan2(num, den))


def compute_phase_shifts(l_max: int, k: float, q: float, a: float, node_tol: float = 1e-12) -> np.ndarray:
    if l_max < 0:
        raise ValueError("l_max must be non-negative.")
    return np.asarray(
        [phase_shift_square_well(l, k, q, a, node_tol=node_tol) for l in range(l_max + 1)],
        dtype=float,
    )


def build_partial_s_matrix(deltas: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
    """Construct diagonal partial-wave S_l = eta_l * exp(2 i delta_l)."""
    if eta is None:
        eta = np.ones_like(deltas)
    if eta.shape != deltas.shape:
        raise ValueError("eta must have the same shape as deltas.")
    if np.any((eta < 0.0) | (eta > 1.0)):
        raise ValueError("eta values must lie in [0, 1].")
    return eta.astype(float) * np.exp(2j * deltas)


def to_matrix(S_l: np.ndarray) -> np.ndarray:
    """Embed partial-wave amplitudes into a diagonal S matrix."""
    return np.diag(S_l)


def matrix_unitarity_residual(S: np.ndarray) -> float:
    ident = np.eye(S.shape[0], dtype=complex)
    residual = S @ S.conj().T - ident
    return float(np.max(np.abs(residual)))


def scattering_amplitude(theta: float, k: float, S_l: np.ndarray) -> complex:
    """Compute f(theta) from partial-wave S_l.

    f(theta) = (1/(2ik)) * sum_l (2l+1)(S_l - 1) P_l(cos theta)
    """
    mu = math.cos(theta)
    series = 0.0j
    for l, s_l in enumerate(S_l):
        p_l = special.eval_legendre(l, mu)
        series += (2 * l + 1) * (s_l - 1.0) * p_l
    return series / (2j * k)


def differential_cross_section(theta_rad: np.ndarray, k: float, S_l: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for theta in theta_rad:
        amp = scattering_amplitude(float(theta), k, S_l)
        values.append(float(np.abs(amp) ** 2))
    return np.asarray(values, dtype=float)


def sigma_elastic(k: float, S_l: np.ndarray) -> float:
    l_vals = np.arange(S_l.size)
    return float((math.pi / (k**2)) * np.sum((2 * l_vals + 1) * np.abs(S_l - 1.0) ** 2))


def sigma_reaction(k: float, S_l: np.ndarray) -> float:
    l_vals = np.arange(S_l.size)
    reaction = (math.pi / (k**2)) * np.sum((2 * l_vals + 1) * (1.0 - np.abs(S_l) ** 2))
    return float(np.clip(reaction, 0.0, None))


def sigma_total_from_reS(k: float, S_l: np.ndarray) -> float:
    l_vals = np.arange(S_l.size)
    return float((2.0 * math.pi / (k**2)) * np.sum((2 * l_vals + 1) * (1.0 - np.real(S_l))))


def sigma_total_optical(k: float, S_l: np.ndarray) -> float:
    f0 = scattering_amplitude(0.0, k, S_l)
    return float((4.0 * math.pi / k) * f0.imag)


def make_phase_table(deltas: np.ndarray, eta: np.ndarray, S_l: np.ndarray) -> pd.DataFrame:
    l_vals = np.arange(deltas.size)
    return pd.DataFrame(
        {
            "l": l_vals,
            "delta_rad": deltas,
            "delta_deg": np.rad2deg(deltas),
            "eta": eta,
            "|S_l|": np.abs(S_l),
            "arg(S_l)_deg": np.rad2deg(np.angle(S_l)),
            "| |S_l|-1 |": np.abs(np.abs(S_l) - 1.0),
        }
    )


def run_case(case_name: str, k: float, deltas: np.ndarray, eta: np.ndarray, angles_deg: np.ndarray) -> None:
    S_l = build_partial_s_matrix(deltas, eta=eta)
    S = to_matrix(S_l)

    sigma_el = sigma_elastic(k, S_l)
    sigma_re = sigma_reaction(k, S_l)
    sigma_tot_from_parts = sigma_el + sigma_re
    sigma_tot_from_res = sigma_total_from_reS(k, S_l)
    sigma_tot_opt = sigma_total_optical(k, S_l)

    theta_rad = np.deg2rad(angles_deg)
    dsdo = differential_cross_section(theta_rad, k, S_l)

    unitarity_matrix_res = matrix_unitarity_residual(S)
    optical_res = abs(sigma_tot_opt - sigma_tot_from_parts)
    sumrule_res = abs(sigma_tot_from_res - sigma_tot_from_parts)

    phase_table = make_phase_table(deltas=deltas, eta=eta, S_l=S_l)
    angle_table = pd.DataFrame({"theta_deg": angles_deg, "d_sigma_d_omega": dsdo})

    print("=" * 92)
    print(f"Case: {case_name}")
    print("-" * 92)
    print(phase_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("-" * 92)
    print(
        "cross sections: "
        f"sigma_el={sigma_el:.8f}, sigma_re={sigma_re:.8f}, "
        f"sigma_tot(parts)={sigma_tot_from_parts:.8f}, "
        f"sigma_tot(ReS)={sigma_tot_from_res:.8f}, sigma_tot(optical)={sigma_tot_opt:.8f}"
    )
    print(
        "consistency: "
        f"matrix_unitarity_res={unitarity_matrix_res:.3e}, "
        f"optical_res={optical_res:.3e}, "
        f"sumrule_res={sumrule_res:.3e}"
    )
    print("-" * 92)
    print(angle_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))


def main() -> None:
    cfg = SMatrixConfig()

    k, q = wave_numbers(
        energy=cfg.energy,
        well_depth=cfg.well_depth,
        mass=cfg.mass,
        hbar=cfg.hbar,
    )
    deltas = compute_phase_shifts(
        l_max=cfg.l_max,
        k=k,
        q=q,
        a=cfg.well_radius,
        node_tol=cfg.node_tol,
    )

    angles_deg = np.asarray(cfg.sample_angles_deg, dtype=float)

    print("S-Matrix Theory MVP (partial-wave representation)")
    print(
        f"Parameters: E={cfg.energy:.3f}, V0={cfg.well_depth:.3f}, a={cfg.well_radius:.3f}, "
        f"l_max={cfg.l_max}, m={cfg.mass:.3f}, hbar={cfg.hbar:.3f}, k={k:.6f}, q={q:.6f}"
    )

    # Case A: purely elastic scattering (eta_l = 1), unitary S matrix.
    eta_elastic = np.ones_like(deltas)
    run_case("elastic (eta_l = 1)", k=k, deltas=deltas, eta=eta_elastic, angles_deg=angles_deg)

    # Case B: illustrative inelasticity in low-l channels.
    eta_inelastic = np.ones_like(deltas)
    if eta_inelastic.size >= 1:
        eta_inelastic[0] = 0.86
    if eta_inelastic.size >= 2:
        eta_inelastic[1] = 0.93
    run_case("inelastic toy example (eta_0=0.86, eta_1=0.93)", k=k, deltas=deltas, eta=eta_inelastic, angles_deg=angles_deg)


if __name__ == "__main__":
    main()
