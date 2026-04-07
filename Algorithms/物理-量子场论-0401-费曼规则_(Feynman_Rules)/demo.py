"""Minimal runnable MVP for Feynman Rules in a toy scalar QFT.

This script explicitly maps interaction terms to Feynman-rule factors and uses
those factors to build tree-level 2->2 scattering amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class QFTConfig:
    """Numerical configuration for the toy phi^3 + phi^4 model."""

    mass_external: float = 1.0
    mass_mediator: float = 1.45
    width_mediator: float = 0.08
    epsilon: float = 1e-9
    g3: float = 0.70
    lambda4: float = 0.85
    e_cm: float = 3.20
    n_theta: int = 41


@dataclass(frozen=True)
class InteractionTerm:
    """Interaction term metadata used to derive vertex factors."""

    name: str
    n_fields: int
    coupling: float


@dataclass(frozen=True)
class Kinematics2to2:
    """COM-frame Mandelstam variables for equal-mass 2->2 scattering."""

    s: float
    t: float
    u: float
    theta_deg: float


def build_interactions(cfg: QFTConfig) -> tuple[InteractionTerm, InteractionTerm]:
    """Return phi^3 and phi^4 interaction terms for the toy model."""
    term_phi3 = InteractionTerm(name="phi^3", n_fields=3, coupling=cfg.g3)
    term_phi4 = InteractionTerm(name="phi^4", n_fields=4, coupling=cfg.lambda4)
    return term_phi3, term_phi4


def vertex_factor(term: InteractionTerm) -> complex:
    """Translate an interaction term to its momentum-space vertex factor.

    For L_int containing -g/n! * phi^n, the vertex factor is -i g.
    """
    if term.n_fields < 3:
        raise ValueError("This MVP expects interaction terms with n_fields >= 3.")
    return complex(0.0, -term.coupling)


def propagator_factor(q2: float, mass: float, width: float, epsilon: float) -> complex:
    """Scalar propagator with finite width regularization.

    Delta(q^2) = i / (q^2 - m^2 + i*m*Gamma + i*eps)
    """
    denom = q2 - mass**2 + 1j * (mass * width + epsilon)
    return 1j / denom


def mandelstam_2to2_equal_mass(e_cm: float, theta_deg: float, mass: float) -> Kinematics2to2:
    """Compute (s, t, u) for equal-mass 2->2 scattering in COM frame."""
    if e_cm <= 2.0 * mass:
        raise ValueError("e_cm must satisfy e_cm > 2 * mass.")

    s = e_cm**2
    p_sq = s / 4.0 - mass**2
    if p_sq <= 0.0:
        raise ValueError("Computed COM momentum^2 is non-positive.")

    cos_theta = np.cos(np.deg2rad(theta_deg))
    t = -2.0 * p_sq * (1.0 - cos_theta)
    u = -2.0 * p_sq * (1.0 + cos_theta)
    return Kinematics2to2(s=s, t=t, u=u, theta_deg=theta_deg)


def diagram_amplitudes(kin: Kinematics2to2, cfg: QFTConfig) -> dict[str, complex]:
    """Compute diagram-level amplitudes from explicit Feynman rules."""
    term_phi3, term_phi4 = build_interactions(cfg)
    v3 = vertex_factor(term_phi3)
    v4 = vertex_factor(term_phi4)

    m_contact = v4
    m_s = v3 * propagator_factor(kin.s, cfg.mass_mediator, cfg.width_mediator, cfg.epsilon) * v3
    m_t = v3 * propagator_factor(kin.t, cfg.mass_mediator, cfg.width_mediator, cfg.epsilon) * v3
    m_u = v3 * propagator_factor(kin.u, cfg.mass_mediator, cfg.width_mediator, cfg.epsilon) * v3

    m_total = m_contact + m_s + m_t + m_u
    return {
        "contact": m_contact,
        "s_channel": m_s,
        "t_channel": m_t,
        "u_channel": m_u,
        "total": m_total,
    }


def differential_cross_section(amplitude: complex, s: float, mass: float) -> float:
    """Compute dσ/dΩ = |M|^2 / (64 π^2 s) for equal-mass 2->2 scattering."""
    if s <= 4.0 * mass**2:
        raise ValueError("s must satisfy s > 4*m^2.")
    return float((abs(amplitude) ** 2) / (64.0 * np.pi**2 * s))


def build_rule_table(cfg: QFTConfig) -> pd.DataFrame:
    """Build a human-readable table of the Feynman rules used in the demo."""
    term_phi3, term_phi4 = build_interactions(cfg)
    v3 = vertex_factor(term_phi3)
    v4 = vertex_factor(term_phi4)

    rows = [
        {
            "object": "phi propagator",
            "symbolic_rule": "i/(q^2-m_med^2+i m_med Gamma+i eps)",
            "sample_q2": float(cfg.e_cm**2),
            "sample_value_real": float(np.real(propagator_factor(cfg.e_cm**2, cfg.mass_mediator, cfg.width_mediator, cfg.epsilon))),
            "sample_value_imag": float(np.imag(propagator_factor(cfg.e_cm**2, cfg.mass_mediator, cfg.width_mediator, cfg.epsilon))),
        },
        {
            "object": "phi^3 vertex",
            "symbolic_rule": "-i g3",
            "sample_q2": np.nan,
            "sample_value_real": float(np.real(v3)),
            "sample_value_imag": float(np.imag(v3)),
        },
        {
            "object": "phi^4 vertex",
            "symbolic_rule": "-i lambda4",
            "sample_q2": np.nan,
            "sample_value_real": float(np.real(v4)),
            "sample_value_imag": float(np.imag(v4)),
        },
    ]
    return pd.DataFrame(rows)


def build_scattering_table(cfg: QFTConfig) -> pd.DataFrame:
    """Compute angle-resolved amplitudes and differential cross section."""
    theta_grid = np.linspace(0.0, 180.0, cfg.n_theta)
    rows: list[dict[str, float]] = []

    for theta in theta_grid:
        kin = mandelstam_2to2_equal_mass(cfg.e_cm, float(theta), cfg.mass_external)
        amps = diagram_amplitudes(kin, cfg)
        dsdo = differential_cross_section(amps["total"], kin.s, cfg.mass_external)

        rows.append(
            {
                "theta_deg": float(theta),
                "s": kin.s,
                "t": kin.t,
                "u": kin.u,
                "M_contact_real": float(np.real(amps["contact"])),
                "M_contact_imag": float(np.imag(amps["contact"])),
                "M_s_real": float(np.real(amps["s_channel"])),
                "M_s_imag": float(np.imag(amps["s_channel"])),
                "M_t_real": float(np.real(amps["t_channel"])),
                "M_t_imag": float(np.imag(amps["t_channel"])),
                "M_u_real": float(np.real(amps["u_channel"])),
                "M_u_imag": float(np.imag(amps["u_channel"])),
                "M_total_real": float(np.real(amps["total"])),
                "M_total_imag": float(np.imag(amps["total"])),
                "M_abs": float(abs(amps["total"])),
                "dsigma_domega": dsdo,
            }
        )

    return pd.DataFrame(rows)


def integrate_total_cross_section(theta_deg: np.ndarray, dsigma_domega: np.ndarray) -> float:
    """Integrate sigma_total = 2π ∫ sin(theta) (dσ/dΩ) dtheta."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    dsigma_domega = np.asarray(dsigma_domega, dtype=float)

    theta_rad = np.deg2rad(theta_deg)
    integrand = np.sin(theta_rad) * dsigma_domega
    return float(2.0 * np.pi * simpson(integrand, x=theta_rad))


def verify_vertex_rules(cfg: QFTConfig) -> tuple[float, float]:
    """Return absolute deviations from expected toy-model vertex rules."""
    term_phi3, term_phi4 = build_interactions(cfg)
    v3 = vertex_factor(term_phi3)
    v4 = vertex_factor(term_phi4)

    err_v3 = abs(v3 - complex(0.0, -cfg.g3))
    err_v4 = abs(v4 - complex(0.0, -cfg.lambda4))
    return float(err_v3), float(err_v4)


def verify_crossing_symmetry(cfg: QFTConfig, angles_deg: list[float]) -> float:
    """Check M(theta) = M(180-theta) for identical-scalar tree-level scattering."""
    max_dev = 0.0
    for theta in angles_deg:
        kin_a = mandelstam_2to2_equal_mass(cfg.e_cm, theta, cfg.mass_external)
        kin_b = mandelstam_2to2_equal_mass(cfg.e_cm, 180.0 - theta, cfg.mass_external)
        m_a = diagram_amplitudes(kin_a, cfg)["total"]
        m_b = diagram_amplitudes(kin_b, cfg)["total"]
        max_dev = max(max_dev, float(abs(m_a - m_b)))
    return max_dev


def contact_only_check(cfg: QFTConfig) -> tuple[float, float, float]:
    """Compare numeric sigma to analytic sigma in the g3=0 contact-only limit."""
    reduced_cfg = replace(cfg, g3=0.0)
    table = build_scattering_table(reduced_cfg)
    sigma_num = integrate_total_cross_section(
        theta_deg=table["theta_deg"].to_numpy(),
        dsigma_domega=table["dsigma_domega"].to_numpy(),
    )

    s = reduced_cfg.e_cm**2
    sigma_analytic = (reduced_cfg.lambda4**2) / (16.0 * np.pi * s)
    rel_err = abs(sigma_num - sigma_analytic) / max(abs(sigma_analytic), 1e-14)
    return sigma_num, sigma_analytic, float(rel_err)


def main() -> None:
    cfg = QFTConfig()

    rule_table = build_rule_table(cfg)
    scattering_table = build_scattering_table(cfg)
    sigma_total = integrate_total_cross_section(
        theta_deg=scattering_table["theta_deg"].to_numpy(),
        dsigma_domega=scattering_table["dsigma_domega"].to_numpy(),
    )

    err_v3, err_v4 = verify_vertex_rules(cfg)
    crossing_max_dev = verify_crossing_symmetry(cfg, angles_deg=[12.0, 33.0, 67.0, 111.0, 149.0])
    sigma_contact_num, sigma_contact_analytic, contact_rel_err = contact_only_check(cfg)

    print("=== Toy Model Config ===")
    print(
        f"m={cfg.mass_external:.3f}, m_med={cfg.mass_mediator:.3f}, "
        f"Gamma={cfg.width_mediator:.3f}, eps={cfg.epsilon:.1e}, "
        f"g3={cfg.g3:.3f}, lambda4={cfg.lambda4:.3f}, "
        f"E_cm={cfg.e_cm:.3f}, n_theta={cfg.n_theta}"
    )
    print()

    print("=== Feynman Rule Table ===")
    print(rule_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Angle-Resolved Scattering Table ===")
    print(scattering_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Summary ===")
    print(f"sigma_total (full tree) = {sigma_total:.10f}")
    print(f"vertex rule error |V3 + i g3| = {err_v3:.3e}")
    print(f"vertex rule error |V4 + i lambda4| = {err_v4:.3e}")
    print(f"crossing max |M(theta)-M(180-theta)| = {crossing_max_dev:.3e}")
    print(f"contact-only sigma (numeric)  = {sigma_contact_num:.10f}")
    print(f"contact-only sigma (analytic) = {sigma_contact_analytic:.10f}")
    print(f"contact-only relative error   = {contact_rel_err:.3e}")

    # Deterministic validation checks for this MVP.
    assert np.all(scattering_table["dsigma_domega"].to_numpy() >= 0.0), "dσ/dΩ must be non-negative."
    assert err_v3 < 1e-14 and err_v4 < 1e-14, "Vertex rule translation mismatch."
    assert crossing_max_dev < 1e-10, f"Crossing symmetry check failed: {crossing_max_dev:.3e}"
    assert contact_rel_err < 1e-3, f"Contact-limit consistency failed: {contact_rel_err:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
