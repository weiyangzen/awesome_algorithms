"""Tree-level 2->2 scattering MVP for a toy scalar QFT.

The script demonstrates a transparent tree-level workflow:
1) build Mandelstam kinematics,
2) compute contact + s/t/u exchange diagram contributions,
3) map amplitude to differential cross section,
4) integrate over solid angle to get total cross section,
5) verify crossing symmetry and a contact-only analytic limit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class TreeLevelConfig:
    """Configuration for toy tree-level scalar scattering."""

    mass_external: float = 1.0
    mass_mediator: float = 1.6
    width_mediator: float = 0.12
    lambda4: float = 0.90
    g3: float = 0.75
    e_cm: float = 3.40
    n_theta: int = 41


@dataclass(frozen=True)
class Kinematics2to2:
    """COM-frame Mandelstam variables for equal-mass 2->2 scattering."""

    s: float
    t: float
    u: float
    theta_deg: float


def mandelstam_2to2_equal_mass(e_cm: float, theta_deg: float, mass: float) -> Kinematics2to2:
    """Compute (s, t, u) for equal-mass 2->2 kinematics in COM frame."""
    if e_cm <= 2.0 * mass:
        raise ValueError("e_cm must satisfy e_cm > 2*mass.")

    s = e_cm**2
    p_sq = s / 4.0 - mass**2
    if p_sq <= 0.0:
        raise ValueError("Computed COM momentum^2 is non-positive.")

    cos_theta = np.cos(np.deg2rad(theta_deg))
    t = -2.0 * p_sq * (1.0 - cos_theta)
    u = -2.0 * p_sq * (1.0 + cos_theta)

    return Kinematics2to2(s=s, t=t, u=u, theta_deg=theta_deg)


def scalar_propagator(q2: float, mass: float, width: float) -> complex:
    """Breit-Wigner-like scalar propagator denominator convention."""
    return 1.0 / (q2 - mass**2 + 1j * mass * width)


def tree_level_components(kin: Kinematics2to2, cfg: TreeLevelConfig) -> dict[str, complex]:
    """Compute diagram-wise tree-level contributions for toy scalar theory.

    Reduced amplitude convention:
      A = A_contact + A_s + A_t + A_u
      A_contact = -lambda4
      A_x = g3^2 / (x - m_med^2 + i m_med Gamma)
    """
    contact = complex(-cfg.lambda4, 0.0)
    s_channel = cfg.g3**2 * scalar_propagator(kin.s, cfg.mass_mediator, cfg.width_mediator)
    t_channel = cfg.g3**2 * scalar_propagator(kin.t, cfg.mass_mediator, cfg.width_mediator)
    u_channel = cfg.g3**2 * scalar_propagator(kin.u, cfg.mass_mediator, cfg.width_mediator)
    total = contact + s_channel + t_channel + u_channel

    return {
        "contact": contact,
        "s_channel": s_channel,
        "t_channel": t_channel,
        "u_channel": u_channel,
        "total": total,
    }


def differential_cross_section(amplitude: complex, s: float, mass: float) -> float:
    """Compute dσ/dΩ for equal-mass scalar 2->2 scattering.

    For equal masses and COM kinematics, p_f / p_i = 1:
      dσ/dΩ = |A|^2 / (64 π^2 s)
    """
    if s <= 4.0 * mass**2:
        raise ValueError("s must be above threshold 4m^2.")
    return float((abs(amplitude) ** 2) / (64.0 * np.pi**2 * s))


def build_scattering_table(cfg: TreeLevelConfig) -> pd.DataFrame:
    """Sweep angles and return a diagram-decomposed scattering table."""
    theta_grid = np.linspace(0.0, 180.0, cfg.n_theta)
    rows: list[dict[str, float]] = []

    for theta in theta_grid:
        kin = mandelstam_2to2_equal_mass(e_cm=cfg.e_cm, theta_deg=float(theta), mass=cfg.mass_external)
        parts = tree_level_components(kin=kin, cfg=cfg)
        dsdo = differential_cross_section(parts["total"], s=kin.s, mass=cfg.mass_external)

        rows.append(
            {
                "theta_deg": float(theta),
                "s": kin.s,
                "t": kin.t,
                "u": kin.u,
                "contact_real": float(np.real(parts["contact"])),
                "s_real": float(np.real(parts["s_channel"])),
                "s_imag": float(np.imag(parts["s_channel"])),
                "t_real": float(np.real(parts["t_channel"])),
                "t_imag": float(np.imag(parts["t_channel"])),
                "u_real": float(np.real(parts["u_channel"])),
                "u_imag": float(np.imag(parts["u_channel"])),
                "amp_total_real": float(np.real(parts["total"])),
                "amp_total_imag": float(np.imag(parts["total"])),
                "amp_abs": float(abs(parts["total"])),
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


def verify_crossing_symmetry(cfg: TreeLevelConfig, angles_deg: list[float]) -> float:
    """Check A(theta) == A(180-theta) for identical-scalar t/u crossing at tree level."""
    max_dev = 0.0
    for theta in angles_deg:
        kin_a = mandelstam_2to2_equal_mass(cfg.e_cm, theta, cfg.mass_external)
        kin_b = mandelstam_2to2_equal_mass(cfg.e_cm, 180.0 - theta, cfg.mass_external)
        amp_a = tree_level_components(kin_a, cfg)["total"]
        amp_b = tree_level_components(kin_b, cfg)["total"]
        max_dev = max(max_dev, float(abs(amp_a - amp_b)))
    return max_dev


def contact_only_consistency_check(cfg: TreeLevelConfig) -> tuple[float, float, float]:
    """Compare numerical and analytic sigma for the g3=0 isotropic contact limit."""
    contact_cfg = replace(cfg, g3=0.0)
    table = build_scattering_table(contact_cfg)

    sigma_num = integrate_total_cross_section(
        theta_deg=table["theta_deg"].to_numpy(),
        dsigma_domega=table["dsigma_domega"].to_numpy(),
    )
    s = contact_cfg.e_cm**2
    sigma_analytic = (contact_cfg.lambda4**2) / (16.0 * np.pi * s)
    rel_err = abs(sigma_num - sigma_analytic) / max(abs(sigma_analytic), 1e-14)
    return sigma_num, sigma_analytic, rel_err


def main() -> None:
    cfg = TreeLevelConfig()

    table = build_scattering_table(cfg)
    sigma_total = integrate_total_cross_section(
        theta_deg=table["theta_deg"].to_numpy(),
        dsigma_domega=table["dsigma_domega"].to_numpy(),
    )

    crossing_max_dev = verify_crossing_symmetry(cfg, angles_deg=[10.0, 35.0, 70.0, 120.0, 155.0])
    sigma_contact_num, sigma_contact_analytic, contact_rel_err = contact_only_consistency_check(cfg)

    print("=== Tree-Level Toy Config ===")
    print(
        f"m={cfg.mass_external:.3f}, m_med={cfg.mass_mediator:.3f}, Gamma_med={cfg.width_mediator:.3f}, "
        f"lambda4={cfg.lambda4:.3f}, g3={cfg.g3:.3f}, E_cm={cfg.e_cm:.3f}, n_theta={cfg.n_theta}"
    )
    print()

    print("=== Diagram-Decomposed Scattering Table ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Summary ===")
    print(f"sigma_total (full tree) = {sigma_total:.10f}")
    print(f"crossing max |A(theta)-A(180-theta)| = {crossing_max_dev:.3e}")
    print(f"contact-only sigma (numeric)  = {sigma_contact_num:.10f}")
    print(f"contact-only sigma (analytic) = {sigma_contact_analytic:.10f}")
    print(f"contact-only relative error   = {contact_rel_err:.3e}")

    # Sanity checks for deterministic MVP behavior.
    assert np.all(table["dsigma_domega"].to_numpy() >= 0.0), "dσ/dΩ must be non-negative."
    assert crossing_max_dev < 1e-10, f"Crossing symmetry check failed: {crossing_max_dev:.3e}"
    assert contact_rel_err < 1e-3, f"Contact-limit consistency check failed: {contact_rel_err:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
