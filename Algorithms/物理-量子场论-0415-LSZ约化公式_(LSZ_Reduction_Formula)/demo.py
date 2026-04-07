"""LSZ reduction formula MVP (toy scalar QFT).

This script builds a transparent numerical toy model for LSZ reduction:
1) estimate the field-strength residue Z from a synthetic 2-point function,
2) build a synthetic connected 4-point Green function with external propagators,
3) amputate external legs and divide by Z factors to recover scattering amplitude.

The purpose is educational and algorithmic, not precision phenomenology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LSZToyConfig:
    """Configuration of a toy scalar QFT setup used for LSZ demonstration."""

    mass: float = 1.0
    epsilon: float = 1e-3
    z_true: float = 0.82
    lambda_contact: float = 0.70
    resonance_g: float = 0.55
    resonance_mass: float = 2.30
    resonance_width: float = 0.18
    regular_c0: float = 0.08
    regular_c1: float = -0.06


@dataclass(frozen=True)
class ScatteringKinematics:
    """Basic 2->2 scalar kinematics (Mandelstam variables)."""

    s: float
    t: float
    u: float
    theta_deg: float


def mandelstam_2to2_equal_mass(e_cm: float, theta_deg: float, mass: float) -> ScatteringKinematics:
    """Construct on-shell Mandelstam variables for identical masses.

    For COM frame with total energy sqrt(s) = e_cm:
      p^2 = s/4 - m^2
      t = -2 p^2 (1-cos theta)
      u = -2 p^2 (1+cos theta)
    and s + t + u = 4 m^2.
    """
    if e_cm <= 2.0 * mass:
        raise ValueError("e_cm must be larger than 2*mass for physical 2->2 scattering.")

    s = e_cm**2
    p_sq = s / 4.0 - mass**2
    if p_sq <= 0.0:
        raise ValueError("Computed momentum^2 must be positive.")

    cos_theta = np.cos(np.deg2rad(theta_deg))
    t = -2.0 * p_sq * (1.0 - cos_theta)
    u = -2.0 * p_sq * (1.0 + cos_theta)
    return ScatteringKinematics(s=s, t=t, u=u, theta_deg=theta_deg)


def amputated_amplitude(kin: ScatteringKinematics, cfg: LSZToyConfig) -> complex:
    """Toy amputated 2->2 amplitude M(s,t,u).

    We include:
    - a contact term (-lambda),
    - an s-channel Breit-Wigner-like term.
    """
    s = kin.s
    denominator = s - cfg.resonance_mass**2 + 1j * cfg.resonance_mass * cfg.resonance_width
    return -cfg.lambda_contact + (cfg.resonance_g**2 / denominator)


def two_point_connected(p2: np.ndarray, cfg: LSZToyConfig) -> np.ndarray:
    """Synthetic connected 2-point function near a one-particle pole.

    G2(p^2) = Z / (p^2 - m^2 + i eps) + regular(p^2)
    regular(p^2) = c0 + c1 * (p^2 - m^2)
    """
    p2 = np.asarray(p2, dtype=float)
    delta = p2 - cfg.mass**2
    pole = cfg.z_true / (delta + 1j * cfg.epsilon)
    regular = cfg.regular_c0 + cfg.regular_c1 * delta
    return pole + regular


def estimate_field_strength_z(p2_samples: np.ndarray, g2_samples: np.ndarray, cfg: LSZToyConfig) -> complex:
    """Estimate pole residue Z by linear extrapolation of (delta+i eps) * G2.

    Define y(delta) = (delta + i eps) G2(delta).
    Near shell, y(delta) = Z + O(delta), so intercept at delta=0 estimates Z.
    """
    p2_samples = np.asarray(p2_samples, dtype=float)
    g2_samples = np.asarray(g2_samples, dtype=complex)

    if p2_samples.ndim != 1 or g2_samples.ndim != 1:
        raise ValueError("Samples must be 1D arrays.")
    if p2_samples.size != g2_samples.size or p2_samples.size < 4:
        raise ValueError("Need >=4 matched 2-point samples.")

    delta = p2_samples - cfg.mass**2
    y = (delta + 1j * cfg.epsilon) * g2_samples

    # Fit y ≈ a0 + a1 * delta (complex least squares); a0 is residue estimate.
    design = np.column_stack([np.ones_like(delta), delta])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    z_est = coeff[0]
    return complex(z_est)


def connected_four_point(
    p2_external: np.ndarray,
    kin: ScatteringKinematics,
    cfg: LSZToyConfig,
) -> complex:
    """Synthetic connected 4-point Green function in momentum space.

    We encode the external legs explicitly:
      G4 = Z^2 * M(s,t,u) / prod_i (p_i^2 - m^2 + i eps)

    Then LSZ extraction is:
      M = [prod_i (p_i^2 - m^2 + i eps)] * G4 / Z^2
    """
    p2_external = np.asarray(p2_external, dtype=float)
    if p2_external.shape != (4,):
        raise ValueError("p2_external must have shape (4,).")

    deltas = p2_external - cfg.mass**2
    denominator = np.prod(deltas + 1j * cfg.epsilon)
    if denominator == 0:
        raise ValueError("External denominator became zero; adjust off-shell points.")

    amp = amputated_amplitude(kin=kin, cfg=cfg)
    return (cfg.z_true**2) * amp / denominator


def lsz_reduce_amplitude(g4: complex, p2_external: np.ndarray, z_est: complex, cfg: LSZToyConfig) -> complex:
    """Apply LSZ amputation formula to recover scattering amplitude."""
    p2_external = np.asarray(p2_external, dtype=float)
    deltas = p2_external - cfg.mass**2
    amputator = np.prod(deltas + 1j * cfg.epsilon)
    if z_est == 0:
        raise ValueError("z_est must be non-zero for LSZ reduction.")
    return amputator * g4 / (z_est**2)


def run_lsz_demo(cfg: LSZToyConfig) -> tuple[pd.DataFrame, complex, np.ndarray]:
    """Run a deterministic LSZ reduction experiment and return report data."""
    rng = np.random.default_rng(42)

    # 1) Estimate Z from two-point data near pole.
    delta_grid = np.array([-0.060, -0.040, -0.025, -0.012, 0.012, 0.025, 0.040, 0.060])
    p2_samples = cfg.mass**2 + delta_grid
    g2_samples = two_point_connected(p2_samples, cfg)
    z_est = estimate_field_strength_z(p2_samples, g2_samples, cfg)

    # 2) Build several 2->2 kinematics points and off-shell external legs.
    e_cm = 3.2 * cfg.mass
    thetas = [25.0, 55.0, 95.0, 140.0]

    rows: list[dict[str, object]] = []
    offshell_spread = []

    for case_id, theta in enumerate(thetas, start=1):
        kin = mandelstam_2to2_equal_mass(e_cm=e_cm, theta_deg=theta, mass=cfg.mass)
        for sample_id in range(3):
            # Keep each leg close to shell so LSZ limit is well represented.
            delta_ext = rng.uniform(-0.020, 0.020, size=4)
            p2_external = cfg.mass**2 + delta_ext

            g4 = connected_four_point(p2_external=p2_external, kin=kin, cfg=cfg)
            m_true = amputated_amplitude(kin=kin, cfg=cfg)
            m_lsz = lsz_reduce_amplitude(g4=g4, p2_external=p2_external, z_est=z_est, cfg=cfg)

            rel_err = abs(m_lsz - m_true) / max(abs(m_true), 1e-14)
            offshell_spread.append(float(np.max(np.abs(delta_ext))))

            rows.append(
                {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "theta_deg": theta,
                    "s": kin.s,
                    "t": kin.t,
                    "u": kin.u,
                    "max_offshell_abs": float(np.max(np.abs(delta_ext))),
                    "amp_true_real": float(np.real(m_true)),
                    "amp_true_imag": float(np.imag(m_true)),
                    "amp_lsz_real": float(np.real(m_lsz)),
                    "amp_lsz_imag": float(np.imag(m_lsz)),
                    "rel_error": float(rel_err),
                }
            )

    df = pd.DataFrame(rows)
    return df, z_est, np.array(offshell_spread, dtype=float)


def main() -> None:
    cfg = LSZToyConfig()

    df, z_est, offshell_spread = run_lsz_demo(cfg)

    z_error = abs(z_est - cfg.z_true)
    max_rel_error = float(df["rel_error"].max())
    mean_rel_error = float(df["rel_error"].mean())

    print("=== LSZ Toy Configuration ===")
    print(
        f"mass={cfg.mass:.3f}, epsilon={cfg.epsilon:.1e}, Z_true={cfg.z_true:.6f}, "
        f"lambda={cfg.lambda_contact:.3f}, g={cfg.resonance_g:.3f}"
    )
    print(f"Estimated residue Z_est={z_est.real:.6f}{z_est.imag:+.6f}j")
    print(f"|Z_est - Z_true| = {z_error:.3e}")
    print()

    print("=== LSZ Reduction Report ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()

    print("=== Error Summary ===")
    print(f"max relative error : {max_rel_error:.3e}")
    print(f"mean relative error: {mean_rel_error:.3e}")
    print(f"max |p_i^2-m^2|    : {float(np.max(offshell_spread)):.3e}")

    # Core sanity checks for algorithmic correctness.
    assert z_error < 5e-3, f"Residue estimation too far from true Z: {z_error:.3e}"
    assert max_rel_error < 2e-2, f"LSZ reconstruction error too large: {max_rel_error:.3e}"
    assert np.all(df["max_offshell_abs"].values < 0.03), "Off-shell perturbation is unexpectedly large."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
