"""Minimal runnable MVP for the Cosmological Principle (PHYS-0053).

The script builds two synthetic galaxy catalogs:
1) A catalog consistent with large-scale isotropy + homogeneity.
2) A catalog that violates both assumptions.

It then evaluates both catalogs using explicit, auditable statistics:
- Angular isotropy chi-square test on equal-area sky bins.
- Radial homogeneity chi-square test on equal-volume shells via u=(r/R)^3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


TWO_PI = 2.0 * np.pi
EPS = 1e-12


@dataclass
class PrincipleConfig:
    n_galaxies: int = 12_000
    r_max_mpc: float = 3_000.0
    mu_bins: int = 10
    phi_bins: int = 20
    radial_bins: int = 12
    alpha: float = 0.01
    seed_reference: int = 53
    seed_violation: int = 530


@dataclass
class CatalogDiagnostics:
    label: str
    n_samples: int
    chi2_isotropy: float
    p_isotropy: float
    dipole_amplitude: float
    chi2_homogeneity: float
    p_homogeneity: float
    radial_shell_cv: float
    accept_cosmological_principle: bool


def sample_isotropic_homogeneous_catalog(cfg: PrincipleConfig, seed: int) -> np.ndarray:
    """Uniform number density in a sphere and isotropic angular distribution."""
    rng = np.random.default_rng(seed)
    n = cfg.n_galaxies

    # Uniform in volume: F(r)=(r/R)^3 -> r=R*u^(1/3).
    u = rng.random(n)
    r = cfg.r_max_mpc * np.cbrt(u)
    mu = rng.uniform(-1.0, 1.0, size=n)  # mu = cos(theta), equal-area sampling.
    phi = rng.uniform(0.0, TWO_PI, size=n)

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * mu
    return np.column_stack([x, y, z])


def sample_mu_with_dipole(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Sample mu in [-1,1] from p(mu) ∝ (1 + beta*mu), |beta|<1 via rejection."""
    if not (-1.0 < beta < 1.0):
        raise ValueError("beta must satisfy -1 < beta < 1")

    accepted: list[np.ndarray] = []
    remaining = n
    max_pdf = 1.0 + abs(beta)

    while remaining > 0:
        batch = max(1024, int(remaining * 1.7))
        cand = rng.uniform(-1.0, 1.0, size=batch)
        prob = (1.0 + beta * cand) / max_pdf
        keep = rng.random(batch) < prob
        chosen = cand[keep]
        if chosen.size > 0:
            accepted.append(chosen[:remaining])
            remaining -= min(remaining, chosen.size)

    return np.concatenate(accepted)


def sample_anisotropic_inhomogeneous_catalog(cfg: PrincipleConfig, seed: int) -> np.ndarray:
    """Deliberately violate isotropy and homogeneity in a controlled way."""
    rng = np.random.default_rng(seed)
    n = cfg.n_galaxies

    # Radial law: F(r)=(r/R)^eta with eta!=3 induces non-constant density.
    eta = 2.2
    u = rng.random(n)
    r = cfg.r_max_mpc * np.power(u, 1.0 / eta)

    # Angular law: dipole-weighted mu breaks isotropy.
    mu = sample_mu_with_dipole(n=n, beta=0.65, rng=rng)
    phi = rng.uniform(0.0, TWO_PI, size=n)

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * mu
    return np.column_stack([x, y, z])


def _safe_norm_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, EPS, None)


def compute_catalog_diagnostics(points: np.ndarray, cfg: PrincipleConfig, label: str) -> CatalogDiagnostics:
    """Compute isotropy and homogeneity diagnostics for one catalog."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if points.shape[0] == 0:
        raise ValueError("points cannot be empty")

    n = points.shape[0]
    r = np.linalg.norm(points, axis=1)
    unit = _safe_norm_rows(points)
    mu = np.clip(unit[:, 2], -1.0, 1.0)
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), TWO_PI)

    # Isotropy test: equal-area bins in (mu, phi).
    mu_edges = np.linspace(-1.0, 1.0, cfg.mu_bins + 1)
    phi_edges = np.linspace(0.0, TWO_PI, cfg.phi_bins + 1)
    counts_iso, _, _ = np.histogram2d(mu, phi, bins=[mu_edges, phi_edges])
    expected_iso = n / (cfg.mu_bins * cfg.phi_bins)
    chi2_isotropy = float(np.sum((counts_iso - expected_iso) ** 2 / expected_iso))
    dof_iso = cfg.mu_bins * cfg.phi_bins - 1
    p_isotropy = float(chi2.sf(chi2_isotropy, dof_iso))

    # Dipole amplitude: mean unit direction norm, should be near zero for isotropy.
    dipole_amplitude = float(np.linalg.norm(np.mean(unit, axis=0)))

    # Homogeneity test: for uniform volume, u=(r/R)^3 should be uniform on [0,1].
    u_volume = np.clip((r / cfg.r_max_mpc) ** 3, 0.0, 1.0)
    u_edges = np.linspace(0.0, 1.0, cfg.radial_bins + 1)
    counts_homo, _ = np.histogram(u_volume, bins=u_edges)
    expected_homo = n / cfg.radial_bins
    chi2_homogeneity = float(np.sum((counts_homo - expected_homo) ** 2 / expected_homo))
    dof_homo = cfg.radial_bins - 1
    p_homogeneity = float(chi2.sf(chi2_homogeneity, dof_homo))
    radial_shell_cv = float(np.std(counts_homo) / max(np.mean(counts_homo), EPS))

    accept = (p_isotropy > cfg.alpha) and (p_homogeneity > cfg.alpha)

    return CatalogDiagnostics(
        label=label,
        n_samples=n,
        chi2_isotropy=chi2_isotropy,
        p_isotropy=p_isotropy,
        dipole_amplitude=dipole_amplitude,
        chi2_homogeneity=chi2_homogeneity,
        p_homogeneity=p_homogeneity,
        radial_shell_cv=radial_shell_cv,
        accept_cosmological_principle=accept,
    )


def diagnostics_to_frame(diags: list[CatalogDiagnostics]) -> pd.DataFrame:
    rows = []
    for d in diags:
        rows.append(
            {
                "catalog": d.label,
                "N": d.n_samples,
                "chi2_iso": d.chi2_isotropy,
                "p_iso": d.p_isotropy,
                "dipole_amp": d.dipole_amplitude,
                "chi2_homo": d.chi2_homogeneity,
                "p_homo": d.p_homogeneity,
                "shell_cv": d.radial_shell_cv,
                "CP_accept": d.accept_cosmological_principle,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = PrincipleConfig()

    ref_points = sample_isotropic_homogeneous_catalog(cfg, seed=cfg.seed_reference)
    bad_points = sample_anisotropic_inhomogeneous_catalog(cfg, seed=cfg.seed_violation)

    ref_diag = compute_catalog_diagnostics(ref_points, cfg, label="reference_isotropic_homogeneous")
    bad_diag = compute_catalog_diagnostics(bad_points, cfg, label="violating_catalog")

    df = diagnostics_to_frame([ref_diag, bad_diag])
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")

    print("=== Cosmological Principle MVP (PHYS-0053) ===")
    print(
        f"Config: N={cfg.n_galaxies}, Rmax={cfg.r_max_mpc:.1f} Mpc, "
        f"angular_bins={cfg.mu_bins}x{cfg.phi_bins}, radial_bins={cfg.radial_bins}, alpha={cfg.alpha:.3f}"
    )
    print("\nDiagnostics:")
    print(df.to_string(index=False))

    checks = {
        "reference catalog accepted": ref_diag.accept_cosmological_principle,
        "violating catalog rejected": not bad_diag.accept_cosmological_principle,
        "reference dipole amplitude < 0.03": ref_diag.dipole_amplitude < 0.03,
        "violating dipole amplitude > 0.08": bad_diag.dipole_amplitude > 0.08,
    }

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
