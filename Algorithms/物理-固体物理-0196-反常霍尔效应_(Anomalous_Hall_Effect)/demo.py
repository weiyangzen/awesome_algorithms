"""Minimal runnable MVP for Anomalous Hall Effect (AHE).

Model:
- 2D massive Dirac Hamiltonian H(k) = kx*sx + ky*sy + m*sz (hbar=vF=1)
- Band energies: E_+(k)=+sqrt(k^2+m^2), E_-(k)=-sqrt(k^2+m^2)
- Berry curvatures:
    Omega_-(k) = -m / (2 * (k^2 + m^2)^(3/2))
    Omega_+(k) = -Omega_-(k)

Intrinsic anomalous Hall conductivity (in e^2/h units):
    sigma_xy = -(1/2pi) * integral d^2k [ f_+(k)*Omega_+(k) + f_-(k)*Omega_-(k) ]
where f_+/f_- are Fermi-Dirac occupations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class AHEConfig:
    mass: float
    temperature: float
    k_max: float
    nk: int
    mu_values: np.ndarray

    def validate(self) -> None:
        if abs(self.mass) <= 0.0:
            raise ValueError("mass must be non-zero")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.k_max <= 0.0:
            raise ValueError(f"k_max must be positive, got {self.k_max}")
        if self.nk < 51:
            raise ValueError(f"nk is too small for stable 2D integration: {self.nk}")
        if self.nk % 2 == 0:
            raise ValueError(f"nk should be odd for Simpson integration, got {self.nk}")
        if self.mu_values.ndim != 1 or self.mu_values.size < 1:
            raise ValueError("mu_values must be a 1D array with at least 1 entry")
        if not np.all(np.diff(self.mu_values) >= 0.0):
            raise ValueError("mu_values must be sorted in non-decreasing order")


@dataclass(frozen=True)
class DiracGrid:
    kx_axis: np.ndarray
    ky_axis: np.ndarray
    energy_abs: np.ndarray
    omega_lower: np.ndarray
    omega_upper: np.ndarray


def build_dirac_grid(cfg: AHEConfig) -> DiracGrid:
    axis = np.linspace(-cfg.k_max, cfg.k_max, cfg.nk, dtype=np.float64)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")

    energy_abs = np.sqrt(kx * kx + ky * ky + cfg.mass * cfg.mass)
    omega_lower = -cfg.mass / (2.0 * energy_abs**3)
    omega_upper = -omega_lower

    return DiracGrid(
        kx_axis=axis,
        ky_axis=axis,
        energy_abs=energy_abs,
        omega_lower=omega_lower,
        omega_upper=omega_upper,
    )


def fermi_dirac(energy: np.ndarray, mu: float, temperature: float) -> np.ndarray:
    if temperature == 0.0:
        return (energy <= mu).astype(np.float64)

    x = (energy - mu) / temperature
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (np.exp(x) + 1.0)


def intrinsic_sigma_xy(mu: float, cfg: AHEConfig, grid: DiracGrid) -> float:
    f_plus = fermi_dirac(energy=grid.energy_abs, mu=mu, temperature=cfg.temperature)
    f_minus = fermi_dirac(energy=-grid.energy_abs, mu=mu, temperature=cfg.temperature)

    integrand = -(
        f_plus * grid.omega_upper
        + f_minus * grid.omega_lower
    )

    # Two-step Simpson integration over ky then kx.
    inner = simpson(integrand, x=grid.ky_axis, axis=1)
    sigma_xy = float(simpson(inner, x=grid.kx_axis) / (2.0 * np.pi))
    return sigma_xy


def sigma_xy_t0_continuum(mass: float, mu: float) -> float:
    """Continuum T=0 reference for one massive Dirac cone (in e^2/h units)."""
    abs_mu = abs(mu)
    abs_m = abs(mass)
    if abs_mu <= abs_m:
        return 0.5 * np.sign(mass)
    return mass / (2.0 * abs_mu)


def run_sweep(cfg: AHEConfig) -> pd.DataFrame:
    cfg.validate()
    grid = build_dirac_grid(cfg)

    rows: list[dict[str, float | str]] = []
    for mu in cfg.mu_values:
        sigma = intrinsic_sigma_xy(mu=float(mu), cfg=cfg, grid=grid)
        sigma_ref = sigma_xy_t0_continuum(mass=cfg.mass, mu=float(mu))
        rows.append(
            {
                "mu": float(mu),
                "sigma_xy_e2_over_h": sigma,
                "sigma_t0_continuum": sigma_ref,
                "abs_err_vs_t0_continuum": abs(sigma - sigma_ref),
                "regime": "in_gap" if abs(mu) <= abs(cfg.mass) else "metallic",
            }
        )

    return pd.DataFrame(rows)


def run_consistency_checks(df: pd.DataFrame, cfg: AHEConfig) -> None:
    sigma0 = float(df.loc[np.isclose(df["mu"], 0.0), "sigma_xy_e2_over_h"].iloc[0])
    if sigma0 * cfg.mass <= 0.0:
        raise AssertionError("sigma_xy(mu=0) should have same sign as mass")

    mu_max = float(np.max(np.abs(cfg.mu_values)))
    tail = df.loc[np.isclose(np.abs(df["mu"]), mu_max), "sigma_xy_e2_over_h"].abs().max()
    if tail >= abs(sigma0):
        raise AssertionError("|sigma_xy| at large |mu| should be smaller than in-gap value")

    # Particle-hole symmetry of this model gives sigma(mu) ~ sigma(-mu).
    for mu in cfg.mu_values:
        if mu < 0.0:
            sigma_neg = float(df.loc[np.isclose(df["mu"], mu), "sigma_xy_e2_over_h"].iloc[0])
            sigma_pos = float(df.loc[np.isclose(df["mu"], -mu), "sigma_xy_e2_over_h"].iloc[0])
            if abs(sigma_neg - sigma_pos) > 2.0e-4:
                raise AssertionError(
                    f"symmetry check failed at mu={mu}: sigma(-mu)={sigma_neg}, sigma(mu)={sigma_pos}"
                )

    mean_abs_err = float(df["abs_err_vs_t0_continuum"].mean())
    if mean_abs_err > 0.06:
        raise AssertionError(
            "MVP deviates too much from continuum T=0 reference; "
            f"mean_abs_err={mean_abs_err:.6f}"
        )


def main() -> None:
    cfg = AHEConfig(
        mass=0.8,
        temperature=0.03,
        k_max=12.0,
        nk=501,
        mu_values=np.array([-3.0, -2.0, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 2.0, 3.0]),
    )

    df = run_sweep(cfg)
    run_consistency_checks(df, cfg)

    # Independent sign-reversal check by flipping magnetization mass term.
    cfg_flipped = AHEConfig(
        mass=-cfg.mass,
        temperature=cfg.temperature,
        k_max=cfg.k_max,
        nk=cfg.nk,
        mu_values=np.array([0.0]),
    )
    df_flipped = run_sweep(cfg_flipped)

    sigma0 = float(df.loc[np.isclose(df["mu"], 0.0), "sigma_xy_e2_over_h"].iloc[0])
    sigma0_flip = float(df_flipped.loc[0, "sigma_xy_e2_over_h"])
    if abs(sigma0 + sigma0_flip) > 5.0e-4:
        raise AssertionError(
            "mass sign-flip check failed: "
            f"sigma(m)+sigma(-m)={sigma0 + sigma0_flip:.6e}"
        )

    print("Anomalous Hall Effect MVP (2D massive Dirac, intrinsic contribution)")
    print(
        f"mass={cfg.mass:.3f}, temperature={cfg.temperature:.3f}, "
        f"k_max={cfg.k_max:.1f}, nk={cfg.nk}"
    )
    print(df.to_string(index=False, justify="center", col_space=14))
    print(f"sigma_xy(mu=0, mass={cfg.mass:+.2f})  = {sigma0:.8f} e^2/h")
    print(f"sigma_xy(mu=0, mass={-cfg.mass:+.2f}) = {sigma0_flip:.8f} e^2/h")
    print("All checks passed.")


if __name__ == "__main__":
    main()
