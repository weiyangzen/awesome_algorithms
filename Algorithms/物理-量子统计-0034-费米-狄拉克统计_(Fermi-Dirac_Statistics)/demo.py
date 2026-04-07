"""Minimal runnable MVP for Fermi-Dirac statistics.

This script models an ideal 3D Fermi gas in normalized units and solves
the temperature-dependent chemical potential under fixed total particle number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special


@dataclass(frozen=True)
class FDConfig:
    """Configuration for the Fermi-Dirac statistics MVP."""

    fermi_energy: float = 1.0
    total_density: float = 1.0
    temperatures: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80, 1.20)
    energy_levels: tuple[float, ...] = (0.10, 0.30, 0.80, 1.00, 1.40, 2.20)
    quad_epsabs: float = 1e-9
    quad_epsrel: float = 1e-8
    max_bracket_expand: int = 80


def dos_prefactor_from_fermi_energy(fermi_energy: float, total_density: float) -> float:
    """Compute DOS prefactor A in g(E)=A*sqrt(E) from T=0 normalization.

    At T=0, f(E)=1 for E<E_F and 0 otherwise:
    n = A * int_0^{E_F} sqrt(E) dE = A * (2/3) * E_F^(3/2)
    """
    return 1.5 * total_density / (fermi_energy ** 1.5)


def fermi_dirac_occupation(energy: np.ndarray | float, mu: float, temperature: float) -> np.ndarray:
    """Return Fermi-Dirac occupation f(E)=1/(exp((E-mu)/T)+1)."""
    e = np.asarray(energy, dtype=float)
    x = (e - mu) / temperature
    return special.expit(-x)


def _density_integrand_t(t: float, mu: float, temperature: float) -> float:
    """Integrand after substitution E=t^2 for number density integral."""
    e = t * t
    f = float(special.expit(-(e - mu) / temperature))
    return 2.0 * t * t * f


def _energy_integrand_t(t: float, mu: float, temperature: float) -> float:
    """Integrand after substitution E=t^2 for energy density integral."""
    e = t * t
    f = float(special.expit(-(e - mu) / temperature))
    return 2.0 * t**4 * f


def _integration_t_upper(mu: float, temperature: float) -> float:
    """Choose a finite integration bound from thermal tail scale."""
    e_cut = max(mu, 0.0) + 40.0 * temperature + 20.0
    return float(np.sqrt(e_cut))


def number_density(mu: float, temperature: float, prefactor_a: float, cfg: FDConfig) -> float:
    """Compute n(T, mu)=A*int sqrt(E)*f(E) dE."""
    t_upper = _integration_t_upper(mu=mu, temperature=temperature)
    value, _ = integrate.quad(
        _density_integrand_t,
        0.0,
        t_upper,
        args=(mu, temperature),
        epsabs=cfg.quad_epsabs,
        epsrel=cfg.quad_epsrel,
        limit=400,
    )
    return float(prefactor_a * value)


def energy_density(mu: float, temperature: float, prefactor_a: float, cfg: FDConfig) -> float:
    """Compute u(T, mu)=A*int E*sqrt(E)*f(E) dE."""
    t_upper = _integration_t_upper(mu=mu, temperature=temperature)
    value, _ = integrate.quad(
        _energy_integrand_t,
        0.0,
        t_upper,
        args=(mu, temperature),
        epsabs=cfg.quad_epsabs,
        epsrel=cfg.quad_epsrel,
        limit=400,
    )
    return float(prefactor_a * value)


def solve_mu_for_fixed_density(temperature: float, prefactor_a: float, cfg: FDConfig) -> float:
    """Solve mu(T) from n(T,mu)=constant by monotonic root search."""

    def residual(mu: float) -> float:
        return number_density(mu, temperature, prefactor_a, cfg) - cfg.total_density

    mu_low = -8.0 * max(temperature, 0.1) - 6.0
    mu_high = max(2.0 * cfg.fermi_energy, 2.0)
    f_low = residual(mu_low)
    f_high = residual(mu_high)

    expand_count = 0
    while f_low > 0.0 and expand_count < cfg.max_bracket_expand:
        mu_low -= (2.0 + 0.5 * expand_count)
        f_low = residual(mu_low)
        expand_count += 1

    while f_high < 0.0 and expand_count < cfg.max_bracket_expand:
        mu_high += (2.0 + 0.5 * expand_count)
        f_high = residual(mu_high)
        expand_count += 1

    if not (f_low <= 0.0 <= f_high):
        raise RuntimeError("Failed to bracket chemical potential root.")

    mu_star = optimize.brentq(residual, mu_low, mu_high, xtol=1e-12, rtol=1e-10, maxiter=200)
    return float(mu_star)


def sommerfeld_mu_approx(temperature: float, fermi_energy: float) -> float:
    """Low-T Sommerfeld expansion: mu≈E_F[1-(pi^2/12)(T/E_F)^2]."""
    x = temperature / fermi_energy
    return float(fermi_energy * (1.0 - (np.pi**2 / 12.0) * x * x))


def analyze(cfg: FDConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary table and occupation samples."""
    prefactor_a = dos_prefactor_from_fermi_energy(cfg.fermi_energy, cfg.total_density)
    energies = np.array(cfg.energy_levels, dtype=float)

    summary_records: list[dict[str, float]] = []
    occ_records: list[dict[str, float]] = []

    for temperature in cfg.temperatures:
        mu = solve_mu_for_fixed_density(temperature, prefactor_a, cfg)
        density = number_density(mu, temperature, prefactor_a, cfg)
        u = energy_density(mu, temperature, prefactor_a, cfg)
        mu_sommerfeld = sommerfeld_mu_approx(temperature, cfg.fermi_energy)

        summary_records.append(
            {
                "T": float(temperature),
                "mu": float(mu),
                "mu_sommerfeld": float(mu_sommerfeld),
                "delta_mu": float(mu - mu_sommerfeld),
                "density": float(density),
                "energy_density": float(u),
                "f(E=mu)": float(special.expit(0.0)),
            }
        )

        occ = fermi_dirac_occupation(energies, mu=mu, temperature=temperature)
        row: dict[str, float] = {"T": float(temperature), "mu": float(mu)}
        for e, occ_value in zip(energies, occ):
            row[f"f(E={e:.2f})"] = float(occ_value)
        occ_records.append(row)

    summary_df = pd.DataFrame.from_records(summary_records)
    occ_df = pd.DataFrame.from_records(occ_records)
    return summary_df, occ_df


def run_consistency_checks(summary_df: pd.DataFrame, occ_df: pd.DataFrame, cfg: FDConfig) -> None:
    """Check basic physical consistency of the generated outputs."""
    # Fixed-density solve should stay close to target.
    max_density_err = float(np.max(np.abs(summary_df["density"].to_numpy() - cfg.total_density)))
    assert max_density_err < 2e-6, f"density mismatch too large: {max_density_err:.3e}"

    # mu(T) should be non-increasing for this fixed-density ideal gas setup.
    mu = summary_df.sort_values("T")["mu"].to_numpy()
    assert np.all(np.diff(mu) <= 5e-4), "mu(T) should decrease as T increases."

    # Occupation bounds and f(E=mu)=0.5 identity.
    occ_cols = [c for c in occ_df.columns if c.startswith("f(E=")]
    occ_values = occ_df[occ_cols].to_numpy()
    assert np.all((occ_values >= 0.0) & (occ_values <= 1.0)), "Fermi occupation must lie in [0,1]."
    assert np.allclose(summary_df["f(E=mu)"].to_numpy(), 0.5, atol=1e-12)

    # High-energy tail should approach Maxwell-Boltzmann form.
    t_last = float(cfg.temperatures[-1])
    mu_last = float(summary_df.sort_values("T")["mu"].to_numpy()[-1])
    e_tail = mu_last + 10.0 * t_last
    f_fd = float(fermi_dirac_occupation(np.array([e_tail]), mu_last, t_last)[0])
    f_mb = float(np.exp(-(e_tail - mu_last) / t_last))
    rel_err = abs(f_fd - f_mb) / max(f_fd, 1e-15)
    assert rel_err < 8e-3, f"FD tail not close to MB limit: rel_err={rel_err:.3e}"


def main() -> None:
    cfg = FDConfig()
    summary_df, occ_df = analyze(cfg)
    run_consistency_checks(summary_df, occ_df, cfg)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    print("=== Fermi-Dirac Statistics MVP ===")
    print("Normalized units: k_B=1, E_F=1, total density=1, DOS=A*sqrt(E)")
    print()

    print("[Thermodynamic summary]")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()

    print("[Sample occupation numbers]")
    print(occ_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()
