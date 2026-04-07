"""Minimal runnable MVP for synchrotron radiation (PHYS-0185).

The script models a relativistic charge moving on a circular trajectory:
1) Compute total synchrotron power from the Lienard formula.
2) Build the universal spectral kernel F(x)=x*int_x^inf K_{5/3}(xi) dxi.
3) Normalize the spectrum and verify that integrating dP/domega recovers total power.
4) Check physically expected scaling and report diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, e, epsilon_0
from scipy.special import kv


@dataclass(frozen=True)
class SynchrotronConfig:
    """Configuration for the MVP numerical experiment."""

    charge_coulomb: float = e
    rho_meter: float = 7.5
    beta: float = 0.999
    beta_high: float = 0.9995
    x_min: float = 1.0e-4
    x_max: float = 40.0
    num_x: int = 3200


def lorentz_gamma(beta: float) -> float:
    """Lorentz factor gamma = 1/sqrt(1-beta^2)."""
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must satisfy 0 < beta < 1")
    return float(1.0 / np.sqrt(1.0 - beta * beta))


def total_synchrotron_power(charge_coulomb: float, beta: float, rho_meter: float) -> float:
    """Total radiated power for circular motion in SI units.

    P = (q^2 c / (6*pi*epsilon0*rho^2)) * beta^4 * gamma^4
    """
    if rho_meter <= 0.0:
        raise ValueError("rho_meter must be positive")
    gamma = lorentz_gamma(beta)
    prefactor = (charge_coulomb * charge_coulomb * c) / (6.0 * np.pi * epsilon_0 * rho_meter * rho_meter)
    return float(prefactor * (beta**4) * (gamma**4))


def critical_angular_frequency(beta: float, rho_meter: float) -> float:
    """Critical angular frequency omega_c = (3/2) * gamma^3 * c / rho."""
    if rho_meter <= 0.0:
        raise ValueError("rho_meter must be positive")
    gamma = lorentz_gamma(beta)
    return float(1.5 * (gamma**3) * c / rho_meter)


def build_synchrotron_kernel_table(x_min: float, x_max: float, num_x: int) -> pd.DataFrame:
    """Build F(x)=x*int_x^inf K_{5/3}(xi) dxi on a log grid.

    We approximate the upper infinite limit by x_max; choosing x_max=O(10^1)
    makes the omitted tail negligible due to exp(-x) decay of K_nu(x).
    """
    if not (0.0 < x_min < x_max):
        raise ValueError("Require 0 < x_min < x_max")
    if num_x < 200:
        raise ValueError("num_x must be at least 200 for stable integration")

    x = np.geomspace(x_min, x_max, num_x, dtype=np.float64)
    k53 = kv(5.0 / 3.0, x)

    tail_integral = np.zeros_like(x)
    dx = np.diff(x)
    for i in range(x.size - 2, -1, -1):
        tail_integral[i] = tail_integral[i + 1] + 0.5 * (k53[i] + k53[i + 1]) * dx[i]

    F = x * tail_integral
    return pd.DataFrame({"x": x, "K_5_3": k53, "tail_integral": tail_integral, "F": F})


def normalized_shape_from_kernel(kernel_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Return x, normalized g(x), and normalization integral int F(x) dx."""
    x = kernel_df["x"].to_numpy()
    F = kernel_df["F"].to_numpy()
    normalization = float(np.trapezoid(F, x))
    if normalization <= 0.0:
        raise RuntimeError("Kernel normalization must be positive")
    g = F / normalization
    return x, g, normalization


def spectral_power_distribution(
    charge_coulomb: float,
    beta: float,
    rho_meter: float,
    x: np.ndarray,
    g: np.ndarray,
) -> pd.DataFrame:
    """Build dP/domega based on normalized shape g(x), x=omega/omega_c."""
    total_power = total_synchrotron_power(charge_coulomb, beta, rho_meter)
    omega_c = critical_angular_frequency(beta, rho_meter)

    omega = x * omega_c
    dP_domega = (total_power / omega_c) * g

    return pd.DataFrame(
        {
            "x": x,
            "omega_rad_s": omega,
            "g_x": g,
            "dP_domega_W_s": dP_domega,
        }
    )


def summarize_case(config: SynchrotronConfig, beta: float, x: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """Compute one case summary for a given beta."""
    spectrum_df = spectral_power_distribution(config.charge_coulomb, beta, config.rho_meter, x, g)
    omega = spectrum_df["omega_rad_s"].to_numpy()
    dP_domega = spectrum_df["dP_domega_W_s"].to_numpy()

    power_integrated = float(np.trapezoid(dP_domega, omega))
    power_formula = total_synchrotron_power(config.charge_coulomb, beta, config.rho_meter)
    power_rel_error = abs(power_integrated - power_formula) / power_formula

    peak_idx = int(np.argmax(spectrum_df["g_x"].to_numpy()))
    x_peak = float(spectrum_df.iloc[peak_idx]["x"])

    return {
        "beta": float(beta),
        "gamma": lorentz_gamma(beta),
        "omega_c_rad_s": critical_angular_frequency(beta, config.rho_meter),
        "power_formula_W": power_formula,
        "power_integrated_W": power_integrated,
        "power_rel_error": power_rel_error,
        "x_peak": x_peak,
    }


def main() -> None:
    config = SynchrotronConfig()

    kernel_df = build_synchrotron_kernel_table(config.x_min, config.x_max, config.num_x)
    x, g, normalization = normalized_shape_from_kernel(kernel_df)

    g_integral = float(np.trapezoid(g, x))
    normalization_analytic = float(8.0 * np.pi / (9.0 * np.sqrt(3.0)))
    normalization_rel_error = abs(normalization - normalization_analytic) / normalization_analytic

    case_base = summarize_case(config, config.beta, x, g)
    case_high = summarize_case(config, config.beta_high, x, g)

    omega_ratio_numeric = case_high["omega_c_rad_s"] / case_base["omega_c_rad_s"]
    omega_ratio_theory = (case_high["gamma"] / case_base["gamma"]) ** 3
    omega_ratio_rel_error = abs(omega_ratio_numeric - omega_ratio_theory) / omega_ratio_theory

    power_ratio_numeric = case_high["power_integrated_W"] / case_base["power_integrated_W"]
    power_ratio_theory = (
        (case_high["beta"] ** 4) * (case_high["gamma"] ** 4)
    ) / ((case_base["beta"] ** 4) * (case_base["gamma"] ** 4))
    power_ratio_rel_error = abs(power_ratio_numeric - power_ratio_theory) / power_ratio_theory

    checks = {
        "normalized kernel integral close to 1": abs(g_integral - 1.0) < 3.0e-4,
        "kernel normalization matches analytic constant": normalization_rel_error < 2.0e-3,
        "integrated power recovers formula (base beta)": case_base["power_rel_error"] < 4.0e-4,
        "integrated power recovers formula (high beta)": case_high["power_rel_error"] < 4.0e-4,
        "spectral peak x in expected range [0.2, 0.4]": 0.2 < case_base["x_peak"] < 0.4,
        "critical frequency gamma^3 scaling": omega_ratio_rel_error < 1.0e-12,
        "power beta^4*gamma^4 scaling": power_ratio_rel_error < 4.0e-4,
    }

    print("=== Synchrotron Radiation MVP (PHYS-0185) ===")
    print(
        "Config: q={q:.6e} C, rho={rho:.3f} m, beta(base/high)=({b0:.6f}, {b1:.6f}), x-range=[{xmin:.1e}, {xmax:.1f}], N={n}".format(
            q=config.charge_coulomb,
            rho=config.rho_meter,
            b0=config.beta,
            b1=config.beta_high,
            xmin=config.x_min,
            xmax=config.x_max,
            n=config.num_x,
        )
    )

    print("\n[Kernel diagnostics]")
    print(f"int F(x) dx (numeric) = {normalization:.9f}")
    print(f"int F(x) dx (analytic) = {normalization_analytic:.9f}")
    print(f"normalization_rel_error = {normalization_rel_error:.3e}")
    print(f"int g(x) dx = {g_integral:.9f}")

    case_table = pd.DataFrame([case_base, case_high])
    print("\n[Case summary]")
    print(case_table.to_string(index=False))

    print("\n[Scaling diagnostics]")
    print(f"omega_ratio_numeric = {omega_ratio_numeric:.9e}")
    print(f"omega_ratio_theory  = {omega_ratio_theory:.9e}")
    print(f"omega_ratio_rel_error = {omega_ratio_rel_error:.3e}")
    print(f"power_ratio_numeric = {power_ratio_numeric:.9e}")
    print(f"power_ratio_theory  = {power_ratio_theory:.9e}")
    print(f"power_ratio_rel_error = {power_ratio_rel_error:.3e}")

    print("\n[Spectrum sample rows]")
    spectrum_sample = spectral_power_distribution(config.charge_coulomb, config.beta, config.rho_meter, x, g)
    sample_idx = np.linspace(0, spectrum_sample.shape[0] - 1, 8, dtype=int)
    print(spectrum_sample.iloc[sample_idx].to_string(index=False))

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
