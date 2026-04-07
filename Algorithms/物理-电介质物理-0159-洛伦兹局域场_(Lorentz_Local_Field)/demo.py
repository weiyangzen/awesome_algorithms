"""Lorentz local field MVP (PHYS-0158).

This demo computes local field enhancement and dielectric constant from
molecular-level parameters under isotropic linear dielectric assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON_0 = 8.8541878128e-12  # F/m


def lorentz_coupling_factor(number_density: float, polarizability: float) -> float:
    """Return x = N * alpha / (3 * epsilon_0)."""
    if number_density < 0:
        raise ValueError("number_density must be non-negative.")
    if polarizability < 0:
        raise ValueError("polarizability must be non-negative.")
    return (number_density * polarizability) / (3.0 * EPSILON_0)


def local_field(macro_field: float, number_density: float, polarizability: float) -> float:
    """Compute E_loc = E_macro / (1 - x), where x = N*alpha/(3*epsilon_0)."""
    x = lorentz_coupling_factor(number_density, polarizability)
    if x >= 1.0:
        raise ValueError(
            "Lorentz model breakdown: N*alpha/(3*epsilon_0) must be < 1.0."
        )
    return macro_field / (1.0 - x)


def polarization(
    number_density: float, polarizability: float, local_field_value: float
) -> float:
    """Compute polarization P = N * alpha * E_loc."""
    return number_density * polarizability * local_field_value


def susceptibility_from_micro(number_density: float, polarizability: float) -> float:
    """Compute chi_e from micro parameters and Lorentz local-field correction."""
    x = lorentz_coupling_factor(number_density, polarizability)
    if x >= 1.0:
        raise ValueError(
            "Susceptibility diverges when N*alpha/(3*epsilon_0) >= 1.0."
        )
    return (number_density * polarizability / EPSILON_0) / (1.0 - x)


def epsilon_r_from_chi(chi_e: float) -> float:
    """Compute relative permittivity epsilon_r = 1 + chi_e."""
    return 1.0 + chi_e


def epsilon_r_clausius_mossotti(number_density: float, polarizability: float) -> float:
    """Compute epsilon_r from Clausius-Mossotti form."""
    x = lorentz_coupling_factor(number_density, polarizability)
    if x >= 1.0:
        raise ValueError(
            "Clausius-Mossotti denominator becomes non-positive when x >= 1.0."
        )
    return (1.0 + 2.0 * x) / (1.0 - x)


def analyze_material(
    name: str, number_density: float, polarizability: float, macro_field: float
) -> dict[str, float | str]:
    """Analyze one material and return a diagnostic record."""
    x = lorentz_coupling_factor(number_density, polarizability)
    e_loc = local_field(macro_field, number_density, polarizability)
    p = polarization(number_density, polarizability, e_loc)
    chi_e = susceptibility_from_micro(number_density, polarizability)
    eps_r_a = epsilon_r_from_chi(chi_e)
    eps_r_b = epsilon_r_clausius_mossotti(number_density, polarizability)
    rel_err = abs(eps_r_a - eps_r_b) / max(abs(eps_r_a), 1e-18)

    # Reconstruct E_loc using E_macro + P/(3*epsilon_0) as a direct consistency check.
    e_loc_reconstructed = macro_field + p / (3.0 * EPSILON_0)
    rel_err_eloc = abs(e_loc - e_loc_reconstructed) / max(abs(e_loc), 1e-18)

    return {
        "material": name,
        "N_m^-3": number_density,
        "alpha_Cm^2/V": polarizability,
        "x=Nalpha/(3eps0)": x,
        "E_macro_V/m": macro_field,
        "E_loc_V/m": e_loc,
        "E_loc/E_macro": e_loc / macro_field if macro_field != 0 else np.nan,
        "P_C/m^2": p,
        "chi_e": chi_e,
        "eps_r_from_1+chi": eps_r_a,
        "eps_r_CM": eps_r_b,
        "rel_err_eps_r": rel_err,
        "rel_err_E_loc_reconstruct": rel_err_eloc,
    }


def main() -> None:
    """Run a minimal non-interactive Lorentz local-field demonstration."""
    macro_field = 1.0e5  # V/m
    materials = [
        ("稀薄气体", 2.5e25, 1.7e-40),
        ("中等极化介质", 3.0e27, 1.2e-39),
        ("高极化介质", 8.0e27, 1.4e-39),
    ]

    rows = [
        analyze_material(name, number_density, polarizability, macro_field)
        for name, number_density, polarizability in materials
    ]
    df = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda value: f"{value:.6e}")

    print("Lorentz Local Field MVP (PHYS-0158)")
    print(df.to_string(index=False))

    max_rel_err_eps = float(df["rel_err_eps_r"].max())
    max_rel_err_eloc = float(df["rel_err_E_loc_reconstruct"].max())
    print("\nConsistency checks:")
    print(f"- max relative error of epsilon_r two paths: {max_rel_err_eps:.3e}")
    print(f"- max relative error of E_loc reconstruction: {max_rel_err_eloc:.3e}")


if __name__ == "__main__":
    main()
