"""Minimal runnable MVP for the van der Waals equation.

Equation of state for n moles:
    P = nRT / (V - nb) - a (n/V)^2

This script demonstrates:
1) Forward evaluation: given (T, V) compute P and compressibility factor Z.
2) Inverse evaluation: given (T, P) solve the cubic in V and report physical roots.
3) Critical-point consistency checks from analytic formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class VanDerWaalsParams:
    """van der Waals parameters in SI units.

    Attributes:
        a: attraction parameter [Pa*m^6/mol^2]
        b: covolume parameter [m^3/mol]
        n_mol: amount of substance [mol]
    """

    a: float
    b: float
    n_mol: float = 1.0

    def validate(self) -> None:
        if self.a <= 0.0:
            raise ValueError(f"Parameter a must be positive, got {self.a}")
        if self.b <= 0.0:
            raise ValueError(f"Parameter b must be positive, got {self.b}")
        if self.n_mol <= 0.0:
            raise ValueError(f"n_mol must be positive, got {self.n_mol}")



def ideal_gas_pressure(volume: float, temperature: float, n_mol: float) -> float:
    """Ideal-gas pressure P = nRT/V."""
    if volume <= 0.0:
        raise ValueError(f"Volume must be positive for ideal gas, got {volume}")
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if n_mol <= 0.0:
        raise ValueError(f"n_mol must be positive, got {n_mol}")
    return float(n_mol * R_UNIVERSAL * temperature / volume)



def vdw_pressure(volume: float, temperature: float, params: VanDerWaalsParams) -> float:
    """van der Waals pressure P(V, T)."""
    params.validate()
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    min_volume = params.n_mol * params.b
    if volume <= min_volume:
        raise ValueError(
            f"Volume must satisfy V > n*b={min_volume:.6e} m^3, got {volume:.6e}"
        )

    repulsion = params.n_mol * R_UNIVERSAL * temperature / (volume - min_volume)
    attraction = params.a * (params.n_mol / volume) ** 2
    return float(repulsion - attraction)



def critical_constants(params: VanDerWaalsParams) -> dict[str, float]:
    """Return critical constants for the van der Waals model.

    Tc = 8a / (27Rb)
    Pc = a / (27b^2)
    Vc = 3nb  (total volume for n moles)
    """
    params.validate()
    tc = 8.0 * params.a / (27.0 * R_UNIVERSAL * params.b)
    pc = params.a / (27.0 * params.b * params.b)
    vc = 3.0 * params.n_mol * params.b
    return {"T_c_K": float(tc), "P_c_Pa": float(pc), "V_c_m3": float(vc)}



def dP_dV(volume: float, temperature: float, params: VanDerWaalsParams) -> float:
    """First derivative (dP/dV)_T for van der Waals EOS."""
    nb = params.n_mol * params.b
    term_1 = -params.n_mol * R_UNIVERSAL * temperature / (volume - nb) ** 2
    term_2 = 2.0 * params.a * (params.n_mol**2) / (volume**3)
    return float(term_1 + term_2)



def d2P_dV2(volume: float, temperature: float, params: VanDerWaalsParams) -> float:
    """Second derivative (d2P/dV2)_T for van der Waals EOS."""
    nb = params.n_mol * params.b
    term_1 = 2.0 * params.n_mol * R_UNIVERSAL * temperature / (volume - nb) ** 3
    term_2 = -6.0 * params.a * (params.n_mol**2) / (volume**4)
    return float(term_1 + term_2)



def cubic_coefficients_for_volume(
    target_pressure: float,
    temperature: float,
    params: VanDerWaalsParams,
) -> np.ndarray:
    """Build cubic coefficients for solving volume from (P, T).

    From:
        (P + a(n/V)^2)(V - nb) = nRT
    Multiply by V^2 and rearrange:
        P V^3 - (Pnb + nRT)V^2 + a n^2 V - a n^3 b = 0
    """
    params.validate()
    if target_pressure <= 0.0:
        raise ValueError(f"target_pressure must be positive, got {target_pressure}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    n = params.n_mol
    b = params.b
    a = params.a
    coeff = np.array(
        [
            target_pressure,
            -(target_pressure * n * b + n * R_UNIVERSAL * temperature),
            a * n * n,
            -a * (n**3) * b,
        ],
        dtype=np.float64,
    )
    return coeff



def solve_volume_roots_from_cubic(
    target_pressure: float,
    temperature: float,
    params: VanDerWaalsParams,
    imag_tol: float = 1e-10,
) -> list[float]:
    """Solve cubic and keep physically meaningful real roots V > n*b."""
    coeff = cubic_coefficients_for_volume(target_pressure, temperature, params)
    roots = np.roots(coeff)

    threshold = params.n_mol * params.b * (1.0 + 1e-12)
    real_roots = [float(z.real) for z in roots if abs(z.imag) <= imag_tol and z.real > threshold]
    real_roots.sort()
    return real_roots



def solve_supercritical_volume_brentq(
    target_pressure: float,
    temperature: float,
    params: VanDerWaalsParams,
) -> float:
    """Solve V(P, T) with Brent for supercritical temperatures (single root regime)."""
    crit = critical_constants(params)
    if temperature <= crit["T_c_K"]:
        raise ValueError(
            f"Brent supercritical solver expects T > Tc={crit['T_c_K']:.6f} K, got {temperature}"
        )

    lower = params.n_mol * params.b * (1.0 + 1e-9)

    def residual(volume: float) -> float:
        return vdw_pressure(volume, temperature, params) - target_pressure

    upper = 0.02  # m^3, and enlarged if needed
    f_low = residual(lower)
    f_up = residual(upper)
    while f_low * f_up > 0.0 and upper < 20.0:
        upper *= 2.0
        f_up = residual(upper)

    if f_low * f_up > 0.0:
        raise RuntimeError("Failed to bracket a root for Brent method")

    root = brentq(residual, lower, upper, xtol=1e-12, maxiter=200)
    return float(root)



def build_isotherm_table(
    params: VanDerWaalsParams,
    temperatures: list[float],
    volume_grid: np.ndarray,
) -> pd.DataFrame:
    """Generate forward-evaluation table for isotherms."""
    rows: list[dict[str, float]] = []
    for temp in temperatures:
        for volume in volume_grid:
            p_vdw = vdw_pressure(float(volume), temp, params)
            p_ideal = ideal_gas_pressure(float(volume), temp, params.n_mol)
            z = p_vdw * float(volume) / (params.n_mol * R_UNIVERSAL * temp)
            rows.append(
                {
                    "T_K": temp,
                    "V_m3": float(volume),
                    "P_vdw_Pa": p_vdw,
                    "P_ideal_Pa": p_ideal,
                    "Z_vdw": z,
                }
            )
    return pd.DataFrame(rows)



def build_inverse_check_table(
    params: VanDerWaalsParams,
    cases: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Generate inverse-evaluation table from (P, T) to V roots."""
    rows: list[dict[str, float | str | int]] = []
    for case in cases:
        name = str(case["name"])
        temp = float(case["T_K"])
        p_target = float(case["P_target_Pa"])

        roots = solve_volume_roots_from_cubic(p_target, temp, params)
        if not roots:
            raise RuntimeError(f"No physical real roots for case {name}")

        for idx, volume in enumerate(roots, start=1):
            p_reconstructed = vdw_pressure(volume, temp, params)
            rows.append(
                {
                    "case": name,
                    "solver": "cubic_roots",
                    "T_K": temp,
                    "P_target_Pa": p_target,
                    "root_index": idx,
                    "num_real_roots": len(roots),
                    "V_root_m3": volume,
                    "P_reconstructed_Pa": p_reconstructed,
                    "abs_pressure_error_Pa": abs(p_reconstructed - p_target),
                    "rel_pressure_error": abs(p_reconstructed - p_target) / p_target,
                }
            )

        crit = critical_constants(params)
        if temp > crit["T_c_K"]:
            v_brent = solve_supercritical_volume_brentq(p_target, temp, params)
            p_brent = vdw_pressure(v_brent, temp, params)
            rows.append(
                {
                    "case": name,
                    "solver": "brentq",
                    "T_K": temp,
                    "P_target_Pa": p_target,
                    "root_index": 1,
                    "num_real_roots": len(roots),
                    "V_root_m3": v_brent,
                    "P_reconstructed_Pa": p_brent,
                    "abs_pressure_error_Pa": abs(p_brent - p_target),
                    "rel_pressure_error": abs(p_brent - p_target) / p_target,
                }
            )

    return pd.DataFrame(rows)



def main() -> None:
    # CO2-like parameters in SI units.
    params = VanDerWaalsParams(
        a=0.3592,      # Pa*m^6/mol^2
        b=4.267e-5,    # m^3/mol
        n_mol=1.0,     # mol
    )

    crit = critical_constants(params)
    tc = crit["T_c_K"]
    pc = crit["P_c_Pa"]
    vc = crit["V_c_m3"]

    # Analytic critical-point checks from EOS derivatives.
    dp = dP_dV(vc, tc, params)
    d2p = d2P_dV2(vc, tc, params)

    volume_grid = np.geomspace(params.n_mol * params.b * 1.05, 1.2e-3, num=8)
    temperatures = [280.0, tc, 330.0]
    isotherm_df = build_isotherm_table(params, temperatures, volume_grid)

    inverse_cases: list[dict[str, float | str]] = [
        {"name": "subcritical_multiroot", "T_K": 280.0, "P_target_Pa": 5.0e6},
        {"name": "near_critical", "T_K": tc, "P_target_Pa": pc},
        {"name": "supercritical_single", "T_K": 330.0, "P_target_Pa": 8.0e6},
    ]
    inverse_df = build_inverse_check_table(params, inverse_cases)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("van der Waals Equation MVP")
    print("Equation: P = nRT/(V-nb) - a(n/V)^2")
    print()
    print("Critical constants (model):")
    print(pd.DataFrame([crit]).to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print(f"Derivative checks at critical point: dP/dV={dp:.3e}, d2P/dV2={d2p:.3e}")
    print()

    print("Isotherm samples (forward evaluation):")
    print(isotherm_df.to_string(index=False, float_format=lambda x: f"{x:.10e}"))
    print()

    print("Inverse checks (solve V from P,T):")
    print(inverse_df.to_string(index=False, float_format=lambda x: f"{x:.10e}"))

    # Deterministic assertions.
    n = params.n_mol
    b = params.b
    a = params.a
    dp_scale = abs(-n * R_UNIVERSAL * tc / (vc - n * b) ** 2) + abs(2.0 * a * n * n / vc**3)
    d2p_scale = abs(2.0 * n * R_UNIVERSAL * tc / (vc - n * b) ** 3) + abs(6.0 * a * n * n / vc**4)
    assert abs(dp) / dp_scale < 1e-12, f"Critical derivative relative check failed: {abs(dp) / dp_scale}"
    assert (
        abs(d2p) / d2p_scale < 1e-12
    ), f"Critical second derivative relative check failed: {abs(d2p) / d2p_scale}"

    max_rel_residual = float(inverse_df["rel_pressure_error"].max())
    assert max_rel_residual < 1e-9, f"Inverse pressure relative residual too large: {max_rel_residual}"

    # Require at least one multi-root case to demonstrate non-ideal phase-like behavior.
    has_multi_root_case = bool((inverse_df["num_real_roots"] >= 3).any())
    assert has_multi_root_case, "Expected at least one case with 3 real roots"

    print("All checks passed.")


if __name__ == "__main__":
    main()
