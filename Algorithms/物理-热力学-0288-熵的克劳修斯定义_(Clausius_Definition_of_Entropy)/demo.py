"""Minimal runnable MVP for Clausius definition of entropy.

Core definition used in this script:
    dS = deltaQ_rev / T
For two equilibrium states:
    Delta S = integral(reversible path) deltaQ_rev / T

We verify path independence for an ideal gas by comparing:
1) Closed-form state function formula,
2) Reversible path A integral (isochoric + isothermal),
3) Reversible path B integral (isothermal + isochoric).

Then we contrast with an irreversible free expansion example:
    Delta S = integral(actual deltaQ/T) + S_gen,
where S_gen > 0 for irreversible processes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class IdealGasState:
    """Thermodynamic equilibrium state for a fixed amount of ideal gas."""

    temperature: float
    volume: float
    n_mol: float
    cv_molar: float

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0 K, got {self.temperature}")
        if self.volume <= 0.0:
            raise ValueError(f"Volume must be > 0 m^3, got {self.volume}")
        if self.n_mol <= 0.0:
            raise ValueError(f"n_mol must be > 0, got {self.n_mol}")
        if self.cv_molar <= 0.0:
            raise ValueError(f"cv_molar must be > 0, got {self.cv_molar}")



def _validate_state_pair(s1: IdealGasState, s2: IdealGasState) -> None:
    s1.validate()
    s2.validate()
    if not np.isclose(s1.n_mol, s2.n_mol, rtol=0.0, atol=1e-12):
        raise ValueError("State pair must have the same n_mol")
    if not np.isclose(s1.cv_molar, s2.cv_molar, rtol=0.0, atol=1e-12):
        raise ValueError("State pair must have the same cv_molar")



def entropy_change_closed_form(s1: IdealGasState, s2: IdealGasState) -> float:
    """Ideal-gas entropy state function formula.

    Delta S = n*Cv*ln(T2/T1) + n*R*ln(V2/V1)
    """
    _validate_state_pair(s1, s2)
    n = s1.n_mol
    cv = s1.cv_molar
    return float(
        n * cv * np.log(s2.temperature / s1.temperature)
        + n * R_UNIVERSAL * np.log(s2.volume / s1.volume)
    )



def _integrand_isochoric(temp: float, n_mol: float, cv_molar: float) -> float:
    """For reversible isochoric step: deltaQ_rev/T = n*Cv*dT/T."""
    return n_mol * cv_molar / temp



def _integrand_isothermal(volume: float, n_mol: float) -> float:
    """For reversible isothermal step: deltaQ_rev/T = n*R*dV/V."""
    return n_mol * R_UNIVERSAL / volume



def entropy_change_path_a(s1: IdealGasState, s2: IdealGasState) -> tuple[float, float]:
    """Path A: isochoric(T1->T2, at V1) + isothermal(V1->V2, at T2)."""
    _validate_state_pair(s1, s2)
    n = s1.n_mol
    cv = s1.cv_molar

    ds_isochoric, err_1 = quad(
        _integrand_isochoric,
        s1.temperature,
        s2.temperature,
        args=(n, cv),
    )
    ds_isothermal, err_2 = quad(
        _integrand_isothermal,
        s1.volume,
        s2.volume,
        args=(n,),
    )
    return float(ds_isochoric + ds_isothermal), float(err_1 + err_2)



def entropy_change_path_b(s1: IdealGasState, s2: IdealGasState) -> tuple[float, float]:
    """Path B: isothermal(V1->V2, at T1) + isochoric(T1->T2, at V2)."""
    _validate_state_pair(s1, s2)
    n = s1.n_mol
    cv = s1.cv_molar

    ds_isothermal, err_1 = quad(
        _integrand_isothermal,
        s1.volume,
        s2.volume,
        args=(n,),
    )
    ds_isochoric, err_2 = quad(
        _integrand_isochoric,
        s1.temperature,
        s2.temperature,
        args=(n, cv),
    )
    return float(ds_isothermal + ds_isochoric), float(err_1 + err_2)



def irreversible_free_expansion_budget(
    s_initial: IdealGasState,
    s_final: IdealGasState,
) -> dict[str, float]:
    """Entropy balance for adiabatic free expansion of ideal gas.

    Actual path has no boundary heat exchange: integral(actual deltaQ/T) = 0.
    Entropy generation is therefore S_gen = Delta S_system.
    """
    _validate_state_pair(s_initial, s_final)
    delta_s_system = entropy_change_closed_form(s_initial, s_final)
    actual_integral_q_over_t = 0.0
    s_gen = delta_s_system - actual_integral_q_over_t
    return {
        "delta_s_system": float(delta_s_system),
        "integral_actual_q_over_t": float(actual_integral_q_over_t),
        "s_gen": float(s_gen),
    }



def main() -> None:
    # Example 1: two states with both T and V changes.
    cv_diatomic_like = 2.5 * R_UNIVERSAL  # use constant Cv in this MVP
    state_1 = IdealGasState(temperature=300.0, volume=0.010, n_mol=1.0, cv_molar=cv_diatomic_like)
    state_2 = IdealGasState(temperature=450.0, volume=0.020, n_mol=1.0, cv_molar=cv_diatomic_like)

    ds_closed = entropy_change_closed_form(state_1, state_2)
    ds_a, err_a = entropy_change_path_a(state_1, state_2)
    ds_b, err_b = entropy_change_path_b(state_1, state_2)

    compare_table = pd.DataFrame(
        [
            {"route": "closed_form", "delta_s_J_per_K": ds_closed, "estimated_quad_error": 0.0},
            {"route": "reversible_path_A", "delta_s_J_per_K": ds_a, "estimated_quad_error": err_a},
            {"route": "reversible_path_B", "delta_s_J_per_K": ds_b, "estimated_quad_error": err_b},
        ]
    )

    # Example 2: irreversible adiabatic free expansion (T constant for ideal gas).
    free_expansion_start = IdealGasState(
        temperature=300.0,
        volume=0.010,
        n_mol=1.0,
        cv_molar=cv_diatomic_like,
    )
    free_expansion_end = IdealGasState(
        temperature=300.0,
        volume=0.020,
        n_mol=1.0,
        cv_molar=cv_diatomic_like,
    )
    free_expansion = irreversible_free_expansion_budget(free_expansion_start, free_expansion_end)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("Clausius Definition of Entropy MVP")
    print("dS = deltaQ_rev/T, Delta S computed from reversible path integral")
    print(compare_table.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print()
    print("Irreversible free expansion entropy balance:")
    print(pd.DataFrame([free_expansion]).to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Deterministic checks for path independence and irreversible entropy generation.
    tol = 1e-9
    assert abs(ds_closed - ds_a) <= tol, f"Path A mismatch: {ds_closed} vs {ds_a}"
    assert abs(ds_closed - ds_b) <= tol, f"Path B mismatch: {ds_closed} vs {ds_b}"
    assert free_expansion["integral_actual_q_over_t"] == 0.0
    assert free_expansion["s_gen"] > 0.0, "Irreversible free expansion must generate entropy"

    print("All checks passed.")


if __name__ == "__main__":
    main()
