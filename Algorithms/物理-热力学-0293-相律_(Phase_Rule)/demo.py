"""Minimal runnable MVP for Gibbs phase rule (phase rule).

Core equation used in this demo:
    F = C - P + 2 - R - M - S

Where
- F: degrees of freedom (variance)
- C: number of independent components
- P: number of coexisting phases
- R: number of independent reactions
- M: number of fixed intensive variables (e.g., fixed T and/or fixed p)
- S: extra independent constraints

The script builds deterministic scenarios and phase-count tables to verify:
1) invariant/univariant/bivariant classification;
2) feasible vs infeasible phase coexistence;
3) maximum possible coexisting phases under different constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhaseRuleScenario:
    """A deterministic thermodynamic setup for phase-rule evaluation."""

    label: str
    components: int
    phases: int
    independent_reactions: int = 0
    fixed_temperature: bool = False
    fixed_pressure: bool = False
    extra_constraints: int = 0

    @property
    def fixed_intensive_count(self) -> int:
        return int(self.fixed_temperature) + int(self.fixed_pressure)



def _validate_non_negative_int(name: str, value: int, min_value: int = 0) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")



def phase_rule_degrees_of_freedom(
    components: int,
    phases: int,
    independent_reactions: int = 0,
    fixed_intensive: int = 0,
    extra_constraints: int = 0,
) -> int:
    """Compute F from the generalized Gibbs phase rule."""
    _validate_non_negative_int("components", components, min_value=1)
    _validate_non_negative_int("phases", phases, min_value=1)
    _validate_non_negative_int("independent_reactions", independent_reactions)
    _validate_non_negative_int("fixed_intensive", fixed_intensive)
    _validate_non_negative_int("extra_constraints", extra_constraints)

    return components - phases + 2 - independent_reactions - fixed_intensive - extra_constraints



def max_coexisting_phases(
    components: int,
    independent_reactions: int = 0,
    fixed_intensive: int = 0,
    extra_constraints: int = 0,
) -> int:
    """Largest P that can keep F >= 0 under current constraints."""
    _validate_non_negative_int("components", components, min_value=1)
    _validate_non_negative_int("independent_reactions", independent_reactions)
    _validate_non_negative_int("fixed_intensive", fixed_intensive)
    _validate_non_negative_int("extra_constraints", extra_constraints)

    return components + 2 - independent_reactions - fixed_intensive - extra_constraints



def classify_variance(freedom: int) -> str:
    """Map numeric F to common thermodynamic variance labels."""
    if freedom < 0:
        return "infeasible"
    if freedom == 0:
        return "invariant"
    if freedom == 1:
        return "univariant"
    if freedom == 2:
        return "bivariant"
    return f"multivariant(F={freedom})"



def analyze_scenario(scenario: PhaseRuleScenario) -> dict[str, int | str | bool]:
    """Evaluate one scenario and return a row for reporting."""
    fixed_intensive = scenario.fixed_intensive_count
    freedom = phase_rule_degrees_of_freedom(
        components=scenario.components,
        phases=scenario.phases,
        independent_reactions=scenario.independent_reactions,
        fixed_intensive=fixed_intensive,
        extra_constraints=scenario.extra_constraints,
    )
    p_max = max_coexisting_phases(
        components=scenario.components,
        independent_reactions=scenario.independent_reactions,
        fixed_intensive=fixed_intensive,
        extra_constraints=scenario.extra_constraints,
    )

    return {
        "label": scenario.label,
        "C": scenario.components,
        "P": scenario.phases,
        "R": scenario.independent_reactions,
        "M_fixed": fixed_intensive,
        "S_extra": scenario.extra_constraints,
        "F": freedom,
        "variance": classify_variance(freedom),
        "equilibrium_possible": freedom >= 0,
        "P_max_from_rule": p_max,
    }



def enumerate_phase_counts(
    components: int,
    independent_reactions: int = 0,
    fixed_temperature: bool = False,
    fixed_pressure: bool = False,
    extra_constraints: int = 0,
) -> pd.DataFrame:
    """Build a phase-count table showing feasible/infeasible regions."""
    fixed_intensive = int(fixed_temperature) + int(fixed_pressure)
    p_max = max_coexisting_phases(
        components=components,
        independent_reactions=independent_reactions,
        fixed_intensive=fixed_intensive,
        extra_constraints=extra_constraints,
    )

    # Show a little beyond theoretical maximum to display infeasible rows.
    p_upper = max(2, p_max + 2)
    p_values = np.arange(1, p_upper + 1, dtype=int)
    freedoms = (
        components
        - p_values
        + 2
        - independent_reactions
        - fixed_intensive
        - extra_constraints
    )

    table = pd.DataFrame(
        {
            "P": p_values,
            "F": freedoms,
            "variance": [classify_variance(int(f)) for f in freedoms],
            "equilibrium_possible": freedoms >= 0,
        }
    )
    table.insert(0, "C", components)
    table["R"] = independent_reactions
    table["M_fixed"] = fixed_intensive
    table["S_extra"] = extra_constraints
    table["P_max_from_rule"] = p_max
    return table



def main() -> None:
    scenarios = [
        PhaseRuleScenario(
            label="Pure water: single phase",
            components=1,
            phases=1,
        ),
        PhaseRuleScenario(
            label="Pure water: liquid-vapor line",
            components=1,
            phases=2,
        ),
        PhaseRuleScenario(
            label="Pure water triple point",
            components=1,
            phases=3,
        ),
        PhaseRuleScenario(
            label="Pure water LV at fixed P",
            components=1,
            phases=2,
            fixed_pressure=True,
        ),
        PhaseRuleScenario(
            label="Binary alloy two phases at fixed P",
            components=2,
            phases=2,
            fixed_pressure=True,
        ),
        PhaseRuleScenario(
            label="Reactive 3-species, 1 reaction, two phases",
            components=3,
            phases=2,
            independent_reactions=1,
        ),
    ]

    summary = pd.DataFrame([analyze_scenario(s) for s in scenarios])

    water_table = enumerate_phase_counts(components=1)
    binary_fixed_p_table = enumerate_phase_counts(
        components=2,
        fixed_pressure=True,
    )

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", None)

    print("Phase Rule MVP")
    print("Equation: F = C - P + 2 - R - M - S")
    print()
    print("=== Scenario Summary ===")
    print(summary.to_string(index=False))
    print()
    print("=== Pure Component (C=1) Phase Count Sweep ===")
    print(water_table.to_string(index=False))
    print()
    print("=== Binary System at Fixed Pressure (C=2, M=1) Phase Count Sweep ===")
    print(binary_fixed_p_table.to_string(index=False))

    # Deterministic acceptance checks.
    assert phase_rule_degrees_of_freedom(components=1, phases=3) == 0
    assert phase_rule_degrees_of_freedom(components=1, phases=2, fixed_intensive=1) == 0
    assert max_coexisting_phases(components=1) == 3
    assert max_coexisting_phases(components=2, fixed_intensive=1) == 3

    pure_triple_f = int(summary.loc[summary["label"] == "Pure water triple point", "F"].iloc[0])
    binary_two_phase_fixed_p_f = int(
        summary.loc[summary["label"] == "Binary alloy two phases at fixed P", "F"].iloc[0]
    )
    assert pure_triple_f == 0
    assert binary_two_phase_fixed_p_f == 1

    # At C=1, P=4 should be infeasible (F=-1).
    water_p4_f = int(water_table.loc[water_table["P"] == 4, "F"].iloc[0])
    water_p4_ok = bool(water_table.loc[water_table["P"] == 4, "equilibrium_possible"].iloc[0])
    assert water_p4_f == -1
    assert not water_p4_ok

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
