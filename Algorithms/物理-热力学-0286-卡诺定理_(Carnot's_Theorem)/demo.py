"""Minimal runnable MVP for Carnot's Theorem.

The theorem statements covered by this script are:
1) No heat engine operating between the same two reservoirs can exceed
   the efficiency of a reversible (Carnot) engine.
2) All reversible engines between the same reservoirs have the same efficiency:
      eta_carnot = 1 - Tc / Th
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineCase:
    """One engine performance record under two thermal reservoirs."""

    name: str
    th: float
    tc: float
    q_in: float
    w_out: float
    claimed_reversible: bool

    def validate(self) -> None:
        if self.th <= 0.0 or self.tc <= 0.0:
            raise ValueError(f"{self.name}: temperatures must be positive")
        if self.tc >= self.th:
            raise ValueError(f"{self.name}: require Tc < Th, got Tc={self.tc}, Th={self.th}")
        if self.q_in <= 0.0:
            raise ValueError(f"{self.name}: q_in must be positive")
        if self.w_out < 0.0:
            raise ValueError(f"{self.name}: w_out must be non-negative")
        if self.w_out >= self.q_in:
            raise ValueError(
                f"{self.name}: w_out must be strictly less than q_in for cyclic heat engine"
            )


def carnot_efficiency(th: float, tc: float) -> float:
    """Maximum possible efficiency between two reservoirs."""
    if th <= 0.0 or tc <= 0.0 or tc >= th:
        raise ValueError("Need Th > Tc > 0")
    return 1.0 - tc / th


def actual_efficiency(case: EngineCase) -> float:
    """Measured/declared thermal efficiency from cycle energy data."""
    case.validate()
    return case.w_out / case.q_in


def classify_case(case: EngineCase, tol: float = 1e-10) -> str:
    """Classify one case against Carnot theorem constraints."""
    eta = actual_efficiency(case)
    eta_c = carnot_efficiency(case.th, case.tc)
    delta = eta - eta_c

    if delta > tol:
        return "violates_carnot_bound"
    if case.claimed_reversible and abs(delta) <= tol:
        return "reversible_consistent"
    if case.claimed_reversible and delta < -tol:
        return "reversible_claim_inconsistent"
    return "irreversible_allowed"


def evaluate_case(case: EngineCase, tol: float = 1e-10) -> dict[str, float | bool | str]:
    """Return derived metrics for reporting and checks."""
    eta = actual_efficiency(case)
    eta_c = carnot_efficiency(case.th, case.tc)
    q_out = case.q_in - case.w_out
    return {
        "name": case.name,
        "Th(K)": float(case.th),
        "Tc(K)": float(case.tc),
        "q_in": float(case.q_in),
        "w_out": float(case.w_out),
        "q_out": float(q_out),
        "eta_actual": float(eta),
        "eta_carnot": float(eta_c),
        "eta_margin(eta_carnot-eta)": float(eta_c - eta),
        "claimed_reversible": bool(case.claimed_reversible),
        "classification": classify_case(case, tol=tol),
    }


def build_demo_cases() -> list[EngineCase]:
    """Create deterministic examples for each logic branch."""
    return [
        # Two reversible engines with same reservoirs -> same eta_carnot.
        EngineCase(
            name="reversible_A_600_300",
            th=600.0,
            tc=300.0,
            q_in=1000.0,
            w_out=500.0,
            claimed_reversible=True,
        ),
        EngineCase(
            name="reversible_B_600_300",
            th=600.0,
            tc=300.0,
            q_in=1500.0,
            w_out=750.0,
            claimed_reversible=True,
        ),
        # Physical irreversible engine (below Carnot limit).
        EngineCase(
            name="irreversible_allowed_600_300",
            th=600.0,
            tc=300.0,
            q_in=1000.0,
            w_out=420.0,
            claimed_reversible=False,
        ),
        # Intentionally impossible: above Carnot limit.
        EngineCase(
            name="forbidden_above_carnot_600_300",
            th=600.0,
            tc=300.0,
            q_in=1000.0,
            w_out=560.0,
            claimed_reversible=False,
        ),
    ]


def main() -> None:
    tol = 1e-10
    cases = build_demo_cases()
    rows = [evaluate_case(c, tol=tol) for c in cases]
    table = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Carnot Theorem MVP")
    print("Efficiency bound: eta_actual <= eta_carnot = 1 - Tc/Th")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    row_a = table.loc[table["name"] == "reversible_A_600_300"].iloc[0]
    row_b = table.loc[table["name"] == "reversible_B_600_300"].iloc[0]
    row_irr = table.loc[table["name"] == "irreversible_allowed_600_300"].iloc[0]
    row_bad = table.loc[table["name"] == "forbidden_above_carnot_600_300"].iloc[0]

    # Theorem branch 1: reversible engines between same reservoirs share same eta.
    assert np.isclose(float(row_a["eta_actual"]), float(row_b["eta_actual"]), atol=tol)
    assert np.isclose(float(row_a["eta_actual"]), float(row_a["eta_carnot"]), atol=tol)
    assert np.isclose(float(row_b["eta_actual"]), float(row_b["eta_carnot"]), atol=tol)

    # Theorem branch 2: any irreversible engine must lie below the Carnot limit.
    assert float(row_irr["eta_actual"]) < float(row_irr["eta_carnot"]) - 1e-6

    # Forbidden branch: greater-than-Carnot efficiencies are impossible.
    assert float(row_bad["eta_actual"]) > float(row_bad["eta_carnot"]) + 1e-6

    assert str(row_a["classification"]) == "reversible_consistent"
    assert str(row_b["classification"]) == "reversible_consistent"
    assert str(row_irr["classification"]) == "irreversible_allowed"
    assert str(row_bad["classification"]) == "violates_carnot_bound"

    print("All checks passed.")


if __name__ == "__main__":
    main()
