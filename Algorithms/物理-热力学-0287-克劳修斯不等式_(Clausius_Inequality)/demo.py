"""Minimal runnable MVP for Clausius Inequality.

We evaluate the cyclic Clausius inequality in discrete form:
    I = ∮ (delta Q / T_b) <= 0
where T_b is the boundary temperature at each heat-exchange step.

Sign convention in this script:
- Q > 0: heat enters the working system.
- Q < 0: heat leaves the working system.

For a cycle, entropy generation is
    S_gen = -I >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeatStep:
    """One heat-exchange step in a cycle."""

    segment: str
    q: float
    t_boundary: float


@dataclass(frozen=True)
class ClausiusCycle:
    """A full cycle represented by finite heat-exchange steps."""

    name: str
    steps: tuple[HeatStep, ...]

    def validate(self) -> None:
        if len(self.steps) == 0:
            raise ValueError("Cycle must contain at least one step")
        for i, step in enumerate(self.steps):
            if step.t_boundary <= 0.0:
                raise ValueError(
                    f"Step {i} has non-positive boundary temperature: {step.t_boundary}"
                )



def clausius_integral(cycle: ClausiusCycle) -> float:
    """Compute I = sum_i (Q_i / T_i), a discrete approximation of ∮δQ/T."""
    cycle.validate()
    q = np.array([s.q for s in cycle.steps], dtype=np.float64)
    t = np.array([s.t_boundary for s in cycle.steps], dtype=np.float64)
    return float(np.sum(q / t))



def entropy_generation(cycle: ClausiusCycle) -> float:
    """For a cycle, S_gen = -∮δQ/T."""
    return -clausius_integral(cycle)



def cycle_work(cycle: ClausiusCycle) -> float:
    """Net work output from first-law closure for a cycle: W = sum(Q_i)."""
    cycle.validate()
    return float(sum(s.q for s in cycle.steps))



def classify_cycle(cycle: ClausiusCycle, tol: float = 1e-9) -> str:
    """Classify cycle by Clausius inequality."""
    i_val = clausius_integral(cycle)
    if i_val > tol:
        return "violates_clausius"
    if abs(i_val) <= tol:
        return "reversible_limit"
    return "irreversible_allowed"



def evaluate_cycle(cycle: ClausiusCycle, tol: float = 1e-9) -> dict[str, float | str]:
    i_val = clausius_integral(cycle)
    s_gen = -i_val
    q_in = float(sum(s.q for s in cycle.steps if s.q > 0.0))
    q_out = float(sum(-s.q for s in cycle.steps if s.q < 0.0))
    work_out = cycle_work(cycle)

    eta = np.nan
    if q_in > 0.0:
        eta = work_out / q_in

    return {
        "cycle": cycle.name,
        "clausius_integral": i_val,
        "entropy_generation": s_gen,
        "q_in": q_in,
        "q_out": q_out,
        "work_out": work_out,
        "efficiency": float(eta),
        "classification": classify_cycle(cycle, tol=tol),
    }



def build_demo_cycles() -> list[ClausiusCycle]:
    """Create three cycles under the same reservoirs for comparison."""
    t_hot = 500.0
    t_cold = 300.0

    # Reversible reference: Qc = -Qh * Tc/Th -> integral exactly 0.
    q_hot_ref = 1200.0
    q_cold_rev = -q_hot_ref * t_cold / t_hot
    reversible = ClausiusCycle(
        name="reversible_carnot_like",
        steps=(
            HeatStep(segment="isothermal_hot", q=q_hot_ref, t_boundary=t_hot),
            HeatStep(segment="adiabatic_expand", q=0.0, t_boundary=430.0),
            HeatStep(segment="isothermal_cold", q=q_cold_rev, t_boundary=t_cold),
            HeatStep(segment="adiabatic_compress", q=0.0, t_boundary=360.0),
        ),
    )

    # Physical irreversible cycle: larger heat rejection to the cold side.
    irreversible = ClausiusCycle(
        name="irreversible_engine_like",
        steps=(
            HeatStep(segment="isothermal_hot", q=q_hot_ref, t_boundary=t_hot),
            HeatStep(segment="adiabatic_expand", q=0.0, t_boundary=430.0),
            HeatStep(segment="isothermal_cold", q=-800.0, t_boundary=t_cold),
            HeatStep(segment="adiabatic_compress", q=0.0, t_boundary=360.0),
        ),
    )

    # Intentionally unphysical: appears to exceed Clausius bound.
    violating = ClausiusCycle(
        name="forbidden_positive_integral",
        steps=(
            HeatStep(segment="isothermal_hot", q=q_hot_ref, t_boundary=t_hot),
            HeatStep(segment="adiabatic_expand", q=0.0, t_boundary=430.0),
            HeatStep(segment="isothermal_cold", q=-650.0, t_boundary=t_cold),
            HeatStep(segment="adiabatic_compress", q=0.0, t_boundary=360.0),
        ),
    )

    return [reversible, irreversible, violating]



def main() -> None:
    tol = 1e-10
    cycles = build_demo_cycles()
    rows = [evaluate_cycle(c, tol=tol) for c in cycles]
    table = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("Clausius Inequality MVP")
    print("Sign convention: Q>0 into system, Q<0 out of system")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    # Deterministic checks.
    row_rev = table.loc[table["cycle"] == "reversible_carnot_like"].iloc[0]
    row_irr = table.loc[table["cycle"] == "irreversible_engine_like"].iloc[0]
    row_bad = table.loc[table["cycle"] == "forbidden_positive_integral"].iloc[0]

    assert abs(float(row_rev["clausius_integral"])) <= tol, (
        "Reversible reference should satisfy equality: "
        f"I={float(row_rev['clausius_integral'])}"
    )
    assert float(row_irr["clausius_integral"]) < -1e-6, (
        "Irreversible cycle should satisfy strict inequality: "
        f"I={float(row_irr['clausius_integral'])}"
    )
    assert float(row_bad["clausius_integral"]) > 1e-6, (
        "Forbidden cycle should have positive integral: "
        f"I={float(row_bad['clausius_integral'])}"
    )

    assert str(row_rev["classification"]) == "reversible_limit"
    assert str(row_irr["classification"]) == "irreversible_allowed"
    assert str(row_bad["classification"]) == "violates_clausius"

    print("All checks passed.")


if __name__ == "__main__":
    main()
