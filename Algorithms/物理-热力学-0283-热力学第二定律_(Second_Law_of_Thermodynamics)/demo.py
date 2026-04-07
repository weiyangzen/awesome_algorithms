"""Minimal runnable MVP for the Second Law of Thermodynamics.

Sign convention used for a cyclic heat engine in this script:
- Qh > 0: heat absorbed from the hot reservoir into the engine
- Qc > 0: heat rejected from the engine to the cold reservoir
- W_out > 0: work output from the engine

For a cycle, the working fluid returns to its initial state, so:
    Delta S_engine = 0

Second-law checks used here:
1) Delta S_universe = -Qh/Th + Qc/Tc >= 0
2) Clausius inequality for cycle: integral(deltaQ/T) = Qh/Th - Qc/Tc <= 0
3) Efficiency bound: eta <= eta_carnot = 1 - Tc/Th
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class HeatEngineCase:
    """One heat-engine claim under two-reservoir assumptions."""

    name: str
    t_hot: float  # K
    t_cold: float  # K
    q_hot: float  # J
    eta_actual: float  # W_out / Qh

    def validate(self) -> None:
        if self.t_hot <= 0.0:
            raise ValueError(f"t_hot must be positive, got {self.t_hot}")
        if self.t_cold <= 0.0:
            raise ValueError(f"t_cold must be positive, got {self.t_cold}")
        if self.t_cold >= self.t_hot:
            raise ValueError(
                f"Require t_cold < t_hot for heat-engine mode, got {self.t_cold} >= {self.t_hot}"
            )
        if self.q_hot <= 0.0:
            raise ValueError(f"q_hot must be positive, got {self.q_hot}")
        if self.eta_actual < 0.0:
            raise ValueError(f"eta_actual must be non-negative, got {self.eta_actual}")


def carnot_efficiency(t_hot: float, t_cold: float) -> float:
    """Analytical Carnot efficiency bound for a two-reservoir heat engine."""
    if t_hot <= 0.0 or t_cold <= 0.0:
        raise ValueError("Temperatures must be positive")
    if t_cold >= t_hot:
        raise ValueError("Require t_cold < t_hot")
    return 1.0 - (t_cold / t_hot)


def work_output(q_hot: float, eta: float) -> float:
    return q_hot * eta


def heat_rejected(q_hot: float, eta: float) -> float:
    return q_hot * (1.0 - eta)


def entropy_generation_universe(case: HeatEngineCase) -> float:
    """Compute Delta S_universe for one full engine cycle."""
    case.validate()
    q_cold = heat_rejected(case.q_hot, case.eta_actual)
    return (-case.q_hot / case.t_hot) + (q_cold / case.t_cold)


def clausius_cycle_integral(case: HeatEngineCase) -> float:
    """Compute cyclic integral of deltaQ/T for the working fluid."""
    case.validate()
    q_cold = heat_rejected(case.q_hot, case.eta_actual)
    return (case.q_hot / case.t_hot) - (q_cold / case.t_cold)


def max_efficiency_from_entropy_balance(t_hot: float, t_cold: float) -> float:
    """Solve Delta S_universe(eta) = 0 numerically for eta in [0, 1)."""
    if t_hot <= 0.0 or t_cold <= 0.0:
        raise ValueError("Temperatures must be positive")
    if t_cold >= t_hot:
        raise ValueError("Require t_cold < t_hot")

    def objective(eta: float) -> float:
        # q_hot cancels, so use q_hot = 1 for conditioning clarity.
        return (-1.0 / t_hot) + ((1.0 - eta) / t_cold)

    # objective(0) > 0 and objective(1-) < 0, guaranteeing one root.
    return float(brentq(objective, 0.0, 1.0 - 1e-12, xtol=1e-14, rtol=1e-12, maxiter=200))


def evaluate_case(case: HeatEngineCase, tol: float = 1e-10) -> dict[str, float | str | bool]:
    case.validate()

    eta_carnot = carnot_efficiency(case.t_hot, case.t_cold)
    eta_limit_numeric = max_efficiency_from_entropy_balance(case.t_hot, case.t_cold)
    w_out = work_output(case.q_hot, case.eta_actual)
    q_cold = heat_rejected(case.q_hot, case.eta_actual)

    ds_universe = entropy_generation_universe(case)
    clausius_val = clausius_cycle_integral(case)

    second_law_ok_entropy = ds_universe >= -tol
    second_law_ok_eff = case.eta_actual <= eta_carnot + tol

    return {
        "case": case.name,
        "Th(K)": case.t_hot,
        "Tc(K)": case.t_cold,
        "Qh(J)": case.q_hot,
        "eta_actual": case.eta_actual,
        "eta_carnot": eta_carnot,
        "eta_limit_numeric": eta_limit_numeric,
        "W_out(J)": w_out,
        "Qc(J)": q_cold,
        "DeltaS_universe(J/K)": ds_universe,
        "CycleIntegral_deltaQ_over_T": clausius_val,
        "second_law_ok_entropy": second_law_ok_entropy,
        "second_law_ok_eff": second_law_ok_eff,
        "second_law_ok": bool(second_law_ok_entropy and second_law_ok_eff),
        "eta_gap_to_carnot": eta_carnot - case.eta_actual,
    }


def main() -> None:
    tol = 1e-10

    cases = [
        HeatEngineCase(
            name="reversible_reference",
            t_hot=600.0,
            t_cold=300.0,
            q_hot=1200.0,
            eta_actual=0.50,
        ),
        HeatEngineCase(
            name="irreversible_but_physical",
            t_hot=600.0,
            t_cold=300.0,
            q_hot=1200.0,
            eta_actual=0.35,
        ),
        HeatEngineCase(
            name="super_carnot_claim",
            t_hot=600.0,
            t_cold=300.0,
            q_hot=1200.0,
            eta_actual=0.62,
        ),
    ]

    rows = [evaluate_case(case, tol=tol) for case in cases]
    frame = pd.DataFrame(rows)

    # A tiny sweep to show DeltaS_universe decreases monotonically with eta.
    eta_grid = np.linspace(0.0, 0.95, 80, dtype=np.float64)
    t_hot = 600.0
    t_cold = 300.0
    ds_grid = (-1.0 / t_hot) + ((1.0 - eta_grid) / t_cold)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Second Law of Thermodynamics MVP")
    print("Checks: DeltaS_universe >= 0, integral(deltaQ/T) <= 0, eta <= eta_carnot")
    print("\nCase summary:")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # 1) Clausius integral and universe entropy should be exact negatives.
    diff = frame["CycleIntegral_deltaQ_over_T"].to_numpy(dtype=np.float64) + frame[
        "DeltaS_universe(J/K)"
    ].to_numpy(dtype=np.float64)
    assert np.all(np.abs(diff) <= 1e-12), "Clausius-vs-entropy identity broken"

    # 2) Numeric root and analytical Carnot limit should agree.
    eta_carnot_vals = frame["eta_carnot"].to_numpy(dtype=np.float64)
    eta_numeric_vals = frame["eta_limit_numeric"].to_numpy(dtype=np.float64)
    assert np.allclose(eta_carnot_vals, eta_numeric_vals, atol=1e-12), (
        "Numerical entropy-balance root differs from Carnot formula"
    )

    # 3) Reversible reference: entropy generation ~ 0.
    ds_rev = float(
        frame.loc[frame["case"] == "reversible_reference", "DeltaS_universe(J/K)"].iloc[0]
    )
    assert abs(ds_rev) <= tol

    # 4) Irreversible physical case: positive entropy generation.
    ds_irrev = float(
        frame.loc[
            frame["case"] == "irreversible_but_physical", "DeltaS_universe(J/K)"
        ].iloc[0]
    )
    assert ds_irrev > tol

    # 5) Super-Carnot claim: violates second law (negative entropy generation).
    ds_violation = float(
        frame.loc[frame["case"] == "super_carnot_claim", "DeltaS_universe(J/K)"].iloc[0]
    )
    assert ds_violation < -tol
    is_marked_invalid = bool(
        frame.loc[frame["case"] == "super_carnot_claim", "second_law_ok"].iloc[0]
    )
    assert not is_marked_invalid

    # 6) Entropy generation should be strictly decreasing with eta on fixed (Th, Tc).
    assert np.all(np.diff(ds_grid) < 0.0)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
