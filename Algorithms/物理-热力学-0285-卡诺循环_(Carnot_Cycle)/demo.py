"""Minimal runnable MVP for Carnot Cycle.

The script constructs an ideal-gas reversible Carnot cycle:
1 -> 2: isothermal expansion at Th
2 -> 3: adiabatic expansion   Th -> Tc
3 -> 4: isothermal compression at Tc
4 -> 1: adiabatic compression Tc -> Th

No black-box thermodynamics solver is used. All state and energy terms are
computed from explicit textbook equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class CarnotCase:
    """Input parameters for one Carnot-cycle configuration."""

    name: str
    th: float  # hot-reservoir temperature (K)
    tc: float  # cold-reservoir temperature (K)
    gamma: float  # heat capacity ratio Cp/Cv (> 1)
    n_mol: float  # amount of substance (mol)
    v1: float  # state-1 volume (m^3)
    r_iso: float  # isothermal expansion ratio V2/V1 (> 1)

    def validate(self) -> None:
        if self.th <= 0.0 or self.tc <= 0.0:
            raise ValueError(f"{self.name}: temperatures must be positive")
        if self.tc >= self.th:
            raise ValueError(f"{self.name}: require Tc < Th, got Tc={self.tc}, Th={self.th}")
        if self.gamma <= 1.0:
            raise ValueError(f"{self.name}: gamma must be > 1")
        if self.n_mol <= 0.0:
            raise ValueError(f"{self.name}: n_mol must be positive")
        if self.v1 <= 0.0:
            raise ValueError(f"{self.name}: v1 must be positive")
        if self.r_iso <= 1.0:
            raise ValueError(f"{self.name}: r_iso must be > 1")


@dataclass(frozen=True)
class StatePoint:
    """Thermodynamic state point of ideal gas."""

    label: str
    t: float
    v: float
    p: float


def ideal_gas_pressure(n_mol: float, t: float, v: float) -> float:
    """Compute pressure via ideal-gas equation pV = nRT."""
    if n_mol <= 0.0 or t <= 0.0 or v <= 0.0:
        raise ValueError("Need n_mol > 0, t > 0, v > 0")
    return n_mol * R_UNIVERSAL * t / v


def adiabatic_end_volume(v_start: float, t_start: float, t_end: float, gamma: float) -> float:
    """For reversible adiabatic process: T * V^(gamma-1) = const."""
    if v_start <= 0.0 or t_start <= 0.0 or t_end <= 0.0:
        raise ValueError("Need positive volume and temperatures")
    if gamma <= 1.0:
        raise ValueError("Need gamma > 1")
    exponent = 1.0 / (gamma - 1.0)
    return v_start * (t_start / t_end) ** exponent


def build_cycle_states(case: CarnotCase) -> dict[str, StatePoint]:
    """Compute the four state points of one ideal-gas Carnot cycle."""
    case.validate()

    v1 = case.v1
    v2 = case.r_iso * v1
    v3 = adiabatic_end_volume(v2, case.th, case.tc, case.gamma)
    v4 = adiabatic_end_volume(v1, case.th, case.tc, case.gamma)

    t1 = case.th
    t2 = case.th
    t3 = case.tc
    t4 = case.tc

    p1 = ideal_gas_pressure(case.n_mol, t1, v1)
    p2 = ideal_gas_pressure(case.n_mol, t2, v2)
    p3 = ideal_gas_pressure(case.n_mol, t3, v3)
    p4 = ideal_gas_pressure(case.n_mol, t4, v4)

    return {
        "1": StatePoint(label="1", t=t1, v=v1, p=p1),
        "2": StatePoint(label="2", t=t2, v=v2, p=p2),
        "3": StatePoint(label="3", t=t3, v=v3, p=p3),
        "4": StatePoint(label="4", t=t4, v=v4, p=p4),
    }


def evaluate_carnot_cycle(case: CarnotCase) -> dict[str, float | str]:
    """Compute heat/work/efficiency metrics of one reversible Carnot cycle."""
    states = build_cycle_states(case)

    v1 = states["1"].v
    v2 = states["2"].v
    v3 = states["3"].v
    v4 = states["4"].v

    # Isothermal heat transfer for ideal gas reversible process.
    q_hot = case.n_mol * R_UNIVERSAL * case.th * math.log(v2 / v1)
    q_cold_mag = case.n_mol * R_UNIVERSAL * case.tc * math.log(v3 / v4)

    w_net = q_hot - q_cold_mag
    eta_actual = w_net / q_hot
    eta_carnot = 1.0 - case.tc / case.th
    eta_gap = eta_actual - eta_carnot

    # Clausius integral for one reversible cycle should be zero.
    clausius_integral = q_hot / case.th - q_cold_mag / case.tc
    entropy_generation = -clausius_integral

    return {
        "case": case.name,
        "Th(K)": case.th,
        "Tc(K)": case.tc,
        "gamma": case.gamma,
        "n(mol)": case.n_mol,
        "r_iso(V2/V1)": case.r_iso,
        "Q_hot(J)": q_hot,
        "Q_cold_mag(J)": q_cold_mag,
        "W_net(J)": w_net,
        "eta_actual": eta_actual,
        "eta_carnot": eta_carnot,
        "eta_gap(actual-carnot)": eta_gap,
        "clausius_integral": clausius_integral,
        "entropy_generation": entropy_generation,
    }


def states_to_frame(case: CarnotCase) -> pd.DataFrame:
    """Return a table of state points and two process invariants."""
    s = build_cycle_states(case)
    rows = [
        {"case": case.name, "state": "1", "T(K)": s["1"].t, "V(m^3)": s["1"].v, "P(Pa)": s["1"].p},
        {"case": case.name, "state": "2", "T(K)": s["2"].t, "V(m^3)": s["2"].v, "P(Pa)": s["2"].p},
        {"case": case.name, "state": "3", "T(K)": s["3"].t, "V(m^3)": s["3"].v, "P(Pa)": s["3"].p},
        {"case": case.name, "state": "4", "T(K)": s["4"].t, "V(m^3)": s["4"].v, "P(Pa)": s["4"].p},
    ]
    frame = pd.DataFrame(rows)

    # In ideal reversible Carnot cycle: V3/V4 = V2/V1 = r_iso.
    frame["V_ratio_check"] = np.nan
    frame.loc[0, "V_ratio_check"] = (s["2"].v / s["1"].v) - case.r_iso
    frame.loc[1, "V_ratio_check"] = (s["3"].v / s["4"].v) - case.r_iso
    return frame


def build_demo_cases() -> list[CarnotCase]:
    """Create deterministic cases to verify core Carnot-cycle properties."""
    return [
        CarnotCase(
            name="air_like_r2",
            th=600.0,
            tc=300.0,
            gamma=1.4,
            n_mol=1.0,
            v1=0.010,
            r_iso=2.0,
        ),
        CarnotCase(
            name="air_like_r35",
            th=600.0,
            tc=300.0,
            gamma=1.4,
            n_mol=1.0,
            v1=0.010,
            r_iso=3.5,
        ),
        CarnotCase(
            name="monoatomic_like_r25",
            th=600.0,
            tc=300.0,
            gamma=1.67,
            n_mol=0.8,
            v1=0.012,
            r_iso=2.5,
        ),
        CarnotCase(
            name="small_deltaT_reference",
            th=700.0,
            tc=500.0,
            gamma=1.4,
            n_mol=1.2,
            v1=0.009,
            r_iso=2.2,
        ),
    ]


def main() -> None:
    tol = 1e-10
    cases = build_demo_cases()

    metric_rows = [evaluate_carnot_cycle(c) for c in cases]
    metrics = pd.DataFrame(metric_rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Carnot Cycle MVP (ideal gas, reversible)")
    print("Expected efficiency: eta = 1 - Tc/Th")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Print one compact state table for each case.
    for c in cases:
        print(f"\nState points for case: {c.name}")
        print(states_to_frame(c).to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # Core Carnot equalities for each case.
    for row in metric_rows:
        assert np.isclose(row["eta_actual"], row["eta_carnot"], atol=tol), (
            f"Efficiency mismatch in {row['case']}: "
            f"{row['eta_actual']} vs {row['eta_carnot']}"
        )
        assert abs(row["clausius_integral"]) <= tol, (
            f"Reversible cycle should satisfy Clausius equality: {row['case']}"
        )
        assert row["Q_hot(J)"] > 0.0
        assert row["Q_cold_mag(J)"] > 0.0
        assert row["W_net(J)"] > 0.0

    # Same reservoirs -> same efficiency, independent of r_iso and gamma.
    same_reservoir_rows = [
        r for r in metric_rows if np.isclose(r["Th(K)"], 600.0) and np.isclose(r["Tc(K)"], 300.0)
    ]
    eta_values = np.array([r["eta_actual"] for r in same_reservoir_rows], dtype=np.float64)
    assert np.max(np.abs(eta_values - eta_values[0])) <= tol
    assert np.isclose(eta_values[0], 0.5, atol=tol)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
