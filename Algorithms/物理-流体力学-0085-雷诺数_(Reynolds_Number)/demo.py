"""Minimal runnable MVP for Reynolds Number (Re) in fluid mechanics.

The demo computes Reynolds number for multiple deterministic flow cases,
classifies flow regimes with scenario-aware thresholds, and runs sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowCase:
    name: str
    scenario: str
    rho: float  # density [kg/m^3]
    velocity: float  # characteristic velocity [m/s]
    length: float  # characteristic length [m]
    mu: float  # dynamic viscosity [Pa*s]
    expected_regime: str


@dataclass
class ReynoldsRecord:
    name: str
    scenario: str
    rho: float
    mu: float
    nu: float
    velocity: float
    length: float
    reynolds: float
    regime: str


PIPE_LAMINAR_MAX = 2300.0
PIPE_TURBULENT_MIN = 4000.0
PLATE_TRANSITION_CRITICAL = 5.0e5


def reynolds_number_from_mu(rho: float, velocity: float, length: float, mu: float) -> float:
    """Compute Reynolds number via Re = rho * V * L / mu."""
    return (rho * velocity * length) / mu


def reynolds_number_from_nu(velocity: float, length: float, nu: float) -> float:
    """Compute Reynolds number via Re = V * L / nu."""
    return (velocity * length) / nu


def classify_reynolds(reynolds: float, scenario: str) -> str:
    """Classify flow regime using scenario-aware thresholds.

    - pipe: laminar / transitional / turbulent with canonical 2300/4000 limits
    - flat_plate: laminar_boundary_layer / turbulent_boundary_layer with Rex=5e5
    - fallback: broad laminar/transitional/turbulent bands
    """
    if scenario == "pipe":
        if reynolds < PIPE_LAMINAR_MAX:
            return "laminar"
        if reynolds <= PIPE_TURBULENT_MIN:
            return "transitional"
        return "turbulent"

    if scenario == "flat_plate":
        if reynolds < PLATE_TRANSITION_CRITICAL:
            return "laminar_boundary_layer"
        return "turbulent_boundary_layer"

    if reynolds < 1.0:
        return "creeping"
    if reynolds < PIPE_LAMINAR_MAX:
        return "laminar"
    if reynolds <= PIPE_TURBULENT_MIN:
        return "transitional"
    return "turbulent"


def validate_case(case: FlowCase) -> None:
    if case.rho <= 0.0:
        raise ValueError(f"{case.name}: rho must be positive.")
    if case.velocity <= 0.0:
        raise ValueError(f"{case.name}: velocity must be positive.")
    if case.length <= 0.0:
        raise ValueError(f"{case.name}: length must be positive.")
    if case.mu <= 0.0:
        raise ValueError(f"{case.name}: mu must be positive.")


def evaluate_case(case: FlowCase) -> ReynoldsRecord:
    validate_case(case)

    nu = case.mu / case.rho
    re_mu = reynolds_number_from_mu(case.rho, case.velocity, case.length, case.mu)
    re_nu = reynolds_number_from_nu(case.velocity, case.length, nu)

    # Both formulas are mathematically equivalent; any mismatch comes from floating error.
    if not np.isclose(re_mu, re_nu, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"Formula mismatch in {case.name}: via mu={re_mu:.6g}, via nu={re_nu:.6g}"
        )

    regime = classify_reynolds(re_mu, case.scenario)
    return ReynoldsRecord(
        name=case.name,
        scenario=case.scenario,
        rho=case.rho,
        mu=case.mu,
        nu=nu,
        velocity=case.velocity,
        length=case.length,
        reynolds=float(re_mu),
        regime=regime,
    )


def build_demo_cases() -> list[FlowCase]:
    """Deterministic benchmark cases spanning laminar/transitional/turbulent ranges."""
    return [
        FlowCase(
            name="Glycerin Microchannel",
            scenario="pipe",
            rho=1260.0,
            velocity=0.03,
            length=0.002,
            mu=1.49,
            expected_regime="laminar",
        ),
        FlowCase(
            name="Water Pipe - Low Speed",
            scenario="pipe",
            rho=998.0,
            velocity=0.02,
            length=0.05,
            mu=1.002e-3,
            expected_regime="laminar",
        ),
        FlowCase(
            name="Water Pipe - Mid Speed",
            scenario="pipe",
            rho=998.0,
            velocity=0.06,
            length=0.05,
            mu=1.002e-3,
            expected_regime="transitional",
        ),
        FlowCase(
            name="Water Pipe - High Speed",
            scenario="pipe",
            rho=998.0,
            velocity=0.12,
            length=0.05,
            mu=1.002e-3,
            expected_regime="turbulent",
        ),
        FlowCase(
            name="Air Duct",
            scenario="pipe",
            rho=1.225,
            velocity=6.0,
            length=0.30,
            mu=1.80e-5,
            expected_regime="turbulent",
        ),
        FlowCase(
            name="Flat Plate - Short Run",
            scenario="flat_plate",
            rho=1.225,
            velocity=5.0,
            length=0.50,
            mu=1.80e-5,
            expected_regime="laminar_boundary_layer",
        ),
        FlowCase(
            name="Flat Plate - Long Run",
            scenario="flat_plate",
            rho=1.225,
            velocity=8.0,
            length=2.00,
            mu=1.80e-5,
            expected_regime="turbulent_boundary_layer",
        ),
    ]


def run_checks(cases: list[FlowCase], records: list[ReynoldsRecord]) -> None:
    if len(cases) != len(records):
        raise AssertionError("Case count and record count mismatch.")

    re_values = np.array([r.reynolds for r in records], dtype=float)
    if not np.all(np.isfinite(re_values)):
        raise AssertionError("Reynolds values must be finite.")
    if not np.all(re_values > 0.0):
        raise AssertionError("Reynolds values must be positive.")

    for case, record in zip(cases, records):
        if case.expected_regime != record.regime:
            raise AssertionError(
                f"Regime mismatch for {case.name}: expected={case.expected_regime}, got={record.regime}"
            )

    # Monotonic check under same fluid/geometry: higher velocity should yield higher Re.
    water_records = [r for r in records if r.name.startswith("Water Pipe")]
    water_re = np.array([r.reynolds for r in water_records], dtype=float)
    if not np.all(np.diff(water_re) > 0.0):
        raise AssertionError("Water pipe Reynolds numbers should increase with speed.")


def records_to_dataframe(records: list[ReynoldsRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [r.name for r in records],
            "scenario": [r.scenario for r in records],
            "rho_kg_m3": [r.rho for r in records],
            "mu_Pa_s": [r.mu for r in records],
            "nu_m2_s": [r.nu for r in records],
            "V_m_s": [r.velocity for r in records],
            "L_m": [r.length for r in records],
            "Re": [r.reynolds for r in records],
            "regime": [r.regime for r in records],
        }
    )


def main() -> None:
    cases = build_demo_cases()
    records = [evaluate_case(case) for case in cases]
    run_checks(cases, records)

    df = records_to_dataframe(records)

    print("Reynolds Number MVP Demo")
    print("formula: Re = rho * V * L / mu = V * L / nu")
    print()
    print(
        df.to_string(
            index=False,
            formatters={
                "rho_kg_m3": lambda x: f"{x:.3f}",
                "mu_Pa_s": lambda x: f"{x:.3e}",
                "nu_m2_s": lambda x: f"{x:.3e}",
                "V_m_s": lambda x: f"{x:.3f}",
                "L_m": lambda x: f"{x:.3f}",
                "Re": lambda x: f"{x:.2f}",
            },
        )
    )

    summary = df.groupby("regime", as_index=False).agg(
        count=("Re", "size"),
        mean_re=("Re", "mean"),
        min_re=("Re", "min"),
        max_re=("Re", "max"),
    )

    print("\nRegime summary")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_re": lambda x: f"{x:.2f}",
                "min_re": lambda x: f"{x:.2f}",
                "max_re": lambda x: f"{x:.2f}",
            },
        )
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
