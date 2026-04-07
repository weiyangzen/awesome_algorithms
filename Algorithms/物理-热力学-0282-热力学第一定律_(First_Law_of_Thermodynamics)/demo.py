"""Minimal runnable MVP for First Law of Thermodynamics.

Sign convention used in this script:
- Q_in > 0: heat enters the system
- W_by > 0: work done by the system

First law for a closed system:
    delta_U = Q_in - W_by
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class ThermoState:
    """State of an ideal-gas closed system."""

    t: float  # temperature in K
    v: float  # volume in m^3
    n_mol: float  # amount of substance in mol

    def validate(self) -> None:
        if self.t <= 0.0:
            raise ValueError(f"Temperature must be positive, got {self.t}")
        if self.v <= 0.0:
            raise ValueError(f"Volume must be positive, got {self.v}")
        if self.n_mol <= 0.0:
            raise ValueError(f"n_mol must be positive, got {self.n_mol}")


@dataclass(frozen=True)
class ProcessResult:
    """One thermodynamic process segment with explicit energy accounting."""

    name: str
    state_in: ThermoState
    state_out: ThermoState
    q_in: float
    w_by: float
    delta_u: float

    def residual(self) -> float:
        """Return delta_U - (Q_in - W_by); should be near zero."""
        return self.delta_u - (self.q_in - self.w_by)


def ideal_pressure(state: ThermoState) -> float:
    """Ideal-gas pressure from pV = nRT."""
    state.validate()
    return state.n_mol * R_UNIVERSAL * state.t / state.v


def step_isochoric(state: ThermoState, t_out: float, cv_molar: float) -> ProcessResult:
    """Isochoric process: dV = 0 => W_by = 0."""
    state.validate()
    if t_out <= 0.0:
        raise ValueError("t_out must be positive")
    if cv_molar <= 0.0:
        raise ValueError("cv_molar must be positive")

    out = ThermoState(t=t_out, v=state.v, n_mol=state.n_mol)
    d_t = out.t - state.t
    delta_u = state.n_mol * cv_molar * d_t
    w_by = 0.0
    q_in = delta_u
    return ProcessResult(
        name="isochoric",
        state_in=state,
        state_out=out,
        q_in=q_in,
        w_by=w_by,
        delta_u=delta_u,
    )


def step_isobaric(state: ThermoState, t_out: float, cv_molar: float) -> ProcessResult:
    """Isobaric process: p = const => W_by = p * (V_out - V_in)."""
    state.validate()
    if t_out <= 0.0:
        raise ValueError("t_out must be positive")
    if cv_molar <= 0.0:
        raise ValueError("cv_molar must be positive")

    p_const = ideal_pressure(state)
    v_out = state.n_mol * R_UNIVERSAL * t_out / p_const
    out = ThermoState(t=t_out, v=v_out, n_mol=state.n_mol)

    d_t = out.t - state.t
    delta_u = state.n_mol * cv_molar * d_t
    w_by = p_const * (out.v - state.v)
    q_in = delta_u + w_by
    return ProcessResult(
        name="isobaric",
        state_in=state,
        state_out=out,
        q_in=q_in,
        w_by=w_by,
        delta_u=delta_u,
    )


def step_isothermal(state: ThermoState, v_out: float) -> ProcessResult:
    """Isothermal ideal-gas process: delta_U = 0, W_by = nRT ln(V_out/V_in)."""
    state.validate()
    if v_out <= 0.0:
        raise ValueError("v_out must be positive")

    out = ThermoState(t=state.t, v=v_out, n_mol=state.n_mol)
    w_by = state.n_mol * R_UNIVERSAL * state.t * math.log(out.v / state.v)
    delta_u = 0.0
    q_in = w_by
    return ProcessResult(
        name="isothermal",
        state_in=state,
        state_out=out,
        q_in=q_in,
        w_by=w_by,
        delta_u=delta_u,
    )


def numerical_isothermal_work(state: ThermoState, v_out: float, n_steps: int = 4000) -> float:
    """Numerically integrate p(V) dV for an isothermal ideal-gas segment."""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    state.validate()
    if v_out <= 0.0:
        raise ValueError("v_out must be positive")

    v_grid = np.linspace(state.v, v_out, n_steps, dtype=np.float64)
    p_grid = state.n_mol * R_UNIVERSAL * state.t / v_grid
    return float(np.trapezoid(p_grid, v_grid))


def build_path_a(initial: ThermoState, cv_molar: float) -> list[ProcessResult]:
    """Path A: isochoric heating -> isobaric expansion -> isothermal compression."""
    r1 = step_isochoric(initial, t_out=420.0, cv_molar=cv_molar)
    r2 = step_isobaric(r1.state_out, t_out=560.0, cv_molar=cv_molar)
    r3 = step_isothermal(r2.state_out, v_out=0.0095)
    return [r1, r2, r3]


def build_path_b(initial: ThermoState, cv_molar: float) -> list[ProcessResult]:
    """Path B: direct isochoric heating -> isothermal compression."""
    r1 = step_isochoric(initial, t_out=560.0, cv_molar=cv_molar)
    r2 = step_isothermal(r1.state_out, v_out=0.0095)
    return [r1, r2]


def path_to_frame(path_name: str, path: list[ProcessResult]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for idx, seg in enumerate(path, start=1):
        rows.append(
            {
                "path": path_name,
                "step": idx,
                "process": seg.name,
                "T_in(K)": seg.state_in.t,
                "T_out(K)": seg.state_out.t,
                "V_in(m^3)": seg.state_in.v,
                "V_out(m^3)": seg.state_out.v,
                "Q_in(J)": seg.q_in,
                "W_by(J)": seg.w_by,
                "delta_U(J)": seg.delta_u,
                "residual": seg.residual(),
            }
        )
    return pd.DataFrame(rows)


def summarize_path(path_name: str, path: list[ProcessResult]) -> dict[str, float | str]:
    q_total = float(sum(seg.q_in for seg in path))
    w_total = float(sum(seg.w_by for seg in path))
    du_total = float(sum(seg.delta_u for seg in path))
    start = path[0].state_in
    end = path[-1].state_out
    return {
        "path": path_name,
        "T_start(K)": start.t,
        "T_end(K)": end.t,
        "V_start(m^3)": start.v,
        "V_end(m^3)": end.v,
        "Q_total(J)": q_total,
        "W_total(J)": w_total,
        "delta_U_total(J)": du_total,
        "residual_total": du_total - (q_total - w_total),
    }


def main() -> None:
    tol = 1e-9
    cv_molar = 2.5 * R_UNIVERSAL  # constant Cv approximation for a diatomic-like gas
    initial = ThermoState(t=300.0, v=0.010, n_mol=1.0)

    path_a = build_path_a(initial, cv_molar=cv_molar)
    path_b = build_path_b(initial, cv_molar=cv_molar)

    segment_frame = pd.concat(
        [path_to_frame("A", path_a), path_to_frame("B", path_b)],
        ignore_index=True,
    )
    summary_frame = pd.DataFrame(
        [summarize_path("A", path_a), summarize_path("B", path_b)]
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("First Law of Thermodynamics MVP")
    print("Sign convention: delta_U = Q_in - W_by")
    print("\nSegment-level accounting:")
    print(segment_frame.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    print("\nPath totals:")
    print(summary_frame.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # 1) First-law residuals should be numerically near zero for every segment/path.
    for seg in list(path_a) + list(path_b):
        assert abs(seg.residual()) <= tol, f"Segment residual too large: {seg}"
    assert np.all(np.abs(summary_frame["residual_total"].to_numpy(dtype=np.float64)) <= tol)

    # 2) Path-independence of internal energy for ideal gas: delta_U depends only on delta_T.
    end_a = path_a[-1].state_out
    end_b = path_b[-1].state_out
    assert np.isclose(end_a.t, end_b.t, atol=tol)
    assert np.isclose(end_a.v, end_b.v, atol=tol)

    expected_du = initial.n_mol * cv_molar * (end_a.t - initial.t)
    du_a = summary_frame.loc[summary_frame["path"] == "A", "delta_U_total(J)"].iloc[0]
    du_b = summary_frame.loc[summary_frame["path"] == "B", "delta_U_total(J)"].iloc[0]
    assert np.isclose(du_a, expected_du, atol=tol)
    assert np.isclose(du_b, expected_du, atol=tol)

    # 3) Same endpoints but different path => Q and W can differ.
    q_a = summary_frame.loc[summary_frame["path"] == "A", "Q_total(J)"].iloc[0]
    q_b = summary_frame.loc[summary_frame["path"] == "B", "Q_total(J)"].iloc[0]
    w_a = summary_frame.loc[summary_frame["path"] == "A", "W_total(J)"].iloc[0]
    w_b = summary_frame.loc[summary_frame["path"] == "B", "W_total(J)"].iloc[0]
    assert not np.isclose(q_a, q_b, atol=1e-6)
    assert not np.isclose(w_a, w_b, atol=1e-6)

    # 4) Validate analytic isothermal work using numerical integral.
    for seg in list(path_a) + list(path_b):
        if seg.name == "isothermal":
            w_num = numerical_isothermal_work(seg.state_in, seg.state_out.v, n_steps=8000)
            assert np.isclose(w_num, seg.w_by, atol=2e-4), (
                f"Isothermal work mismatch: analytic={seg.w_by}, numeric={w_num}"
            )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
