"""Minimal runnable MVP for the planar circular Restricted Three-Body Problem (CR3BP).

Model:
- Two primaries with normalized masses (1-mu) and mu on circular orbits.
- Rotating frame with primaries fixed at x=-mu and x=1-mu.
- A third body with negligible mass evolves under the primaries' gravity.

This script integrates two deterministic test trajectories, then reports:
- Jacobi-constant drift
- minimum distance to each primary
- Poincare-style upward y=0 crossing count
- consistency of the zero-velocity relation

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class CR3BPConfig:
    mu: float = 0.0121505856
    t_final: float = 40.0
    samples: int = 4000
    rtol: float = 1e-10
    atol: float = 1e-12
    min_distance_guard: float = 0.02


@dataclass
class CaseResult:
    name: str
    t: np.ndarray
    states: np.ndarray  # shape (4, n)
    terminated_early: bool


def validate_config(config: CR3BPConfig) -> None:
    if not np.isfinite(config.mu):
        raise ValueError("mu must be finite.")
    if not (0.0 < config.mu < 0.5):
        raise ValueError(f"mu must be in (0, 0.5), got {config.mu}.")
    if config.t_final <= 0.0:
        raise ValueError("t_final must be > 0.")
    if config.samples < 2:
        raise ValueError("samples must be >= 2.")
    if config.rtol <= 0.0 or config.atol <= 0.0:
        raise ValueError("rtol and atol must be > 0.")
    if config.min_distance_guard <= 0.0:
        raise ValueError("min_distance_guard must be > 0.")


def validate_initial_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError(f"initial state must have shape (4,), got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial state contains non-finite values.")
    return state


def distances_to_primaries(x: np.ndarray | float, y: np.ndarray | float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1.0 + mu, y)
    return r1, r2


def effective_potential(x: np.ndarray | float, y: np.ndarray | float, mu: float) -> np.ndarray:
    r1, r2 = distances_to_primaries(x=x, y=y, mu=mu)
    return (1.0 - mu) / r1 + mu / r2 + 0.5 * (x * x + y * y)


def grad_effective_potential(x: float, y: float, mu: float) -> Tuple[float, float]:
    r1, r2 = distances_to_primaries(x=x, y=y, mu=mu)
    d_omega_dx = x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - 1.0 + mu) / (r2**3)
    d_omega_dy = y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    return float(d_omega_dx), float(d_omega_dy)


def planar_cr3bp_rhs(_t: float, state: np.ndarray, mu: float) -> np.ndarray:
    x, y, vx, vy = state
    d_omega_dx, d_omega_dy = grad_effective_potential(x=x, y=y, mu=mu)
    return np.array(
        [
            vx,
            vy,
            2.0 * vy + d_omega_dx,
            -2.0 * vx + d_omega_dy,
        ],
        dtype=float,
    )


def jacobi_constant(states: np.ndarray, mu: float) -> np.ndarray:
    x, y, vx, vy = states
    omega = effective_potential(x=x, y=y, mu=mu)
    return 2.0 * omega - (vx * vx + vy * vy)


def count_upward_y_crossings(states: np.ndarray) -> int:
    y = states[1]
    vy = states[3]
    if y.size < 2:
        return 0
    mask = (y[:-1] <= 0.0) & (y[1:] > 0.0) & (vy[1:] > 0.0)
    return int(np.count_nonzero(mask))


def make_collision_event(primary_x: float, guard_radius: float):
    def event(_t: float, state: np.ndarray) -> float:
        x, y = state[0], state[1]
        return float(np.hypot(x - primary_x, y) - guard_radius)

    event.terminal = True
    event.direction = -1.0
    return event


def propagate_case(name: str, initial_state: np.ndarray, config: CR3BPConfig) -> CaseResult:
    y0 = validate_initial_state(initial_state)

    t_eval = np.linspace(0.0, config.t_final, config.samples)
    event1 = make_collision_event(primary_x=-config.mu, guard_radius=config.min_distance_guard)
    event2 = make_collision_event(primary_x=1.0 - config.mu, guard_radius=config.min_distance_guard)

    sol = solve_ivp(
        fun=lambda t, s: planar_cr3bp_rhs(t, s, config.mu),
        t_span=(0.0, config.t_final),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        events=[event1, event2],
        rtol=config.rtol,
        atol=config.atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for case {name}: {sol.message}")

    terminated_early = any(evt.size > 0 for evt in sol.t_events)
    if sol.y.shape[1] == 0:
        raise RuntimeError(f"No trajectory points produced for case {name}.")

    return CaseResult(name=name, t=sol.t, states=sol.y, terminated_early=terminated_early)


def summarize_case(case: CaseResult, mu: float) -> Dict[str, float | int | str]:
    x, y, vx, vy = case.states
    speed = np.hypot(vx, vy)
    r1, r2 = distances_to_primaries(x=x, y=y, mu=mu)

    jacobi = jacobi_constant(case.states, mu=mu)
    c0 = float(jacobi[0])
    jacobi_drift = float(np.max(np.abs(jacobi - c0)))

    zvc_margin = 2.0 * effective_potential(x=x, y=y, mu=mu) - c0
    speed_sq = vx * vx + vy * vy
    kinetic_recon_error = float(np.max(np.abs(zvc_margin - speed_sq)))

    return {
        "case": case.name,
        "C0": c0,
        "max_jacobi_drift": jacobi_drift,
        "min_r1": float(np.min(r1)),
        "min_r2": float(np.min(r2)),
        "max_speed": float(np.max(speed)),
        "min_zvc_margin": float(np.min(zvc_margin)),
        "max_kinetic_recon_error": kinetic_recon_error,
        "upward_y_crossings": count_upward_y_crossings(case.states),
        "terminated_early": int(case.terminated_early),
        "steps": int(case.t.size),
    }


def sample_trajectory_table(case: CaseResult, mu: float, rows: int = 8) -> pd.DataFrame:
    n = case.t.size
    rows = min(rows, n)
    idx = np.linspace(0, n - 1, rows, dtype=int)

    states = case.states[:, idx]
    x, y, vx, vy = states
    r1, r2 = distances_to_primaries(x=x, y=y, mu=mu)
    jacobi = jacobi_constant(states=states, mu=mu)

    return pd.DataFrame(
        {
            "t": case.t[idx],
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "r1": r1,
            "r2": r2,
            "C": jacobi,
        }
    )


def run_demo(config: CR3BPConfig) -> Tuple[pd.DataFrame, List[CaseResult]]:
    x_l4 = 0.5 - config.mu
    y_l4 = np.sqrt(3.0) / 2.0

    cases = [
        (
            "Near-L4 tadpole-like",
            np.array([x_l4 - 0.025, y_l4, 0.0, 0.0], dtype=float),
        ),
        (
            "Inner-region transit-like",
            np.array([0.70, 0.0, 0.0, 0.12], dtype=float),
        ),
    ]

    trajectories: List[CaseResult] = []
    summaries: List[Dict[str, float | int | str]] = []

    for name, initial_state in cases:
        case = propagate_case(name=name, initial_state=initial_state, config=config)
        trajectories.append(case)
        summaries.append(summarize_case(case=case, mu=config.mu))

    summary_df = pd.DataFrame.from_records(summaries)
    return summary_df, trajectories


def assert_quality(summary_df: pd.DataFrame, config: CR3BPConfig) -> None:
    max_drift = float(summary_df["max_jacobi_drift"].max())
    min_r1 = float(summary_df["min_r1"].min())
    min_r2 = float(summary_df["min_r2"].min())
    min_margin = float(summary_df["min_zvc_margin"].min())
    terminated_count = int(summary_df["terminated_early"].sum())

    if max_drift > 1e-7:
        raise RuntimeError(f"Jacobi drift too large: {max_drift:.3e}")
    if min_r1 <= config.min_distance_guard or min_r2 <= config.min_distance_guard:
        raise RuntimeError(
            "Trajectory entered collision guard radius: "
            f"min_r1={min_r1:.6f}, min_r2={min_r2:.6f}, guard={config.min_distance_guard:.6f}"
        )
    if min_margin < -1e-7:
        raise RuntimeError(f"Zero-velocity margin violated: min={min_margin:.3e}")
    if terminated_count != 0:
        raise RuntimeError("At least one trajectory terminated early by collision event.")


def main() -> None:
    config = CR3BPConfig()
    validate_config(config)

    pd.set_option("display.width", 160)

    summary_df, trajectories = run_demo(config=config)

    print("=== Restricted Three-Body Problem (Planar Circular, Rotating Frame) ===")
    print(
        f"mu={config.mu:.10f}, t_final={config.t_final:.2f}, samples={config.samples}, "
        f"rtol={config.rtol:.1e}, atol={config.atol:.1e}, guard={config.min_distance_guard:.3f}"
    )
    print()
    print("Summary metrics:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    for case in trajectories:
        print("\n" + "-" * 92)
        print(f"Case: {case.name}")
        sample_df = sample_trajectory_table(case=case, mu=config.mu, rows=8)
        print(sample_df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    assert_quality(summary_df=summary_df, config=config)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
