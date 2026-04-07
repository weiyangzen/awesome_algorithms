"""Three-Body Problem MVP.

This script solves the planar Newtonian three-body problem with equal masses
using SciPy's ODE integrator. It is intentionally transparent:
- Explicit state packing/unpacking
- Explicit pairwise gravity acceleration
- Explicit diagnostics for energy/momentum/center-of-mass drift

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class ThreeBodyConfig:
    gravitational_constant: float = 1.0
    softening: float = 1e-6
    t0: float = 0.0
    t1: float = 6.3259
    num_steps: int = 1201
    rtol: float = 1e-10
    atol: float = 1e-12


def build_figure_eight_initial_state() -> Tuple[np.ndarray, np.ndarray]:
    """Classic equal-mass figure-eight initial condition in 2D."""
    masses = np.array([1.0, 1.0, 1.0], dtype=float)

    positions = np.array(
        [
            [-0.97000436, 0.24308753],
            [0.97000436, -0.24308753],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [0.4662036850, 0.4323657300],
            [0.4662036850, 0.4323657300],
            [-0.93240737, -0.86473146],
        ],
        dtype=float,
    )

    return masses, pack_state(positions, velocities)


def pack_state(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def unpack_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=float)
    positions = state[:6].reshape(3, 2)
    velocities = state[6:].reshape(3, 2)
    return positions, velocities


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
    softening: float,
) -> np.ndarray:
    accelerations = np.zeros_like(positions)
    for i in range(3):
        ai = np.zeros(2, dtype=float)
        for j in range(3):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            dist_sq = float(np.dot(diff, diff)) + softening**2
            inv_dist_cubed = dist_sq ** (-1.5)
            ai += gravitational_constant * masses[j] * diff * inv_dist_cubed
        accelerations[i] = ai
    return accelerations


def three_body_rhs(
    t: float,
    state: np.ndarray,
    masses: np.ndarray,
    cfg: ThreeBodyConfig,
) -> np.ndarray:
    del t
    positions, velocities = unpack_state(state)
    accelerations = compute_accelerations(
        positions=positions,
        masses=masses,
        gravitational_constant=cfg.gravitational_constant,
        softening=cfg.softening,
    )
    return pack_state(velocities, accelerations)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    d01 = np.linalg.norm(positions[0] - positions[1])
    d02 = np.linalg.norm(positions[0] - positions[2])
    d12 = np.linalg.norm(positions[1] - positions[2])
    return np.array([d01, d02, d12], dtype=float)


def total_energy(state: np.ndarray, masses: np.ndarray, cfg: ThreeBodyConfig) -> float:
    positions, velocities = unpack_state(state)

    kinetic = 0.5 * float(np.sum(masses[:, None] * velocities**2))

    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            diff = positions[j] - positions[i]
            distance = np.sqrt(float(np.dot(diff, diff)) + cfg.softening**2)
            potential -= cfg.gravitational_constant * masses[i] * masses[j] / distance

    return kinetic + potential


def total_momentum(state: np.ndarray, masses: np.ndarray) -> np.ndarray:
    _, velocities = unpack_state(state)
    return np.sum(masses[:, None] * velocities, axis=0)


def center_of_mass(state: np.ndarray, masses: np.ndarray) -> np.ndarray:
    positions, _ = unpack_state(state)
    mass_sum = float(np.sum(masses))
    return np.sum(masses[:, None] * positions, axis=0) / mass_sum


def run_simulation(
    masses: np.ndarray,
    state0: np.ndarray,
    cfg: ThreeBodyConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(cfg.t0, cfg.t1, cfg.num_steps)
    solution = solve_ivp(
        fun=three_body_rhs,
        t_span=(cfg.t0, cfg.t1),
        y0=state0,
        t_eval=t_eval,
        args=(masses, cfg),
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    return solution.t, solution.y.T


def build_diagnostics(
    time_grid: np.ndarray,
    states: np.ndarray,
    masses: np.ndarray,
    cfg: ThreeBodyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    body_rows = []
    global_rows = []

    initial_energy = total_energy(states[0], masses, cfg)
    initial_momentum_norm = float(np.linalg.norm(total_momentum(states[0], masses)))
    initial_com_norm = float(np.linalg.norm(center_of_mass(states[0], masses)))

    for t, state in zip(time_grid, states):
        positions, velocities = unpack_state(state)
        energy = total_energy(state, masses, cfg)
        momentum_norm = float(np.linalg.norm(total_momentum(state, masses)))
        com_norm = float(np.linalg.norm(center_of_mass(state, masses)))
        distances = pairwise_distances(positions)

        global_rows.append(
            {
                "t": float(t),
                "energy": energy,
                "rel_energy_drift": (energy - initial_energy) / abs(initial_energy),
                "momentum_norm": momentum_norm,
                "com_norm": com_norm,
                "d01": float(distances[0]),
                "d02": float(distances[1]),
                "d12": float(distances[2]),
                "min_pair_dist": float(np.min(distances)),
            }
        )

        for body_idx in range(3):
            radius = float(np.linalg.norm(positions[body_idx]))
            speed = float(np.linalg.norm(velocities[body_idx]))
            body_rows.append(
                {
                    "t": float(t),
                    "body": int(body_idx),
                    "x": float(positions[body_idx, 0]),
                    "y": float(positions[body_idx, 1]),
                    "vx": float(velocities[body_idx, 0]),
                    "vy": float(velocities[body_idx, 1]),
                    "radius": radius,
                    "speed": speed,
                }
            )

    body_df = pd.DataFrame(body_rows)
    global_df = pd.DataFrame(global_rows)

    final_state_error = float(np.linalg.norm(states[-1] - states[0]))

    summary = {
        "max_abs_rel_energy_drift": float(np.max(np.abs(global_df["rel_energy_drift"]))),
        "max_momentum_norm": float(np.max(global_df["momentum_norm"])),
        "max_com_norm": float(np.max(global_df["com_norm"])),
        "min_pair_dist": float(np.min(global_df["min_pair_dist"])),
        "period_closure_error": final_state_error,
        "initial_momentum_norm": initial_momentum_norm,
        "initial_com_norm": initial_com_norm,
    }

    return body_df, global_df, summary


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 160)

    cfg = ThreeBodyConfig()
    masses, state0 = build_figure_eight_initial_state()

    time_grid, states = run_simulation(masses=masses, state0=state0, cfg=cfg)
    body_df, global_df, summary = build_diagnostics(
        time_grid=time_grid,
        states=states,
        masses=masses,
        cfg=cfg,
    )

    print("Three-Body Problem MVP (planar Newtonian gravity)")
    print(
        f"Integration: method=DOP853, t_span=[{cfg.t0:.4f}, {cfg.t1:.4f}], "
        f"num_steps={cfg.num_steps}, rtol={cfg.rtol:.1e}, atol={cfg.atol:.1e}"
    )
    print("\nGlobal diagnostics (head):")
    print(global_df.head(8).to_string(index=False, float_format=lambda v: f"{v: .6e}"))
    print("\nBody trajectory samples (head):")
    print(body_df.head(12).to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print("\nSummary checks:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6e}")

    if summary["max_abs_rel_energy_drift"] > 5e-7:
        raise RuntimeError("Energy drift too large; integration accuracy check failed.")
    if summary["max_momentum_norm"] > 5e-9:
        raise RuntimeError("Momentum conservation check failed.")
    if summary["max_com_norm"] > 5e-9:
        raise RuntimeError("Center-of-mass drift check failed.")
    if summary["min_pair_dist"] < 0.1:
        raise RuntimeError("Bodies became too close; trajectory likely unstable for this setup.")
    if summary["period_closure_error"] > 5e-3:
        raise RuntimeError("Figure-eight closure error is too large.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
