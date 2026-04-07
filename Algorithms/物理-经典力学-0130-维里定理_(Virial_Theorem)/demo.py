"""Virial theorem MVP using 2D central power-law potentials.

Virial theorem core identity:
    dG/dt = 2T - r·∇V,
where G = r·p is the scalar virial.
For homogeneous potential V(r)=alpha*r^n (n != 0), r·∇V = nV, so time-average gives:
    2<T> = n<V>
for bounded motion over a long enough time window.

This script validates the theorem numerically for two scenarios:
1) isotropic harmonic oscillator (n=2),
2) Kepler-like gravity (n=-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Scenario:
    """One power-law potential test case."""

    name: str
    n: float
    alpha: float
    r0: tuple[float, float]
    v0: tuple[float, float]
    t_end: float
    num_points: int
    rtol: float = 1e-9
    atol: float = 1e-11


@dataclass(frozen=True)
class SimulationResult:
    """Container for one scenario's trajectory and diagnostics."""

    scenario: Scenario
    trajectory: pd.DataFrame
    summary: dict[str, float]


def validate_scenario(s: Scenario) -> None:
    """Sanity checks for physical and numerical parameters."""

    if s.num_points < 200:
        raise ValueError("num_points must be >= 200 for stable time averages.")
    if s.t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if s.n == 0.0:
        raise ValueError("n=0 is excluded for this homogeneous potential form.")
    if not np.isfinite(s.alpha):
        raise ValueError("alpha must be finite.")

    r0 = np.array(s.r0, dtype=float)
    v0 = np.array(s.v0, dtype=float)
    if r0.shape != (2,) or v0.shape != (2,):
        raise ValueError("r0 and v0 must be length-2 vectors.")
    if not np.all(np.isfinite(r0)) or not np.all(np.isfinite(v0)):
        raise ValueError("Initial state must be finite.")
    if float(np.linalg.norm(r0)) <= 1e-8:
        raise ValueError("Initial radius too close to zero; singular force risk.")


def potential_energy(radius: np.ndarray, alpha: float, n: float) -> np.ndarray:
    """Homogeneous potential V(r)=alpha*r^n."""

    return alpha * np.power(radius, n)


def acceleration_xy(x: float, y: float, alpha: float, n: float) -> tuple[float, float]:
    """Central-force acceleration from V(r)=alpha*r^n (unit mass)."""

    r2 = x * x + y * y
    r = float(np.sqrt(r2))
    r_safe = max(r, 1e-10)

    # F = -grad(V) = -alpha*n*r^(n-2) * r_vec
    coeff = -alpha * n * (r_safe ** (n - 2.0))
    ax = coeff * x
    ay = coeff * y
    return ax, ay


def rhs(_t: float, y: np.ndarray, alpha: float, n: float) -> np.ndarray:
    """State derivative for y=[x, y, vx, vy]."""

    x, y_pos, vx, vy = y
    ax, ay = acceleration_xy(x, y_pos, alpha=alpha, n=n)
    return np.array([vx, vy, ax, ay], dtype=float)


def run_scenario(s: Scenario) -> SimulationResult:
    """Integrate one scenario and compute virial diagnostics."""

    validate_scenario(s)

    t = np.linspace(0.0, s.t_end, s.num_points)
    y0 = np.array([s.r0[0], s.r0[1], s.v0[0], s.v0[1]], dtype=float)

    sol = solve_ivp(
        fun=lambda tt, yy: rhs(tt, yy, alpha=s.alpha, n=s.n),
        t_span=(0.0, s.t_end),
        y0=y0,
        t_eval=t,
        method="DOP853",
        rtol=s.rtol,
        atol=s.atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for {s.name}: {sol.message}")

    x = sol.y[0]
    y_pos = sol.y[1]
    vx = sol.y[2]
    vy = sol.y[3]

    radius = np.sqrt(x * x + y_pos * y_pos)
    speed2 = vx * vx + vy * vy

    kinetic = 0.5 * speed2
    potential = potential_energy(radius=radius, alpha=s.alpha, n=s.n)
    total_energy = kinetic + potential

    # Instantaneous virial balance for homogeneous potential:
    # dG/dt = 2T - nV, G = r·p
    virial_inst = 2.0 * kinetic - s.n * potential
    scalar_virial = x * vx + y_pos * vy

    mean_kinetic = float(np.mean(kinetic))
    mean_potential = float(np.mean(potential))
    lhs = 2.0 * mean_kinetic
    rhs_val = s.n * mean_potential
    abs_err = float(abs(lhs - rhs_val))
    rel_err = float(abs_err / max(abs(lhs), abs(rhs_val), 1e-12))

    e0 = float(total_energy[0])
    rel_energy_drift = np.abs((total_energy - e0) / max(abs(e0), 1e-12))

    # Also verify <dG/dt> ~= (G_end-G_start)/T ~= 0 for bounded dynamics.
    dg_dt_time_avg = float((scalar_virial[-1] - scalar_virial[0]) / s.t_end)

    traj = pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y_pos,
            "vx": vx,
            "vy": vy,
            "r": radius,
            "kinetic_T": kinetic,
            "potential_V": potential,
            "energy_E": total_energy,
            "virial_G": scalar_virial,
            "virial_balance_2T_minus_nV": virial_inst,
        }
    )

    summary = {
        "n": float(s.n),
        "alpha": float(s.alpha),
        "mean_T": mean_kinetic,
        "mean_V": mean_potential,
        "lhs_2_mean_T": float(lhs),
        "rhs_n_mean_V": float(rhs_val),
        "virial_abs_error": abs_err,
        "virial_rel_error": rel_err,
        "mean_2T_minus_nV": float(np.mean(virial_inst)),
        "mean_abs_2T_minus_nV": float(np.mean(np.abs(virial_inst))),
        "dg_dt_time_avg": dg_dt_time_avg,
        "max_rel_energy_drift": float(np.max(rel_energy_drift)),
        "min_radius": float(np.min(radius)),
        "max_radius": float(np.max(radius)),
    }

    return SimulationResult(scenario=s, trajectory=traj, summary=summary)


def default_scenarios() -> list[Scenario]:
    """Built-in cases that demonstrate the theorem under different n."""

    return [
        Scenario(
            name="harmonic_oscillator_n2",
            n=2.0,
            alpha=0.5,  # V = 0.5 * r^2
            r0=(1.2, 0.0),
            v0=(0.0, 0.7),
            t_end=125.66370614359172,  # 40*pi, aligns with full periods
            num_points=10000,
        ),
        Scenario(
            name="kepler_like_n_minus_1",
            n=-1.0,
            alpha=-1.0,  # V = -1/r
            r0=(1.0, 0.0),
            v0=(0.0, 0.85),
            t_end=120.0,
            num_points=12000,
        ),
    ]


def format_summary_table(results: list[SimulationResult]) -> pd.DataFrame:
    """Collect summaries into a compact display table."""

    rows: list[dict[str, str]] = []
    for res in results:
        s = res.summary
        rows.append(
            {
                "scenario": res.scenario.name,
                "n": f"{s['n']:.1f}",
                "2<T>": f"{s['lhs_2_mean_T']:.8f}",
                "n<V>": f"{s['rhs_n_mean_V']:.8f}",
                "rel_err": f"{s['virial_rel_error']:.3e}",
                "mean(2T-nV)": f"{s['mean_2T_minus_nV']:.3e}",
                "<dG/dt>": f"{s['dg_dt_time_avg']:.3e}",
                "max_energy_drift": f"{s['max_rel_energy_drift']:.3e}",
                "r_range": f"[{s['min_radius']:.4f}, {s['max_radius']:.4f}]",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    results = [run_scenario(s) for s in default_scenarios()]

    print("Virial Theorem MVP (central power-law potentials)")
    print("identity for bounded trajectories: 2<T> = n<V>\n")

    summary_df = format_summary_table(results)
    print("summary:")
    print(summary_df.to_string(index=False))

    for res in results:
        print(f"\ntrajectory_head ({res.scenario.name}):")
        print(res.trajectory.head(3).to_string(index=False))
        print(f"\ntrajectory_tail ({res.scenario.name}):")
        print(res.trajectory.tail(3).to_string(index=False))

    # Keep thresholds practical across both scenarios.
    for res in results:
        rel_err = res.summary["virial_rel_error"]
        energy_drift = res.summary["max_rel_energy_drift"]
        if rel_err > 2e-3:
            raise AssertionError(
                f"Virial average mismatch too large for {res.scenario.name}: {rel_err:.3e}"
            )
        if energy_drift > 5e-5:
            raise AssertionError(
                f"Energy drift too large for {res.scenario.name}: {energy_drift:.3e}"
            )


if __name__ == "__main__":
    main()
