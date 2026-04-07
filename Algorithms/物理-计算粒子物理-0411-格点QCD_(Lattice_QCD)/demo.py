"""Minimal runnable MVP for Lattice QCD (PHYS-0392).

This script implements a pedagogical lattice gauge Monte Carlo with Wilson action
on a 2D periodic lattice using compact U(1) link variables.

Why U(1), not full SU(3)+fermion determinant?
- Full production Lattice QCD requires much heavier infrastructure.
- This MVP keeps the source-level algorithm transparent:
  Wilson action -> Metropolis update -> thermalization -> measurements.
- The code still preserves the core lattice-field-theory workflow used in QCD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import sem


@dataclass(frozen=True)
class LatticeConfig:
    lt: int = 8
    lx: int = 8
    beta: float = 1.15
    proposal_width: float = 0.75
    thermal_sweeps: int = 140
    measurement_sweeps: int = 220
    sweeps_per_measure: int = 2
    seed: int = 2026


@dataclass(frozen=True)
class LoopRequest:
    r_values: tuple[int, ...] = (1, 2, 3)
    t_values: tuple[int, ...] = (1, 2)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Map angle(s) to (-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def plaquette_angle(theta: np.ndarray, t: int, x: int, cfg: LatticeConfig) -> float:
    """Plaquette angle in 2D for site (t, x).

    Using mu=0 (time), mu=1 (space):
    theta_p = theta_0(t,x) + theta_1(t+1,x) - theta_0(t,x+1) - theta_1(t,x)
    """

    tp = (t + 1) % cfg.lt
    xp = (x + 1) % cfg.lx

    return float(
        theta[0, t, x]
        + theta[1, tp, x]
        - theta[0, t, xp]
        - theta[1, t, x]
    )


def affected_plaquettes(mu: int, t: int, x: int, cfg: LatticeConfig) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return plaquette anchors touched by a single link update."""

    if mu == 0:
        return ((t, x), (t, (x - 1) % cfg.lx))
    if mu == 1:
        return ((t, x), ((t - 1) % cfg.lt, x))
    raise ValueError("mu must be 0 or 1")


def local_cos_sum(theta: np.ndarray, anchors: tuple[tuple[int, int], tuple[int, int]], cfg: LatticeConfig) -> float:
    return float(
        np.cos(plaquette_angle(theta, anchors[0][0], anchors[0][1], cfg))
        + np.cos(plaquette_angle(theta, anchors[1][0], anchors[1][1], cfg))
    )


def metropolis_link_update(
    theta: np.ndarray,
    mu: int,
    t: int,
    x: int,
    cfg: LatticeConfig,
    rng: np.random.Generator,
) -> bool:
    """Single-link Metropolis update under Wilson action.

    Wilson gauge action here: S = -beta * sum_p cos(theta_p)
    Acceptance uses min(1, exp(-Delta S)).
    """

    anchors = affected_plaquettes(mu, t, x, cfg)
    old_local = local_cos_sum(theta, anchors, cfg)

    old_angle = theta[mu, t, x]
    proposal = wrap_angle(old_angle + rng.uniform(-cfg.proposal_width, cfg.proposal_width))
    theta[mu, t, x] = proposal

    new_local = local_cos_sum(theta, anchors, cfg)
    delta_s = -cfg.beta * (new_local - old_local)

    if delta_s <= 0.0 or rng.uniform() < np.exp(-delta_s):
        return True

    theta[mu, t, x] = old_angle
    return False


def single_sweep(theta: np.ndarray, cfg: LatticeConfig, rng: np.random.Generator) -> tuple[int, int]:
    """One full lattice sweep over all links."""

    accepted = 0
    total = 0

    # Deterministic link ordering is fine for this pedagogical MVP.
    for mu in range(2):
        for t in range(cfg.lt):
            for x in range(cfg.lx):
                accepted += int(metropolis_link_update(theta, mu, t, x, cfg, rng))
                total += 1

    return accepted, total


def average_plaquette(theta: np.ndarray, cfg: LatticeConfig) -> float:
    vals = np.empty((cfg.lt, cfg.lx), dtype=float)
    for t in range(cfg.lt):
        for x in range(cfg.lx):
            vals[t, x] = np.cos(plaquette_angle(theta, t, x, cfg))
    return float(np.mean(vals))


def wilson_loop_angle(theta: np.ndarray, t0: int, x0: int, r: int, tau: int, cfg: LatticeConfig) -> float:
    """Oriented rectangular loop angle with size spatial=r, temporal=tau."""

    angle = 0.0

    # Up edge: +time at x = x0
    for dt in range(tau):
        t = (t0 + dt) % cfg.lt
        angle += theta[0, t, x0]

    # Right edge: +space at t = t0 + tau
    t_top = (t0 + tau) % cfg.lt
    for dx in range(r):
        x = (x0 + dx) % cfg.lx
        angle += theta[1, t_top, x]

    # Down edge: -time at x = x0 + r
    x_right = (x0 + r) % cfg.lx
    for dt in range(tau):
        t = (t0 + dt) % cfg.lt
        angle -= theta[0, t, x_right]

    # Left edge: -space at t = t0
    for dx in range(r):
        x = (x0 + dx) % cfg.lx
        angle -= theta[1, t0, x]

    return float(wrap_angle(angle))


def wilson_loop(theta: np.ndarray, r: int, tau: int, cfg: LatticeConfig) -> float:
    vals = np.empty((cfg.lt, cfg.lx), dtype=float)
    for t in range(cfg.lt):
        for x in range(cfg.lx):
            vals[t, x] = np.cos(wilson_loop_angle(theta, t, x, r, tau, cfg))
    return float(np.mean(vals))


def run_simulation(cfg: LatticeConfig, loops: LoopRequest) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    theta = rng.uniform(-np.pi, np.pi, size=(2, cfg.lt, cfg.lx))

    thermal_acceptance: list[float] = []
    for _ in range(cfg.thermal_sweeps):
        accepted, total = single_sweep(theta, cfg, rng)
        thermal_acceptance.append(accepted / total)

    records: list[dict[str, float]] = []
    measurement_acceptance: list[float] = []

    for measure_idx in range(1, cfg.measurement_sweeps + 1):
        accepted_block = 0
        total_block = 0
        for _ in range(cfg.sweeps_per_measure):
            accepted, total = single_sweep(theta, cfg, rng)
            accepted_block += accepted
            total_block += total

        acc_ratio = accepted_block / total_block
        measurement_acceptance.append(acc_ratio)

        row: dict[str, float] = {
            "measure": float(measure_idx),
            "plaquette": average_plaquette(theta, cfg),
            "acc_ratio": acc_ratio,
        }

        for r in loops.r_values:
            for tau in loops.t_values:
                key = f"W_R{r}_T{tau}"
                row[key] = wilson_loop(theta, r=r, tau=tau, cfg=cfg)

        records.append(row)

    if not records:
        raise RuntimeError("No measurements collected")

    df = pd.DataFrame(records)

    return {
        "cfg": cfg,
        "loops": loops,
        "history": df,
        "thermal_acceptance": np.array(thermal_acceptance, dtype=float),
        "measurement_acceptance": np.array(measurement_acceptance, dtype=float),
    }


def estimate_static_potential(df: pd.DataFrame, r_values: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for r in r_values:
        w_name = f"W_R{r}_T1"
        w_mean = float(df[w_name].mean())
        potential = -np.log(max(w_mean, 1e-12))
        rows.append({"R": float(r), "<W(R,1)>": w_mean, "V_eff(R)": potential})
    return pd.DataFrame(rows)


def estimate_creutz_ratio(df: pd.DataFrame) -> float:
    """Creutz ratio chi(2,2) from loop means.

    chi(2,2) = -ln( W(2,2) * W(1,1) / (W(2,1) * W(1,2)) )
    """

    w22 = float(df["W_R2_T2"].mean())
    w11 = float(df["W_R1_T1"].mean())
    w21 = float(df["W_R2_T1"].mean())
    w12 = float(df["W_R1_T2"].mean())

    ratio = (max(w22, 1e-12) * max(w11, 1e-12)) / (max(w21, 1e-12) * max(w12, 1e-12))
    return float(-np.log(ratio))


def main() -> None:
    cfg = LatticeConfig(
        lt=8,
        lx=8,
        beta=1.15,
        proposal_width=0.75,
        thermal_sweeps=140,
        measurement_sweeps=220,
        sweeps_per_measure=2,
        seed=2026,
    )
    loops = LoopRequest(r_values=(1, 2, 3), t_values=(1, 2))

    result = run_simulation(cfg, loops)
    history = result["history"]
    thermal_acceptance = result["thermal_acceptance"]
    measurement_acceptance = result["measurement_acceptance"]

    potential_df = estimate_static_potential(history, loops.r_values)
    chi_22 = estimate_creutz_ratio(history)

    summary = {
        "n_measurements": float(len(history)),
        "plaquette_mean": float(history["plaquette"].mean()),
        "plaquette_sem": float(sem(history["plaquette"], ddof=1)),
        "acc_mean": float(np.mean(measurement_acceptance)),
        "acc_thermal_mean": float(np.mean(thermal_acceptance)),
        "W_R1_T1_mean": float(history["W_R1_T1"].mean()),
        "W_R1_T1_sem": float(sem(history["W_R1_T1"], ddof=1)),
        "chi_22": chi_22,
    }

    checks = {
        "no NaN in measurement table": bool(np.isfinite(history.to_numpy()).all()),
        "thermal acceptance in [0.10, 0.95]": 0.10 <= summary["acc_thermal_mean"] <= 0.95,
        "measurement acceptance in [0.10, 0.95]": 0.10 <= summary["acc_mean"] <= 0.95,
        "plaquette mean in [-1, 1]": -1.0 <= summary["plaquette_mean"] <= 1.0,
        "|W_R1_T1| <= 1": abs(summary["W_R1_T1_mean"]) <= 1.0,
        "effective potentials are finite": bool(np.isfinite(potential_df["V_eff(R)"].to_numpy()).all()),
        "chi_22 finite": bool(np.isfinite(chi_22)),
        "measurement count correct": len(history) == cfg.measurement_sweeps,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    print("=== Lattice QCD MVP (Pedagogical U(1) Wilson Lattice) | PHYS-0392 ===")
    print(f"Lattice: Lt={cfg.lt}, Lx={cfg.lx}, beta={cfg.beta:.3f}, proposal_width={cfg.proposal_width:.3f}")
    print(f"Thermal sweeps={cfg.thermal_sweeps}, measurements={cfg.measurement_sweeps}, sweeps/measure={cfg.sweeps_per_measure}")
    print()

    print("Recent measurements (tail 8):")
    print(
        history[
            [
                "measure",
                "plaquette",
                "W_R1_T1",
                "W_R2_T1",
                "W_R1_T2",
                "acc_ratio",
            ]
        ]
        .tail(8)
        .to_string(index=False)
    )

    print()
    print("Summary:")
    print(pd.DataFrame([summary]).to_string(index=False))

    print()
    print("Estimated effective static potential from W(R,1):")
    print(potential_df.to_string(index=False))

    print()
    print("Checks:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"- {name}: {status}")

    all_passed = all(checks.values())
    print()
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
