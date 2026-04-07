"""Minimal runnable MVP for Langevin Dynamics in computational physics.

This script simulates a 1D underdamped Langevin oscillator with a BAOAB
splitting integrator and verifies equilibrium statistics against theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LangevinConfig:
    """Configuration for a 1D underdamped Langevin simulation."""

    mass: float = 1.0
    spring_k: float = 4.0
    gamma: float = 1.5
    kbt: float = 1.0
    dt: float = 0.005
    n_steps: int = 3000
    burn_in_steps: int = 800
    n_trajectories: int = 2500
    sample_stride: int = 150
    seed: int = 42

    def validate(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.spring_k <= 0.0:
            raise ValueError("spring_k must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.kbt <= 0.0:
            raise ValueError("kbt must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.n_steps <= 10:
            raise ValueError("n_steps must be > 10")
        if not (0 <= self.burn_in_steps < self.n_steps):
            raise ValueError("burn_in_steps must be in [0, n_steps)")
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")


def harmonic_force(x: np.ndarray, spring_k: float) -> np.ndarray:
    """Return force F(x) = -k x for U(x)=0.5*k*x^2."""

    return -spring_k * x


def theoretical_moments(cfg: LangevinConfig) -> dict[str, float]:
    """Return canonical-equilibrium moments for harmonic oscillator."""

    return {
        "x_mean": 0.0,
        "v_mean": 0.0,
        "x2": cfg.kbt / cfg.spring_k,
        "v2": cfg.kbt / cfg.mass,
        "potential_energy": 0.5 * cfg.kbt,
        "kinetic_energy": 0.5 * cfg.kbt,
    }


def run_baoab(cfg: LangevinConfig) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Simulate trajectories and return summary metrics and checkpoints."""

    cfg.validate()
    rng = np.random.default_rng(cfg.seed)

    # Start away from equilibrium to show relaxation.
    x = rng.normal(loc=0.0, scale=2.0 * np.sqrt(cfg.kbt / cfg.spring_k), size=cfg.n_trajectories)
    v = rng.normal(loc=0.0, scale=2.0 * np.sqrt(cfg.kbt / cfg.mass), size=cfg.n_trajectories)

    half_dt = 0.5 * cfg.dt
    exp_factor = np.exp(-cfg.gamma * cfg.dt)
    noise_scale = np.sqrt((cfg.kbt / cfg.mass) * (1.0 - exp_factor * exp_factor))

    sum_x = 0.0
    sum_v = 0.0
    sum_x2 = 0.0
    sum_v2 = 0.0
    sample_count = 0

    checkpoint_rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps):
        # BAOAB splitting: B (half kick) -> A (half drift) -> O (OU thermostat)
        # -> A (half drift) -> B (half kick)
        v += half_dt * harmonic_force(x, cfg.spring_k) / cfg.mass
        x += half_dt * v
        v = exp_factor * v + noise_scale * rng.standard_normal(cfg.n_trajectories)
        x += half_dt * v
        v += half_dt * harmonic_force(x, cfg.spring_k) / cfg.mass

        if step >= cfg.burn_in_steps:
            sum_x += x.sum()
            sum_v += v.sum()
            sum_x2 += np.square(x).sum()
            sum_v2 += np.square(v).sum()
            sample_count += cfg.n_trajectories

        if step % cfg.sample_stride == 0:
            checkpoint_rows.append(
                {
                    "step": float(step),
                    "time": step * cfg.dt,
                    "x_probe": float(x[0]),
                    "v_probe": float(v[0]),
                    "ensemble_x2": float(np.mean(np.square(x))),
                    "ensemble_v2": float(np.mean(np.square(v))),
                }
            )

    empirical = {
        "x_mean": sum_x / sample_count,
        "v_mean": sum_v / sample_count,
        "x2": sum_x2 / sample_count,
        "v2": sum_v2 / sample_count,
    }
    empirical["potential_energy"] = 0.5 * cfg.spring_k * empirical["x2"]
    empirical["kinetic_energy"] = 0.5 * cfg.mass * empirical["v2"]

    theory = theoretical_moments(cfg)
    rows = []
    for key in ["x_mean", "v_mean", "x2", "v2", "potential_energy", "kinetic_energy"]:
        target = theory[key]
        value = empirical[key]
        if abs(target) < 1e-12:
            rel_err = abs(value - target)
        else:
            rel_err = abs(value - target) / abs(target)
        rows.append(
            {
                "metric": key,
                "empirical": value,
                "theory": target,
                "abs_or_rel_error": rel_err,
            }
        )

    summary_df = pd.DataFrame(rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    return empirical, summary_df, checkpoint_df


def main() -> None:
    cfg = LangevinConfig()
    empirical, summary_df, checkpoint_df = run_baoab(cfg)

    print("=== Langevin Dynamics MVP (Underdamped, BAOAB, Harmonic Potential) ===")
    print(
        "Config:",
        {
            "mass": cfg.mass,
            "spring_k": cfg.spring_k,
            "gamma": cfg.gamma,
            "kbt": cfg.kbt,
            "dt": cfg.dt,
            "n_steps": cfg.n_steps,
            "burn_in_steps": cfg.burn_in_steps,
            "n_trajectories": cfg.n_trajectories,
            "seed": cfg.seed,
        },
    )
    print("\nSummary vs theory:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nCheckpoint snapshots (tail):")
    print(checkpoint_df.tail(6).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    x2_theory = cfg.kbt / cfg.spring_k
    v2_theory = cfg.kbt / cfg.mass
    x2_rel_err = abs(empirical["x2"] - x2_theory) / x2_theory
    v2_rel_err = abs(empirical["v2"] - v2_theory) / v2_theory

    # Deterministic, practical acceptance thresholds for this compact MVP.
    assert abs(empirical["x_mean"]) < 0.03
    assert abs(empirical["v_mean"]) < 0.03
    assert x2_rel_err < 0.05
    assert v2_rel_err < 0.05

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
