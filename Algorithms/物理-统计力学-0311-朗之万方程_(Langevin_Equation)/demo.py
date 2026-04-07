"""Minimal runnable MVP for Langevin equation in statistical mechanics.

Model: overdamped Brownian particle in harmonic trap
    dx_t = -(k/gamma) x_t dt + sqrt(2 k_B T / gamma) dW_t
This is an Ornstein-Uhlenbeck process, simulated by Euler-Maruyama.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LangevinConfig:
    k_spring: float = 3.0
    gamma: float = 1.2
    temperature: float = 1.5
    k_boltzmann: float = 1.0
    dt: float = 1e-3
    n_steps: int = 18000
    n_trajectories: int = 256
    burn_in_steps: int = 6000
    seed: int = 20260407


def simulate_trajectories(cfg: LangevinConfig) -> np.ndarray:
    """Simulate x(t) trajectories by Euler-Maruyama."""
    if cfg.burn_in_steps >= cfg.n_steps:
        raise ValueError("burn_in_steps must be smaller than n_steps")

    rng = np.random.default_rng(cfg.seed)
    x = np.zeros((cfg.n_steps, cfg.n_trajectories), dtype=np.float64)

    drift_coeff = cfg.k_spring / cfg.gamma
    diffusion_coeff = np.sqrt(2.0 * cfg.k_boltzmann * cfg.temperature / cfg.gamma)
    noise_scale = diffusion_coeff * np.sqrt(cfg.dt)

    for step in range(1, cfg.n_steps):
        prev = x[step - 1]
        noise = rng.standard_normal(cfg.n_trajectories)
        x[step] = prev - drift_coeff * prev * cfg.dt + noise_scale * noise

    return x


def stationary_statistics(cfg: LangevinConfig, x: np.ndarray) -> dict[str, float]:
    """Compute stationary mean/variance from post-burn-in samples."""
    stationary = x[cfg.burn_in_steps :]
    flat = stationary.ravel()

    sample_mean = float(flat.mean())
    sample_var = float(flat.var(ddof=1))

    theory_mean = 0.0
    theory_var = cfg.k_boltzmann * cfg.temperature / cfg.k_spring

    rel_var_error = abs(sample_var - theory_var) / theory_var

    return {
        "sample_mean": sample_mean,
        "sample_variance": sample_var,
        "theory_mean": theory_mean,
        "theory_variance": theory_var,
        "relative_variance_error": rel_var_error,
    }


def autocorrelation_table(cfg: LangevinConfig, x: np.ndarray, lag_steps: list[int]) -> pd.DataFrame:
    """Compare empirical and analytical normalized autocorrelation."""
    stationary = x[cfg.burn_in_steps :]
    centered = stationary - stationary.mean()
    variance = centered.var()

    rows: list[dict[str, float]] = []
    relax_rate = cfg.k_spring / cfg.gamma

    for lag in lag_steps:
        if lag <= 0:
            continue
        if lag >= centered.shape[0]:
            continue

        lhs = centered[:-lag]
        rhs = centered[lag:]
        empirical = float((lhs * rhs).mean() / variance)

        lag_time = lag * cfg.dt
        theory = float(np.exp(-relax_rate * lag_time))

        rows.append(
            {
                "lag_step": float(lag),
                "lag_time": lag_time,
                "empirical_corr": empirical,
                "theory_corr": theory,
                "abs_error": abs(empirical - theory),
            }
        )

    return pd.DataFrame(rows)


def run_mvp(cfg: LangevinConfig) -> tuple[dict[str, float], pd.DataFrame]:
    x = simulate_trajectories(cfg)
    stats = stationary_statistics(cfg, x)
    ac_df = autocorrelation_table(cfg, x, lag_steps=[1, 2, 5, 10, 20, 50, 100, 200, 400])
    return stats, ac_df


def main() -> None:
    cfg = LangevinConfig()
    stats, ac_df = run_mvp(cfg)

    tau = cfg.gamma / cfg.k_spring
    print("=== Langevin Equation MVP: Overdamped Harmonic Brownian Motion ===")
    print(
        f"k={cfg.k_spring}, gamma={cfg.gamma}, T={cfg.temperature}, "
        f"dt={cfg.dt}, trajectories={cfg.n_trajectories}, steps={cfg.n_steps}"
    )
    print(f"Relaxation time tau = gamma/k = {tau:.6f}")
    print()

    print("Stationary statistics")
    print(f"- sample mean           : {stats['sample_mean']:.6f}")
    print(f"- theory mean           : {stats['theory_mean']:.6f}")
    print(f"- sample variance       : {stats['sample_variance']:.6f}")
    print(f"- theory variance kBT/k : {stats['theory_variance']:.6f}")
    print(f"- relative var error    : {stats['relative_variance_error']:.4%}")
    print()

    print("Autocorrelation check (empirical vs. exp(-t/tau))")
    print(ac_df.to_string(index=False, justify="center", col_space=10, float_format=lambda v: f"{v:.6f}"))
    max_ac_error = float(ac_df["abs_error"].max()) if not ac_df.empty else float("nan")
    print()
    print(f"Max autocorrelation absolute error: {max_ac_error:.6f}")

    # Lightweight self-checks for automated validation.
    if abs(stats["sample_mean"]) > 0.03:
        raise AssertionError("stationary mean deviates too much from 0")
    if stats["relative_variance_error"] > 0.08:
        raise AssertionError("stationary variance deviates too much from k_B T / k")
    if max_ac_error > 0.06:
        raise AssertionError("autocorrelation mismatch is too large")

    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()
