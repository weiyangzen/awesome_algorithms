"""Simulated Annealing MVP for a rugged energy landscape.

We optimize the 2D Rastrigin energy (a multi-minima benchmark):
    E(x) = 10d + sum_i [x_i^2 - 10 cos(2*pi*x_i)]
Global minimum is E(0,0)=0.

This script provides:
1) A source-level implementation of simulated annealing (SA).
2) A same-budget random-search baseline.
3) Reproducible multi-seed statistics and lightweight assertions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SAConfig:
    """Configuration for the simulated annealing run."""

    dim: int = 2
    low: float = -5.12
    high: float = 5.12
    n_steps: int = 12_000
    t0: float = 5.0
    cooling: float = 0.9993
    proposal_sigma: float = 0.4
    seed: int = 20260407
    trace_every: int = 500


def rastrigin_energy(x: np.ndarray) -> float:
    """Rastrigin energy function with many local minima."""
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * math.pi * x)))


def _temperature(step: int, cfg: SAConfig) -> float:
    """Exponential cooling schedule with a lower numerical floor."""
    return max(1e-8, cfg.t0 * (cfg.cooling**step))


def simulated_annealing(cfg: SAConfig) -> tuple[dict[str, float | int | np.ndarray], pd.DataFrame]:
    """Run one simulated annealing trajectory.

    Returns:
        summary: final scalar metrics and best state.
        trace_df: coarse time-series diagnostics.
    """
    rng = np.random.default_rng(cfg.seed)

    x = rng.uniform(cfg.low, cfg.high, size=cfg.dim)
    current_e = rastrigin_energy(x)
    best_x = x.copy()
    best_e = current_e

    accepted_total = 0
    uphill_accepted_total = 0

    trace_rows: list[dict[str, float | int]] = []

    for step in range(1, cfg.n_steps + 1):
        temp = _temperature(step, cfg)

        proposal = np.clip(
            x + rng.normal(0.0, cfg.proposal_sigma, size=cfg.dim),
            cfg.low,
            cfg.high,
        )
        proposal_e = rastrigin_energy(proposal)
        delta = proposal_e - current_e

        accepted = False
        if delta <= 0.0:
            accepted = True
        else:
            accept_prob = math.exp(-delta / temp)
            if rng.random() < accept_prob:
                accepted = True

        if accepted:
            accepted_total += 1
            if delta > 0.0:
                uphill_accepted_total += 1
            x = proposal
            current_e = proposal_e

            if current_e < best_e:
                best_e = current_e
                best_x = x.copy()

        if step == 1 or step % cfg.trace_every == 0 or step == cfg.n_steps:
            trace_rows.append(
                {
                    "step": step,
                    "temperature": temp,
                    "current_energy": current_e,
                    "best_energy": best_e,
                    "accepted_ratio": accepted_total / step,
                    "uphill_accept_ratio": uphill_accepted_total / max(accepted_total, 1),
                }
            )

    summary: dict[str, float | int | np.ndarray] = {
        "seed": cfg.seed,
        "n_steps": cfg.n_steps,
        "start_energy": trace_rows[0]["current_energy"],
        "final_energy": current_e,
        "best_energy": best_e,
        "accepted_ratio": accepted_total / cfg.n_steps,
        "uphill_accept_ratio": uphill_accepted_total / max(accepted_total, 1),
        "best_state": best_x,
    }
    return summary, pd.DataFrame(trace_rows)


def random_search_baseline(cfg: SAConfig, seed: int) -> float:
    """Same evaluation budget baseline: independent random samples."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(cfg.low, cfg.high, size=(cfg.n_steps + 1, cfg.dim))
    energies = 10.0 * cfg.dim + np.sum(
        points * points - 10.0 * np.cos(2.0 * math.pi * points),
        axis=1,
    )
    return float(energies.min())


def run_multi_seed_benchmark(cfg: SAConfig, n_runs: int = 30) -> pd.DataFrame:
    """Compare SA vs random search over multiple seeds."""
    rows: list[dict[str, float | int | str]] = []

    for i in range(n_runs):
        sa_cfg = replace(cfg, seed=cfg.seed + 17 * i)
        sa_summary, _ = simulated_annealing(sa_cfg)

        rs_best = random_search_baseline(cfg, seed=cfg.seed + 100_000 + i)

        rows.append(
            {
                "method": "simulated_annealing",
                "run": i,
                "best_energy": float(sa_summary["best_energy"]),
            }
        )
        rows.append({"method": "random_search", "run": i, "best_energy": rs_best})

    return pd.DataFrame(rows)


def summarize_benchmark(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark table for concise reporting."""
    stats = (
        benchmark_df.groupby("method", as_index=False)["best_energy"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    return stats


def main() -> None:
    cfg = SAConfig()

    summary, trace_df = simulated_annealing(cfg)
    benchmark_df = run_multi_seed_benchmark(cfg, n_runs=30)
    benchmark_stats = summarize_benchmark(benchmark_df)

    best_state = np.asarray(summary["best_state"], dtype=float)

    print("Simulated Annealing MVP")
    print("-" * 72)
    print("Objective: minimize 2D Rastrigin energy on [-5.12, 5.12]^2")
    print("Global optimum: E(x*) = 0 at x* = (0, 0)")
    print()
    print("[Single Run Summary]")
    print(f"seed               : {int(summary['seed'])}")
    print(f"steps              : {int(summary['n_steps'])}")
    print(f"best_energy        : {float(summary['best_energy']):.8f}")
    print(f"final_energy       : {float(summary['final_energy']):.8f}")
    print(f"accepted_ratio     : {float(summary['accepted_ratio']):.5f}")
    print(f"uphill_accept_ratio: {float(summary['uphill_accept_ratio']):.5f}")
    print(f"best_state         : [{best_state[0]:.6f}, {best_state[1]:.6f}]")
    print()
    print("[Annealing Trace]")
    print(trace_df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print()
    print("[Benchmark: 30 Runs, same evaluation budget]")
    print(benchmark_stats.to_string(index=False, float_format=lambda v: f"{v:.8f}"))

    sa_median = float(
        benchmark_stats.loc[
            benchmark_stats["method"] == "simulated_annealing", "median"
        ].iloc[0]
    )
    rs_median = float(
        benchmark_stats.loc[benchmark_stats["method"] == "random_search", "median"].iloc[
            0
        ]
    )

    # Lightweight, reproducible checks.
    assert float(summary["best_energy"]) < 0.1, "Single-run SA did not approach optimum."
    assert 0.005 < float(summary["accepted_ratio"]) < 0.2, "Acceptance ratio out of expected range."
    assert sa_median < 0.25 * rs_median, "SA should significantly beat random-search median."

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
