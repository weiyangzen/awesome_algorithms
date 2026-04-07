"""muVT (grand-canonical) ensemble MVP via ideal-gas GCMC.

This script samples particle-number fluctuations at fixed chemical potential
mu, volume V, and temperature T using insertion/deletion Metropolis moves.
For an ideal classical gas, the exact distribution is Poisson, which gives
closed-form checks for <N> and Var(N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MuVTConfig:
    temperature: float = 1.0
    volume: float = 40.0
    lambda_th: float = 1.0
    dimension: int = 3
    chemical_potentials: tuple[float, ...] = (-2.0, -1.0, -0.2)
    burn_in: int = 5000
    sample_steps: int = 14000
    thin: int = 4
    seed: int = 20260407


def activity(beta: float, mu: float, lambda_th: float, dimension: int) -> float:
    """Return z = exp(beta*mu)/lambda_th^d in reduced units."""
    return math.exp(beta * mu) / (lambda_th**dimension)


def theoretical_stats(mu: float, cfg: MuVTConfig) -> dict[str, float]:
    """Analytical moments for ideal-gas muVT number distribution."""
    beta = 1.0 / cfg.temperature
    z = activity(beta, mu, cfg.lambda_th, cfg.dimension)
    lam = cfg.volume * z
    return {
        "mean_n_theory": lam,
        "var_n_theory": lam,
        "fano_theory": 1.0,
    }


def poisson_pmf_truncated(max_k: int, lam: float) -> np.ndarray:
    """Compute Poisson PMF on k=0..max_k and renormalize over this finite window."""
    ks = np.arange(max_k + 1, dtype=np.float64)
    log_p = -lam + ks * math.log(lam) - np.array([math.lgamma(k + 1.0) for k in ks])
    p = np.exp(log_p)
    p_sum = float(np.sum(p))
    if p_sum <= 0.0:
        raise RuntimeError("Invalid Poisson PMF normalization.")
    return p / p_sum


def run_gcmc_ideal(mu: float, cfg: MuVTConfig, rng: np.random.Generator) -> dict[str, float]:
    """Run one mu-point grand-canonical Monte Carlo for the ideal gas."""
    beta = 1.0 / cfg.temperature
    z = activity(beta, mu, cfg.lambda_th, cfg.dimension)
    vz = cfg.volume * z

    total_steps = cfg.burn_in + cfg.sample_steps * cfg.thin
    n_particles = int(rng.poisson(vz))

    insert_attempts = 0
    insert_accepts = 0
    delete_attempts = 0
    delete_accepts = 0

    samples = np.empty(cfg.sample_steps, dtype=np.int64)
    saved = 0

    for step in range(total_steps):
        do_insert = bool(rng.random() < 0.5)

        if do_insert:
            insert_attempts += 1
            log_accept = math.log(vz) - math.log(n_particles + 1.0)
            if math.log(rng.random()) < min(0.0, log_accept):
                n_particles += 1
                insert_accepts += 1
        else:
            delete_attempts += 1
            if n_particles > 0:
                log_accept = math.log(float(n_particles)) - math.log(vz)
                if math.log(rng.random()) < min(0.0, log_accept):
                    n_particles -= 1
                    delete_accepts += 1

        if step >= cfg.burn_in and (step - cfg.burn_in) % cfg.thin == 0:
            samples[saved] = n_particles
            saved += 1

    if saved != cfg.sample_steps:
        raise RuntimeError(f"Saved sample count mismatch: {saved} != {cfg.sample_steps}")

    samples_f = samples.astype(np.float64)
    mean_emp = float(np.mean(samples_f))
    var_emp = float(np.var(samples_f, ddof=1))
    fano_emp = var_emp / mean_emp if mean_emp > 0.0 else float("nan")

    theory = theoretical_stats(mu, cfg)

    mean_rel_err = abs(mean_emp - theory["mean_n_theory"]) / theory["mean_n_theory"]
    var_rel_err = abs(var_emp - theory["var_n_theory"]) / theory["var_n_theory"]
    fano_abs_err = abs(fano_emp - theory["fano_theory"]) if math.isfinite(fano_emp) else float("inf")

    max_k = int(max(np.max(samples), math.ceil(theory["mean_n_theory"] + 7.0 * math.sqrt(theory["mean_n_theory"])) + 3))
    empirical_hist = np.bincount(samples, minlength=max_k + 1).astype(np.float64)
    empirical_p = empirical_hist / float(np.sum(empirical_hist))
    theory_p = poisson_pmf_truncated(max_k, theory["mean_n_theory"])
    distribution_l1 = float(np.sum(np.abs(empirical_p - theory_p)))

    return {
        "mu": mu,
        "mean_n_emp": mean_emp,
        "mean_n_theory": theory["mean_n_theory"],
        "var_n_emp": var_emp,
        "var_n_theory": theory["var_n_theory"],
        "fano_emp": fano_emp,
        "fano_theory": theory["fano_theory"],
        "mean_rel_error": mean_rel_err,
        "var_rel_error": var_rel_err,
        "fano_abs_error": fano_abs_err,
        "distribution_l1": distribution_l1,
        "insert_acceptance": insert_accepts / insert_attempts,
        "delete_acceptance": delete_accepts / delete_attempts,
    }


def main() -> None:
    cfg = MuVTConfig()
    rows: list[dict[str, float]] = []

    for idx, mu in enumerate(cfg.chemical_potentials):
        run_rng = np.random.default_rng(cfg.seed + 10007 * idx)
        rows.append(run_gcmc_ideal(mu, cfg, run_rng))

    df = pd.DataFrame(rows).sort_values("mu").reset_index(drop=True)

    with pd.option_context("display.width", 180, "display.max_columns", 30):
        print("muVT ensemble check using ideal-gas GCMC (insertion/deletion moves)")
        print(
            "Parameters:"
            f" T={cfg.temperature}, V={cfg.volume}, lambda_th={cfg.lambda_th},"
            f" burn_in={cfg.burn_in}, sample_steps={cfg.sample_steps}, thin={cfg.thin}"
        )
        print(df.round(6).to_string(index=False))

    max_rel = float(df[["mean_rel_error", "var_rel_error"]].to_numpy().max())
    max_l1 = float(df["distribution_l1"].max())
    min_ins = float(df["insert_acceptance"].min())
    min_del = float(df["delete_acceptance"].min())

    print(
        "\nSummary:"
        f" max_relative_moment_error={max_rel:.4f},"
        f" max_distribution_l1={max_l1:.4f},"
        f" min_insert_acceptance={min_ins:.3f},"
        f" min_delete_acceptance={min_del:.3f}"
    )

    means = df["mean_n_emp"].to_numpy()
    assert bool(np.all(np.diff(means) > 0.0)), "Empirical <N> should increase with mu."
    assert max_rel < 0.10, f"Moment relative error too large: {max_rel:.4f}"
    assert max_l1 < 0.16, f"Empirical N distribution too far from Poisson: {max_l1:.4f}"
    assert 0.05 < min_ins < 0.98 and 0.05 < min_del < 0.98, "Acceptance rates unhealthy."


if __name__ == "__main__":
    main()
