"""MVP for Boltzmann distribution on discrete energy levels.

The script is deterministic and non-interactive:
1) compute theoretical Boltzmann probabilities,
2) sample states from those probabilities,
3) compare empirical frequencies with theory,
4) estimate beta/temperature from sampled energies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp


@dataclass(frozen=True)
class BoltzmannConfig:
    temperature: float = 1.8
    k_b: float = 1.0
    energy_levels: Tuple[float, ...] = (0.0, 0.7, 1.5, 2.8, 4.2, 6.0)
    n_samples: int = 160_000
    seed: int = 20260407

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.k_b <= 0.0:
            raise ValueError("k_b must be > 0")
        if self.n_samples < 5_000:
            raise ValueError("n_samples must be >= 5000 for stable statistics")
        if len(self.energy_levels) < 2:
            raise ValueError("at least two energy levels are required")

        energies = np.asarray(self.energy_levels, dtype=np.float64)
        if not np.all(np.isfinite(energies)):
            raise ValueError("energy levels must be finite numbers")

        # The demo assumes nondecreasing levels to check monotone probabilities.
        if np.any(np.diff(energies) < 0.0):
            raise ValueError("energy levels must be nondecreasing")


def beta_from_temperature(cfg: BoltzmannConfig) -> float:
    return 1.0 / (cfg.k_b * cfg.temperature)


def boltzmann_probabilities(energies: np.ndarray, beta: float) -> tuple[np.ndarray, float, float]:
    log_weights = -beta * energies
    log_z = float(logsumexp(log_weights))
    probs = np.exp(log_weights - log_z)
    z = float(np.exp(log_z))
    return probs, z, log_z


def sample_states(probs: np.ndarray, cfg: BoltzmannConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    return rng.choice(probs.size, size=cfg.n_samples, p=probs)


def empirical_probabilities(state_indices: np.ndarray, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(state_indices, minlength=n_levels).astype(np.int64)
    probs = counts / counts.sum()
    return counts, probs


def model_mean_energy(energies: np.ndarray, beta: float) -> float:
    probs, _, _ = boltzmann_probabilities(energies, beta)
    return float(np.dot(probs, energies))


def estimate_beta_mle(energies: np.ndarray, sample_mean_energy: float) -> float:
    def objective(beta: float) -> float:
        return model_mean_energy(energies, beta) - sample_mean_energy

    beta_lo = 1.0e-12
    beta_hi = 1.0

    f_lo = objective(beta_lo)
    f_hi = objective(beta_hi)

    # For positive-temperature Boltzmann distributions, mean energy decreases with beta.
    # Expand the upper bracket until the sign changes.
    max_hi = 1.0e6
    while f_lo * f_hi > 0.0 and beta_hi < max_hi:
        beta_hi *= 2.0
        f_hi = objective(beta_hi)

    if f_lo * f_hi > 0.0:
        raise RuntimeError("failed to bracket MLE beta root")

    return float(brentq(objective, beta_lo, beta_hi, xtol=1e-12, rtol=1e-10, maxiter=200))


def kl_divergence(empirical: np.ndarray, theoretical: np.ndarray) -> float:
    mask = empirical > 0.0
    return float(np.sum(empirical[mask] * np.log(empirical[mask] / theoretical[mask])))


def build_report(cfg: BoltzmannConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    energies = np.asarray(cfg.energy_levels, dtype=np.float64)
    beta_true = beta_from_temperature(cfg)

    theory_probs, z, log_z = boltzmann_probabilities(energies, beta_true)
    sampled_states = sample_states(theory_probs, cfg)
    counts, emp_probs = empirical_probabilities(sampled_states, energies.size)

    mean_energy_theory = float(np.dot(theory_probs, energies))
    mean_energy_emp = float(np.dot(emp_probs, energies))

    beta_hat = estimate_beta_mle(energies, mean_energy_emp)
    temp_hat = 1.0 / (cfg.k_b * beta_hat)

    tv_distance = 0.5 * float(np.sum(np.abs(emp_probs - theory_probs)))
    rmse_prob = float(np.sqrt(np.mean((emp_probs - theory_probs) ** 2)))
    kl_emp_theory = kl_divergence(emp_probs, theory_probs)

    rows = []
    for idx, (energy, p_th, p_emp, count) in enumerate(zip(energies, theory_probs, emp_probs, counts)):
        rel_err = abs(p_emp - p_th) / max(p_th, 1e-15)
        rows.append(
            {
                "level": idx,
                "energy": energy,
                "count": int(count),
                "p_theory": p_th,
                "p_empirical": p_emp,
                "relative_error": rel_err,
            }
        )

    report_df = pd.DataFrame(rows)

    metrics = {
        "beta_true": beta_true,
        "beta_hat": beta_hat,
        "temperature_true": cfg.temperature,
        "temperature_hat": float(temp_hat),
        "partition_function": z,
        "log_partition_function": log_z,
        "mean_energy_theory": mean_energy_theory,
        "mean_energy_empirical": mean_energy_emp,
        "tv_distance": float(tv_distance),
        "rmse_prob": float(rmse_prob),
        "kl_empirical_to_theory": float(kl_emp_theory),
    }

    return report_df, metrics


def run_checks(cfg: BoltzmannConfig, report_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    energies = np.asarray(cfg.energy_levels, dtype=np.float64)
    theory_probs = report_df["p_theory"].to_numpy()

    if np.any(np.diff(energies) > 0.0):
        # Higher energy should not have larger Boltzmann probability.
        if not np.all(np.diff(theory_probs) <= 1e-12):
            raise AssertionError("theoretical probabilities must be nonincreasing with increasing energy")

    temp_rel_err = abs(metrics["temperature_hat"] - metrics["temperature_true"]) / metrics["temperature_true"]
    mean_rel_err = abs(metrics["mean_energy_empirical"] - metrics["mean_energy_theory"]) / max(
        metrics["mean_energy_theory"], 1e-15
    )

    if temp_rel_err > 0.03:
        raise AssertionError(f"temperature relative error too large: {temp_rel_err:.4f}")
    if mean_rel_err > 0.02:
        raise AssertionError(f"mean energy relative error too large: {mean_rel_err:.4f}")
    if metrics["tv_distance"] > 0.015:
        raise AssertionError(f"total variation distance too large: {metrics['tv_distance']:.4f}")
    if metrics["rmse_prob"] > 0.012:
        raise AssertionError(f"probability RMSE too large: {metrics['rmse_prob']:.4f}")
    if metrics["kl_empirical_to_theory"] > 0.0015:
        raise AssertionError(f"KL(emp||theory) too large: {metrics['kl_empirical_to_theory']:.6f}")


def main() -> None:
    cfg = BoltzmannConfig()
    cfg.validate()

    report_df, metrics = build_report(cfg)

    print("=== Boltzmann Distribution MVP (Discrete Energy Levels) ===")
    print(
        "config:",
        f"T={cfg.temperature}, k_B={cfg.k_b}, levels={cfg.energy_levels},",
        f"n_samples={cfg.n_samples}, seed={cfg.seed}",
    )
    print()
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(
        "beta_true={beta_true:.6f}, beta_hat={beta_hat:.6f}, "
        "temperature_true={temperature_true:.6f}, temperature_hat={temperature_hat:.6f}".format(**metrics)
    )
    print(
        "Z={partition_function:.6f}, logZ={log_partition_function:.6f}, "
        "<E>_theory={mean_energy_theory:.6f}, <E>_emp={mean_energy_empirical:.6f}".format(**metrics)
    )
    print(
        "TV={tv_distance:.6f}, RMSE={rmse_prob:.6f}, KL(emp||theory)={kl_empirical_to_theory:.6f}".format(**metrics)
    )

    run_checks(cfg, report_df, metrics)
    print("All checks passed.")


if __name__ == "__main__":
    main()
