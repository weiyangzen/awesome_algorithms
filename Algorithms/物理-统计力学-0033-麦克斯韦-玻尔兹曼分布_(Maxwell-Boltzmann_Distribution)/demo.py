"""MVP for Maxwell-Boltzmann speed distribution in 3D.

The script is deterministic and non-interactive:
1) sample velocity components from Gaussian laws implied by thermal equilibrium,
2) convert to speed samples,
3) compare empirical distribution/moments against Maxwell-Boltzmann theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import erf


@dataclass(frozen=True)
class MBConfig:
    temperature: float = 2.0
    mass: float = 1.0
    k_b: float = 1.0
    n_samples: int = 120_000
    n_bins: int = 90
    speed_max_factor: float = 4.2
    seed: int = 20260407

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.mass <= 0.0:
            raise ValueError("mass must be > 0")
        if self.k_b <= 0.0:
            raise ValueError("k_b must be > 0")
        if self.n_samples < 5_000:
            raise ValueError("n_samples must be >= 5000 for stable statistics")
        if self.n_bins < 20:
            raise ValueError("n_bins must be >= 20")
        if self.speed_max_factor <= 1.0:
            raise ValueError("speed_max_factor must be > 1")


def thermal_sigma(cfg: MBConfig) -> float:
    return float(np.sqrt(cfg.k_b * cfg.temperature / cfg.mass))


def sample_speeds(cfg: MBConfig) -> np.ndarray:
    sigma = thermal_sigma(cfg)
    rng = np.random.default_rng(cfg.seed)
    velocities = rng.normal(loc=0.0, scale=sigma, size=(cfg.n_samples, 3))
    return np.linalg.norm(velocities, axis=1)


def maxwell_boltzmann_pdf(v: np.ndarray, cfg: MBConfig) -> np.ndarray:
    sigma = thermal_sigma(cfg)
    z = np.asarray(v, dtype=np.float64) / sigma
    return np.sqrt(2.0 / np.pi) * (z * z / sigma) * np.exp(-0.5 * z * z)


def maxwell_boltzmann_cdf(v: np.ndarray, cfg: MBConfig) -> np.ndarray:
    sigma = thermal_sigma(cfg)
    z = np.asarray(v, dtype=np.float64) / sigma
    return erf(z / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * z * np.exp(-0.5 * z * z)


def theoretical_stats(cfg: MBConfig) -> Dict[str, float]:
    factor = cfg.k_b * cfg.temperature / cfg.mass
    return {
        "most_probable_speed": float(np.sqrt(2.0 * factor)),
        "mean_speed": float(np.sqrt(8.0 * factor / np.pi)),
        "rms_speed": float(np.sqrt(3.0 * factor)),
    }


def empirical_stats(speeds: np.ndarray, cfg: MBConfig) -> Dict[str, float]:
    mean_v = float(np.mean(speeds))
    rms_v = float(np.sqrt(np.mean(speeds * speeds)))
    most_probable = float(np.histogram(speeds, bins=cfg.n_bins)[1][np.argmax(np.histogram(speeds, bins=cfg.n_bins)[0])])
    t_hat = float(cfg.mass * np.mean(speeds * speeds) / (3.0 * cfg.k_b))
    return {
        "most_probable_speed": most_probable,
        "mean_speed": mean_v,
        "rms_speed": rms_v,
        "temperature_hat": t_hat,
    }


def compute_ks_distance(speeds: np.ndarray, cfg: MBConfig) -> float:
    x = np.sort(speeds)
    n = x.size
    theo = maxwell_boltzmann_cdf(x, cfg)
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - theo)
    d_minus = np.max(theo - (i - 1.0) / n)
    return float(max(d_plus, d_minus))


def summarize_distribution(speeds: np.ndarray, cfg: MBConfig) -> tuple[pd.DataFrame, float, float]:
    theo = theoretical_stats(cfg)
    emp = empirical_stats(speeds, cfg)

    vmax = theo["rms_speed"] * cfg.speed_max_factor
    hist, edges = np.histogram(speeds, bins=cfg.n_bins, range=(0.0, vmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = maxwell_boltzmann_pdf(centers, cfg)
    pdf_rmse = float(np.sqrt(np.mean((hist - pdf) ** 2)))
    ks_dist = compute_ks_distance(speeds, cfg)

    rows = []
    for key in ("most_probable_speed", "mean_speed", "rms_speed"):
        t_val = theo[key]
        e_val = emp[key]
        rel_err = abs(e_val - t_val) / t_val
        rows.append(
            {
                "metric": key,
                "theory": t_val,
                "empirical": e_val,
                "relative_error": rel_err,
            }
        )

    t_rel_err = abs(emp["temperature_hat"] - cfg.temperature) / cfg.temperature
    rows.append(
        {
            "metric": "temperature_from_second_moment",
            "theory": cfg.temperature,
            "empirical": emp["temperature_hat"],
            "relative_error": t_rel_err,
        }
    )
    return pd.DataFrame(rows), ks_dist, pdf_rmse


def main() -> None:
    cfg = MBConfig()
    cfg.validate()

    speeds = sample_speeds(cfg)
    summary_df, ks_dist, pdf_rmse = summarize_distribution(speeds, cfg)

    print("=== Maxwell-Boltzmann Distribution MVP ===")
    print(
        "config:",
        f"T={cfg.temperature}, m={cfg.mass}, k_B={cfg.k_b},",
        f"n_samples={cfg.n_samples}, n_bins={cfg.n_bins}, seed={cfg.seed}",
    )
    print()
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"KS distance (empirical CDF vs theory CDF): {ks_dist:.6f}")
    print(f"PDF RMSE (histogram vs theory PDF):       {pdf_rmse:.6f}")

    mean_speed = float(summary_df.loc[summary_df["metric"] == "mean_speed", "empirical"].iloc[0])
    most_probable_speed = float(summary_df.loc[summary_df["metric"] == "most_probable_speed", "empirical"].iloc[0])
    rms_speed = float(summary_df.loc[summary_df["metric"] == "rms_speed", "empirical"].iloc[0])

    if not (most_probable_speed < mean_speed < rms_speed):
        raise AssertionError("Expected most_probable < mean < rms for Maxwell-Boltzmann speeds")

    t_rel_err = float(
        summary_df.loc[summary_df["metric"] == "temperature_from_second_moment", "relative_error"].iloc[0]
    )
    mean_rel_err = float(summary_df.loc[summary_df["metric"] == "mean_speed", "relative_error"].iloc[0])

    if t_rel_err > 0.025:
        raise AssertionError(f"Temperature inversion relative error too large: {t_rel_err:.4f}")
    if mean_rel_err > 0.02:
        raise AssertionError(f"Mean-speed relative error too large: {mean_rel_err:.4f}")
    if ks_dist > 0.01:
        raise AssertionError(f"KS distance too large: {ks_dist:.4f}")
    if pdf_rmse > 0.04:
        raise AssertionError(f"PDF RMSE too large: {pdf_rmse:.4f}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
