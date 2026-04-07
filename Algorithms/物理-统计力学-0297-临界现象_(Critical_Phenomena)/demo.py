"""MVP for critical phenomena using a 2D Ising model (Metropolis sampling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    lattice_size: int = 24
    temperatures: tuple[float, ...] = (1.8, 2.0, 2.15, 2.25, 2.35, 2.5, 2.8, 3.2)
    warmup_sweeps: int = 300
    sample_steps: int = 240
    sample_interval: int = 3
    seed: int = 20260407


class Ising2D:
    """Square-lattice Ising model with periodic boundary conditions."""

    def __init__(self, lattice_size: int, rng: np.random.Generator) -> None:
        self.lattice_size = lattice_size
        self.n_spins = lattice_size * lattice_size
        self.spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(lattice_size, lattice_size))

    def metropolis_sweep(self, beta: float, rng: np.random.Generator) -> None:
        """Attempt one spin-flip update per site on average."""
        l = self.lattice_size
        spins = self.spins
        for _ in range(self.n_spins):
            i = int(rng.integers(0, l))
            j = int(rng.integers(0, l))
            s_ij = int(spins[i, j])
            neighbor_sum = int(
                spins[(i - 1) % l, j]
                + spins[(i + 1) % l, j]
                + spins[i, (j - 1) % l]
                + spins[i, (j + 1) % l]
            )
            delta_e = 2 * s_ij * neighbor_sum
            if delta_e <= 0 or rng.random() < np.exp(-beta * delta_e):
                spins[i, j] = np.int8(-s_ij)

    def total_energy(self) -> float:
        """Return total energy E (J=1, h=0), counting each bond once."""
        s = self.spins
        return float(-np.sum(s * (np.roll(s, 1, axis=0) + np.roll(s, 1, axis=1))))

    def magnetization(self) -> float:
        return float(np.sum(self.spins))


def simulate_one_temperature(config: SimulationConfig, temperature: float, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model = Ising2D(config.lattice_size, rng)
    beta = 1.0 / temperature

    for _ in range(config.warmup_sweeps):
        model.metropolis_sweep(beta, rng)

    energies: list[float] = []
    magnetizations: list[float] = []
    for _ in range(config.sample_steps):
        for _ in range(config.sample_interval):
            model.metropolis_sweep(beta, rng)
        energies.append(model.total_energy())
        magnetizations.append(model.magnetization())

    n = float(model.n_spins)
    e = np.asarray(energies, dtype=float) / n
    m = np.asarray(magnetizations, dtype=float) / n

    mean_e = float(np.mean(e))
    mean_abs_m = float(np.mean(np.abs(m)))
    susceptibility = float(n * (np.mean(m * m) - np.mean(m) ** 2) / temperature)
    heat_capacity = float(n * (np.mean(e * e) - mean_e * mean_e) / (temperature * temperature))

    return {
        "temperature": float(temperature),
        "energy_per_spin": mean_e,
        "abs_magnetization": mean_abs_m,
        "susceptibility": susceptibility,
        "heat_capacity": heat_capacity,
    }


def run_temperature_scan(config: SimulationConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for idx, temp in enumerate(config.temperatures):
        rows.append(simulate_one_temperature(config, float(temp), seed=config.seed + 7919 * idx))
    return pd.DataFrame(rows).sort_values("temperature", ignore_index=True)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def estimate_critical_temperature(df: pd.DataFrame) -> tuple[float, float, float]:
    t_chi = float(df.loc[df["susceptibility"].idxmax(), "temperature"])
    t_cv = float(df.loc[df["heat_capacity"].idxmax(), "temperature"])
    return t_chi, t_cv, 0.5 * (t_chi + t_cv)


def estimate_beta_exponent(df: pd.DataFrame, tc_est: float) -> tuple[float, float]:
    sub = df[df["temperature"] < tc_est].copy()
    sub["tau"] = tc_est - sub["temperature"]
    sub = sub[(sub["tau"] > 1e-8) & (sub["abs_magnetization"] > 1e-8)]
    if len(sub) < 3:
        return float("nan"), float("nan")
    x = np.log(sub["tau"].to_numpy())
    y = np.log(sub["abs_magnetization"].to_numpy())
    beta_est, _, r2 = linear_fit(x, y)
    return float(beta_est), float(r2)


def main() -> None:
    config = SimulationConfig()
    df = run_temperature_scan(config)
    t_chi, t_cv, tc_est = estimate_critical_temperature(df)
    beta_est, beta_r2 = estimate_beta_exponent(df, tc_est)

    print("=== 2D Ising Critical-Phenomena MVP (Metropolis) ===")
    print(
        "config:",
        f"L={config.lattice_size}, temperatures={list(config.temperatures)},",
        f"warmup_sweeps={config.warmup_sweeps}, sample_steps={config.sample_steps},",
        f"sample_interval={config.sample_interval}, seed={config.seed}",
    )
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print()
    print(f"Peak susceptibility temperature      T_chi* = {t_chi:.4f}")
    print(f"Peak heat-capacity temperature      T_cv*  = {t_cv:.4f}")
    print(f"Combined finite-size Tc estimate    Tc(L)  = {tc_est:.4f}")
    print("Onsager exact Tc (2D Ising, infinite lattice): Tc = 2.2692")
    if np.isfinite(beta_est):
        print(f"Order-parameter exponent fit below Tc: beta ≈ {beta_est:.4f} (R^2={beta_r2:.4f})")
    else:
        print("Order-parameter exponent fit skipped: not enough valid points below Tc.")

    trend_ok = (
        df["abs_magnetization"].iloc[0] > df["abs_magnetization"].iloc[-1]
        and float(df["susceptibility"].max()) > 0.0
        and float(df["heat_capacity"].max()) > 0.0
    )
    print(f"Critical-signature sanity check: {'PASS' if trend_ok else 'CHECK_MANUALLY'}")


if __name__ == "__main__":
    main()
