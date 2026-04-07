"""Minimal MVP for scaling laws in statistical mechanics via 2D Ising sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    lattice_sizes: tuple[int, ...] = (10, 14, 18, 22)
    temperatures: tuple[float, ...] = (2.15, 2.22, 2.27, 2.32, 2.39)
    warmup_sweeps: int = 140
    sample_steps: int = 140
    sample_interval: int = 2
    replicas_per_point: int = 3
    seed: int = 20260407
    tc_exact: float = 2.0 / np.log(1.0 + np.sqrt(2.0))
    beta_over_nu_theory: float = 1.0 / 8.0
    gamma_over_nu_theory: float = 7.0 / 4.0
    nu_theory: float = 1.0


class Ising2D:
    """Square-lattice Ising model with periodic boundary conditions."""

    def __init__(self, lattice_size: int, rng: np.random.Generator) -> None:
        self.lattice_size = lattice_size
        self.n_spins = lattice_size * lattice_size
        self.spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(lattice_size, lattice_size))

    def metropolis_sweep(self, beta: float, rng: np.random.Generator) -> None:
        """Attempt one spin-flip update per site on average."""
        l = self.lattice_size
        s = self.spins
        for _ in range(self.n_spins):
            i = int(rng.integers(0, l))
            j = int(rng.integers(0, l))
            sij = int(s[i, j])
            nn = int(s[(i - 1) % l, j] + s[(i + 1) % l, j] + s[i, (j - 1) % l] + s[i, (j + 1) % l])
            delta_e = 2 * sij * nn
            if delta_e <= 0 or rng.random() < np.exp(-beta * delta_e):
                s[i, j] = np.int8(-sij)

    def total_energy(self) -> float:
        """Return total energy E (J=1, h=0), counting each bond once."""
        s = self.spins
        return float(-np.sum(s * (np.roll(s, 1, axis=0) + np.roll(s, 1, axis=1))))

    def magnetization(self) -> float:
        return float(np.sum(self.spins))


def simulate_point(config: SimulationConfig, lattice_size: int, temperature: float, seed: int) -> dict[str, float]:
    energies: list[float] = []
    magnetizations: list[float] = []
    beta = 1.0 / temperature

    for rep in range(config.replicas_per_point):
        rng = np.random.default_rng(seed + 104_729 * rep)
        model = Ising2D(lattice_size=lattice_size, rng=rng)

        for _ in range(config.warmup_sweeps):
            model.metropolis_sweep(beta=beta, rng=rng)

        for _ in range(config.sample_steps):
            for _ in range(config.sample_interval):
                model.metropolis_sweep(beta=beta, rng=rng)
            energies.append(model.total_energy())
            magnetizations.append(model.magnetization())

    n = float(lattice_size * lattice_size)
    e = np.asarray(energies, dtype=float) / n
    m = np.asarray(magnetizations, dtype=float) / n

    mean_e = float(np.mean(e))
    mean_abs_m = float(np.mean(np.abs(m)))
    susceptibility = float(n * (np.mean(m * m) - np.mean(m) ** 2) / temperature)

    return {
        "lattice_size": float(lattice_size),
        "temperature": float(temperature),
        "energy_per_spin": mean_e,
        "abs_magnetization": mean_abs_m,
        "susceptibility": susceptibility,
    }


def run_scan(config: SimulationConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for li, lattice_size in enumerate(config.lattice_sizes):
        for ti, temperature in enumerate(config.temperatures):
            point_seed = config.seed + 1009 * li + 9173 * ti
            rows.append(simulate_point(config, lattice_size=lattice_size, temperature=temperature, seed=point_seed))

    df = pd.DataFrame(rows)
    df["lattice_size"] = df["lattice_size"].astype(int)
    return df.sort_values(["lattice_size", "temperature"], ignore_index=True)


def fit_log_log(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    log_x = np.log(np.asarray(x, dtype=float))
    log_y = np.log(np.asarray(y, dtype=float))

    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def extract_near_tc_slice(df: pd.DataFrame, tc: float) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, grp in df.groupby("lattice_size", sort=True):
        idx = (grp["temperature"] - tc).abs().idxmin()
        rows.append(df.loc[idx])
    return pd.DataFrame(rows).sort_values("lattice_size", ignore_index=True)


def extract_peak_susceptibility_slice(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, grp in df.groupby("lattice_size", sort=True):
        idx = grp["susceptibility"].idxmax()
        rows.append(df.loc[idx])
    return pd.DataFrame(rows).sort_values("lattice_size", ignore_index=True)


def binwise_cv(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return float("nan")

    edges = np.linspace(xmin, xmax, n_bins + 1)
    cvs: list[float] = []
    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if int(np.sum(mask)) < 3:
            continue
        vals = y[mask]
        denom = float(np.mean(np.abs(vals)))
        if denom <= 1e-12:
            continue
        cvs.append(float(np.std(vals) / denom))

    return float(np.mean(cvs)) if cvs else float("nan")


def compute_scaled_frame(
    df: pd.DataFrame,
    tc: float,
    beta_over_nu: float,
    gamma_over_nu: float,
    nu: float,
) -> tuple[pd.DataFrame, float, float]:
    scaled = df.copy()
    l = scaled["lattice_size"].to_numpy(dtype=float)
    t = scaled["temperature"].to_numpy(dtype=float)

    scaled["x_scaled"] = (t - tc) * np.power(l, 1.0 / nu)
    scaled["m_scaled"] = scaled["abs_magnetization"] * np.power(l, beta_over_nu)
    scaled["chi_scaled"] = scaled["susceptibility"] / np.power(l, gamma_over_nu)

    cv_m = binwise_cv(scaled["x_scaled"].to_numpy(), scaled["m_scaled"].to_numpy())
    cv_chi = binwise_cv(scaled["x_scaled"].to_numpy(), scaled["chi_scaled"].to_numpy())
    return scaled, cv_m, cv_chi


def trend_sanity_check(df: pd.DataFrame) -> bool:
    checks: list[bool] = []
    for _, grp in df.groupby("lattice_size", sort=True):
        g = grp.sort_values("temperature")
        checks.append(float(g["abs_magnetization"].iloc[0]) > float(g["abs_magnetization"].iloc[-1]))
    return bool(np.all(checks))


def main() -> None:
    config = SimulationConfig()
    df = run_scan(config)

    tc_slice = extract_near_tc_slice(df=df, tc=config.tc_exact)
    chi_peak_slice = extract_peak_susceptibility_slice(df=df)
    slope_m, _, r2_m = fit_log_log(
        tc_slice["lattice_size"].to_numpy(dtype=float),
        tc_slice["abs_magnetization"].to_numpy(dtype=float),
    )
    slope_chi, _, r2_chi = fit_log_log(
        chi_peak_slice["lattice_size"].to_numpy(dtype=float),
        chi_peak_slice["susceptibility"].to_numpy(dtype=float),
    )

    beta_over_nu_est = -slope_m
    gamma_over_nu_est = slope_chi

    scaled_df, cv_m, cv_chi = compute_scaled_frame(
        df=df,
        tc=config.tc_exact,
        beta_over_nu=config.beta_over_nu_theory,
        gamma_over_nu=config.gamma_over_nu_theory,
        nu=config.nu_theory,
    )

    print("=== Scaling-Law MVP via 2D Ising (Metropolis) ===")
    print(
        "config:",
        f"L={list(config.lattice_sizes)}, T={list(config.temperatures)},",
        f"warmup={config.warmup_sweeps}, sample_steps={config.sample_steps},",
        f"sample_interval={config.sample_interval}, replicas={config.replicas_per_point}, seed={config.seed}",
    )
    print(f"theory: Tc={config.tc_exact:.6f}, beta/nu={config.beta_over_nu_theory:.4f}, gamma/nu={config.gamma_over_nu_theory:.4f}, nu={config.nu_theory:.4f}")
    print()

    print("[scan table]")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print()

    print("[near-Tc slice for power-law fit]")
    print(tc_slice.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print()

    print("[chi-peak slice for power-law fit]")
    print(chi_peak_slice.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print()

    print("[estimated exponents from near-Tc finite-size power laws]")
    print(f"beta/nu  estimate = {beta_over_nu_est:.4f} (fit R^2={r2_m:.4f})")
    print(f"gamma/nu estimate = {gamma_over_nu_est:.4f} (fit R^2={r2_chi:.4f}, from chi_max(L))")
    print()

    print("[collapse quality using theory exponents]")
    print(f"collapse CV for m-scaled   = {cv_m:.4f}")
    print(f"collapse CV for chi-scaled = {cv_chi:.4f}")
    print()

    print("[scaled sample rows]")
    cols = ["lattice_size", "temperature", "x_scaled", "m_scaled", "chi_scaled"]
    print(scaled_df[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print()

    print(f"trend sanity check (|m| low T > high T for each L): {'PASS' if trend_sanity_check(df) else 'CHECK_MANUALLY'}")


if __name__ == "__main__":
    main()
