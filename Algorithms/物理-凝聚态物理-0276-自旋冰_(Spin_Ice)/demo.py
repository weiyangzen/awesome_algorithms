"""Minimal runnable MVP for Spin Ice using a tetrahedral toy model.

Model summary:
- Each tetrahedron has 4 Ising pseudo-spins sigma_i in {-1, +1}.
- Effective Hamiltonian per tetrahedron:
      E_tet = J_eff * sum_{i<j} sigma_i sigma_j - h * sum_i sigma_i (d_i · n)
  where d_i are local <111> easy-axis unit vectors, and n is external field direction.
- For J_eff > 0 and h = 0, low-energy manifold is the 2-in-2-out ice rule.

This script provides:
1) Metropolis Monte Carlo sampling over many independent tetrahedra,
2) exact 16-state thermodynamics for one tetrahedron (for validation),
3) cross-check with a PyTorch re-implementation of the same energy,
4) Arrhenius fit of monopole density using scikit-learn.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class SpinIceConfig:
    """Configuration for the spin-ice toy simulation."""

    n_tetra: int = 192
    j_eff: float = 1.0
    field_strength: float = 0.0
    field_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)

    t_min: float = 0.35
    t_max: float = 2.80
    n_temps: int = 12

    equil_sweeps: int = 220
    meas_sweeps: int = 320
    arrhenius_t_max: float = 0.90

    seed: int = 20260407


def local_easy_axes() -> np.ndarray:
    """Return 4 local <111> easy axes for one tetrahedron."""
    axes = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return axes / np.sqrt(3.0)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return normalized vector with finite-value checks."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError("field_direction must be a finite 3D vector")
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("field_direction norm is too small")
    return vec / norm


def field_projections(cfg: SpinIceConfig) -> np.ndarray:
    """Compute d_i · n for each local axis d_i."""
    axes = local_easy_axes()
    direction = normalize(np.array(cfg.field_direction, dtype=np.float64))
    return axes @ direction


def random_spins(n_tetra: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random ±1 spin states, shape (n_tetra, 4)."""
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_tetra, 4), replace=True)
    return spins


def pair_sum_per_tetra(spins: np.ndarray) -> np.ndarray:
    """Compute sum_{i<j} sigma_i sigma_j for each tetrahedron."""
    s0 = spins[:, 0]
    s1 = spins[:, 1]
    s2 = spins[:, 2]
    s3 = spins[:, 3]
    return s0 * s1 + s0 * s2 + s0 * s3 + s1 * s2 + s1 * s3 + s2 * s3


def energy_per_tetra(spins: np.ndarray, cfg: SpinIceConfig, proj: np.ndarray) -> np.ndarray:
    """Vectorized tetrahedron energies."""
    pair = pair_sum_per_tetra(spins).astype(np.float64)
    zeeman = (spins.astype(np.float64) @ proj).astype(np.float64)
    return cfg.j_eff * pair - cfg.field_strength * zeeman


def total_energy(spins: np.ndarray, cfg: SpinIceConfig, proj: np.ndarray) -> float:
    """Total energy across all tetrahedra."""
    return float(np.sum(energy_per_tetra(spins, cfg, proj)))


def metropolis_sweep(
    spins: np.ndarray,
    temperature: float,
    cfg: SpinIceConfig,
    proj: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """One Metropolis sweep: propose each sublattice spin once for every tetrahedron.

    Update is vectorized over tetrahedra and sequential over 4 spin sublattices.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    beta = 1.0 / temperature
    for k in range(4):
        s = spins[:, k].astype(np.float64)
        sum_others = spins.sum(axis=1).astype(np.float64) - s

        # Delta E after sigma_k -> -sigma_k.
        delta_e = -2.0 * cfg.j_eff * s * sum_others + 2.0 * cfg.field_strength * s * proj[k]

        accept = delta_e <= 0.0
        uphill = ~accept
        if np.any(uphill):
            probs = np.exp(-beta * delta_e[uphill])
            accept[uphill] = rng.random(np.count_nonzero(uphill)) < probs

        spins[accept, k] *= -1


def measure_state(spins: np.ndarray, cfg: SpinIceConfig, proj: np.ndarray) -> dict[str, float]:
    """Measure energy and defect fractions for current spin state."""
    e_tet = energy_per_tetra(spins, cfg, proj)
    q = spins.sum(axis=1)

    return {
        "energy_total": float(np.sum(e_tet)),
        "ice_fraction": float(np.mean(q == 0)),
        "monopole_fraction": float(np.mean(np.abs(q) == 2)),
        "double_charge_fraction": float(np.mean(np.abs(q) == 4)),
    }


def simulate_one_temperature(
    spins: np.ndarray,
    temperature: float,
    cfg: SpinIceConfig,
    proj: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run equilibration + measurement at one temperature."""
    for _ in range(cfg.equil_sweeps):
        metropolis_sweep(spins, temperature, cfg, proj, rng)

    e_samples: list[float] = []
    ice_samples: list[float] = []
    mono_samples: list[float] = []
    double_samples: list[float] = []

    for _ in range(cfg.meas_sweeps):
        metropolis_sweep(spins, temperature, cfg, proj, rng)
        obs = measure_state(spins, cfg, proj)

        e_samples.append(obs["energy_total"])
        ice_samples.append(obs["ice_fraction"])
        mono_samples.append(obs["monopole_fraction"])
        double_samples.append(obs["double_charge_fraction"])

    e_arr = np.asarray(e_samples, dtype=np.float64)
    n_spins = cfg.n_tetra * 4

    e_mean = float(np.mean(e_arr))
    e2_mean = float(np.mean(e_arr**2))

    return {
        "temperature": temperature,
        "energy_per_spin_mc": e_mean / n_spins,
        "specific_heat_mc": (e2_mean - e_mean * e_mean) / (n_spins * temperature * temperature),
        "ice_fraction_mc": float(np.mean(ice_samples)),
        "monopole_fraction_mc": float(np.mean(mono_samples)),
        "double_charge_fraction_mc": float(np.mean(double_samples)),
    }


def enumerate_tetra_states() -> np.ndarray:
    """Enumerate all 16 Ising states of one tetrahedron."""
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=4)), dtype=np.float64)
    return states


def exact_thermo_table(temperatures: np.ndarray, cfg: SpinIceConfig, proj: np.ndarray) -> pd.DataFrame:
    """Exact thermodynamics from full 16-state enumeration."""
    states = enumerate_tetra_states()
    q = states.sum(axis=1)

    pair = (
        states[:, 0] * states[:, 1]
        + states[:, 0] * states[:, 2]
        + states[:, 0] * states[:, 3]
        + states[:, 1] * states[:, 2]
        + states[:, 1] * states[:, 3]
        + states[:, 2] * states[:, 3]
    )
    zeeman = states @ proj
    energies = cfg.j_eff * pair - cfg.field_strength * zeeman

    rows: list[dict[str, float]] = []
    for temp in temperatures:
        logits = -energies / temp
        log_z = float(logsumexp(logits))
        weights = np.exp(logits - log_z)

        e_mean = float(np.sum(weights * energies))
        e2_mean = float(np.sum(weights * energies * energies))

        # F = -T ln Z, S = (E - F)/T = (E + T ln Z)/T.
        entropy_per_spin = (e_mean + temp * log_z) / (4.0 * temp)

        rows.append(
            {
                "temperature": float(temp),
                "energy_per_spin_exact": e_mean / 4.0,
                "specific_heat_exact": (e2_mean - e_mean * e_mean) / (4.0 * temp * temp),
                "ice_fraction_exact": float(np.sum(weights * (q == 0))),
                "monopole_fraction_exact": float(np.sum(weights * (np.abs(q) == 2))),
                "double_charge_fraction_exact": float(np.sum(weights * (np.abs(q) == 4))),
                "entropy_per_spin_exact": entropy_per_spin,
            }
        )

    return pd.DataFrame(rows)


def torch_energy_consistency(spins: np.ndarray, cfg: SpinIceConfig, proj: np.ndarray) -> float:
    """Cross-check NumPy energy against PyTorch implementation."""
    spins_t = torch.tensor(spins, dtype=torch.float64)
    proj_t = torch.tensor(proj, dtype=torch.float64)

    s0 = spins_t[:, 0]
    s1 = spins_t[:, 1]
    s2 = spins_t[:, 2]
    s3 = spins_t[:, 3]
    pair_t = s0 * s1 + s0 * s2 + s0 * s3 + s1 * s2 + s1 * s3 + s2 * s3
    zeeman_t = spins_t @ proj_t
    e_torch = cfg.j_eff * pair_t - cfg.field_strength * zeeman_t

    e_numpy = torch.tensor(energy_per_tetra(spins, cfg, proj), dtype=torch.float64)
    return float(torch.max(torch.abs(e_torch - e_numpy)).item())


def fit_arrhenius_activation(table: pd.DataFrame, t_max: float) -> dict[str, float]:
    """Fit log(monopole_fraction_exact) = a + b*(1/T), activation ~ -b."""
    mask = (table["temperature"] <= t_max) & (table["monopole_fraction_exact"] > 1e-12)
    subset = table.loc[mask, ["temperature", "monopole_fraction_exact"]].copy()
    if len(subset) < 3:
        raise RuntimeError("Not enough low-temperature points for Arrhenius fit")

    x = (1.0 / subset["temperature"].to_numpy()).reshape(-1, 1)
    y = np.log(subset["monopole_fraction_exact"].to_numpy())

    reg = LinearRegression()
    reg.fit(x, y)
    slope = float(reg.coef_[0])

    return {
        "activation_energy_est": -slope,
        "intercept": float(reg.intercept_),
        "r2": float(reg.score(x, y)),
        "n_points": float(len(subset)),
    }


def main() -> None:
    cfg = SpinIceConfig()
    rng = np.random.default_rng(cfg.seed)

    temperatures = np.linspace(cfg.t_min, cfg.t_max, cfg.n_temps, dtype=np.float64)
    proj = field_projections(cfg)
    spins = random_spins(cfg.n_tetra, rng)

    mc_rows: list[dict[str, float]] = []
    for temp in temperatures:
        row = simulate_one_temperature(spins, float(temp), cfg, proj, rng)
        mc_rows.append(row)

    mc_df = pd.DataFrame(mc_rows)
    exact_df = exact_thermo_table(temperatures, cfg, proj)
    merged = mc_df.merge(exact_df, on="temperature", how="inner")

    energy_rmse = float(
        np.sqrt(
            mean_squared_error(
                merged["energy_per_spin_exact"],
                merged["energy_per_spin_mc"],
            )
        )
    )
    cv_rmse = float(
        np.sqrt(
            mean_squared_error(
                merged["specific_heat_exact"],
                merged["specific_heat_mc"],
            )
        )
    )
    monopole_rmse = float(
        np.sqrt(
            mean_squared_error(
                merged["monopole_fraction_exact"],
                merged["monopole_fraction_mc"],
            )
        )
    )

    torch_max_diff = torch_energy_consistency(spins, cfg, proj)
    arrhenius = fit_arrhenius_activation(merged, cfg.arrhenius_t_max)

    low_t_ice = float(merged.loc[merged["temperature"].idxmin(), "ice_fraction_mc"])
    high_t_ice = float(merged.loc[merged["temperature"].idxmax(), "ice_fraction_mc"])
    high_t_ice_exact = float(merged.loc[merged["temperature"].idxmax(), "ice_fraction_exact"])
    low_t_entropy = float(merged.loc[merged["temperature"].idxmin(), "entropy_per_spin_exact"])
    expected_pauling_limit_independent = np.log(6.0) / 4.0

    out_path = Path(__file__).with_name("thermo_results.csv")
    merged.to_csv(out_path, index=False)

    print("=== Spin Ice (Independent Tetrahedra) MVP ===")
    print("\nConfig:")
    print(pd.Series(asdict(cfg)).to_string())

    print("\nThermodynamic table sample:")
    print(
        merged[
            [
                "temperature",
                "energy_per_spin_mc",
                "energy_per_spin_exact",
                "specific_heat_mc",
                "specific_heat_exact",
                "ice_fraction_mc",
                "monopole_fraction_mc",
            ]
        ].head(8).to_string(index=False, float_format=lambda x: f"{x: .6f}")
    )

    print("\nConsistency metrics:")
    print(f"energy RMSE (MC vs exact)      = {energy_rmse:.4e}")
    print(f"specific-heat RMSE (MC vs exact)= {cv_rmse:.4e}")
    print(f"monopole RMSE (MC vs exact)    = {monopole_rmse:.4e}")
    print(f"torch max |E_torch - E_numpy|  = {torch_max_diff:.4e}")

    print("\nLow-T / High-T indicators:")
    print(f"low-T ice fraction (MC)        = {low_t_ice:.4f}")
    print(f"high-T ice fraction (MC)       = {high_t_ice:.4f}")
    print(f"high-T ice fraction (exact)    = {high_t_ice_exact:.4f}")
    print(f"low-T entropy per spin (exact) = {low_t_entropy:.6f}")
    print(f"ln(6)/4 reference              = {expected_pauling_limit_independent:.6f}")

    print("\nArrhenius fit on exact monopole density:")
    print(f"activation energy estimate     = {arrhenius['activation_energy_est']:.4f}")
    print(f"expected ~ 2*J_eff             = {2.0 * cfg.j_eff:.4f}")
    print(f"R^2                            = {arrhenius['r2']:.6f}")
    print(f"points used                    = {int(arrhenius['n_points'])}")

    assert energy_rmse < 0.06, "Energy mismatch between MC and exact is too large."
    assert cv_rmse < 0.10, "Specific heat mismatch between MC and exact is too large."
    assert monopole_rmse < 0.06, "Monopole fraction mismatch is too large."
    assert torch_max_diff < 1e-12, "Torch and NumPy energy computations disagree."
    assert low_t_ice > 0.80, "Low-temperature ice-rule fraction is unexpectedly low."
    assert abs(high_t_ice - high_t_ice_exact) < 0.05, "High-temperature ice fraction disagrees with exact result."
    assert abs(arrhenius["activation_energy_est"] - 2.0 * cfg.j_eff) < 0.30
    assert abs(low_t_entropy - expected_pauling_limit_independent) < 0.08

    print(f"\nSaved detailed table to: {out_path}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
