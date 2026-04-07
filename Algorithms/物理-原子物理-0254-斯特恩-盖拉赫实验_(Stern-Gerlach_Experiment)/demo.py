"""Minimal runnable MVP for Stern-Gerlach experiment.

This script compares:
1) Quantum spin-1/2 model (two discrete magnetic moment projections)
2) Classical dipole orientation model (continuous projection)

Both models share the same beamline geometry and velocity spread. The output
shows how a quantized spin projection gives two detector spots, while a
classical continuous orientation gives a single broadened distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.constants import atomic_mass, physical_constants
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

MU_B = physical_constants["Bohr magneton"][0]  # J/T
ELECTRON_G = 2.00231930436256


@dataclass(frozen=True)
class SGConfig:
    num_atoms: int = 40000
    seed: int = 7
    field_gradient_t_per_m: float = 1600.0
    magnet_length_m: float = 0.035
    drift_length_m: float = 0.45
    velocity_mean_m_s: float = 800.0
    velocity_std_m_s: float = 80.0
    min_velocity_m_s: float = 350.0
    initial_spot_sigma_m: float = 2.0e-5
    silver_mass_amu: float = 107.8682
    electron_g: float = ELECTRON_G

    @property
    def atom_mass_kg(self) -> float:
        return self.silver_mass_amu * atomic_mass

    @property
    def quantum_mu_eff(self) -> float:
        # |mu_z| for spin-1/2 with m_s=+-1/2 in this simplified model.
        return 0.5 * self.electron_g * MU_B


@dataclass
class PeakProfile:
    grid_mm: np.ndarray
    density: np.ndarray
    peak_positions_mm: np.ndarray

    @property
    def peak_count(self) -> int:
        return int(self.peak_positions_mm.size)


def sample_forward_velocity(cfg: SGConfig, rng: np.random.Generator, size: int) -> np.ndarray:
    vx = rng.normal(loc=cfg.velocity_mean_m_s, scale=cfg.velocity_std_m_s, size=size)
    return np.clip(vx, cfg.min_velocity_m_s, None)


def ballistic_factor(vx_m_s: np.ndarray, cfg: SGConfig) -> np.ndarray:
    """Return k(v) so that z = z0 + a_z * k(v), with a_z = (mu_z * dB/dz)/m."""
    t_m = cfg.magnet_length_m / vx_m_s
    t_d = cfg.drift_length_m / vx_m_s
    return t_m * (0.5 * t_m + t_d)


def simulate_quantum_beam(cfg: SGConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    vx = sample_forward_velocity(cfg=cfg, rng=rng, size=cfg.num_atoms)
    k = ballistic_factor(vx_m_s=vx, cfg=cfg)

    spin_sign = rng.choice(np.array([-1.0, 1.0]), size=cfg.num_atoms)
    m_s = 0.5 * spin_sign
    mu_z = cfg.quantum_mu_eff * spin_sign

    z0 = rng.normal(loc=0.0, scale=cfg.initial_spot_sigma_m, size=cfg.num_atoms)
    acceleration_z = (mu_z * cfg.field_gradient_t_per_m) / cfg.atom_mass_kg
    z_m = z0 + acceleration_z * k

    return pd.DataFrame(
        {
            "model": "quantum",
            "spin_sign": spin_sign,
            "m_s": m_s,
            "mu_z_J_T": mu_z,
            "vx_m_s": vx,
            "k_s2": k,
            "z_m": z_m,
            "z_mm": z_m * 1e3,
        }
    )


def simulate_classical_beam(cfg: SGConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    vx = sample_forward_velocity(cfg=cfg, rng=rng, size=cfg.num_atoms)
    k = ballistic_factor(vx_m_s=vx, cfg=cfg)

    cos_theta = rng.uniform(-1.0, 1.0, size=cfg.num_atoms)
    mu_z = cfg.quantum_mu_eff * cos_theta

    z0 = rng.normal(loc=0.0, scale=cfg.initial_spot_sigma_m, size=cfg.num_atoms)
    acceleration_z = (mu_z * cfg.field_gradient_t_per_m) / cfg.atom_mass_kg
    z_m = z0 + acceleration_z * k

    return pd.DataFrame(
        {
            "model": "classical",
            "cos_theta": cos_theta,
            "mu_z_J_T": mu_z,
            "vx_m_s": vx,
            "k_s2": k,
            "z_m": z_m,
            "z_mm": z_m * 1e3,
        }
    )


def analyze_peaks_kde(samples_mm: np.ndarray, bw_method: float = 0.18) -> PeakProfile:
    samples_mm = np.asarray(samples_mm, dtype=np.float64)
    q_low, q_high = np.quantile(samples_mm, [0.001, 0.999])
    grid_mm = np.linspace(q_low, q_high, 800)
    kde = gaussian_kde(samples_mm, bw_method=bw_method)
    density = kde(grid_mm)

    peaks, _ = find_peaks(
        density,
        height=float(np.max(density) * 0.20),
        distance=100,
        prominence=float(np.max(density) * 0.05),
    )
    peak_positions_mm = grid_mm[peaks]

    return PeakProfile(grid_mm=grid_mm, density=density, peak_positions_mm=peak_positions_mm)


def fit_quantum_gmm(samples_mm: np.ndarray, seed: int) -> dict[str, np.ndarray | float]:
    samples_mm = np.asarray(samples_mm, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.fit(samples_mm)

    means = gmm.means_.ravel()
    weights = gmm.weights_.ravel()
    variances = gmm.covariances_.ravel()

    order = np.argsort(means)
    means = means[order]
    weights = weights[order]
    stds = np.sqrt(variances[order])

    pooled_std = float(np.sqrt(0.5 * (stds[0] ** 2 + stds[1] ** 2)))
    separation = float(abs(means[1] - means[0]) / pooled_std)

    return {
        "means_mm": means,
        "stds_mm": stds,
        "weights": weights,
        "separation_sigma": separation,
    }


def expected_center_mm(cfg: SGConfig, v_ref_m_s: float) -> float:
    k = ballistic_factor(vx_m_s=np.array([v_ref_m_s], dtype=np.float64), cfg=cfg)[0]
    z = (cfg.quantum_mu_eff * cfg.field_gradient_t_per_m / cfg.atom_mass_kg) * k
    return float(z * 1e3)


def estimate_mu_torch_optional(quantum_df: pd.DataFrame, cfg: SGConfig) -> Optional[float]:
    """Estimate |mu_z| from z = beta * (spin_sign * k), beta = mu*grad/m."""
    if torch is None:
        return None

    scale_x = 1.0e8

    x = torch.tensor(
        quantum_df["spin_sign"].to_numpy(dtype=np.float64)
        * quantum_df["k_s2"].to_numpy(dtype=np.float64)
        * scale_x,
        dtype=torch.float64,
    )
    y = torch.tensor(quantum_df["z_m"].to_numpy(dtype=np.float64), dtype=torch.float64)

    beta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([beta], lr=0.05)

    for _ in range(2500):
        optimizer.zero_grad()
        pred = beta * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    return float(beta.detach().cpu().item() * scale_x * cfg.atom_mass_kg / cfg.field_gradient_t_per_m)


def summary_table(quantum_df: pd.DataFrame, classical_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for df in (quantum_df, classical_df):
        model = str(df["model"].iloc[0])
        z = df["z_mm"].to_numpy(dtype=np.float64)
        records.append(
            {
                "model": model,
                "mean_z_mm": float(np.mean(z)),
                "std_z_mm": float(np.std(z)),
                "p05_z_mm": float(np.quantile(z, 0.05)),
                "median_z_mm": float(np.median(z)),
                "p95_z_mm": float(np.quantile(z, 0.95)),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    cfg = SGConfig()

    quantum_df = simulate_quantum_beam(cfg)
    classical_df = simulate_classical_beam(cfg)

    quantum_peaks = analyze_peaks_kde(quantum_df["z_mm"].to_numpy(dtype=np.float64))
    classical_peaks = analyze_peaks_kde(classical_df["z_mm"].to_numpy(dtype=np.float64))

    gmm_result = fit_quantum_gmm(quantum_df["z_mm"].to_numpy(dtype=np.float64), seed=cfg.seed)
    means_mm = np.asarray(gmm_result["means_mm"], dtype=np.float64)
    stds_mm = np.asarray(gmm_result["stds_mm"], dtype=np.float64)
    weights = np.asarray(gmm_result["weights"], dtype=np.float64)
    separation_sigma = float(gmm_result["separation_sigma"])

    v_ref = float(np.mean(quantum_df["vx_m_s"].to_numpy(dtype=np.float64)))
    expected_center = expected_center_mm(cfg, v_ref_m_s=v_ref)

    mu_torch = estimate_mu_torch_optional(quantum_df=quantum_df, cfg=cfg)

    peak_table = pd.DataFrame(
        {
            "model": ["quantum", "classical"],
            "peak_count": [quantum_peaks.peak_count, classical_peaks.peak_count],
            "peak_positions_mm": [
                np.array2string(quantum_peaks.peak_positions_mm, precision=3),
                np.array2string(classical_peaks.peak_positions_mm, precision=3),
            ],
        }
    )

    gmm_table = pd.DataFrame(
        {
            "component": ["left", "right"],
            "mean_mm": means_mm,
            "std_mm": stds_mm,
            "weight": weights,
        }
    )

    print("Stern-Gerlach MVP")
    print(f"num_atoms={cfg.num_atoms}, field_gradient={cfg.field_gradient_t_per_m:.1f} T/m")
    print(f"silver_atom_mass={cfg.atom_mass_kg:.6e} kg")
    print(f"quantum_|mu_z| reference={cfg.quantum_mu_eff:.6e} J/T")
    print(f"mean_forward_velocity={v_ref:.3f} m/s")
    print(f"expected_center_mm_at_mean_velocity={expected_center:.6f}")
    print()

    print("Beam summary (z on detector, mm):")
    print(summary_table(quantum_df, classical_df).to_string(index=False))
    print()

    print("KDE peak analysis:")
    print(peak_table.to_string(index=False))
    print()

    print("Quantum two-Gaussian fit (sklearn GMM):")
    print(gmm_table.to_string(index=False))
    print(f"separation_sigma={separation_sigma:.3f}")

    if mu_torch is not None:
        rel_mu_err = abs(mu_torch - cfg.quantum_mu_eff) / cfg.quantum_mu_eff
        print(f"torch_mu_estimate={mu_torch:.6e} J/T")
        print(f"torch_rel_error={rel_mu_err:.3e}")
    else:
        rel_mu_err = None
        print("torch not available: skipped optional gradient fit")

    # Assertions for automatic validation.
    assert quantum_peaks.peak_count == 2, f"quantum peak count != 2: {quantum_peaks.peak_count}"
    assert classical_peaks.peak_count == 1, f"classical peak count != 1: {classical_peaks.peak_count}"

    assert means_mm[0] < 0.0 < means_mm[1], f"quantum means not split around zero: {means_mm}"
    assert abs(weights[0] - 0.5) < 0.08 and abs(weights[1] - 0.5) < 0.08, (
        f"quantum component weights not near 0.5: {weights}"
    )
    assert abs(abs(means_mm[0]) - expected_center) < 0.5, (
        f"left center mismatch: {means_mm[0]} vs expected {expected_center}"
    )
    assert abs(abs(means_mm[1]) - expected_center) < 0.5, (
        f"right center mismatch: {means_mm[1]} vs expected {expected_center}"
    )
    assert abs(means_mm[0] + means_mm[1]) < 0.2, f"centers not symmetric: {means_mm}"
    assert separation_sigma > 6.0, f"quantum separation too weak: {separation_sigma}"

    if rel_mu_err is not None:
        assert rel_mu_err < 5e-3, f"torch mu estimate too far: {rel_mu_err}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
