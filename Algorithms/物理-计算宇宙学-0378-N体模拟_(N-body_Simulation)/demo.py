"""N-body Simulation MVP (PHYS-0360).

A compact, transparent gravitational N-body demo for computational cosmology.
- Integrator: kick-drift-kick leapfrog
- Force model: softened Newtonian gravity (direct O(N^2) summation)
- Diagnostics: energy conservation, virial ratio, radial density profile

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class NBodyConfig:
    """Configuration for a minimal, deterministic N-body experiment."""

    n_particles: int = 64
    n_steps: int = 320
    dt: float = 0.01
    grav_const: float = 1.0
    softening: float = 0.05
    init_pos_sigma: float = 0.35
    init_vel_sigma: float = 1.0
    init_virial_ratio: float = 0.30  # Q=K/|U| at t=0
    random_seed: int = 7


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the center-of-mass position."""

    return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)


def pairwise_geometry(positions: np.ndarray, softening: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pairwise displacement and softened inverse-distance tensors.

    Returns:
        diff[i,j] = x_j - x_i
        inv_r[i,j] = 1 / sqrt(|x_j-x_i|^2 + eps^2)
        inv_r3[i,j] = inv_r^3
    """

    diff = positions[None, :, :] - positions[:, None, :]
    dist2 = np.sum(diff * diff, axis=2) + softening**2
    np.fill_diagonal(dist2, np.inf)

    inv_r = dist2 ** (-0.5)
    inv_r3 = dist2 ** (-1.5)
    return diff, inv_r, inv_r3


def accelerations(positions: np.ndarray, masses: np.ndarray, cfg: NBodyConfig) -> np.ndarray:
    """Compute acceleration for each particle from all others (direct summation)."""

    diff, _, inv_r3 = pairwise_geometry(positions, cfg.softening)
    acc = cfg.grav_const * np.einsum("ij,ijk,j->ik", inv_r3, diff, masses)
    return acc


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Total kinetic energy."""

    return float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1)))


def potential_energy(positions: np.ndarray, masses: np.ndarray, cfg: NBodyConfig) -> float:
    """Total softened Newtonian potential energy."""

    _, inv_r, _ = pairwise_geometry(positions, cfg.softening)
    mass_matrix = masses[:, None] * masses[None, :]
    upper = np.triu(mass_matrix * inv_r, k=1)
    return float(-cfg.grav_const * np.sum(upper))


def initialize_state(cfg: NBodyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize masses, positions, velocities with deterministic random seed.

    Velocities are re-scaled to enforce the target initial virial ratio Q=K/|U|.
    """

    rng = np.random.default_rng(cfg.random_seed)
    masses = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)

    positions = rng.normal(0.0, cfg.init_pos_sigma, size=(cfg.n_particles, 3))
    positions -= center_of_mass(positions, masses)

    velocities = rng.normal(0.0, cfg.init_vel_sigma, size=(cfg.n_particles, 3))
    v_com = np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)
    velocities -= v_com

    u0 = potential_energy(positions, masses, cfg)
    k0 = kinetic_energy(velocities, masses)
    scale = np.sqrt(cfg.init_virial_ratio * abs(u0) / max(k0, 1e-14))
    velocities *= scale

    return masses, positions, velocities


def leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    acc_now: np.ndarray,
    cfg: NBodyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one step with kick-drift-kick leapfrog."""

    v_half = velocities + 0.5 * cfg.dt * acc_now
    x_new = positions + cfg.dt * v_half
    a_new = accelerations(x_new, masses, cfg)
    v_new = v_half + 0.5 * cfg.dt * a_new
    return x_new, v_new, a_new


def radial_density_profile(
    positions: np.ndarray,
    masses: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute spherical shell density profile around center-of-mass."""

    com = center_of_mass(positions, masses)
    radii = np.linalg.norm(positions - com, axis=1)

    r_max = float(np.max(radii))
    r_min = max(float(np.percentile(radii, 10)) * 0.5, 1e-3)
    if r_max <= r_min:
        r_max = r_min * 1.2

    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    mass_in_shell, edges, _ = binned_statistic(radii, masses, statistic="sum", bins=bins)

    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = mass_in_shell / np.maximum(shell_vol, 1e-16)
    r_mid = np.sqrt(edges[:-1] * edges[1:])

    return pd.DataFrame(
        {
            "r_mid": r_mid,
            "mass_shell": mass_in_shell,
            "rho_shell": rho,
        }
    )


def fit_density_slope_sklearn(profile_df: pd.DataFrame) -> tuple[float, float]:
    """Fit log10(rho) = a + b*log10(r) using scikit-learn."""

    mask = (profile_df["r_mid"] > 0.0) & (profile_df["rho_shell"] > 0.0)
    x = np.log10(profile_df.loc[mask, "r_mid"].to_numpy())
    y = np.log10(profile_df.loc[mask, "rho_shell"].to_numpy())

    if x.size < 2:
        return float("nan"), float("nan")

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def fit_density_slope_torch(profile_df: pd.DataFrame) -> tuple[float, float, float]:
    """Fit log(rho)=b+alpha*log(r) via PyTorch autograd."""

    mask = (profile_df["r_mid"] > 0.0) & (profile_df["rho_shell"] > 0.0)
    x_np = np.log(profile_df.loc[mask, "r_mid"].to_numpy())
    y_np = np.log(profile_df.loc[mask, "rho_shell"].to_numpy())

    if x_np.size < 2:
        return float("nan"), float("nan"), float("nan")

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    alpha = torch.tensor(-2.0, dtype=torch.float32, requires_grad=True)
    log_a = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([alpha, log_a], lr=0.05)

    for _ in range(500):
        opt.zero_grad()
        pred = log_a + alpha * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        rmse = torch.sqrt(torch.mean((log_a + alpha * x - y) ** 2))

    return float(alpha.item()), float(log_a.item()), float(rmse.item())


def run_simulation(cfg: NBodyConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the N-body simulation and return trajectory diagnostics + final state."""

    masses, positions, velocities = initialize_state(cfg)
    acc = accelerations(positions, masses, cfg)

    k_init = kinetic_energy(velocities, masses)
    u_init = potential_energy(positions, masses, cfg)
    e0 = k_init + u_init

    rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps + 1):
        k = kinetic_energy(velocities, masses)
        u = potential_energy(positions, masses, cfg)
        e = k + u

        com = center_of_mass(positions, masses)
        r_rms = float(np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1))))
        virial_ratio_2k_u = float(2.0 * k / max(abs(u), 1e-14))

        rows.append(
            {
                "step": float(step),
                "time": step * cfg.dt,
                "kinetic": k,
                "potential": u,
                "total_energy": e,
                "rel_energy_drift": (e - e0) / max(abs(e0), 1e-14),
                "virial_2K_over_|U|": virial_ratio_2k_u,
                "r_rms": r_rms,
                "com_norm": float(np.linalg.norm(com)),
            }
        )

        if step == cfg.n_steps:
            break

        positions, velocities, acc = leapfrog_step(positions, velocities, masses, acc, cfg)

    return pd.DataFrame(rows), masses, positions, velocities, e0


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = NBodyConfig()
    traj_df, masses, positions, _, e0 = run_simulation(cfg)

    # Compact trajectory sample for terminal display.
    pick = np.linspace(0, len(traj_df) - 1, 8, dtype=int)
    traj_preview = traj_df.iloc[pick].copy()

    # Radial profile + slope diagnostics.
    profile_df = radial_density_profile(positions, masses, n_bins=10)
    slope_sk, intercept_sk = fit_density_slope_sklearn(profile_df)
    slope_torch, loga_torch, rmse_torch = fit_density_slope_torch(profile_df)

    max_abs_drift = float(np.max(np.abs(traj_df["rel_energy_drift"].to_numpy())))
    final_row = traj_df.iloc[-1]

    print("=== N-body Simulation MVP (PHYS-0360) ===")
    print(
        f"N={cfg.n_particles}, steps={cfg.n_steps}, dt={cfg.dt:.4f}, "
        f"G={cfg.grav_const:.3f}, epsilon={cfg.softening:.3f}"
    )
    print(f"Initial total energy E0 = {e0:.6e}")
    print()

    print("[Trajectory Sample]")
    print(traj_preview.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Final Diagnostics]")
    print(f"Final rel_energy_drift      : {final_row['rel_energy_drift']:.6e}")
    print(f"Max |rel_energy_drift|      : {max_abs_drift:.6e}")
    print(f"Final virial_2K_over_|U|    : {final_row['virial_2K_over_|U|']:.6e}")
    print(f"Final r_rms                 : {final_row['r_rms']:.6e}")
    print(f"Final COM norm              : {final_row['com_norm']:.6e}")
    print()

    print("[Radial Density Profile]")
    print(profile_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Density Slope Fits]")
    print(f"sklearn slope b (log10 rho vs log10 r): {slope_sk:.6f}, intercept={intercept_sk:.6f}")
    print(f"torch   alpha (ln rho vs ln r)        : {slope_torch:.6f}, logA={loga_torch:.6f}, RMSE={rmse_torch:.3e}")


if __name__ == "__main__":
    main()
