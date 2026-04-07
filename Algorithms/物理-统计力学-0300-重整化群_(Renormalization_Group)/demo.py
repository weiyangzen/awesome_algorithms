"""Minimal MVP for Renormalization Group (RG) in 2D Ising model.

This script builds an empirical real-space RG map K -> K' by:
1) sampling spin configurations with Metropolis at coupling K,
2) coarse-graining with 2x2 block-spin majority rule,
3) estimating effective K' from coarse spins via Ising pseudolikelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class RGConfig:
    lattice_size: int = 20
    coupling_values: tuple[float, ...] = (0.24, 0.30, 0.36, 0.42, 0.46, 0.52, 0.62, 0.76)
    warmup_sweeps: int = 120
    snapshots_per_k: int = 48
    sweeps_between_snapshots: int = 4
    block_factor: int = 2
    seed: int = 20260407
    k_min_fit: float = 0.0
    k_max_fit: float = 1.5
    exact_kc_2d_ising: float = 0.5 * np.log(1.0 + np.sqrt(2.0))


class Ising2D:
    """Square-lattice Ising model with periodic boundary condition (J=1, h=0)."""

    def __init__(self, lattice_size: int, rng: np.random.Generator) -> None:
        self.lattice_size = lattice_size
        self.n_spins = lattice_size * lattice_size
        self.spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(lattice_size, lattice_size))

    def metropolis_sweep(self, coupling_k: float, rng: np.random.Generator) -> None:
        """One sweep: N single-spin proposals where N=L^2."""
        l = self.lattice_size
        s = self.spins
        for _ in range(self.n_spins):
            i = int(rng.integers(0, l))
            j = int(rng.integers(0, l))
            sij = int(s[i, j])
            nn = int(s[(i - 1) % l, j] + s[(i + 1) % l, j] + s[i, (j - 1) % l] + s[i, (j + 1) % l])
            delta_e = 2 * sij * nn
            if delta_e <= 0 or rng.random() < np.exp(-coupling_k * delta_e):
                s[i, j] = np.int8(-sij)

    def abs_magnetization(self) -> float:
        return float(np.mean(np.abs(self.spins)))


def collect_snapshots(config: RGConfig, coupling_k: float, seed: int) -> np.ndarray:
    """Sample spin snapshots from the fine lattice at coupling K."""
    rng = np.random.default_rng(seed)
    model = Ising2D(lattice_size=config.lattice_size, rng=rng)

    for _ in range(config.warmup_sweeps):
        model.metropolis_sweep(coupling_k=coupling_k, rng=rng)

    snaps: list[np.ndarray] = []
    for _ in range(config.snapshots_per_k):
        for _ in range(config.sweeps_between_snapshots):
            model.metropolis_sweep(coupling_k=coupling_k, rng=rng)
        snaps.append(model.spins.copy())

    return np.asarray(snaps, dtype=np.int8)


def block_majority_2x2(snapshot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2x2 majority-rule block-spin transform with random tie-break."""
    l = snapshot.shape[0]
    if l % 2 != 0:
        raise ValueError("lattice size must be even for 2x2 block transform")

    l2 = l // 2
    block_sum = snapshot.reshape(l2, 2, l2, 2).sum(axis=(1, 3))

    coarse = np.sign(block_sum).astype(np.int8)
    ties = block_sum == 0
    if np.any(ties):
        coarse[ties] = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(np.sum(ties)))
    return coarse


def coarse_grain_snapshots(snapshots: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse_list = [block_majority_2x2(s, rng=rng) for s in snapshots]
    return np.asarray(coarse_list, dtype=np.int8)


def _neighbor_sum(spins: np.ndarray) -> np.ndarray:
    return (
        np.roll(spins, 1, axis=0)
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
    )


def build_local_alignment_values(snapshots: np.ndarray) -> np.ndarray:
    """Return z = s_i * sum_nn s_j for all sites and all snapshots."""
    z_list: list[np.ndarray] = []
    for s in snapshots:
        z_list.append((s * _neighbor_sum(s)).ravel().astype(float))
    return np.concatenate(z_list, axis=0)


def estimate_effective_coupling_from_pseudolikelihood(
    z_values: np.ndarray,
    k_min: float,
    k_max: float,
) -> tuple[float, float]:
    """Estimate K by minimizing negative conditional log-likelihood.

    For Ising conditionals:
        P(s_i | nn) = 1 / (1 + exp(-2 K s_i h_i)), h_i=sum_nn s_j
    NLL(K) = mean(log(1 + exp(-2 K z))), z=s_i*h_i
    """

    def nll(k: float) -> float:
        return float(np.mean(np.logaddexp(0.0, -2.0 * k * z_values)))

    result = minimize_scalar(nll, bounds=(k_min, k_max), method="bounded")
    if not result.success:
        raise RuntimeError("pseudolikelihood optimization failed")
    return float(result.x), float(result.fun)


def run_rg_map(config: RGConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for idx, k in enumerate(config.coupling_values):
        seed_k = config.seed + 12_989 * idx
        fine = collect_snapshots(config=config, coupling_k=k, seed=seed_k)
        coarse = coarse_grain_snapshots(fine, seed=seed_k + 1)

        z = build_local_alignment_values(coarse)
        k_prime, nll = estimate_effective_coupling_from_pseudolikelihood(
            z_values=z,
            k_min=config.k_min_fit,
            k_max=config.k_max_fit,
        )

        fine_abs_m = float(np.mean(np.abs(fine.reshape(fine.shape[0], -1).mean(axis=1))))
        coarse_abs_m = float(np.mean(np.abs(coarse.reshape(coarse.shape[0], -1).mean(axis=1))))

        rows.append(
            {
                "K": float(k),
                "K_prime": k_prime,
                "delta_K": k_prime - float(k),
                "abs_m_fine": fine_abs_m,
                "abs_m_coarse": coarse_abs_m,
                "nll": nll,
            }
        )

    return pd.DataFrame(rows).sort_values("K", ignore_index=True)


def estimate_nontrivial_fixed_point(df: pd.DataFrame, scale_factor: float) -> dict[str, float]:
    """Estimate K* from sign change of delta_K and derive nu from slope dK'/dK."""
    k = df["K"].to_numpy(dtype=float)
    kp = df["K_prime"].to_numpy(dtype=float)
    d = kp - k

    crossing_index = -1
    for i in range(len(d) - 1):
        if d[i] == 0.0 or d[i] * d[i + 1] < 0.0:
            crossing_index = i
            break

    if crossing_index < 0:
        return {
            "k_star": float("nan"),
            "lambda": float("nan"),
            "nu": float("nan"),
        }

    i = crossing_index
    k0, k1 = k[i], k[i + 1]
    d0, d1 = d[i], d[i + 1]
    kp0, kp1 = kp[i], kp[i + 1]

    if abs(d1 - d0) < 1e-12:
        k_star = 0.5 * (k0 + k1)
    else:
        k_star = k0 - d0 * (k1 - k0) / (d1 - d0)

    lam = (kp1 - kp0) / (k1 - k0)
    nu = float(np.log(scale_factor) / np.log(lam)) if lam > 1.0 else float("nan")

    return {
        "k_star": float(k_star),
        "lambda": float(lam),
        "nu": float(nu),
    }


def iterate_flow_by_interpolation(
    df: pd.DataFrame,
    initial_k_values: tuple[float, ...],
    steps: int,
) -> dict[float, list[float]]:
    """Iterate K_{n+1}=R(K_n) using linear interpolation of sampled RG map."""
    k = df["K"].to_numpy(dtype=float)
    kp = df["K_prime"].to_numpy(dtype=float)

    trajectories: dict[float, list[float]] = {}
    for k0 in initial_k_values:
        seq = [float(k0)]
        current = float(k0)
        for _ in range(steps):
            current = float(np.interp(current, k, kp, left=kp[0], right=kp[-1]))
            seq.append(current)
        trajectories[float(k0)] = seq
    return trajectories


def monotonicity_check(df: pd.DataFrame) -> bool:
    k = df["K"].to_numpy(dtype=float)
    kp = df["K_prime"].to_numpy(dtype=float)
    return bool(np.all(np.diff(k) > 0.0) and np.all(np.diff(kp) >= -1e-9))


def main() -> None:
    config = RGConfig()
    if config.lattice_size % config.block_factor != 0:
        raise ValueError("lattice_size must be divisible by block_factor")

    rg_df = run_rg_map(config)
    fp = estimate_nontrivial_fixed_point(df=rg_df, scale_factor=float(config.block_factor))
    trajectories = iterate_flow_by_interpolation(
        df=rg_df,
        initial_k_values=(0.28, 0.44, 0.64),
        steps=6,
    )

    print("=== Renormalization Group MVP: 2D Ising Real-Space RG ===")
    print(
        "config:",
        f"L={config.lattice_size}, K grid={list(config.coupling_values)},",
        f"warmup={config.warmup_sweeps}, snapshots={config.snapshots_per_k},",
        f"stride={config.sweeps_between_snapshots}, block={config.block_factor}, seed={config.seed}",
    )
    print(f"reference exact Kc (2D Ising) = {config.exact_kc_2d_ising:.6f}")
    print()

    print("[empirical RG map]")
    print(rg_df.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))
    print()

    print("[fixed-point estimate from K' - K crossing]")
    print(f"K* estimate          = {fp['k_star']:.6f}")
    print(f"linearized lambda    = {fp['lambda']:.6f}")
    print(f"nu estimate          = {fp['nu']:.6f}")
    if np.isfinite(fp["k_star"]):
        print(f"|K* - Kc_exact|      = {abs(fp['k_star'] - config.exact_kc_2d_ising):.6f}")
    print()

    print("[flow trajectories by interpolated RG map]")
    for k0, seq in trajectories.items():
        seq_str = " -> ".join(f"{v:.5f}" for v in seq)
        print(f"K0={k0:.2f}: {seq_str}")
    print()

    low_delta = float(rg_df.loc[rg_df["K"].idxmin(), "delta_K"])
    high_delta = float(rg_df.loc[rg_df["K"].idxmax(), "delta_K"])
    direction_ok = low_delta < 0.0 and high_delta > 0.0

    print(f"monotonic map check: {'PASS' if monotonicity_check(rg_df) else 'CHECK_MANUALLY'}")
    print(f"phase-direction check (low K flow down, high K flow up): {'PASS' if direction_ok else 'CHECK_MANUALLY'}")


if __name__ == "__main__":
    main()
