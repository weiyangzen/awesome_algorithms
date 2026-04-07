"""Asymptotic states MVP via 1D wave-packet scattering.

This script builds a minimal, auditable numerical analog of asymptotic states:
- Evolve an incoming wave packet with an interacting Hamiltonian H = H0 + V.
- Use a localized potential V(x), so at early/late times particles are effectively free.
- Verify asymptotic behavior by checking late-time stability of momentum distribution.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    mass: float = 1.0
    x_min: float = -240.0
    x_max: float = 240.0
    n_grid: int = 2048

    dt: float = 0.02
    total_time: float = 120.0
    snapshot_times: tuple[float, ...] = (0.0, 40.0, 70.0, 90.0, 100.0, 110.0, 120.0)

    packet_center_x0: float = -100.0
    packet_width_sigma: float = 8.0
    packet_k0: float = 1.5

    potential_strength: float = 0.8
    potential_width: float = 3.0
    interaction_radius: float = 12.0


def make_grid(cfg: Config) -> tuple[np.ndarray, float, np.ndarray]:
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.n_grid, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(cfg.n_grid, d=dx)
    return x, dx, k


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float, k0: float, dx: float) -> np.ndarray:
    envelope = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2))
    phase = np.exp(1j * k0 * x)
    psi = envelope * phase
    return normalize(psi, dx)


def localized_potential(x: np.ndarray, strength: float, width: float) -> np.ndarray:
    return strength * np.exp(-0.5 * (x / width) ** 2)


def evolve_split_operator(
    psi0: np.ndarray,
    potential: np.ndarray,
    k_grid: np.ndarray,
    mass: float,
    dt: float,
    n_steps: int,
    snapshot_indices: set[int],
) -> dict[int, np.ndarray]:
    psi = psi0.copy()
    snapshots: dict[int, np.ndarray] = {}
    if 0 in snapshot_indices:
        snapshots[0] = psi.copy()

    v_half = np.exp(-0.5j * potential * dt)
    t_phase = np.exp(-1j * (k_grid**2) * dt / (2.0 * mass))

    for step in range(1, n_steps + 1):
        psi = v_half * psi
        psi_k = np.fft.fft(psi)
        psi_k *= t_phase
        psi = np.fft.ifft(psi_k)
        psi = v_half * psi

        if step in snapshot_indices:
            snapshots[step] = psi.copy()

    return snapshots


def momentum_density(psi_x: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    n = psi_x.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    phi_k = np.fft.fft(psi_x) * dx / np.sqrt(2.0 * np.pi)

    order = np.argsort(k)
    k_sorted = k[order]
    density = np.abs(phi_k[order]) ** 2
    return k_sorted, density


def channel_probabilities(psi_x: np.ndarray, x: np.ndarray, dx: float, interaction_radius: float) -> tuple[float, float, float]:
    prob = np.abs(psi_x) ** 2
    left = float(np.sum(prob[x < -interaction_radius]) * dx)
    center = float(np.sum(prob[np.abs(x) <= interaction_radius]) * dx)
    right = float(np.sum(prob[x > interaction_radius]) * dx)
    return left, center, right


def total_variation_distance(dens_a: np.ndarray, dens_b: np.ndarray, dk: float) -> float:
    return 0.5 * float(np.sum(np.abs(dens_a - dens_b)) * dk)


def main() -> None:
    cfg = Config()
    x, dx, k_grid = make_grid(cfg)

    psi0 = gaussian_wavepacket(
        x=x,
        x0=cfg.packet_center_x0,
        sigma=cfg.packet_width_sigma,
        k0=cfg.packet_k0,
        dx=dx,
    )
    potential = localized_potential(x, cfg.potential_strength, cfg.potential_width)

    n_steps = int(round(cfg.total_time / cfg.dt))
    snapshot_indices = {int(round(t / cfg.dt)) for t in cfg.snapshot_times}

    snapshots = evolve_split_operator(
        psi0=psi0,
        potential=potential,
        k_grid=k_grid,
        mass=cfg.mass,
        dt=cfg.dt,
        n_steps=n_steps,
        snapshot_indices=snapshot_indices,
    )

    missing = snapshot_indices.difference(snapshots)
    if missing:
        raise RuntimeError(f"Missing snapshots for steps: {sorted(missing)}")

    final_step = int(round(cfg.total_time / cfg.dt))
    psi_final = snapshots[final_step]
    k_final, dens_final = momentum_density(psi_final, dx)
    dk = k_final[1] - k_final[0]

    # Normalize momentum density to avoid small discrete FFT normalization bias.
    dens_final = dens_final / (np.sum(dens_final) * dk)

    rows: list[dict[str, float]] = []
    for t in cfg.snapshot_times:
        step = int(round(t / cfg.dt))
        psi_t = snapshots[step]

        norm_x = float(np.sum(np.abs(psi_t) ** 2) * dx)
        left, center, right = channel_probabilities(psi_t, x, dx, cfg.interaction_radius)

        k_t, dens_t = momentum_density(psi_t, dx)
        if not np.allclose(k_t, k_final):
            raise RuntimeError("Momentum grid mismatch between snapshots.")
        dens_t = dens_t / (np.sum(dens_t) * dk)
        tv_to_final = total_variation_distance(dens_t, dens_final, dk)

        rows.append(
            {
                "time": t,
                "norm": norm_x,
                "P_left": left,
                "P_interaction_region": center,
                "P_right": right,
                "TV(momentum, final)": tv_to_final,
            }
        )

    report = pd.DataFrame(rows)

    # Final in/out channel diagnostics from momentum sign decomposition.
    p_k_pos = float(np.sum(dens_final[k_final > 0.0]) * dk)
    p_k_neg = float(np.sum(dens_final[k_final < 0.0]) * dk)

    print("=== Asymptotic-State MVP: Localized Interaction Scattering ===")
    print(report.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print("\nFinal momentum-channel probabilities:")
    print(f"  P(k>0) (transmission-like) = {p_k_pos:.6f}")
    print(f"  P(k<0) (reflection-like)   = {p_k_neg:.6f}")
    print(f"  P(k>0)+P(k<0)              = {p_k_pos + p_k_neg:.6f}")

    r0 = report.iloc[0]
    r70 = report[report["time"] == 70.0].iloc[0]
    r100 = report[report["time"] == 100.0].iloc[0]
    r110 = report[report["time"] == 110.0].iloc[0]
    r120 = report[report["time"] == 120.0].iloc[0]

    # Numerical sanity checks for "asymptotic state" behavior.
    assert abs(float(r120["norm"]) - 1.0) < 5e-3, "Norm drift is too large."
    assert float(r0["P_interaction_region"]) < 1e-4, "Initial state is not asymptotically free enough."
    assert float(r120["P_interaction_region"]) < 2e-3, "Final state is not asymptotically free enough."

    interaction_peak = float(report["P_interaction_region"].max())
    assert interaction_peak > 0.2, "Interaction is too weak to demonstrate scattering."

    # Late-time momentum distribution should be nearly stationary.
    tv_70 = float(r70["TV(momentum, final)"])
    tv_100 = float(r100["TV(momentum, final)"])
    tv_110 = float(r110["TV(momentum, final)"])
    tv_120 = float(r120["TV(momentum, final)"])
    assert tv_70 > 0.1, "Mid-time state should differ from final asymptotic state."
    assert tv_100 < 3e-2 and tv_110 < 2e-2 and tv_120 < 1e-12, "Late-time momentum is not stable."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
