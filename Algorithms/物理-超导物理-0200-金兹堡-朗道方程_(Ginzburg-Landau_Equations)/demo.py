"""Minimal runnable MVP for the Ginzburg-Landau equations.

This script solves a 2D time-dependent Ginzburg-Landau (TDGL) model in the
zero-vector-potential gauge (A=0):

    dpsi/dt = -(alpha*psi + beta*|psi|^2*psi - xi^2*Laplacian(psi))

The implementation is intentionally transparent:
- finite-difference Laplacian with periodic boundary conditions
- explicit Euler time marching
- explicit free-energy evaluation for sanity checking
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GLConfig:
    """Numerical and physical parameters for a minimal TDGL simulation."""

    nx: int = 72
    ny: int = 72
    dx: float = 1.0
    dt: float = 0.02
    n_steps: int = 1400
    record_every: int = 100
    alpha_sc: float = -1.0
    alpha_defect: float = 0.8
    beta: float = 1.0
    xi: float = 1.0
    defect_radius: float = 8.0
    init_noise: float = 0.08
    seed: int = 7


def laplacian_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """Return 2D Laplacian with periodic boundaries."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def build_alpha_map(cfg: GLConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create alpha(x,y) with a normal-metal circular defect at domain center."""
    yy, xx = np.indices((cfg.ny, cfg.nx))
    cx = 0.5 * (cfg.nx - 1)
    cy = 0.5 * (cfg.ny - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    alpha = np.full((cfg.ny, cfg.nx), cfg.alpha_sc, dtype=float)
    defect_mask = rr <= cfg.defect_radius
    alpha[defect_mask] = cfg.alpha_defect
    return alpha, defect_mask, rr


def free_energy_density(
    psi: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    xi: float,
    dx: float,
) -> np.ndarray:
    """Compute GL free-energy density: alpha|psi|^2 + beta/2|psi|^4 + xi^2|grad psi|^2."""
    abs2 = np.abs(psi) ** 2
    grad_x = (np.roll(psi, -1, axis=1) - psi) / dx
    grad_y = (np.roll(psi, -1, axis=0) - psi) / dx
    grad_term = (np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2) * (xi * xi)
    return alpha * abs2 + 0.5 * beta * (abs2**2) + grad_term


def total_free_energy(psi: np.ndarray, alpha: np.ndarray, cfg: GLConfig) -> float:
    """Integrate free-energy density over the grid."""
    density = free_energy_density(psi=psi, alpha=alpha, beta=cfg.beta, xi=cfg.xi, dx=cfg.dx)
    return float(np.sum(density) * (cfg.dx * cfg.dx))


def radial_profile(abs_psi: np.ndarray, rr: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    """Return radial averages of |psi| around the domain center."""
    r_max = float(np.max(rr))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, float]] = []
    for left, right, center in zip(edges[:-1], edges[1:], centers):
        mask = (rr >= left) & (rr < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "r_left": float(left),
                "r_right": float(right),
                "r_center": float(center),
                "mean_abs_psi": float(np.mean(abs_psi[mask])),
            }
        )
    return pd.DataFrame(rows)


def run_tdgl(cfg: GLConfig) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run explicit TDGL evolution and return field, time history, radial profile."""
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.record_every <= 0:
        raise ValueError("record_every must be positive")
    if cfg.beta <= 0.0:
        raise ValueError("beta must be positive")
    if cfg.alpha_sc >= 0.0:
        raise ValueError("alpha_sc should be negative for superconducting bulk")

    alpha, defect_mask, rr = build_alpha_map(cfg)
    rng = np.random.default_rng(cfg.seed)
    psi = cfg.init_noise * (
        rng.standard_normal((cfg.ny, cfg.nx)) + 1j * rng.standard_normal((cfg.ny, cfg.nx))
    )

    bulk_amplitude = np.sqrt(max(-cfg.alpha_sc / cfg.beta, 0.0))
    history_rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps + 1):
        abs_psi = np.abs(psi)
        if step % cfg.record_every == 0 or step == cfg.n_steps:
            energy = total_free_energy(psi=psi, alpha=alpha, cfg=cfg)
            history_rows.append(
                {
                    "step": float(step),
                    "time": step * cfg.dt,
                    "mean_abs_psi": float(np.mean(abs_psi)),
                    "mean_abs_psi_defect": float(np.mean(abs_psi[defect_mask])),
                    "mean_abs_psi_bulk": float(np.mean(abs_psi[~defect_mask])),
                    "superconducting_fraction": float(np.mean(abs_psi >= 0.8 * bulk_amplitude)),
                    "free_energy": energy,
                }
            )

        if step == cfg.n_steps:
            break

        lap = laplacian_periodic(psi, cfg.dx)
        rhs = -(alpha * psi + cfg.beta * (np.abs(psi) ** 2) * psi - (cfg.xi**2) * lap)
        psi = psi + cfg.dt * rhs

    history = pd.DataFrame(history_rows)
    profile = radial_profile(np.abs(psi), rr, n_bins=12)
    return psi, history, profile


def run_consistency_checks(history: pd.DataFrame) -> dict[str, float]:
    """Simple sanity checks expected from dissipative TDGL dynamics."""
    energies = history["free_energy"].to_numpy(dtype=float)
    if energies.size < 2:
        raise ValueError("History should contain at least two records")

    energy_drop = float(energies[-1] - energies[0])
    energy_up_steps = float(np.sum(np.diff(energies) > 1e-8))

    final_bulk = float(history.iloc[-1]["mean_abs_psi_bulk"])
    final_defect = float(history.iloc[-1]["mean_abs_psi_defect"])
    final_sc_fraction = float(history.iloc[-1]["superconducting_fraction"])

    assert energy_drop < 0.0, "Free energy should decrease overall in gradient flow."
    assert final_bulk > 0.75, "Bulk |psi| should recover close to equilibrium amplitude."
    assert final_defect < final_bulk, "Defect region should suppress the order parameter."
    assert final_sc_fraction > 0.6, "Most of the bulk should enter superconducting state."

    return {
        "energy_drop_total": energy_drop,
        "energy_non_decreasing_records": energy_up_steps,
        "final_mean_abs_psi_bulk": final_bulk,
        "final_mean_abs_psi_defect": final_defect,
        "final_superconducting_fraction": final_sc_fraction,
    }


def main() -> None:
    cfg = GLConfig()
    _, history, profile = run_tdgl(cfg)
    checks = run_consistency_checks(history)

    print("=== Ginzburg-Landau Equations MVP (2D TDGL, A=0) ===")
    print(
        f"grid=({cfg.ny}x{cfg.nx}), dx={cfg.dx}, dt={cfg.dt}, "
        f"steps={cfg.n_steps}, record_every={cfg.record_every}"
    )
    print(
        f"alpha_sc={cfg.alpha_sc}, alpha_defect={cfg.alpha_defect}, beta={cfg.beta}, "
        f"xi={cfg.xi}, defect_radius={cfg.defect_radius}"
    )
    print()

    print("[Time History]")
    with pd.option_context("display.width", 180, "display.precision", 6):
        print(history.to_string(index=False))
    print()

    print("[Final Radial Profile of |psi|]")
    with pd.option_context("display.width", 180, "display.precision", 6):
        print(profile.to_string(index=False))
    print()

    print("[Checks]")
    for k, v in checks.items():
        print(f"- {k}: {v:.6f}")


if __name__ == "__main__":
    main()
