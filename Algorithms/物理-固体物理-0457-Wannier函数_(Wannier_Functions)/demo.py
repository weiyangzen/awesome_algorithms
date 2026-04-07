"""Wannier Functions MVP on a 1D SSH model.

This script demonstrates how to construct a single-band Wannier function from
discrete Bloch eigenvectors and why k-space gauge smoothing matters for spatial
localization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SSHConfig:
    """Configuration of a 1D SSH tight-binding band structure."""

    t1: float = 0.8
    t2: float = 1.2
    nk: int = 121
    seed: int = 20260407


def ssh_hamiltonian(k: float, t1: float, t2: float) -> np.ndarray:
    """Return the 2x2 SSH Bloch Hamiltonian H(k)."""
    offdiag = t1 + t2 * np.exp(-1j * k)
    return np.array([[0.0, offdiag], [np.conj(offdiag), 0.0]], dtype=complex)


def sample_lower_band(config: SSHConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample lower-band cell-periodic eigenvectors u_k on a uniform k-grid."""
    k_grid = np.linspace(-np.pi, np.pi, config.nk, endpoint=False)
    u_k = np.zeros((config.nk, 2), dtype=complex)

    for i, k in enumerate(k_grid):
        h_k = ssh_hamiltonian(k, config.t1, config.t2)
        eigvals, eigvecs = np.linalg.eigh(h_k)
        u_k[i] = eigvecs[:, np.argmin(eigvals)]

    return k_grid, u_k


def apply_random_gauge(u_k: np.ndarray, seed: int) -> np.ndarray:
    """Apply arbitrary U(1) phases to emulate an unsmoothed gauge."""
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=u_k.shape[0]))
    return u_k * phases[:, None]


def parallel_transport_gauge(u_k: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """Fix phase along k-grid by maximizing neighboring overlap."""
    smooth = u_k.copy()

    # Anchor the first vector to remove a trivial global phase.
    anchor_phase = np.angle(smooth[0, 0])
    smooth[0] *= np.exp(-1j * anchor_phase)

    for i in range(1, smooth.shape[0]):
        ov = np.vdot(smooth[i - 1], smooth[i])
        if abs(ov) > eps:
            smooth[i] *= np.conj(ov) / abs(ov)

    # Enforce periodic closure by distributing the residual phase twist.
    closure = np.vdot(smooth[-1], smooth[0])
    gamma = np.angle(closure)
    n = smooth.shape[0]
    for i in range(n):
        smooth[i] *= np.exp(-1j * gamma * i / n)

    return smooth


def build_wannier(u_k: np.ndarray, k_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct discrete Wannier amplitudes w(R, alpha) via inverse Bloch sum."""
    nk = k_grid.size
    half = nk // 2
    if nk % 2 == 0:
        r_cells = np.arange(-half, half)
    else:
        r_cells = np.arange(-half, half + 1)
    if r_cells.size != nk:
        raise ValueError("Real-space grid size must match k-grid size.")

    # We Fourier transform cell-periodic vectors u_k. In this discrete
    # convention, 1/nk keeps total Wannier norm at 1.
    phase = np.exp(-1j * np.outer(r_cells, k_grid)) / nk
    w_r = phase @ u_k  # shape: (nk, 2)
    return r_cells, w_r


def localization_metrics(r_cells: np.ndarray, w_r: np.ndarray) -> dict[str, float]:
    """Compute norm, center, spread and inverse participation ratio."""
    prob = np.sum(np.abs(w_r) ** 2, axis=1)
    norm = float(prob.sum())
    center = float(np.sum(r_cells * prob) / norm)
    second_moment = float(np.sum((r_cells**2) * prob) / norm)
    spread = second_moment - center**2
    ipr = float(np.sum(prob**2))
    return {
        "norm": norm,
        "center": center,
        "spread": spread,
        "ipr": ipr,
    }


def central_profile_dataframe(
    r_cells: np.ndarray,
    w_random: np.ndarray,
    w_smooth: np.ndarray,
    width: int = 6,
) -> pd.DataFrame:
    """Create a compact comparison table of cell probabilities near the center."""
    prob_random = np.sum(np.abs(w_random) ** 2, axis=1)
    prob_smooth = np.sum(np.abs(w_smooth) ** 2, axis=1)
    mask = (r_cells >= -width) & (r_cells <= width)

    return pd.DataFrame(
        {
            "R": r_cells[mask],
            "P_random": prob_random[mask],
            "P_smooth": prob_smooth[mask],
        }
    )


def main() -> None:
    config = SSHConfig()
    k_grid, u_band = sample_lower_band(config)

    u_random = apply_random_gauge(u_band, seed=config.seed)
    u_smooth = parallel_transport_gauge(u_random)

    r_cells, w_random = build_wannier(u_random, k_grid)
    _, w_smooth = build_wannier(u_smooth, k_grid)

    m_random = localization_metrics(r_cells, w_random)
    m_smooth = localization_metrics(r_cells, w_smooth)
    profile_df = central_profile_dataframe(r_cells, w_random, w_smooth, width=6)

    print("Wannier function MVP (1D SSH model, single isolated band)")
    print("-" * 72)
    print(f"t1={config.t1:.3f}, t2={config.t2:.3f}, nk={config.nk}, seed={config.seed}")
    print()
    print("Localization metrics")
    print(
        f"random gauge: norm={m_random['norm']:.8f}, center={m_random['center']:.6f}, "
        f"spread={m_random['spread']:.6f}, ipr={m_random['ipr']:.6f}"
    )
    print(
        f"smooth gauge: norm={m_smooth['norm']:.8f}, center={m_smooth['center']:.6f}, "
        f"spread={m_smooth['spread']:.6f}, ipr={m_smooth['ipr']:.6f}"
    )
    print()
    print("Central real-space probability profile P(R)=sum_alpha |w(R,alpha)|^2")
    print(profile_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    # Reproducible and lightweight checks.
    assert abs(m_random["norm"] - 1.0) < 1e-10, "Random-gauge Wannier is not normalized."
    assert abs(m_smooth["norm"] - 1.0) < 1e-10, "Smooth-gauge Wannier is not normalized."
    assert m_smooth["spread"] < m_random["spread"], (
        "Gauge smoothing should improve Wannier localization in this setup."
    )
    assert m_smooth["ipr"] > m_random["ipr"], (
        "Gauge smoothing should increase IPR (more localized state)."
    )

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
