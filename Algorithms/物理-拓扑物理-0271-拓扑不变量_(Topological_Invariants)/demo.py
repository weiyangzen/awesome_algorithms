"""Topological invariant MVP: Chern number via discrete Berry curvature.

This demo uses the 2D Qi-Wu-Zhang two-band lattice model and computes
its first Chern number with the Fukui-Hatsugai-Suzuki (FHS) plaquette method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qwz_hamiltonian(kx: float, ky: float, m: float) -> np.ndarray:
    """Return 2x2 Bloch Hamiltonian H(k) for the Qi-Wu-Zhang model."""
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = m + np.cos(kx) + np.cos(ky)
    return np.array(
        [[dz, dx - 1j * dy], [dx + 1j * dy, -dz]],
        dtype=np.complex128,
    )


def occupied_eigenvector(kx: float, ky: float, m: float) -> np.ndarray:
    """Return normalized eigenvector of the lower-energy (occupied) band."""
    eigvals, eigvecs = np.linalg.eigh(qwz_hamiltonian(kx, ky, m))
    occ_index = int(np.argmin(eigvals))
    vec = eigvecs[:, occ_index]
    return vec / np.linalg.norm(vec)


def _unit_link(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-14) -> complex:
    """Gauge link U=<v1|v2>/|<v1|v2>| with a small epsilon for stability."""
    overlap = np.vdot(v1, v2)
    norm = np.abs(overlap)
    if norm < eps:
        # Degenerate/ill-conditioned point: keep finite and deterministic.
        return 1.0 + 0.0j
    return overlap / norm


def chern_number_fhs(m: float, grid_size: int = 41) -> tuple[float, float]:
    """Compute Chern number using the FHS discretized Berry-curvature scheme.

    Returns:
        (chern_rounded, chern_raw)
    """
    k_vals = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)

    occ = np.empty((grid_size, grid_size, 2), dtype=np.complex128)
    for ix, kx in enumerate(k_vals):
        for iy, ky in enumerate(k_vals):
            occ[ix, iy] = occupied_eigenvector(kx, ky, m)

    ux = np.empty((grid_size, grid_size), dtype=np.complex128)
    uy = np.empty((grid_size, grid_size), dtype=np.complex128)

    for ix in range(grid_size):
        ix1 = (ix + 1) % grid_size
        for iy in range(grid_size):
            iy1 = (iy + 1) % grid_size
            ux[ix, iy] = _unit_link(occ[ix, iy], occ[ix1, iy])
            uy[ix, iy] = _unit_link(occ[ix, iy], occ[ix, iy1])

    berry_flux_total = 0.0
    for ix in range(grid_size):
        ix1 = (ix + 1) % grid_size
        for iy in range(grid_size):
            iy1 = (iy + 1) % grid_size
            plaquette = ux[ix, iy] * uy[ix1, iy] / (ux[ix, iy1] * uy[ix, iy])
            berry_flux_total += np.angle(plaquette)

    chern_raw = berry_flux_total / (2.0 * np.pi)
    chern_rounded = float(np.rint(chern_raw))
    return chern_rounded, float(chern_raw)


def run_scan(m_values: list[float], grid_size: int = 41) -> pd.DataFrame:
    """Scan multiple mass parameters and return a compact summary table."""
    rows: list[dict[str, float]] = []
    for m in m_values:
        c_int, c_raw = chern_number_fhs(m=m, grid_size=grid_size)
        rows.append(
            {
                "m": m,
                "chern_integer": c_int,
                "chern_raw": c_raw,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    m_values = [-3.0, -1.0, -0.5, 0.5, 1.5, 3.0]
    grid_size = 51

    df = run_scan(m_values=m_values, grid_size=grid_size)

    print("Topological invariant demo: Chern number of Qi-Wu-Zhang model")
    print(f"Brillouin-zone grid: {grid_size} x {grid_size}")
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()
