"""Topological-insulator MVP via BHZ blocks + FHS Chern-number discretization.

This script computes:
1) Chern number of spin-up BHZ block
2) Chern number of spin-down (time-reversal partner) block
3) Spin Chern and derived Z2 index
for a set of mass parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class BHZConfig:
    """Configuration for phase-diagram sampling."""

    nk: int = 41
    masses: tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0)
    overlap_eps: float = 1e-14
    integer_tol: float = 5e-3
    tr_pair_tol: float = 1e-2
    gap_tol: float = 1e-2


def bhz_block_hamiltonian(kx: float, ky: float, mass: float, spin: int) -> np.ndarray:
    """Return 2x2 BHZ block Hamiltonian for a given momentum and spin block.

    spin = +1: H_up
    spin = -1: H_down = H_up^* (time-reversal partner block in this decoupled model)
    """

    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = mass + np.cos(kx) + np.cos(ky)

    if spin == +1:
        return dx * SIGMA_X + dy * SIGMA_Y + dz * SIGMA_Z
    if spin == -1:
        return dx * SIGMA_X - dy * SIGMA_Y + dz * SIGMA_Z
    msg = f"spin must be +1 or -1, got {spin}"
    raise ValueError(msg)


def occupied_state(hamiltonian: np.ndarray) -> np.ndarray:
    """Return occupied-band eigenvector (lowest eigenvalue) for a 2x2 Hermitian matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    _ = eigenvalues  # explicit for readability
    return eigenvectors[:, 0]


def occupied_manifold(mass: float, spin: int, nk: int) -> np.ndarray:
    """Compute occupied state on an nk x nk momentum mesh."""

    ks = np.linspace(-math.pi, math.pi, nk, endpoint=False)
    manifold = np.empty((nk, nk, 2), dtype=complex)
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            h_k = bhz_block_hamiltonian(kx=kx, ky=ky, mass=mass, spin=spin)
            manifold[i, j] = occupied_state(h_k)
    return manifold


def normalized_overlap(u1: np.ndarray, u2: np.ndarray, eps: float) -> complex:
    """Gauge link from two neighboring occupied states."""

    overlap = np.vdot(u1, u2)
    denom = max(abs(overlap), eps)
    return overlap / denom


def chern_number_fhs(mass: float, spin: int, nk: int, eps: float) -> float:
    """Compute Chern number by Fukui-Hatsugai-Suzuki lattice formula."""

    manifold = occupied_manifold(mass=mass, spin=spin, nk=nk)
    ux = np.empty((nk, nk), dtype=complex)
    uy = np.empty((nk, nk), dtype=complex)

    for i in range(nk):
        i_next = (i + 1) % nk
        for j in range(nk):
            j_next = (j + 1) % nk
            u_ij = manifold[i, j]
            ux[i, j] = normalized_overlap(u_ij, manifold[i_next, j], eps=eps)
            uy[i, j] = normalized_overlap(u_ij, manifold[i, j_next], eps=eps)

    berry_flux_sum = 0.0
    for i in range(nk):
        i_next = (i + 1) % nk
        for j in range(nk):
            j_next = (j + 1) % nk
            plaquette = ux[i, j] * uy[i_next, j] * np.conj(ux[i, j_next]) * np.conj(uy[i, j])
            berry_flux_sum += float(np.angle(plaquette))

    return berry_flux_sum / (2.0 * math.pi)


def minimum_direct_gap(mass: float, nk: int) -> float:
    """Return minimum direct band gap over BZ for the block spectrum."""

    ks = np.linspace(-math.pi, math.pi, nk, endpoint=False)
    min_gap = float("inf")
    for kx in ks:
        for ky in ks:
            dx = np.sin(kx)
            dy = np.sin(ky)
            dz = mass + np.cos(kx) + np.cos(ky)
            d_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            gap = 2.0 * d_norm
            min_gap = min(min_gap, gap)
    return min_gap


def compute_phase_point(mass: float, cfg: BHZConfig) -> dict[str, float]:
    """Compute topological invariants for one mass value."""

    chern_up = chern_number_fhs(mass=mass, spin=+1, nk=cfg.nk, eps=cfg.overlap_eps)
    chern_down = chern_number_fhs(mass=mass, spin=-1, nk=cfg.nk, eps=cfg.overlap_eps)
    spin_chern = 0.5 * (chern_up - chern_down)
    z2 = int(abs(int(round(spin_chern))) % 2)
    min_gap = minimum_direct_gap(mass=mass, nk=cfg.nk)
    return {
        "mass": mass,
        "chern_up": chern_up,
        "chern_down": chern_down,
        "spin_chern": spin_chern,
        "z2": float(z2),
        "min_gap": min_gap,
    }


def phase_diagram(cfg: BHZConfig) -> pd.DataFrame:
    """Compute phase points for all configured masses."""

    rows = [compute_phase_point(mass, cfg=cfg) for mass in cfg.masses]
    return pd.DataFrame(rows).sort_values("mass").reset_index(drop=True)


def validate_results(df: pd.DataFrame, cfg: BHZConfig) -> tuple[bool, list[str]]:
    """Validate numerical topology checks."""

    checks: list[str] = []
    ok = True

    for _, row in df.iterrows():
        m = float(row["mass"])
        cup = float(row["chern_up"])
        cdown = float(row["chern_down"])
        gap = float(row["min_gap"])
        z2 = int(row["z2"])

        cup_integer = abs(cup - round(cup)) < cfg.integer_tol
        cdown_integer = abs(cdown - round(cdown)) < cfg.integer_tol
        tr_pair = abs(cup + cdown) < cfg.tr_pair_tol
        gap_positive = gap > cfg.gap_tol

        if m in (-1.0, 1.0):
            expected_z2 = 1
        elif m in (-3.0, 3.0):
            expected_z2 = 0
        else:
            expected_z2 = z2
        z2_match = z2 == expected_z2

        check_pass = cup_integer and cdown_integer and tr_pair and gap_positive and z2_match
        ok = ok and check_pass
        checks.append(
            (
                f"m={m:+.1f}: "
                f"integer_up={cup_integer}, integer_down={cdown_integer}, "
                f"tr_pair={tr_pair}, gap_ok={gap_positive}, z2_ok={z2_match}"
            )
        )

    return ok, checks


def main() -> None:
    cfg = BHZConfig()
    df = phase_diagram(cfg)

    printable = df.copy()
    for col in ("chern_up", "chern_down", "spin_chern", "min_gap"):
        printable[col] = printable[col].map(lambda x: f"{x:+.6f}")
    printable["z2"] = printable["z2"].astype(int)

    print("=== 2D BHZ Topological Invariants (FHS discretization) ===")
    print(f"nk = {cfg.nk}, masses = {cfg.masses}")
    print()
    print(printable.to_string(index=False))
    print()

    ok, checks = validate_results(df, cfg=cfg)
    print("=== Validation Checks ===")
    for line in checks:
        print(line)

    status = "PASS" if ok else "FAIL"
    print()
    print(f"Validation: {status}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
