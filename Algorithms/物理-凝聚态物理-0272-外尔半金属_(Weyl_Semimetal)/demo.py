"""Weyl-semimetal MVP via 3D two-band lattice model + slice Chern numbers.

The script computes:
1) kz-resolved Chern number C(kz) on 2D (kx, ky) slices using FHS discretization
2) Analytic Weyl-node locations for |m| < 1
3) Numeric node estimates from Chern jumps
4) Consistency checks and PASS/FAIL exit status
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
class WeylConfig:
    """Configuration for Weyl-semimetal phase diagnostics."""

    mass: float = 0.0
    nk_xy: int = 41
    nk_z: int = 61
    overlap_eps: float = 1e-14
    chern_integer_tol: float = 6e-3
    node_kz_tol: float = 0.20
    gapped_slice_tol: float = 0.10
    finite_gap_floor: float = 0.20


def weyl_hamiltonian(kx: float, ky: float, kz: float, mass: float) -> np.ndarray:
    """Return 2x2 lattice Weyl Hamiltonian H(k)=d(k)·sigma.

    d_x = sin(kx)
    d_y = sin(ky)
    d_z = m + 2 - cos(kx) - cos(ky) - cos(kz)
    """

    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = mass + 2.0 - np.cos(kx) - np.cos(ky) - np.cos(kz)
    return dx * SIGMA_X + dy * SIGMA_Y + dz * SIGMA_Z


def occupied_state(hamiltonian: np.ndarray) -> np.ndarray:
    """Return occupied-band eigenvector for a 2x2 Hermitian matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    _ = eigenvalues
    return eigenvectors[:, 0]


def occupied_manifold_slice(kz: float, mass: float, nk_xy: int) -> np.ndarray:
    """Compute occupied state on the (kx, ky) mesh at fixed kz."""

    ks = np.linspace(-math.pi, math.pi, nk_xy, endpoint=False)
    manifold = np.empty((nk_xy, nk_xy, 2), dtype=complex)

    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            h_k = weyl_hamiltonian(kx=kx, ky=ky, kz=kz, mass=mass)
            manifold[i, j] = occupied_state(h_k)

    return manifold


def normalized_overlap(u1: np.ndarray, u2: np.ndarray, eps: float) -> complex:
    """Compute normalized overlap as a U(1) link variable with epsilon guard."""

    overlap = np.vdot(u1, u2)
    denom = max(abs(overlap), eps)
    return overlap / denom


def chern_number_slice_fhs(kz: float, mass: float, nk_xy: int, eps: float) -> float:
    """Compute Chern number C(kz) by Fukui-Hatsugai-Suzuki discretization."""

    manifold = occupied_manifold_slice(kz=kz, mass=mass, nk_xy=nk_xy)
    ux = np.empty((nk_xy, nk_xy), dtype=complex)
    uy = np.empty((nk_xy, nk_xy), dtype=complex)

    for i in range(nk_xy):
        i_next = (i + 1) % nk_xy
        for j in range(nk_xy):
            j_next = (j + 1) % nk_xy
            u_ij = manifold[i, j]
            ux[i, j] = normalized_overlap(u_ij, manifold[i_next, j], eps=eps)
            uy[i, j] = normalized_overlap(u_ij, manifold[i, j_next], eps=eps)

    flux_sum = 0.0
    for i in range(nk_xy):
        i_next = (i + 1) % nk_xy
        for j in range(nk_xy):
            j_next = (j + 1) % nk_xy
            plaquette = ux[i, j] * uy[i_next, j] * np.conj(ux[i, j_next]) * np.conj(uy[i, j])
            flux_sum += float(np.angle(plaquette))

    return flux_sum / (2.0 * math.pi)


def min_direct_gap_slice(kz: float, mass: float, nk_xy: int) -> float:
    """Return the minimum direct gap on one fixed-kz slice."""

    ks = np.linspace(-math.pi, math.pi, nk_xy, endpoint=False)
    min_gap = float("inf")

    for kx in ks:
        for ky in ks:
            dx = np.sin(kx)
            dy = np.sin(ky)
            dz = mass + 2.0 - np.cos(kx) - np.cos(ky) - np.cos(kz)
            gap = 2.0 * math.sqrt(dx * dx + dy * dy + dz * dz)
            min_gap = min(min_gap, gap)

    return min_gap


def direct_gap_point(kx: float, ky: float, kz: float, mass: float) -> float:
    """Return direct gap at one momentum point."""

    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = mass + 2.0 - np.cos(kx) - np.cos(ky) - np.cos(kz)
    return 2.0 * math.sqrt(dx * dx + dy * dy + dz * dz)


def scan_kz_slices(cfg: WeylConfig) -> pd.DataFrame:
    """Compute C(kz) and slice gaps on a uniform kz mesh."""

    kzs = np.linspace(-math.pi, math.pi, cfg.nk_z)
    rows: list[dict[str, float]] = []

    for kz in kzs:
        chern = chern_number_slice_fhs(kz=kz, mass=cfg.mass, nk_xy=cfg.nk_xy, eps=cfg.overlap_eps)
        min_gap = min_direct_gap_slice(kz=kz, mass=cfg.mass, nk_xy=cfg.nk_xy)
        rows.append(
            {
                "kz": kz,
                "chern": chern,
                "chern_round": float(int(round(chern))),
                "min_gap_slice": min_gap,
            }
        )

    return pd.DataFrame(rows)


def expected_weyl_nodes(mass: float) -> tuple[float, ...]:
    """Return analytic Weyl-node kz positions for the chosen lattice model."""

    if abs(mass) >= 1.0:
        return ()
    k0 = float(math.acos(mass))
    return (-k0, +k0)


def detect_nodes_from_chern_jumps(df: pd.DataFrame) -> list[float]:
    """Estimate node positions from adjacent Chern-number jumps."""

    kz = df["kz"].to_numpy()
    c_raw = df["chern"].to_numpy()
    c_rounded = np.round(c_raw).astype(int)

    nodes: list[float] = []
    for i in range(len(df) - 1):
        c1 = c_rounded[i]
        c2 = c_rounded[i + 1]
        if c1 == c2:
            continue

        # Linear interpolation of C(kz) toward the midpoint between two integer plateaus.
        target = 0.5 * (c1 + c2)
        denom = c_raw[i + 1] - c_raw[i]
        if abs(denom) < 1e-12:
            k_est = 0.5 * (kz[i] + kz[i + 1])
        else:
            frac = (target - c_raw[i]) / denom
            frac = min(1.0, max(0.0, frac))
            k_est = kz[i] + frac * (kz[i + 1] - kz[i])

        nodes.append(float(k_est))

    nodes.sort()
    return nodes


def nearest_row(df: pd.DataFrame, target_kz: float) -> pd.Series:
    """Pick the dataframe row with kz closest to target_kz."""

    idx = int(np.argmin(np.abs(df["kz"].to_numpy() - target_kz)))
    return df.iloc[idx]


def validate_results(df: pd.DataFrame, cfg: WeylConfig) -> tuple[bool, list[str]]:
    """Run consistency checks for Weyl-semimetal diagnostics."""

    checks: list[str] = []
    ok = True

    theory_nodes = expected_weyl_nodes(cfg.mass)
    numeric_nodes = detect_nodes_from_chern_jumps(df)

    # Check 1: model in Weyl phase with two analytic nodes
    check_weyl_phase = len(theory_nodes) == 2
    ok = ok and check_weyl_phase
    checks.append(f"weyl_phase(|m|<1): {check_weyl_phase}")

    # Check 2: gapped slices away from nodes should have quantized Chern
    gapped = df[df["min_gap_slice"] > cfg.gapped_slice_tol]
    quantized = bool(np.all(np.abs(gapped["chern"] - np.round(gapped["chern"])) < cfg.chern_integer_tol))
    ok = ok and quantized
    checks.append(f"quantized_chern_on_gapped_slices: {quantized}")

    # Check 3: interior slice (kz~0) should be non-trivial, exterior (kz~pi) should be trivial
    row_mid = nearest_row(df, 0.0)
    row_edge = nearest_row(df, math.pi)
    interior_nontrivial = abs(int(round(float(row_mid["chern"])))) == 1
    exterior_trivial = int(round(float(row_edge["chern"]))) == 0
    ok = ok and interior_nontrivial and exterior_trivial
    checks.append(f"interior_nontrivial(kz≈0): {interior_nontrivial}")
    checks.append(f"exterior_trivial(kz≈pi): {exterior_trivial}")

    # Check 4: numeric nodes should match analytic nodes
    node_match = len(numeric_nodes) >= len(theory_nodes)
    if node_match:
        for k_theory in theory_nodes:
            min_dist = min(abs(k_num - k_theory) for k_num in numeric_nodes)
            if min_dist >= cfg.node_kz_tol:
                node_match = False
                break
    ok = ok and node_match
    checks.append(f"node_locations_match: {node_match}")

    # Check 5: gap should collapse exactly at theoretical nodes and remain finite away from nodes
    gap_at_nodes = [direct_gap_point(0.0, 0.0, kz, cfg.mass) for kz in theory_nodes]
    min_node_gap = max(gap_at_nodes) if gap_at_nodes else float("inf")
    finite_gap_mid = float(row_mid["min_gap_slice"]) > cfg.finite_gap_floor
    near_node_gapless = min_node_gap < 1e-6
    ok = ok and finite_gap_mid and near_node_gapless
    checks.append(f"finite_gap_at_kz≈0: {finite_gap_mid}")
    checks.append(f"gapless_at_theoretical_nodes: {near_node_gapless}")

    return ok, checks


def main() -> None:
    cfg = WeylConfig()
    df = scan_kz_slices(cfg)

    theory_nodes = expected_weyl_nodes(cfg.mass)
    numeric_nodes = detect_nodes_from_chern_jumps(df)

    print("=== Weyl Semimetal MVP (3D lattice two-band model) ===")
    print(f"mass = {cfg.mass:+.3f}, nk_xy = {cfg.nk_xy}, nk_z = {cfg.nk_z}")
    print(f"theoretical nodes (kz): {[round(v, 6) for v in theory_nodes]}")
    print(f"detected nodes (kz):    {[round(v, 6) for v in numeric_nodes]}")
    print()

    preview_idx = np.unique(np.linspace(0, len(df) - 1, 13, dtype=int))
    preview = df.iloc[preview_idx].copy()
    preview["kz"] = preview["kz"].map(lambda x: f"{x:+.6f}")
    preview["chern"] = preview["chern"].map(lambda x: f"{x:+.6f}")
    preview["chern_round"] = preview["chern_round"].astype(int)
    preview["min_gap_slice"] = preview["min_gap_slice"].map(lambda x: f"{x:.6e}")

    print("=== kz-slice preview ===")
    print(preview.to_string(index=False))
    print()

    ok, checks = validate_results(df, cfg)
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
