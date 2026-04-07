"""Minimal runnable MVP for Chern number (QWZ model).

This script computes the first Chern number of a 2D two-band lattice model
using the Fukui-Hatsugai-Suzuki (FHS) lattice-gauge discretization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChernResult:
    """Container for one parameter point."""

    m: float
    grid_size: int
    chern_raw: float
    chern_rounded: int
    max_flux_abs: float


def qwz_hamiltonian(kx: float, ky: float, m: float) -> np.ndarray:
    """Return 2x2 Qi-Wu-Zhang Hamiltonian H(k).

    H(k) = sin(kx) * sigma_x + sin(ky) * sigma_y + (m + cos(kx) + cos(ky)) * sigma_z
    """
    dx = math.sin(kx)
    dy = math.sin(ky)
    dz = m + math.cos(kx) + math.cos(ky)

    # Explicit Pauli-composed matrix (Hermitian).
    return np.array(
        [[dz, dx - 1j * dy], [dx + 1j * dy, -dz]],
        dtype=np.complex128,
    )


def occupied_eigenvector(kx: float, ky: float, m: float) -> np.ndarray:
    """Lowest-energy normalized eigenvector at momentum (kx, ky)."""
    h = qwz_hamiltonian(kx, ky, m)
    evals, evecs = np.linalg.eigh(h)
    # eigh returns ascending eigenvalues; band index 0 is occupied lower band.
    u = evecs[:, np.argmin(evals)]
    # Re-normalize defensively against tiny numerical drift.
    return u / np.linalg.norm(u)


def build_occupied_grid(m: float, n_k: int) -> np.ndarray:
    """Build periodic occupied-band eigenvectors on an n_k x n_k BZ grid."""
    ks = np.linspace(-math.pi, math.pi, n_k, endpoint=False)
    u = np.empty((n_k, n_k, 2), dtype=np.complex128)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            u[ix, iy] = occupied_eigenvector(float(kx), float(ky), m)
    return u


def safe_u1_phase(z: complex, eps: float = 1e-14) -> complex:
    """Project a complex overlap to U(1) with a tiny safety floor."""
    amp = abs(z)
    if amp < eps:
        # Gap closings are excluded in the chosen test points, so this branch is
        # only a numerical fallback.
        return 1.0 + 0.0j
    return z / amp


def fhs_chern_number(m: float, n_k: int = 51) -> ChernResult:
    """Compute Chern number with Fukui-Hatsugai-Suzuki lattice algorithm."""
    u = build_occupied_grid(m=m, n_k=n_k)

    # Ux[ix,iy] = <u(ix,iy) | u(ix+1,iy)> / |...|
    # Uy[ix,iy] = <u(ix,iy) | u(ix,iy+1)> / |...|
    ux = np.empty((n_k, n_k), dtype=np.complex128)
    uy = np.empty((n_k, n_k), dtype=np.complex128)

    for ix in range(n_k):
        ixp = (ix + 1) % n_k
        for iy in range(n_k):
            iyp = (iy + 1) % n_k
            ux[ix, iy] = safe_u1_phase(np.vdot(u[ix, iy], u[ixp, iy]))
            uy[ix, iy] = safe_u1_phase(np.vdot(u[ix, iy], u[ix, iyp]))

    # Plaquette flux:
    # F = Arg( Ux(k) Uy(k+dx) Ux(k+dy)^(-1) Uy(k)^(-1) )
    # where inverse on U(1) equals complex conjugate.
    flux = np.empty((n_k, n_k), dtype=np.float64)
    for ix in range(n_k):
        ixp = (ix + 1) % n_k
        for iy in range(n_k):
            iyp = (iy + 1) % n_k
            loop = ux[ix, iy] * uy[ixp, iy] * np.conjugate(ux[ix, iyp]) * np.conjugate(uy[ix, iy])
            flux[ix, iy] = float(np.angle(loop))

    chern_raw = float(np.sum(flux) / (2.0 * math.pi))
    chern_rounded = int(np.rint(chern_raw))

    return ChernResult(
        m=m,
        grid_size=n_k,
        chern_raw=chern_raw,
        chern_rounded=chern_rounded,
        max_flux_abs=float(np.max(np.abs(flux))),
    )


def main() -> None:
    # Stay away from gap-closing values m in {-2, 0, 2}.
    test_points = [-3.0, -1.0, 1.0, 3.0]
    expected = {
        -3.0: 0,
        -1.0: -1,
        1.0: 1,
        3.0: 0,
    }

    n_k = 61
    print("Chern number via FHS lattice gauge on QWZ model")
    print(f"grid size: {n_k} x {n_k}")
    print("-" * 72)
    print(f"{'m':>8} {'chern_raw':>14} {'chern_round':>12} {'expected':>10} {'ok':>6}")

    for m in test_points:
        res = fhs_chern_number(m=m, n_k=n_k)
        ok = res.chern_rounded == expected[m]
        print(
            f"{res.m:8.3f} {res.chern_raw:14.8f} {res.chern_rounded:12d} "
            f"{expected[m]:10d} {str(ok):>6}"
        )
        assert abs(res.chern_raw - res.chern_rounded) < 1e-3, (
            f"Chern number not close to integer at m={m}: {res.chern_raw}"
        )
        assert ok, (
            f"Unexpected phase at m={m}: got {res.chern_rounded}, expected {expected[m]}"
        )

    print("-" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
