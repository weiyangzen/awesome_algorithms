"""Electrostatic Potential Theory MVP.

This script demonstrates core electrostatic potential theory with a deterministic,
non-interactive workflow:
1) Compute potential of multiple point charges by linear superposition.
2) Compute electric field in two ways:
   - Direct Coulomb summation
   - Numerical gradient of potential (E = -grad(V))
3) Check quantitative consistency and print diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.constants import epsilon_0, pi


@dataclass(frozen=True)
class PointCharge:
    """Point charge in 2D (embedded in 3D electrostatic law)."""

    q_coulomb: float
    x: float
    y: float


@dataclass(frozen=True)
class PotentialTheoryResult:
    """Container for MVP diagnostics."""

    nx: int
    ny: int
    hx: float
    hy: float
    superposition_rel_err: float
    field_consistency_rel_err: float
    dipole_origin_potential: float
    potential_min: float
    potential_max: float


def coulomb_constant(epsilon_r: float = 1.0) -> float:
    """Return k = 1 / (4*pi*epsilon_0*epsilon_r)."""
    if epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive.")
    return 1.0 / (4.0 * pi * epsilon_0 * epsilon_r)


def build_grid(
    nx: int = 161,
    ny: int = 161,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build a uniform Cartesian grid."""
    if nx < 11 or ny < 11:
        raise ValueError("nx and ny must be >= 11.")
    if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        raise ValueError("Invalid domain limits.")

    x = np.linspace(xlim[0], xlim[1], nx, dtype=np.float64)
    y = np.linspace(ylim[0], ylim[1], ny, dtype=np.float64)
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xx, yy, hx, hy


def potential_from_charges(
    xx: np.ndarray,
    yy: np.ndarray,
    charges: Sequence[PointCharge],
    epsilon_r: float = 1.0,
    softening: float = 5.0e-3,
) -> np.ndarray:
    """Compute potential V(xx, yy) by direct superposition.

    A small softening radius avoids singular values exactly at charge locations and
    keeps the demo numerically stable on finite grids.
    """
    if softening <= 0.0:
        raise ValueError("softening must be positive.")
    if len(charges) == 0:
        return np.zeros_like(xx, dtype=np.float64)

    k = coulomb_constant(epsilon_r=epsilon_r)
    qs = np.array([c.q_coulomb for c in charges], dtype=np.float64)[:, None, None]
    cx = np.array([c.x for c in charges], dtype=np.float64)[:, None, None]
    cy = np.array([c.y for c in charges], dtype=np.float64)[:, None, None]

    dx = xx[None, :, :] - cx
    dy = yy[None, :, :] - cy
    r = np.sqrt(dx * dx + dy * dy + softening * softening)

    return k * np.sum(qs / r, axis=0)


def electric_field_direct(
    xx: np.ndarray,
    yy: np.ndarray,
    charges: Sequence[PointCharge],
    epsilon_r: float = 1.0,
    softening: float = 5.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute E = (Ex, Ey) by direct Coulomb superposition."""
    if softening <= 0.0:
        raise ValueError("softening must be positive.")
    if len(charges) == 0:
        zeros = np.zeros_like(xx, dtype=np.float64)
        return zeros, zeros

    k = coulomb_constant(epsilon_r=epsilon_r)
    qs = np.array([c.q_coulomb for c in charges], dtype=np.float64)[:, None, None]
    cx = np.array([c.x for c in charges], dtype=np.float64)[:, None, None]
    cy = np.array([c.y for c in charges], dtype=np.float64)[:, None, None]

    dx = xx[None, :, :] - cx
    dy = yy[None, :, :] - cy
    r2 = dx * dx + dy * dy + softening * softening
    r3 = np.power(r2, 1.5)

    ex = k * np.sum(qs * dx / r3, axis=0)
    ey = k * np.sum(qs * dy / r3, axis=0)
    return ex, ey


def electric_field_from_potential(
    v: np.ndarray,
    hx: float,
    hy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute E = -grad(V) with second-order finite differences via numpy.gradient."""
    dv_dx, dv_dy = np.gradient(v, hx, hy, edge_order=2)
    return -dv_dx, -dv_dy


def min_distance_to_charges(
    xx: np.ndarray,
    yy: np.ndarray,
    charges: Sequence[PointCharge],
) -> np.ndarray:
    """Distance-to-nearest-charge map for masking singular neighborhoods."""
    if len(charges) == 0:
        return np.full_like(xx, np.inf, dtype=np.float64)

    dists = []
    for c in charges:
        d = np.sqrt((xx - c.x) ** 2 + (yy - c.y) ** 2)
        dists.append(d)
    return np.min(np.stack(dists, axis=0), axis=0)


def potential_at_points(
    points_xy: np.ndarray,
    charges: Sequence[PointCharge],
    epsilon_r: float = 1.0,
    softening: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate potential at arbitrary point coordinates."""
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (m, 2).")

    k = coulomb_constant(epsilon_r=epsilon_r)
    vals = np.zeros(points_xy.shape[0], dtype=np.float64)
    for i, (px, py) in enumerate(points_xy):
        r = np.sqrt(
            np.array([(px - c.x) ** 2 + (py - c.y) ** 2 for c in charges], dtype=np.float64)
            + softening * softening
        )
        q = np.array([c.q_coulomb for c in charges], dtype=np.float64)
        vals[i] = k * np.sum(q / r)
    return vals


def run_mvp() -> tuple[PotentialTheoryResult, pd.DataFrame]:
    """Execute one deterministic electrostatic potential experiment."""
    epsilon_r = 2.5
    softening = 7.5e-3

    charges = [
        PointCharge(q_coulomb=2.0e-9, x=-0.35, y=-0.10),
        PointCharge(q_coulomb=-1.5e-9, x=0.20, y=0.05),
        PointCharge(q_coulomb=1.0e-9, x=0.00, y=0.42),
    ]

    xx, yy, hx, hy = build_grid(nx=161, ny=161)

    v_total = potential_from_charges(xx, yy, charges, epsilon_r=epsilon_r, softening=softening)

    # Superposition identity check: V(sum qi) == sum_i V(qi).
    v_sum = np.zeros_like(v_total)
    for c in charges:
        v_sum += potential_from_charges(xx, yy, [c], epsilon_r=epsilon_r, softening=softening)
    superposition_rel_err = float(np.linalg.norm(v_total - v_sum) / np.linalg.norm(v_sum))

    ex_direct, ey_direct = electric_field_direct(
        xx, yy, charges, epsilon_r=epsilon_r, softening=softening
    )
    ex_grad, ey_grad = electric_field_from_potential(v_total, hx=hx, hy=hy)

    # Avoid singular neighborhoods where finite-difference gradient is least accurate.
    dist = min_distance_to_charges(xx, yy, charges)
    mask = dist > (3.5 * max(hx, hy))

    diff_sq = (ex_direct - ex_grad) ** 2 + (ey_direct - ey_grad) ** 2
    ref_sq = ex_direct**2 + ey_direct**2
    field_consistency_rel_err = float(
        np.sqrt(np.sum(diff_sq[mask])) / np.sqrt(np.sum(ref_sq[mask]))
    )

    # Symmetry/physical sanity check with ideal dipole: V(0,0) should be 0.
    dipole = [
        PointCharge(q_coulomb=1.0e-9, x=-0.30, y=0.0),
        PointCharge(q_coulomb=-1.0e-9, x=0.30, y=0.0),
    ]
    dipole_origin_potential = float(
        potential_at_points(
            points_xy=np.array([[0.0, 0.0]], dtype=np.float64),
            charges=dipole,
            epsilon_r=epsilon_r,
            softening=1.0e-6,
        )[0]
    )

    # Build a compact centerline profile table.
    iy0 = yy.shape[1] // 2
    sample_ix = np.linspace(0, xx.shape[0] - 1, 11, dtype=int)
    profile = pd.DataFrame(
        {
            "x": xx[sample_ix, iy0],
            "y": yy[sample_ix, iy0],
            "V_total": v_total[sample_ix, iy0],
            "Ex_direct": ex_direct[sample_ix, iy0],
            "Ex_from_-gradV": ex_grad[sample_ix, iy0],
        }
    )
    profile["|Ex_diff|"] = np.abs(profile["Ex_direct"] - profile["Ex_from_-gradV"])

    result = PotentialTheoryResult(
        nx=v_total.shape[0],
        ny=v_total.shape[1],
        hx=hx,
        hy=hy,
        superposition_rel_err=superposition_rel_err,
        field_consistency_rel_err=field_consistency_rel_err,
        dipole_origin_potential=dipole_origin_potential,
        potential_min=float(np.min(v_total)),
        potential_max=float(np.max(v_total)),
    )
    return result, profile


def quality_checks(result: PotentialTheoryResult) -> None:
    """Hard checks for deterministic validation."""
    if result.superposition_rel_err > 1.0e-12:
        raise RuntimeError(
            f"Superposition check failed: rel_err={result.superposition_rel_err:.3e}"
        )

    if result.field_consistency_rel_err > 1.6e-1:
        raise RuntimeError(
            "Field consistency check failed: "
            f"rel_err={result.field_consistency_rel_err:.3e}"
        )

    if abs(result.dipole_origin_potential) > 1.0e-6:
        raise RuntimeError(
            "Dipole symmetry check failed: "
            f"V(0,0)={result.dipole_origin_potential:.3e}"
        )



def main() -> None:
    result, profile = run_mvp()
    quality_checks(result)

    print("=== Electrostatic Potential Theory MVP ===")
    print("Model            : V(r)=sum_i k*q_i/|r-r_i|,  E=-grad(V)")
    print("Grid             : {} x {}".format(result.nx, result.ny))
    print(f"Spacing          : hx={result.hx:.5f}, hy={result.hy:.5f}")
    print(f"Potential range  : [{result.potential_min:.6e}, {result.potential_max:.6e}] V")
    print(f"Superposition err: {result.superposition_rel_err:.6e}")
    print(f"Field consistency: {result.field_consistency_rel_err:.6e}")
    print(f"Dipole V(0,0)    : {result.dipole_origin_potential:.6e} V")

    print("\n--- Centerline sample (y~0) ---")
    print(profile.to_string(index=False, float_format=lambda v: f"{v:.6e}"))


if __name__ == "__main__":
    main()
