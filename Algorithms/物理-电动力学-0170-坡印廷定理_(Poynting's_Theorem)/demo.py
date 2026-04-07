"""Minimal runnable MVP for Poynting's Theorem.

This script verifies, on a 1D vacuum plane wave:
1) Algebraic identity: Sz = c * u
2) Local conservation: du/dt + dSz/dz ≈ 0
3) Integral conservation: d/dt ∫u dz + [Sz(z2)-Sz(z1)] ≈ 0
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


EPS0 = 8.854_187_812_8e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)


def build_plane_wave_fields(
    nt: int,
    nz: int,
    domain_t: float,
    domain_z: float,
    e0: float,
    wavelength: float,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Construct Ey(z,t), Bx(z,t) for a +z propagating vacuum plane wave.

    Ey = E0 cos(kz - wt + phase), Bx = -Ey / c.
    """
    if nt < 5 or nz < 5:
        raise ValueError("nt and nz must be >= 5")
    if domain_t <= 0.0 or domain_z <= 0.0:
        raise ValueError("domain_t and domain_z must be positive")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")

    t = np.linspace(0.0, float(domain_t), int(nt), dtype=np.float64)
    z = np.linspace(0.0, float(domain_z), int(nz), dtype=np.float64)

    k = 2.0 * np.pi / float(wavelength)
    omega = C0 * k

    phase_grid = k * z[None, :] - omega * t[:, None] + float(phase)
    ey = float(e0) * np.cos(phase_grid)
    bx = -(ey / C0)

    return t, z, ey, bx, float(k), float(omega)


def compute_energy_density_and_flux(ey: np.ndarray, bx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute u and Sz for fields E=(0,Ey,0), B=(Bx,0,0)."""
    u = 0.5 * (EPS0 * ey * ey + (bx * bx) / MU0)
    sz = -(ey * bx) / MU0
    return u, sz


def rms(values: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(values * values)))


def check_plane_wave_identity(u: np.ndarray, sz: np.ndarray) -> Dict[str, float]:
    """Check Sz = c*u for the constructed vacuum plane wave."""
    ref = C0 * u
    abs_err = np.abs(sz - ref)
    rel_err = float(np.max(abs_err) / max(1e-12, np.max(np.abs(ref))))
    return {
        "plane_wave_identity_max_abs_err": float(np.max(abs_err)),
        "plane_wave_identity_rel_err": rel_err,
    }


def compute_local_residual(t: np.ndarray, z: np.ndarray, u: np.ndarray, sz: np.ndarray) -> Dict[str, float]:
    """Compute local residual for du/dt + dSz/dz = 0."""
    dt = float(t[1] - t[0])
    dz = float(z[1] - z[0])

    du_dt = np.gradient(u, dt, axis=0, edge_order=2)
    dsz_dz = np.gradient(sz, dz, axis=1, edge_order=2)
    residual = du_dt + dsz_dz

    interior = residual[2:-2, 2:-2]
    du_dt_i = du_dt[2:-2, 2:-2]
    dsz_dz_i = dsz_dz[2:-2, 2:-2]

    scale = rms(du_dt_i) + rms(dsz_dz_i)
    rel = rms(interior) / max(1e-12, scale)

    return {
        "local_rms_residual": rms(interior),
        "local_max_abs_residual": float(np.max(np.abs(interior))),
        "local_rel_residual": float(rel),
    }


def compute_integral_residual(
    t: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    sz: np.ndarray,
    left_ratio: float = 0.2,
    right_ratio: float = 0.8,
) -> Dict[str, float]:
    """Compute integral residual for d/dt∫u dz + [Sz(z2)-Sz(z1)] = 0."""
    if not (0.0 < left_ratio < right_ratio < 1.0):
        raise ValueError("Require 0 < left_ratio < right_ratio < 1")

    nz = z.shape[0]
    i0 = int(left_ratio * (nz - 1))
    i1 = int(right_ratio * (nz - 1))
    if i1 <= i0 + 2:
        raise ValueError("Integration slab too thin")

    z_slice = z[i0 : i1 + 1]
    u_slice = u[:, i0 : i1 + 1]

    u_total = np.trapezoid(u_slice, x=z_slice, axis=1)
    dt = float(t[1] - t[0])
    du_total_dt = np.gradient(u_total, dt, edge_order=2)
    flux_diff = sz[:, i1] - sz[:, i0]

    residual = du_total_dt + flux_diff
    interior = residual[2:-2]

    scale = rms(du_total_dt[2:-2]) + rms(flux_diff[2:-2])
    rel = rms(interior) / max(1e-12, scale)

    return {
        "slab_left_z": float(z[i0]),
        "slab_right_z": float(z[i1]),
        "integral_rms_residual": rms(interior),
        "integral_max_abs_residual": float(np.max(np.abs(interior))),
        "integral_rel_residual": float(rel),
    }


def run_demo() -> Dict[str, float]:
    """Run all checks for Poynting theorem on a vacuum plane wave."""
    wavelength = 0.8
    domain_z = 4.0 * wavelength
    k = 2.0 * np.pi / wavelength
    omega = C0 * k
    period = 2.0 * np.pi / omega

    t, z, ey, bx, _, _ = build_plane_wave_fields(
        nt=1401,
        nz=1201,
        domain_t=8.0 * period,
        domain_z=domain_z,
        e0=12.0,
        wavelength=wavelength,
        phase=0.23,
    )

    u, sz = compute_energy_density_and_flux(ey, bx)

    identity_report = check_plane_wave_identity(u, sz)
    local_report = compute_local_residual(t, z, u, sz)
    integral_report = compute_integral_residual(t, z, u, sz)

    avg_transport_speed = float(np.mean(sz) / max(1e-12, np.mean(u)))

    report: Dict[str, float] = {
        "grid_nt": float(t.shape[0]),
        "grid_nz": float(z.shape[0]),
        "wavelength_m": float(wavelength),
        "period_s": float(period),
        "avg_transport_speed_m_s": avg_transport_speed,
    }
    report.update(identity_report)
    report.update(local_report)
    report.update(integral_report)

    return report


def main() -> None:
    report = run_demo()

    print("=== Poynting Theorem MVP: 1D vacuum plane wave ===")
    for key, value in report.items():
        print(f"{key:>32s}: {value:.6e}")

    assert report["plane_wave_identity_rel_err"] < 1e-10, (
        "Plane-wave identity Sz=c*u failed with large relative error"
    )
    assert report["local_rel_residual"] < 2.5e-3, "Local Poynting residual too large"
    assert report["integral_rel_residual"] < 2.5e-3, "Integral Poynting residual too large"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
