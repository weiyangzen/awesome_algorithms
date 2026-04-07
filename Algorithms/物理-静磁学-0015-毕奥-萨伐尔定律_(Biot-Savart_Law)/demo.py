"""Minimal runnable MVP for Biot-Savart Law (PHYS-0015)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import mu_0


def build_circular_loop(radius: float, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Return segment start/end points for a circular loop in the x-y plane.

    The current direction follows increasing polar angle theta.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_segments < 8:
        raise ValueError("n_segments must be >= 8 for a meaningful loop discretization")

    theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    points = np.stack([x, y, z], axis=1)
    return points[:-1], points[1:]


def biot_savart_field_from_segments(
    obs_points: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    current: float,
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute magnetic field B at observation points using segment midpoint rule.

    Formula used:
        dB = mu0 / (4*pi) * I * (dl x r) / |r|^3

    where dl is the segment vector and r points from segment midpoint to observation.
    """
    if seg_starts.shape != seg_ends.shape:
        raise ValueError("seg_starts and seg_ends must have the same shape")

    obs = np.atleast_2d(np.asarray(obs_points, dtype=float))
    starts = np.asarray(seg_starts, dtype=float)
    ends = np.asarray(seg_ends, dtype=float)

    if obs.shape[1] != 3 or starts.shape[1] != 3:
        raise ValueError("obs_points and segments must be 3D vectors")

    dl = ends - starts  # (N, 3)
    mids = 0.5 * (starts + ends)  # (N, 3)

    # r shape: (M, N, 3), where M=#observation points, N=#segments
    r = obs[:, None, :] - mids[None, :, :]
    r_norm = np.linalg.norm(r, axis=2)

    cross_term = np.cross(dl[None, :, :], r, axis=2)
    denom = np.where(r_norm > eps, r_norm**3, np.inf)

    contribution = cross_term / denom[:, :, None]
    b_field = (mu_0 * current / (4.0 * np.pi)) * np.sum(contribution, axis=1)
    return b_field


def analytic_loop_axis_field(radius: float, current: float, z: np.ndarray) -> np.ndarray:
    """Analytic Bz of a circular current loop along its axis."""
    z = np.asarray(z, dtype=float)
    return mu_0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def main() -> None:
    radius = 0.10  # meters
    current = 5.0  # amperes

    print("Biot-Savart Law MVP (PHYS-0015)")
    print("=" * 72)

    # 1) Convergence check at loop center against analytic value.
    n_list = [50, 100, 200, 400, 800, 1600]
    center = np.array([[0.0, 0.0, 0.0]])
    b_center_exact = mu_0 * current / (2.0 * radius)

    rows = []
    for n_segments in n_list:
        seg_s, seg_e = build_circular_loop(radius, n_segments)
        b_num = biot_savart_field_from_segments(center, seg_s, seg_e, current)[0, 2]
        rel_err = abs((b_num - b_center_exact) / b_center_exact)
        rows.append(
            {
                "n_segments": n_segments,
                "Bz_numeric_T": b_num,
                "Bz_exact_T": b_center_exact,
                "relative_error": rel_err,
            }
        )

    df_center = pd.DataFrame(rows)
    print("Center-field convergence (z=0):")
    print(df_center.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # 2) On-axis profile check with one high-resolution discretization.
    n_segments_axis = 2400
    seg_s, seg_e = build_circular_loop(radius, n_segments_axis)

    z_values = np.linspace(-3.0 * radius, 3.0 * radius, 13)
    obs = np.column_stack([np.zeros_like(z_values), np.zeros_like(z_values), z_values])

    b_numeric = biot_savart_field_from_segments(obs, seg_s, seg_e, current)
    b_numeric_z = b_numeric[:, 2]
    b_exact_z = analytic_loop_axis_field(radius, current, z_values)

    rel_axis = np.abs((b_numeric_z - b_exact_z) / b_exact_z)
    df_axis = pd.DataFrame(
        {
            "z_over_R": z_values / radius,
            "Bz_numeric_T": b_numeric_z,
            "Bz_exact_T": b_exact_z,
            "relative_error": rel_axis,
        }
    )

    print("\nOn-axis profile check (high-resolution discretization):")
    print(df_axis.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # 3) Deterministic assertions for MVP self-validation.
    max_center_rel = float(df_center["relative_error"].max())
    max_axis_rel = float(df_axis["relative_error"].max())

    assert max_center_rel < 3.5e-3, f"Center convergence too poor: {max_center_rel}"
    assert max_axis_rel < 8e-3, f"Axis profile mismatch too large: {max_axis_rel}"

    # Symmetry sanity checks: Bx and By should be near zero on axis.
    max_transverse = float(np.max(np.linalg.norm(b_numeric[:, :2], axis=1)))
    assert max_transverse < 1e-11, f"Unexpected transverse component on axis: {max_transverse}"

    print("=" * 72)
    print(
        "All checks passed: "
        f"max_center_rel={max_center_rel:.3e}, "
        f"max_axis_rel={max_axis_rel:.3e}, "
        f"max_transverse={max_transverse:.3e}"
    )


if __name__ == "__main__":
    main()
