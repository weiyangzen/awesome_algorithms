"""Minimal runnable MVP for magnetic dipole field (PHYS-0020)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MU0 = 4.0 * np.pi * 1e-7
EPS = 1e-12


@dataclass
class DivergenceMetrics:
    max_abs_divergence: float
    rms_divergence: float


@dataclass
class ScalingMetrics:
    max_abs_ratio_error: float
    table: pd.DataFrame


@dataclass
class LoopComparisonMetrics:
    max_rel_error_far_field: float
    table: pd.DataFrame


def dipole_field(points: np.ndarray, m: np.ndarray, min_radius: float = 1e-6) -> np.ndarray:
    """Compute magnetic dipole field at multiple 3D points.

    Parameters
    ----------
    points:
        Array of shape (N, 3), each row is (x, y, z) in meters.
    m:
        Dipole moment vector of shape (3,) in A*m^2.
    min_radius:
        Safety threshold to avoid evaluating the point singularity at r=0.
    """
    pts = np.asarray(points, dtype=float)
    m = np.asarray(m, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if m.shape != (3,):
        raise ValueError("m must have shape (3,)")

    r_norm = np.linalg.norm(pts, axis=1)
    if np.any(r_norm < min_radius):
        raise ValueError(
            f"some points are closer than min_radius={min_radius} m to the origin; "
            "point dipole field is singular at r=0"
        )

    r_hat = pts / r_norm[:, None]
    m_dot_rhat = r_hat @ m
    coeff = MU0 / (4.0 * np.pi * r_norm**3)

    field = coeff[:, None] * (3.0 * m_dot_rhat[:, None] * r_hat - m)
    return field


def divergence_check(m: np.ndarray) -> DivergenceMetrics:
    """Numerically verify div(B)≈0 on a box that excludes the origin."""
    x = np.linspace(0.4, 1.0, 17)
    y = np.linspace(0.45, 1.05, 15)
    z = np.linspace(0.5, 1.1, 13)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    B = dipole_field(pts, m).reshape(X.shape + (3,))

    dBx_dx = np.gradient(B[..., 0], x, axis=0, edge_order=2)
    dBy_dy = np.gradient(B[..., 1], y, axis=1, edge_order=2)
    dBz_dz = np.gradient(B[..., 2], z, axis=2, edge_order=2)

    div_b = dBx_dx + dBy_dy + dBz_dz
    interior = div_b[1:-1, 1:-1, 1:-1]

    return DivergenceMetrics(
        max_abs_divergence=float(np.max(np.abs(interior))),
        rms_divergence=float(np.sqrt(np.mean(interior**2))),
    )


def inverse_cubic_scaling_check(m: np.ndarray) -> ScalingMetrics:
    """Check |B(r)| / |B(2r)| ≈ 8 for fixed directions."""
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 2.0, 1.0],
            [2.0, 1.0, -1.0],
            [-2.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    r1 = 0.7
    r2 = 1.4

    b1 = dipole_field(directions * r1, m)
    b2 = dipole_field(directions * r2, m)

    norm_b1 = np.linalg.norm(b1, axis=1)
    norm_b2 = np.linalg.norm(b2, axis=1)
    ratios = norm_b1 / norm_b2

    table = pd.DataFrame(
        {
            "dir_x": directions[:, 0],
            "dir_y": directions[:, 1],
            "dir_z": directions[:, 2],
            "|B(r)|/|B(2r)|": ratios,
            "abs_error_to_8": np.abs(ratios - 8.0),
        }
    )

    return ScalingMetrics(
        max_abs_ratio_error=float(np.max(np.abs(ratios - 8.0))),
        table=table,
    )


def loop_axis_field_exact(z: np.ndarray, radius: float, current: float) -> np.ndarray:
    """Exact Bz on the axis of a circular loop (Biot-Savart closed form)."""
    return MU0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def loop_axis_field_dipole(z: np.ndarray, radius: float, current: float) -> np.ndarray:
    """Dipole far-field approximation for the same loop on its axis."""
    magnetic_moment = current * np.pi * radius**2
    return MU0 * magnetic_moment / (2.0 * np.pi * z**3)


def build_loop_comparison_table(radius: float, current: float, z_samples: np.ndarray) -> LoopComparisonMetrics:
    """Compare exact loop axis field and dipole approximation."""
    exact = loop_axis_field_exact(z_samples, radius, current)
    approx = loop_axis_field_dipole(z_samples, radius, current)

    rel_error = np.abs(approx - exact) / np.maximum(np.abs(exact), EPS)
    ratio = z_samples / radius

    table = pd.DataFrame(
        {
            "z(m)": z_samples,
            "z/a": ratio,
            "B_exact(T)": exact,
            "B_dipole(T)": approx,
            "relative_error": rel_error,
        }
    )

    far_field_mask = ratio >= 5.0
    max_rel_error_far = float(np.max(rel_error[far_field_mask]))

    return LoopComparisonMetrics(max_rel_error_far_field=max_rel_error_far, table=table)


def build_sample_point_table(m: np.ndarray) -> pd.DataFrame:
    """Compute field values at representative points."""
    points = np.array(
        [
            [0.8, 0.1, 0.5],
            [0.5, -0.4, 0.9],
            [1.2, 0.3, -0.2],
            [0.6, 0.7, 0.4],
        ],
        dtype=float,
    )
    B = dipole_field(points, m)

    return pd.DataFrame(
        {
            "x(m)": points[:, 0],
            "y(m)": points[:, 1],
            "z(m)": points[:, 2],
            "Bx(T)": B[:, 0],
            "By(T)": B[:, 1],
            "Bz(T)": B[:, 2],
            "|B|(T)": np.linalg.norm(B, axis=1),
        }
    )


def main() -> None:
    m = np.array([0.15, -0.05, 0.20], dtype=float)

    sample_table = build_sample_point_table(m)
    div_metrics = divergence_check(m)
    scaling_metrics = inverse_cubic_scaling_check(m)

    loop_radius = 0.1
    loop_current = 5.0
    z_samples = np.array([0.2, 0.3, 0.5, 0.7, 1.0], dtype=float)
    loop_metrics = build_loop_comparison_table(loop_radius, loop_current, z_samples)

    checks = {
        "divergence max < 3e-9": div_metrics.max_abs_divergence < 3e-9,
        "divergence rms < 5e-10": div_metrics.rms_divergence < 5e-10,
        "inverse-cubic ratio error < 1e-12": scaling_metrics.max_abs_ratio_error < 1e-12,
        "far-field loop relative error (z/a>=5) < 6.5%": loop_metrics.max_rel_error_far_field < 0.065,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Magnetic Dipole Field MVP (PHYS-0020) ===")
    print(f"dipole moment m = {m} A*m^2")

    print("\nSample field values:")
    print(sample_table.to_string(index=False))

    print("\nDivergence check on an origin-free box:")
    print(f"max |div B| = {div_metrics.max_abs_divergence:.3e}")
    print(f"rms |div B| = {div_metrics.rms_divergence:.3e}")

    print("\nInverse-cubic scaling check (expected ratio = 8):")
    print(scaling_metrics.table.to_string(index=False))
    print(f"max ratio error = {scaling_metrics.max_abs_ratio_error:.3e}")

    print("\nLoop-axis exact vs dipole approximation:")
    print(loop_metrics.table.to_string(index=False))
    print(f"max far-field relative error (z/a>=5) = {loop_metrics.max_rel_error_far_field:.3%}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
