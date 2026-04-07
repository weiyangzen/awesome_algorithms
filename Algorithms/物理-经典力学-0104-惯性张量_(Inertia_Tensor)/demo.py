"""Minimal runnable MVP for inertia tensor computation and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InertiaConfig:
    """Configuration for the inertia-tensor MVP experiment."""

    box_lengths: tuple[float, float, float] = (2.0, 1.4, 0.8)
    grid_shape: tuple[int, int, int] = (35, 27, 21)
    total_mass: float = 12.0
    world_shift: tuple[float, float, float] = (0.45, -0.30, 0.25)
    rotation_angles_rad: tuple[float, float, float] = (0.60, -0.35, 0.40)
    tol_symmetry: float = 1e-10
    tol_parallel_axis: float = 1e-9
    tol_rotation_covariance: float = 1e-9
    tol_analytic_relative: float = 2.0e-2


def build_uniform_box_cloud(
    lengths: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic uniform point cloud inside a rectangular box.

    The box is centered at origin and aligned with x/y/z axes.
    """

    if any(l <= 0.0 for l in lengths):
        raise ValueError("All box lengths must be positive.")
    if any(n < 2 for n in grid_shape):
        raise ValueError("Each grid dimension must be >= 2.")
    if total_mass <= 0.0:
        raise ValueError("total_mass must be positive.")

    lx, ly, lz = lengths
    nx, ny, nz = grid_shape

    # Use voxel centers instead of boundary nodes for better integral approximation.
    xs = ((np.arange(nx, dtype=float) + 0.5) / nx - 0.5) * lx
    ys = ((np.arange(ny, dtype=float) + 0.5) / ny - 0.5) * ly
    zs = ((np.arange(nz, dtype=float) + 0.5) / nz - 0.5) * lz

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(float)

    n_points = points.shape[0]
    masses = np.full(n_points, total_mass / n_points, dtype=float)
    return points, masses


def center_of_mass(points: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute center of mass for weighted point cloud."""

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")
    return np.sum(points * masses[:, None], axis=0) / total_mass


def inertia_tensor(points: np.ndarray, masses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Compute inertia tensor about an arbitrary origin.

    Formula:
        I = sum_i m_i * (||r_i||^2 E - r_i r_i^T)
    where r_i is point position relative to the chosen origin.
    """

    rel = points - origin[None, :]
    rel_sq = np.einsum("ni,ni->n", rel, rel)
    identity = np.eye(3, dtype=float)

    outer = np.einsum("ni,nj->nij", rel, rel)
    terms = rel_sq[:, None, None] * identity[None, :, :] - outer
    tensor = np.einsum("n,nij->ij", masses, terms)
    return 0.5 * (tensor + tensor.T)


def parallel_axis_term(total_mass: float, displacement: np.ndarray) -> np.ndarray:
    """Parallel-axis correction term M (||d||^2 E - d d^T)."""

    d2 = float(displacement @ displacement)
    return total_mass * (d2 * np.eye(3, dtype=float) - np.outer(displacement, displacement))


def rotation_matrix_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """Construct rotation matrix R = Rz(az) Ry(ay) Rx(ax)."""

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def analytic_box_inertia_at_com(lengths: tuple[float, float, float], total_mass: float) -> np.ndarray:
    """Analytic inertia tensor for a uniform box at COM in its body-aligned axes."""

    lx, ly, lz = lengths
    ixx = total_mass * (ly**2 + lz**2) / 12.0
    iyy = total_mass * (lx**2 + lz**2) / 12.0
    izz = total_mass * (lx**2 + ly**2) / 12.0
    return np.diag([ixx, iyy, izz]).astype(float)


def principal_moments_axes(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending principal moments and corresponding orthonormal axes."""

    eigvals, eigvecs = np.linalg.eigh(tensor)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def max_offdiag_abs(matrix: np.ndarray) -> float:
    """Maximum absolute off-diagonal entry."""

    temp = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(temp, 0.0)
    return float(np.max(np.abs(temp)))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute elementwise difference."""

    return float(np.max(np.abs(a - b)))


def matrix_str(matrix: np.ndarray) -> str:
    """Stable compact matrix formatting for console output."""

    return np.array2string(matrix, precision=6, suppress_small=False)


def main() -> None:
    cfg = InertiaConfig()

    points_local, masses = build_uniform_box_cloud(cfg.box_lengths, cfg.grid_shape, cfg.total_mass)
    points_world = points_local + np.array(cfg.world_shift, dtype=float)[None, :]

    total_mass = float(np.sum(masses))
    world_origin = np.zeros(3, dtype=float)
    com_world = center_of_mass(points_world, masses)

    i_origin = inertia_tensor(points_world, masses, world_origin)
    i_com_direct = inertia_tensor(points_world, masses, com_world)

    displacement = com_world - world_origin
    shift_term = parallel_axis_term(total_mass, displacement)
    i_origin_from_parallel = i_com_direct + shift_term
    i_com_from_parallel = i_origin - shift_term

    rot = rotation_matrix_xyz(*cfg.rotation_angles_rad)
    points_rot = points_world @ rot.T
    i_origin_rot_direct = inertia_tensor(points_rot, masses, world_origin)
    i_origin_rot_expected = rot @ i_origin @ rot.T

    i_com_analytic = analytic_box_inertia_at_com(cfg.box_lengths, total_mass)

    principal_vals, principal_vecs = principal_moments_axes(i_com_direct)
    diag_direct = np.diag(i_com_direct)
    diag_analytic = np.diag(i_com_analytic)
    rel_diag_err = np.abs((diag_direct - diag_analytic) / diag_analytic)

    metrics = {
        "symmetry_error_origin": max_abs_diff(i_origin, i_origin.T),
        "symmetry_error_com": max_abs_diff(i_com_direct, i_com_direct.T),
        "parallel_axis_error_origin": max_abs_diff(i_origin, i_origin_from_parallel),
        "parallel_axis_error_com": max_abs_diff(i_com_direct, i_com_from_parallel),
        "rotation_covariance_error": max_abs_diff(i_origin_rot_direct, i_origin_rot_expected),
        "analytic_diag_relative_error_max": float(np.max(rel_diag_err)),
        "com_offdiag_max": max_offdiag_abs(i_com_direct),
        "min_principal_moment": float(np.min(principal_vals)),
    }

    metrics_df = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in metrics.items()]
    )
    diag_df = pd.DataFrame(
        {
            "axis": ["x", "y", "z"],
            "I_discrete": diag_direct,
            "I_analytic": diag_analytic,
            "relative_error": rel_diag_err,
        }
    )
    principal_df = pd.DataFrame(
        {
            "mode": [1, 2, 3],
            "principal_moment": principal_vals,
            "axis_x": principal_vecs[0, :],
            "axis_y": principal_vecs[1, :],
            "axis_z": principal_vecs[2, :],
        }
    )

    print("Inertia Tensor MVP")
    print(
        f"box_lengths={cfg.box_lengths}, grid_shape={cfg.grid_shape}, "
        f"total_mass={total_mass:.6f}, world_shift={cfg.world_shift}"
    )
    print(f"center_of_mass={com_world}")

    print("\nI_origin:")
    print(matrix_str(i_origin))
    print("\nI_com_direct:")
    print(matrix_str(i_com_direct))

    print("\nprincipal_moments_and_axes:")
    print(principal_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    print("\nanalytic_vs_discrete_diagonal:")
    print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    print("\nmetrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    passed = (
        metrics["symmetry_error_origin"] <= cfg.tol_symmetry
        and metrics["symmetry_error_com"] <= cfg.tol_symmetry
        and metrics["parallel_axis_error_origin"] <= cfg.tol_parallel_axis
        and metrics["parallel_axis_error_com"] <= cfg.tol_parallel_axis
        and metrics["rotation_covariance_error"] <= cfg.tol_rotation_covariance
        and metrics["analytic_diag_relative_error_max"] <= cfg.tol_analytic_relative
        and metrics["min_principal_moment"] > 0.0
    )

    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("Inertia tensor validation failed.")


if __name__ == "__main__":
    main()
