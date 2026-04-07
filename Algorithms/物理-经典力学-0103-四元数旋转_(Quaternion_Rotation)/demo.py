"""Quaternion Rotation MVP for classical mechanics.

This script implements source-level quaternion rotation, not a black-box call:
1) Axis-angle -> quaternion
2) Quaternion multiply / conjugate / normalize
3) Quaternion -> rotation matrix
4) Point rotation by q * p * q_conj (and vectorized equivalent)
5) Composition checks + SciPy/sklearn/Torch cross-validation

No interactive input is required.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error


def check_vector(name: str, x: np.ndarray, dim: int) -> None:
    if x.ndim != 1 or x.shape[0] != dim:
        raise ValueError(f"{name} must be a 1D vector of length {dim}, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_points(name: str, points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an (N, 3) array, got shape={points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values.")


def normalize_quaternion(q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    check_vector("quaternion", q, 4)
    nrm = float(np.linalg.norm(q))
    if not np.isfinite(nrm) or nrm <= eps:
        raise ValueError("quaternion norm is too small for normalization.")
    return (q / nrm).astype(float)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    check_vector("quaternion", q, 4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first quaternions q=[w, x, y, z]."""
    check_vector("q1", q1, 4)
    check_vector("q2", q2, 4)
    a = q1.astype(float)
    b = q2.astype(float)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    check_vector("axis", axis, 3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-15:
        raise ValueError("axis norm is too small.")
    axis_unit = axis / axis_norm
    half = 0.5 * float(angle)
    s = float(np.sin(half))
    q = np.array(
        [
            float(np.cos(half)),
            float(axis_unit[0] * s),
            float(axis_unit[1] * s),
            float(axis_unit[2] * s),
        ],
        dtype=float,
    )
    return normalize_quaternion(q)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert scalar-first unit quaternion to 3x3 active rotation matrix."""
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    check_vector("vector", v, 3)
    qn = normalize_quaternion(q)
    p = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    qp = quaternion_multiply(qn, p)
    qpq = quaternion_multiply(qp, quaternion_conjugate(qn))
    return qpq[1:]


def rotate_points_numpy(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized rotation formula for unit quaternion q=[s, u]."""
    check_points("points", points)
    qn = normalize_quaternion(q)
    s = float(qn[0])
    u = qn[1:]
    u_batch = np.broadcast_to(u, points.shape)
    cross1 = np.cross(u_batch, points)
    cross2 = np.cross(u_batch, cross1)
    return points + 2.0 * (s * cross1 + cross2)


def rotate_points_torch(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Torch version for batch consistency checking."""
    check_points("points", points)
    qn = normalize_quaternion(q)

    points_t = torch.as_tensor(points, dtype=torch.float64)
    q_t = torch.as_tensor(qn, dtype=torch.float64)
    s_t = q_t[0]
    u_t = q_t[1:]
    u_batch = u_t.unsqueeze(0).expand_as(points_t)

    cross1_t = torch.cross(u_batch, points_t, dim=1)
    cross2_t = torch.cross(u_batch, cross1_t, dim=1)
    rotated_t = points_t + 2.0 * (s_t * cross1_t + cross2_t)
    return rotated_t.cpu().numpy()


def compose_active_rotations(q_first: np.ndarray, q_second: np.ndarray) -> np.ndarray:
    """Apply q_first then q_second -> q_total = q_second * q_first."""
    return normalize_quaternion(quaternion_multiply(q_second, q_first))


def scipy_rotation_from_scalar_first(q: np.ndarray) -> Rotation:
    qn = normalize_quaternion(q)
    # SciPy uses [x, y, z, w]; our script uses [w, x, y, z].
    scipy_quat = np.array([qn[1], qn[2], qn[3], qn[0]], dtype=float)
    return Rotation.from_quat(scipy_quat)


def simulate_uniform_rotation(
    axis: np.ndarray,
    omega: float,
    times: Iterable[float],
    body_point: np.ndarray,
    mass: float,
) -> pd.DataFrame:
    check_vector("axis", axis, 3)
    check_vector("body_point", body_point, 3)
    if mass <= 0.0:
        raise ValueError("mass must be > 0.")

    axis_unit = axis / float(np.linalg.norm(axis))
    omega_vec = float(omega) * axis_unit

    rows = []
    for t in times:
        t_val = float(t)
        angle = float(omega) * t_val
        q_t = axis_angle_to_quaternion(axis_unit, angle)
        r_t = rotate_vector_by_quaternion(body_point, q_t)
        v_t = np.cross(omega_vec, r_t)
        ke = 0.5 * mass * float(v_t @ v_t)

        rows.append(
            {
                "t": t_val,
                "angle": angle,
                "qw": float(q_t[0]),
                "qx": float(q_t[1]),
                "qy": float(q_t[2]),
                "qz": float(q_t[3]),
                "rx": float(r_t[0]),
                "ry": float(r_t[1]),
                "rz": float(r_t[2]),
                "speed": float(np.linalg.norm(v_t)),
                "kinetic_energy": ke,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # 1) Base quaternion from axis-angle
    axis = np.array([1.0, 2.0, -1.0], dtype=float)
    angle = 1.2
    q = axis_angle_to_quaternion(axis, angle)

    # 2) Rotate marker points (manual vs SciPy)
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-1.5, 0.2, 0.7],
        ],
        dtype=float,
    )
    rotated_manual = rotate_points_numpy(points, q)
    rotated_scipy = scipy_rotation_from_scalar_first(q).apply(points)

    diff = rotated_manual - rotated_scipy
    max_point_error = float(np.max(np.linalg.norm(diff, axis=1)))
    mse = float(mean_squared_error(rotated_scipy.reshape(-1), rotated_manual.reshape(-1)))
    rmse = float(np.sqrt(mse))

    # 3) Matrix checks
    r_manual = quaternion_to_matrix(q)
    r_scipy = scipy_rotation_from_scalar_first(q).as_matrix()
    matrix_error = float(np.linalg.norm(r_manual - r_scipy))
    orthogonality_error = float(np.linalg.norm(r_manual.T @ r_manual - np.eye(3)))
    determinant = float(np.linalg.det(r_manual))

    # 4) Composition checks: apply q_a then q_b
    q_a = axis_angle_to_quaternion(np.array([1.0, 0.0, 1.0], dtype=float), 0.7)
    q_b = axis_angle_to_quaternion(np.array([0.0, 1.0, 1.0], dtype=float), -1.1)
    q_ab = compose_active_rotations(q_a, q_b)

    test_vec = np.array([0.4, -0.2, 1.1], dtype=float)
    seq_vec = rotate_vector_by_quaternion(rotate_vector_by_quaternion(test_vec, q_a), q_b)
    once_vec = rotate_vector_by_quaternion(test_vec, q_ab)
    composition_error = float(np.linalg.norm(seq_vec - once_vec))

    matrix_comp_error = float(
        np.linalg.norm(
            quaternion_to_matrix(q_ab) - quaternion_to_matrix(q_b) @ quaternion_to_matrix(q_a)
        )
    )

    # 5) Torch batch consistency check
    rotated_torch = rotate_points_torch(points, q)
    torch_consistency_error = float(np.max(np.linalg.norm(rotated_torch - rotated_manual, axis=1)))

    # 6) Uniform rotation trajectory (classical mechanics linkage)
    times = np.linspace(0.0, 2.5, 11)
    traj = simulate_uniform_rotation(
        axis=np.array([0.3, -0.4, 0.5], dtype=float),
        omega=2.2,
        times=times,
        body_point=np.array([0.8, 0.1, -0.2], dtype=float),
        mass=1.7,
    )
    energy_span = float(traj["kinetic_energy"].max() - traj["kinetic_energy"].min())

    print("Quaternion Rotation MVP")
    print("base quaternion [w, x, y, z] =", np.array2string(q, precision=6))
    print()

    print("Manual vs SciPy on rotated points:")
    print(f"max_point_error={max_point_error:.3e}")
    print(f"rmse={rmse:.3e}")
    print()

    print("Rotation matrix checks:")
    print(f"matrix_error_manual_vs_scipy={matrix_error:.3e}")
    print(f"orthogonality_error={orthogonality_error:.3e}")
    print(f"determinant={determinant:.12f}")
    print()

    print("Composition checks:")
    print(f"vector_composition_error={composition_error:.3e}")
    print(f"matrix_composition_error={matrix_comp_error:.3e}")
    print()

    print("Torch consistency:")
    print(f"torch_vs_numpy_max_error={torch_consistency_error:.3e}")
    print()

    print("Uniform-rotation trajectory:")
    print(traj.round(6).to_string(index=False))
    print()

    summary = traj[["speed", "kinetic_energy"]].agg(["min", "max", "mean"])
    print("Trajectory summary:")
    print(summary.round(6).to_string())
    print(f"kinetic_energy_span={energy_span:.3e}")
    print()

    checks: Dict[str, bool] = {
        "point_rotation_match_scipy": max_point_error < 1e-12,
        "matrix_match_scipy": matrix_error < 1e-12,
        "rotation_matrix_orthogonal": orthogonality_error < 1e-12,
        "det_close_to_one": abs(determinant - 1.0) < 1e-12,
        "composition_vector_ok": composition_error < 1e-12,
        "composition_matrix_ok": matrix_comp_error < 1e-12,
        "torch_matches_numpy": torch_consistency_error < 1e-12,
        "trajectory_finite": bool(np.all(np.isfinite(traj.to_numpy()))),
        "trajectory_energy_nearly_constant": energy_span < 1e-10,
    }

    print("Checks:")
    for key, value in checks.items():
        print(f"- {key}={value}")
    print(f"all_core_checks_pass={all(checks.values())}")


if __name__ == "__main__":
    main()
