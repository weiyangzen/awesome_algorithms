"""Euler Angles (ZXZ) MVP for classical mechanics.

This script implements a compact, auditable pipeline for:
1) Euler angles (ZXZ intrinsic) -> rotation matrix
2) Rotation matrix -> Euler angles (with singular handling)
3) Euler angle rates -> body angular velocity
4) Trajectory-wise angular velocity and rotational kinetic energy

The script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def euler_zxz_to_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Compute R = Rz(phi) * Rx(theta) * Rz(psi)."""
    return rotation_z(phi) @ rotation_x(theta) @ rotation_z(psi)


def matrix_to_euler_zxz(r: np.ndarray, eps: float = 1e-10) -> Tuple[float, float, float, bool]:
    """Recover one valid ZXZ Euler-angle solution from a rotation matrix.

    Returns:
    - phi, theta, psi
    - singular flag (True near theta=0 or theta=pi)
    """
    if r.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got shape={r.shape}")

    ctheta = float(np.clip(r[2, 2], -1.0, 1.0))
    theta = float(np.arccos(ctheta))
    stheta = float(np.sin(theta))

    singular = abs(stheta) < eps
    if not singular:
        # From analytical ZXZ matrix entries:
        # r13 = sin(phi) * sin(theta), r23 = -cos(phi) * sin(theta)
        # r31 = sin(theta) * sin(psi), r32 = sin(theta) * cos(psi)
        phi = float(np.arctan2(r[0, 2], -r[1, 2]))
        psi = float(np.arctan2(r[2, 0], r[2, 1]))
    else:
        # Gimbal lock: phi and psi are not individually identifiable.
        phi = 0.0
        if ctheta > 0.0:
            theta = 0.0
            psi = float(np.arctan2(r[1, 0], r[0, 0]))
        else:
            theta = float(np.pi)
            psi = float(-np.arctan2(r[1, 0], r[0, 0]))

    return wrap_to_pi(phi), theta, wrap_to_pi(psi), singular


def euler_rates_to_body_omega(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> np.ndarray:
    """Map ZXZ Euler rates to body-frame angular velocity components."""
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(psi)
    cp = np.cos(psi)

    wx = phi_dot * st * sp + theta_dot * cp
    wy = phi_dot * st * cp - theta_dot * sp
    wz = phi_dot * ct + psi_dot
    return np.array([wx, wy, wz], dtype=float)


def skew_to_vector(skew: np.ndarray) -> np.ndarray:
    """Convert 3x3 skew-symmetric matrix to vector [wx, wy, wz]."""
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


def omega_from_matrix_derivative(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    dt: float = 1e-6,
) -> np.ndarray:
    """Estimate body angular velocity from R^T * dR/dt using central difference."""
    r = euler_zxz_to_matrix(phi, theta, psi)

    r_plus = euler_zxz_to_matrix(
        phi + phi_dot * dt,
        theta + theta_dot * dt,
        psi + psi_dot * dt,
    )
    r_minus = euler_zxz_to_matrix(
        phi - phi_dot * dt,
        theta - theta_dot * dt,
        psi - psi_dot * dt,
    )

    r_dot = (r_plus - r_minus) / (2.0 * dt)
    omega_skew = r.T @ r_dot
    return skew_to_vector(omega_skew)


def trajectory_state(t: float) -> Tuple[float, float, float, float, float, float]:
    """Return (phi, theta, psi, phi_dot, theta_dot, psi_dot) at time t."""
    phi = 0.2 + 0.8 * t
    theta = 1.0 + 0.25 * np.sin(1.3 * t)
    psi = -0.3 + 1.1 * t

    phi_dot = 0.8
    theta_dot = 0.25 * 1.3 * np.cos(1.3 * t)
    psi_dot = 1.1
    return phi, theta, psi, phi_dot, theta_dot, psi_dot


def batch_kinetic_energy_torch(omegas: np.ndarray, inertia_diag: np.ndarray) -> np.ndarray:
    """Compute T = 0.5 * omega^T * I * omega for each row using PyTorch."""
    omega_t = torch.as_tensor(omegas, dtype=torch.float64)
    inertia_t = torch.diag(torch.as_tensor(inertia_diag, dtype=torch.float64))
    # (N,3) @ (3,3) -> (N,3), then row-wise inner product with omega
    kinetic_t = 0.5 * torch.sum((omega_t @ inertia_t) * omega_t, dim=1)
    return kinetic_t.cpu().numpy()


def build_trajectory_table(times: np.ndarray, inertia_diag: np.ndarray) -> pd.DataFrame:
    rows = []
    omega_list = []

    for t in times:
        phi, theta, psi, phi_dot, theta_dot, psi_dot = trajectory_state(float(t))
        omega = euler_rates_to_body_omega(phi, theta, psi, phi_dot, theta_dot, psi_dot)

        rows.append(
            {
                "t": float(t),
                "phi": float(phi),
                "theta": float(theta),
                "psi": float(psi),
                "wx": float(omega[0]),
                "wy": float(omega[1]),
                "wz": float(omega[2]),
                "omega_norm": float(np.linalg.norm(omega)),
            }
        )
        omega_list.append(omega)

    df = pd.DataFrame(rows)
    omegas = np.vstack(omega_list)
    df["kinetic_energy"] = batch_kinetic_energy_torch(omegas, inertia_diag)
    return df


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    # 1) Single-state representation checks
    phi, theta, psi = 0.9, 1.1, -0.7
    r_manual = euler_zxz_to_matrix(phi, theta, psi)
    r_scipy = Rotation.from_euler("ZXZ", [phi, theta, psi], degrees=False).as_matrix()

    orthogonality_error = float(np.linalg.norm(r_manual.T @ r_manual - np.eye(3)))
    determinant = float(np.linalg.det(r_manual))
    manual_vs_scipy_error = float(np.linalg.norm(r_manual - r_scipy))

    # 2) Round-trip angle recovery
    phi_rec, theta_rec, psi_rec, singular_flag = matrix_to_euler_zxz(r_manual)
    r_recovered = euler_zxz_to_matrix(phi_rec, theta_rec, psi_rec)
    roundtrip_matrix_error = float(np.linalg.norm(r_manual - r_recovered))

    # 3) Angular velocity consistency (formula vs matrix derivative)
    phi_dot, theta_dot, psi_dot = 0.35, -0.22, 0.78
    omega_formula = euler_rates_to_body_omega(phi, theta, psi, phi_dot, theta_dot, psi_dot)
    omega_numeric = omega_from_matrix_derivative(phi, theta, psi, phi_dot, theta_dot, psi_dot)
    omega_error = float(np.linalg.norm(omega_formula - omega_numeric))

    # 4) Batch trajectory + kinetic energy
    times = np.linspace(0.0, 2.0, 9)
    inertia_diag = np.array([2.0, 1.5, 1.0], dtype=float)
    df = build_trajectory_table(times, inertia_diag)

    # 5) Near-singularity demonstration
    r_singular = euler_zxz_to_matrix(0.3, 1e-12, -1.1)
    _, theta_s, _, singular_detected = matrix_to_euler_zxz(r_singular)

    print("Euler Angles (ZXZ) MVP")
    print(f"orthogonality_error={orthogonality_error:.3e}")
    print(f"determinant={determinant:.12f}")
    print(f"manual_vs_scipy_error={manual_vs_scipy_error:.3e}")
    print(f"roundtrip_matrix_error={roundtrip_matrix_error:.3e}")
    print(f"roundtrip_singular_flag={singular_flag}")
    print()

    print("omega_formula=", np.array2string(omega_formula, precision=6))
    print("omega_numeric=", np.array2string(omega_numeric, precision=6))
    print(f"omega_formula_vs_numeric_error={omega_error:.3e}")
    print()

    print("Trajectory table (t, angles, omega, kinetic_energy):")
    print(df.round(6).to_string(index=False))
    print()

    summary = df[["omega_norm", "kinetic_energy"]].agg(["min", "max", "mean"])
    print("Trajectory summary:")
    print(summary.round(6).to_string())
    print()

    checks: Dict[str, bool] = {
        "rotation_is_orthogonal": orthogonality_error < 1e-10,
        "det_close_to_one": abs(determinant - 1.0) < 1e-10,
        "manual_matches_scipy": manual_vs_scipy_error < 1e-10,
        "roundtrip_ok": roundtrip_matrix_error < 1e-10,
        "omega_match": omega_error < 1e-6,
        "all_energy_finite": bool(np.all(np.isfinite(df["kinetic_energy"].to_numpy()))),
        "singular_detected": bool(singular_detected and abs(theta_s) < 1e-8),
    }

    print("Checks:")
    for key, value in checks.items():
        print(f"- {key}={value}")

    print(f"all_core_checks_pass={all(checks.values())}")


if __name__ == "__main__":
    main()
