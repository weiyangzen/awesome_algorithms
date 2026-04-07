"""Rigid Body Kinematics MVP (PHYS-0101).

This script demonstrates core rigid-body kinematics identities:
1) R_dot = [omega]x R
2) v_P = v_O + omega x r
3) a_P = a_O + alpha x r + omega x (omega x r)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class RigidBodyScenario:
    """Scenario parameters for rigid-body kinematics validation."""

    t_end: float = 10.0
    num_steps: int = 2401
    r_body: np.ndarray = field(
        default_factory=lambda: np.array([0.4, -0.2, 0.35], dtype=float)
    )


def skew(w: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [w]x."""

    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float)


def omega_world(t: float) -> np.ndarray:
    """Time-varying angular velocity in inertial frame."""

    return np.array(
        [
            0.8 + 0.15 * np.cos(0.7 * t),
            -0.3 + 0.10 * np.sin(0.5 * t),
            1.2 + 0.12 * np.cos(0.9 * t),
        ],
        dtype=float,
    )


def alpha_world(t: float) -> np.ndarray:
    """Time derivative of omega_world(t)."""

    return np.array(
        [
            -0.15 * 0.7 * np.sin(0.7 * t),
            0.10 * 0.5 * np.cos(0.5 * t),
            -0.12 * 0.9 * np.sin(0.9 * t),
        ],
        dtype=float,
    )


def origin_kinematics(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_O, v_O, a_O) for the reference point O."""

    x_o = np.array(
        [0.3 * t, -0.1 * t + 0.05 * t * t, 0.2 * np.sin(0.6 * t)],
        dtype=float,
    )
    v_o = np.array([0.3, -0.1 + 0.1 * t, 0.12 * np.cos(0.6 * t)], dtype=float)
    a_o = np.array([0.0, 0.1, -0.072 * np.sin(0.6 * t)], dtype=float)
    return x_o, v_o, a_o


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis-angle to rotation matrix via Rodrigues' formula."""

    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0.0:
        raise ValueError("axis norm must be positive")
    k = axis / axis_norm
    kx = skew(k)
    eye = np.eye(3)
    return eye + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def initial_rotation() -> np.ndarray:
    """Construct a non-trivial initial orientation."""

    axis = np.array([1.0, 2.0, -1.0], dtype=float)
    angle = 0.45
    return rodrigues(axis, angle)


def rotation_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """ODE RHS for flattened rotation matrix with inertial omega(t)."""

    r_mat = y.reshape(3, 3)
    r_dot = skew(omega_world(t)) @ r_mat
    return r_dot.reshape(-1)


def project_to_so3(r_mat: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix to SO(3) via SVD."""

    u_mat, _, vt_mat = np.linalg.svd(r_mat)
    r_proj = u_mat @ vt_mat
    if np.linalg.det(r_proj) < 0.0:
        u_mat[:, -1] *= -1.0
        r_proj = u_mat @ vt_mat
    return r_proj


def integrate_rotation(scenario: RigidBodyScenario) -> tuple[np.ndarray, np.ndarray]:
    """Integrate rotation ODE and return (t, R[t])."""

    r0 = initial_rotation()
    t_eval = np.linspace(0.0, scenario.t_end, scenario.num_steps)

    sol = solve_ivp(
        fun=rotation_rhs,
        t_span=(0.0, scenario.t_end),
        y0=r0.reshape(-1),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Rotation ODE failed: {sol.message}")

    raw_rotation = sol.y.T.reshape(-1, 3, 3)
    rotation = np.stack([project_to_so3(r_mat) for r_mat in raw_rotation], axis=0)
    return sol.t, rotation


def point_kinematics(
    t_grid: np.ndarray,
    rotation: np.ndarray,
    scenario: RigidBodyScenario,
) -> dict[str, np.ndarray]:
    """Compute position/velocity/acceleration of a body-fixed point."""

    n_steps = len(t_grid)
    r_world = np.empty((n_steps, 3), dtype=float)
    p_world = np.empty((n_steps, 3), dtype=float)
    v_formula = np.empty((n_steps, 3), dtype=float)
    a_formula = np.empty((n_steps, 3), dtype=float)
    omega_hist = np.empty((n_steps, 3), dtype=float)
    alpha_hist = np.empty((n_steps, 3), dtype=float)

    for i, t in enumerate(t_grid):
        w = omega_world(float(t))
        a = alpha_world(float(t))
        x_o, v_o, a_o = origin_kinematics(float(t))

        r = rotation[i] @ scenario.r_body
        p = x_o + r
        v = v_o + np.cross(w, r)
        acc = a_o + np.cross(a, r) + np.cross(w, np.cross(w, r))

        omega_hist[i] = w
        alpha_hist[i] = a
        r_world[i] = r
        p_world[i] = p
        v_formula[i] = v
        a_formula[i] = acc

    return {
        "r_world": r_world,
        "p_world": p_world,
        "v_formula": v_formula,
        "a_formula": a_formula,
        "omega": omega_hist,
        "alpha": alpha_hist,
    }


def finite_difference_kinematics(
    t_grid: np.ndarray,
    p_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical derivatives of position: velocity and acceleration."""

    v_num = np.gradient(p_world, t_grid, axis=0, edge_order=2)
    a_num = np.gradient(v_num, t_grid, axis=0, edge_order=2)
    return v_num, a_num


def simulate(scenario: RigidBodyScenario) -> dict[str, float]:
    """Run simulation and return diagnostic metrics."""

    t_grid, rotation = integrate_rotation(scenario)
    kin = point_kinematics(t_grid, rotation, scenario)
    v_num, a_num = finite_difference_kinematics(t_grid, kin["p_world"])
    r_num = np.gradient(kin["r_world"], t_grid, axis=0, edge_order=2)

    rot_err = rotation.transpose(0, 2, 1) @ rotation - np.eye(3)[None, :, :]
    orthogonality_max_fro = float(np.max(np.linalg.norm(rot_err, axis=(1, 2))))
    determinant_max_abs_error = float(np.max(np.abs(np.linalg.det(rotation) - 1.0)))

    interior = slice(2, -2)
    v_diff = kin["v_formula"][interior] - v_num[interior]
    a_diff = kin["a_formula"][interior] - a_num[interior]
    transport_diff = r_num[interior] - np.cross(
        kin["omega"][interior], kin["r_world"][interior]
    )

    velocity_rmse = float(np.sqrt(np.mean(v_diff * v_diff)))
    acceleration_rmse = float(np.sqrt(np.mean(a_diff * a_diff)))
    transport_rmse = float(np.sqrt(np.mean(transport_diff * transport_diff)))

    velocity_max_abs = float(np.max(np.abs(v_diff)))
    acceleration_max_abs = float(np.max(np.abs(a_diff)))

    omega_norm = np.linalg.norm(kin["omega"], axis=1)
    omega_norm_range = float(np.max(omega_norm) - np.min(omega_norm))

    return {
        "orthogonality_max_fro": orthogonality_max_fro,
        "determinant_max_abs_error": determinant_max_abs_error,
        "velocity_rmse": velocity_rmse,
        "acceleration_rmse": acceleration_rmse,
        "transport_rmse": transport_rmse,
        "velocity_max_abs": velocity_max_abs,
        "acceleration_max_abs": acceleration_max_abs,
        "omega_norm_range": omega_norm_range,
        "t_end": float(scenario.t_end),
        "num_steps": float(scenario.num_steps),
    }


def print_report(metrics: dict[str, float]) -> None:
    """Print compact diagnostics table."""

    rows = [
        {"metric": "orthogonality_max_fro", "value": f"{metrics['orthogonality_max_fro']:.3e}"},
        {
            "metric": "determinant_max_abs_error",
            "value": f"{metrics['determinant_max_abs_error']:.3e}",
        },
        {"metric": "velocity_rmse", "value": f"{metrics['velocity_rmse']:.3e}"},
        {"metric": "velocity_max_abs", "value": f"{metrics['velocity_max_abs']:.3e}"},
        {"metric": "acceleration_rmse", "value": f"{metrics['acceleration_rmse']:.3e}"},
        {"metric": "acceleration_max_abs", "value": f"{metrics['acceleration_max_abs']:.3e}"},
        {"metric": "transport_rmse", "value": f"{metrics['transport_rmse']:.3e}"},
        {"metric": "omega_norm_range", "value": f"{metrics['omega_norm_range']:.3e}"},
    ]
    df = pd.DataFrame(rows)

    print("=== Rigid Body Kinematics MVP (PHYS-0101) ===")
    print(
        {
            "t_end": metrics["t_end"],
            "num_steps": int(metrics["num_steps"]),
            "equations_checked": [
                "R_dot=[omega]xR",
                "v=v_O+omega×r",
                "a=a_O+alpha×r+omega×(omega×r)",
            ],
        }
    )
    print(df.to_string(index=False))


def main() -> None:
    scenario = RigidBodyScenario()
    metrics = simulate(scenario)
    print_report(metrics)

    if metrics["orthogonality_max_fro"] > 1e-8:
        raise AssertionError("Orthogonality drift too large.")
    if metrics["determinant_max_abs_error"] > 1e-8:
        raise AssertionError("Determinant drift too large.")
    if metrics["velocity_rmse"] > 5e-4:
        raise AssertionError("Velocity identity mismatch is too large.")
    if metrics["acceleration_rmse"] > 8e-3:
        raise AssertionError("Acceleration identity mismatch is too large.")
    if metrics["transport_rmse"] > 5e-4:
        raise AssertionError("Transport theorem mismatch is too large.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
