"""Minimal runnable MVP for Coriolis force in a rotating Earth-fixed frame."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

EARTH_ANGULAR_SPEED = 7.2921159e-5  # rad/s


def check_finite_scalar(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def check_vector3(name: str, value: np.ndarray) -> None:
    if value.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")


def earth_rotation_vector_enu(latitude_deg: float, omega: float = EARTH_ANGULAR_SPEED) -> np.ndarray:
    """
    Earth angular velocity in local ENU coordinates.

    ENU axes: x-east, y-north, z-up.
    """
    check_finite_scalar("latitude_deg", latitude_deg)
    check_finite_scalar("omega", omega)
    if latitude_deg < -90.0 or latitude_deg > 90.0:
        raise ValueError(f"latitude_deg must be in [-90, 90], got {latitude_deg}")
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")

    lat = math.radians(latitude_deg)
    return np.array([0.0, omega * math.cos(lat), omega * math.sin(lat)], dtype=float)


def coriolis_acceleration(velocity_enu: np.ndarray, omega_enu: np.ndarray) -> np.ndarray:
    """Compute Coriolis acceleration a_c = -2 * (Omega x v)."""
    return -2.0 * np.cross(omega_enu, velocity_enu)


def rhs_state(state: np.ndarray, omega_enu: np.ndarray, gravity_enu: np.ndarray) -> np.ndarray:
    """
    State derivative for [x, y, z, vx, vy, vz].

    dx/dt = v
    dv/dt = g + a_coriolis
    """
    vel = state[3:]
    acc = gravity_enu + coriolis_acceleration(velocity_enu=vel, omega_enu=omega_enu)

    out = np.empty_like(state)
    out[:3] = vel
    out[3:] = acc
    return out


def rk4_step(state: np.ndarray, dt: float, omega_enu: np.ndarray, gravity_enu: np.ndarray) -> np.ndarray:
    """Single classic RK4 step for the rotating-frame ODE."""
    k1 = rhs_state(state=state, omega_enu=omega_enu, gravity_enu=gravity_enu)
    k2 = rhs_state(state=state + 0.5 * dt * k1, omega_enu=omega_enu, gravity_enu=gravity_enu)
    k3 = rhs_state(state=state + 0.5 * dt * k2, omega_enu=omega_enu, gravity_enu=gravity_enu)
    k4 = rhs_state(state=state + dt * k3, omega_enu=omega_enu, gravity_enu=gravity_enu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_trajectory(
    latitude_deg: float,
    dt: float,
    steps: int,
    initial_position_enu: np.ndarray,
    initial_velocity_enu: np.ndarray,
    include_coriolis: bool = True,
    gravity_mps2: float = 9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 3D ballistic motion in ENU with optional Coriolis force."""
    check_finite_scalar("dt", dt)
    check_finite_scalar("gravity_mps2", gravity_mps2)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if gravity_mps2 <= 0.0:
        raise ValueError(f"gravity_mps2 must be positive, got {gravity_mps2}")

    pos0 = np.asarray(initial_position_enu, dtype=float)
    vel0 = np.asarray(initial_velocity_enu, dtype=float)
    check_vector3("initial_position_enu", pos0)
    check_vector3("initial_velocity_enu", vel0)

    omega_enu = earth_rotation_vector_enu(latitude_deg) if include_coriolis else np.zeros(3, dtype=float)
    gravity_enu = np.array([0.0, 0.0, -gravity_mps2], dtype=float)

    t = np.empty(steps + 1, dtype=float)
    states = np.empty((steps + 1, 6), dtype=float)
    t[0] = 0.0
    states[0, :3] = pos0
    states[0, 3:] = vel0

    for n in range(steps):
        states[n + 1] = rk4_step(
            state=states[n],
            dt=dt,
            omega_enu=omega_enu,
            gravity_enu=gravity_enu,
        )
        t[n + 1] = t[n] + dt
        if not np.all(np.isfinite(states[n + 1])):
            raise RuntimeError("non-finite state encountered during integration")

    positions = states[:, :3]
    velocities = states[:, 3:]
    coriolis_acc = -2.0 * np.cross(omega_enu, velocities)
    return t, positions, velocities, coriolis_acc


def print_trajectory_sample(
    t: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    rows: int = 8,
) -> None:
    """Print early trajectory rows for sanity checking."""
    count = min(rows, t.shape[0])
    print(" n      t(s)      east(m)      north(m)        up(m)      vx(m/s)      vy(m/s)      vz(m/s)")
    print("-" * 98)
    for i in range(count):
        print(
            f"{i:2d}  {t[i]:8.3f}  {positions[i,0]:11.3f}  {positions[i,1]:11.3f}  "
            f"{positions[i,2]:11.3f}  {velocities[i,0]:11.3f}  {velocities[i,1]:11.3f}  {velocities[i,2]:11.3f}"
        )


def run_case(
    name: str,
    latitude_deg: float,
    dt: float,
    steps: int,
    position0: np.ndarray,
    velocity0: np.ndarray,
) -> None:
    print("=" * 108)
    print(f"Case: {name}")
    print(f"latitude={latitude_deg:.1f} deg, dt={dt:.3f} s, steps={steps}, horizon={dt * steps:.2f} s")
    print(f"initial position ENU (m): {position0}")
    print(f"initial velocity ENU (m/s): {velocity0}")

    t_c, pos_c, vel_c, acc_c = simulate_trajectory(
        latitude_deg=latitude_deg,
        dt=dt,
        steps=steps,
        initial_position_enu=position0,
        initial_velocity_enu=velocity0,
        include_coriolis=True,
    )
    _, pos_n, vel_n, _ = simulate_trajectory(
        latitude_deg=latitude_deg,
        dt=dt,
        steps=steps,
        initial_position_enu=position0,
        initial_velocity_enu=velocity0,
        include_coriolis=False,
    )

    delta = pos_c - pos_n
    final_delta = delta[-1]
    max_abs_delta = np.max(np.abs(delta), axis=0)

    speed_c = np.linalg.norm(vel_c, axis=1)
    speed_n = np.linalg.norm(vel_n, axis=1)
    max_speed_diff = float(np.max(np.abs(speed_c - speed_n)))

    # Coriolis force is perpendicular to velocity in continuous dynamics.
    power_density = np.sum(vel_c * acc_c, axis=1)
    max_abs_v_dot_ac = float(np.max(np.abs(power_density)))

    print("Final displacement difference (with - without Coriolis), ENU in meters:")
    print(f"  east={final_delta[0]: .6f}, north={final_delta[1]: .6f}, up={final_delta[2]: .6f}")
    print("Max absolute displacement difference over whole trajectory, ENU in meters:")
    print(f"  east={max_abs_delta[0]: .6f}, north={max_abs_delta[1]: .6f}, up={max_abs_delta[2]: .6f}")
    print(f"Max speed difference between trajectories: {max_speed_diff:.6e} m/s")
    print(f"Max |v · a_coriolis| in Coriolis run: {max_abs_v_dot_ac:.6e} (m^2/s^3)")

    print("Trajectory sample (with Coriolis):")
    print_trajectory_sample(t=t_c, positions=pos_c, velocities=vel_c, rows=8)


def main() -> None:
    print("Coriolis force MVP in local ENU rotating frame")
    print(f"Earth angular speed: {EARTH_ANGULAR_SPEED:.10e} rad/s")

    latitude_deg = 45.0
    dt = 0.05
    steps = 400  # 20 seconds
    position0 = np.array([0.0, 0.0, 1000.0], dtype=float)

    run_case(
        name="Northward launch (expect eastward deflection in Northern Hemisphere)",
        latitude_deg=latitude_deg,
        dt=dt,
        steps=steps,
        position0=position0,
        velocity0=np.array([0.0, 400.0, 80.0], dtype=float),
    )

    run_case(
        name="Eastward launch (shows south/up Coriolis components at mid-latitude)",
        latitude_deg=latitude_deg,
        dt=dt,
        steps=steps,
        position0=position0,
        velocity0=np.array([400.0, 0.0, 80.0], dtype=float),
    )


if __name__ == "__main__":
    main()
