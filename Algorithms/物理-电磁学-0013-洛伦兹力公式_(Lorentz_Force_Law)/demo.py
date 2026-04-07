"""Minimal runnable MVP for Lorentz Force Law.

This script demonstrates two things:
1) Force-level properties of F = q(E + v x B).
2) Trajectory-level behavior under uniform magnetic field via Boris pusher.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


Vector = np.ndarray


def lorentz_force(charge: float, velocity: Vector, electric_field: Vector, magnetic_field: Vector) -> Tuple[Vector, Vector, Vector]:
    """Return (F_total, F_electric, F_magnetic)."""
    v = np.asarray(velocity, dtype=np.float64)
    e = np.asarray(electric_field, dtype=np.float64)
    b = np.asarray(magnetic_field, dtype=np.float64)

    f_electric = charge * e
    f_magnetic = charge * np.cross(v, b)
    f_total = f_electric + f_magnetic
    return f_total, f_electric, f_magnetic


def boris_push(
    position: Vector,
    velocity: Vector,
    charge: float,
    mass: float,
    electric_field: Vector,
    magnetic_field: Vector,
    dt: float,
) -> Tuple[Vector, Vector]:
    """Advance one step using the non-relativistic Boris pusher.

    This method is robust for Lorentz dynamics and preserves kinetic energy well
    in the pure-magnetic case.
    """
    x = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    e = np.asarray(electric_field, dtype=np.float64)
    b = np.asarray(magnetic_field, dtype=np.float64)

    qmdt2 = (charge / mass) * (dt * 0.5)

    v_minus = v + qmdt2 * e
    t = qmdt2 * b
    t2 = float(np.dot(t, t))
    s = 2.0 * t / (1.0 + t2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_new = v_plus + qmdt2 * e

    x_new = x + v_new * dt
    return x_new, v_new


def kinetic_energy(mass: float, velocity: Vector) -> float:
    """Return kinetic energy 1/2 m |v|^2."""
    return 0.5 * mass * float(np.dot(velocity, velocity))


def simulate_particle(
    mass: float,
    charge: float,
    electric_field: Vector,
    magnetic_field: Vector,
    x0: Vector,
    v0: Vector,
    dt: float,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Simulate single-particle Lorentz dynamics and return full history."""
    times = np.zeros(steps + 1, dtype=np.float64)
    positions = np.zeros((steps + 1, 3), dtype=np.float64)
    velocities = np.zeros((steps + 1, 3), dtype=np.float64)
    kinetic = np.zeros(steps + 1, dtype=np.float64)

    positions[0] = np.asarray(x0, dtype=np.float64)
    velocities[0] = np.asarray(v0, dtype=np.float64)
    kinetic[0] = kinetic_energy(mass, velocities[0])

    for i in range(steps):
        x_new, v_new = boris_push(
            position=positions[i],
            velocity=velocities[i],
            charge=charge,
            mass=mass,
            electric_field=electric_field,
            magnetic_field=magnetic_field,
            dt=dt,
        )
        positions[i + 1] = x_new
        velocities[i + 1] = v_new
        kinetic[i + 1] = kinetic_energy(mass, v_new)
        times[i + 1] = (i + 1) * dt

    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "kinetic": kinetic,
    }


def run_force_property_demo() -> List[Dict[str, float]]:
    """Validate algebraic properties of Lorentz force on deterministic cases."""
    rows: List[Dict[str, float]] = []

    # Case 1: v ⟂ B and E=0 -> |F_B| = |q| v B
    q = 1.5
    v = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    e = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    f_total, f_e, f_b = lorentz_force(q, v, e, b)

    expected_mag = abs(q) * float(np.linalg.norm(v)) * float(np.linalg.norm(b))
    actual_mag = float(np.linalg.norm(f_b))
    rel_err = abs(actual_mag - expected_mag) / expected_mag
    assert rel_err < 1e-12, "Perpendicular-case magnetic force magnitude mismatch."

    rows.append(
        {
            "case": 1.0,
            "f_total_norm": float(np.linalg.norm(f_total)),
            "f_e_norm": float(np.linalg.norm(f_e)),
            "f_b_norm": actual_mag,
            "dot_fb_v": float(np.dot(f_b, v)),
            "dot_fb_b": float(np.dot(f_b, b)),
            "rel_err_mag": rel_err,
        }
    )

    # Case 2: v || B -> magnetic force should vanish.
    q2 = 2.0
    v2 = np.array([0.0, 0.0, 4.0], dtype=np.float64)
    e2 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    b2 = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    f_total2, f_e2, f_b2 = lorentz_force(q2, v2, e2, b2)
    assert float(np.linalg.norm(f_b2)) < 1e-12, "Magnetic force should be zero for v parallel B."

    rows.append(
        {
            "case": 2.0,
            "f_total_norm": float(np.linalg.norm(f_total2)),
            "f_e_norm": float(np.linalg.norm(f_e2)),
            "f_b_norm": float(np.linalg.norm(f_b2)),
            "dot_fb_v": float(np.dot(f_b2, v2)),
            "dot_fb_b": float(np.dot(f_b2, b2)),
            "rel_err_mag": 0.0,
        }
    )

    # Case 3: charge sign reversal -> force sign reversal.
    v3 = np.array([1.1, -0.4, 0.7], dtype=np.float64)
    e3 = np.array([0.3, -0.2, 0.5], dtype=np.float64)
    b3 = np.array([0.0, 1.2, 2.4], dtype=np.float64)
    q3 = 0.8

    f_pos, _, f_b_pos = lorentz_force(q3, v3, e3, b3)
    f_neg, _, f_b_neg = lorentz_force(-q3, v3, e3, b3)

    assert np.allclose(f_neg, -f_pos, atol=1e-12), "Lorentz force should flip with charge sign."

    dot_v = float(np.dot(f_b_pos, v3))
    dot_b = float(np.dot(f_b_pos, b3))
    assert abs(dot_v) < 1e-12 and abs(dot_b) < 1e-12, "Magnetic force should be orthogonal to v and B."
    assert np.allclose(f_b_neg, -f_b_pos, atol=1e-12), "Magnetic component should flip with charge sign."

    rows.append(
        {
            "case": 3.0,
            "f_total_norm": float(np.linalg.norm(f_pos)),
            "f_e_norm": float(np.linalg.norm(q3 * e3)),
            "f_b_norm": float(np.linalg.norm(f_b_pos)),
            "dot_fb_v": dot_v,
            "dot_fb_b": dot_b,
            "rel_err_mag": 0.0,
        }
    )

    return rows


def analyze_uniform_b_case() -> Dict[str, float]:
    """Simulate and validate cyclotron relations in a uniform magnetic field."""
    mass = 2.0
    charge = 1.0
    b = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    e = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    x0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([2.0, 0.0, 1.0], dtype=np.float64)

    b_mag = float(np.linalg.norm(b))
    omega_true = abs(charge) * b_mag / mass
    period_true = 2.0 * np.pi / omega_true
    r_true = mass * float(np.linalg.norm(v0[:2])) / (abs(charge) * b_mag)

    dt = period_true / 400.0
    periods = 6.0
    steps = int(round(periods * period_true / dt))

    history = simulate_particle(
        mass=mass,
        charge=charge,
        electric_field=e,
        magnetic_field=b,
        x0=x0,
        v0=v0,
        dt=dt,
        steps=steps,
    )

    t = history["time"]
    pos = history["position"]
    vel = history["velocity"]
    ke = history["kinetic"]

    energy_rel_drift = float(np.max(np.abs(ke - ke[0])) / ke[0])

    phase = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    omega_num = abs(float(np.polyfit(t, phase, deg=1)[0]))
    freq_rel_err = abs(omega_num - omega_true) / omega_true

    v_perp = np.linalg.norm(vel[:, :2], axis=1)
    r_num = float(np.mean(v_perp)) / omega_true
    radius_rel_err = abs(r_num - r_true) / r_true

    vz_mean = float(np.mean(vel[:, 2]))
    vz_abs_err = abs(vz_mean - float(v0[2]))

    z_slope = float(np.polyfit(t, pos[:, 2], deg=1)[0])
    z_slope_abs_err = abs(z_slope - float(v0[2]))

    assert energy_rel_drift < 1e-10, "Kinetic energy drift is too large for pure-B Boris integration."
    assert freq_rel_err < 2e-3, "Cyclotron frequency mismatch is too large."
    assert radius_rel_err < 2e-3, "Larmor radius mismatch is too large."
    assert vz_abs_err < 1e-10 and z_slope_abs_err < 2e-3, "Parallel motion consistency check failed."

    return {
        "omega_true": omega_true,
        "omega_num": omega_num,
        "freq_rel_err": freq_rel_err,
        "r_true": r_true,
        "r_num": r_num,
        "radius_rel_err": radius_rel_err,
        "energy_rel_drift": energy_rel_drift,
        "vz_mean": vz_mean,
        "vz_abs_err": vz_abs_err,
        "z_slope": z_slope,
        "z_slope_abs_err": z_slope_abs_err,
        "dt": dt,
        "steps": float(steps),
    }


def main() -> None:
    print("=== Demo A: Lorentz force properties ===")
    rows = run_force_property_demo()
    for row in rows:
        print(
            "case={case:.0f} | |F|={f_total_norm:.6f} | |F_E|={f_e_norm:.6f} | "
            "|F_B|={f_b_norm:.6f} | dot(F_B,v)={dot_fb_v:.3e} | dot(F_B,B)={dot_fb_b:.3e}".format(
                **row
            )
        )

    print("\n=== Demo B: Uniform-B trajectory checks (Boris pusher) ===")
    report = analyze_uniform_b_case()
    print("dt={:.6e}, steps={:.0f}".format(report["dt"], report["steps"]))
    print(
        "omega_true={:.6f}, omega_num={:.6f}, rel_err={:.3e}".format(
            report["omega_true"], report["omega_num"], report["freq_rel_err"]
        )
    )
    print(
        "r_true={:.6f}, r_num={:.6f}, rel_err={:.3e}".format(
            report["r_true"], report["r_num"], report["radius_rel_err"]
        )
    )
    print(
        "energy_rel_drift={:.3e}, vz_mean={:.6f}, z_slope={:.6f}".format(
            report["energy_rel_drift"], report["vz_mean"], report["z_slope"]
        )
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
