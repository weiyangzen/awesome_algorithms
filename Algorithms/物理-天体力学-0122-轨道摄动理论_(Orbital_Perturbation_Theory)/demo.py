"""Orbital Perturbation Theory MVP (J2 secular perturbation).

This script demonstrates a classic perturbation-theory workflow:
1) Propagate an orbit with Cowell's method (numerical ODE integration).
2) Add Earth's J2 perturbing acceleration.
3) Estimate secular drift rates of RAAN and argument of perigee from simulation.
4) Compare with first-order analytical perturbation theory.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# Earth constants in km / s units.
MU_EARTH = 398600.4418  # km^3 / s^2
R_EARTH = 6378.1363  # km
J2_EARTH = 1.08262668e-3


@dataclass
class OrbitConfig:
    semi_major_axis_km: float = 7000.0
    eccentricity: float = 0.02
    inclination_deg: float = 50.0
    raan_deg: float = 40.0
    argp_deg: float = 30.0
    true_anomaly_deg: float = 10.0
    duration_hours: float = 72.0
    samples: int = 721


@dataclass
class PropagationResult:
    time_s: np.ndarray
    state: np.ndarray  # shape=(6, N)
    elements: Dict[str, np.ndarray]


def validate_config(config: OrbitConfig) -> None:
    if config.semi_major_axis_km <= 0.0:
        raise ValueError("semi_major_axis_km must be > 0.")
    if not (0.0 < config.eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 < e < 1 for this elliptical MVP.")
    if config.samples < 3:
        raise ValueError("samples must be >= 3.")
    if config.duration_hours <= 0.0:
        raise ValueError("duration_hours must be > 0.")


def deg2rad(angle_deg: float) -> float:
    return float(np.deg2rad(angle_deg))


def rad2deg(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad)


def perifocal_to_eci_rotation(raan: float, inc: float, argp: float) -> np.ndarray:
    cos_o = np.cos(raan)
    sin_o = np.sin(raan)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    cos_w = np.cos(argp)
    sin_w = np.sin(argp)

    # R = R3(raan) * R1(inc) * R3(argp)
    return np.array(
        [
            [cos_o * cos_w - sin_o * sin_w * cos_i, -cos_o * sin_w - sin_o * cos_w * cos_i, sin_o * sin_i],
            [sin_o * cos_w + cos_o * sin_w * cos_i, -sin_o * sin_w + cos_o * cos_w * cos_i, -cos_o * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=float,
    )


def coe_to_rv(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    true_anomaly: float,
    mu: float = MU_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    p = a * (1.0 - e * e)
    r_pf = np.array(
        [
            p * np.cos(true_anomaly) / (1.0 + e * np.cos(true_anomaly)),
            p * np.sin(true_anomaly) / (1.0 + e * np.cos(true_anomaly)),
            0.0,
        ],
        dtype=float,
    )
    v_pf = np.array(
        [
            -np.sqrt(mu / p) * np.sin(true_anomaly),
            np.sqrt(mu / p) * (e + np.cos(true_anomaly)),
            0.0,
        ],
        dtype=float,
    )

    rotation = perifocal_to_eci_rotation(raan=raan, inc=inc, argp=argp)
    r_eci = rotation @ r_pf
    v_eci = rotation @ v_pf
    return r_eci, v_eci


def rv_to_coe(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> Dict[str, float]:
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    energy = 0.5 * v * v - mu / r
    a = -mu / (2.0 * energy)
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    if n > 1e-12:
        raan = np.arctan2(n_vec[1], n_vec[0])
    else:
        raan = 0.0

    if n > 1e-12 and e > 1e-12:
        argp = np.arctan2(
            np.dot(np.cross(n_vec, e_vec), h_vec) / (n * h * e),
            np.dot(n_vec, e_vec) / (n * e),
        )
    else:
        argp = 0.0

    if e > 1e-12:
        nu = np.arctan2(
            np.dot(np.cross(e_vec, r_vec), h_vec) / (e * h * r),
            np.dot(e_vec, r_vec) / (e * r),
        )
    else:
        nu = 0.0

    return {
        "a": float(a),
        "e": float(e),
        "i": float(inc),
        "raan": float(np.mod(raan, 2.0 * np.pi)),
        "argp": float(np.mod(argp, 2.0 * np.pi)),
        "nu": float(np.mod(nu, 2.0 * np.pi)),
    }


def two_body_acceleration(r_vec: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / (r**3)


def j2_acceleration(
    r_vec: np.ndarray,
    mu: float = MU_EARTH,
    radius: float = R_EARTH,
    j2: float = J2_EARTH,
) -> np.ndarray:
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    z2_over_r2 = (z * z) / r2

    factor = 1.5 * j2 * mu * (radius**2) / (r**5)
    return factor * np.array(
        [
            x * (5.0 * z2_over_r2 - 1.0),
            y * (5.0 * z2_over_r2 - 1.0),
            z * (5.0 * z2_over_r2 - 3.0),
        ],
        dtype=float,
    )


def dynamics(_t: float, state: np.ndarray, with_j2: bool) -> np.ndarray:
    r_vec = state[0:3]
    v_vec = state[3:6]

    acc = two_body_acceleration(r_vec)
    if with_j2:
        acc = acc + j2_acceleration(r_vec)

    return np.hstack((v_vec, acc))


def propagate_orbit(config: OrbitConfig, with_j2: bool) -> PropagationResult:
    inc = deg2rad(config.inclination_deg)
    raan = deg2rad(config.raan_deg)
    argp = deg2rad(config.argp_deg)
    nu0 = deg2rad(config.true_anomaly_deg)

    r0, v0 = coe_to_rv(
        a=config.semi_major_axis_km,
        e=config.eccentricity,
        inc=inc,
        raan=raan,
        argp=argp,
        true_anomaly=nu0,
    )
    y0 = np.hstack((r0, v0))

    t_final = config.duration_hours * 3600.0
    t_eval = np.linspace(0.0, t_final, config.samples)

    solution = solve_ivp(
        fun=lambda t, y: dynamics(t, y, with_j2=with_j2),
        t_span=(0.0, t_final),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=120.0,
    )
    if not solution.success:
        raise RuntimeError(f"Orbit propagation failed: {solution.message}")

    elems = {"a": [], "e": [], "i": [], "raan": [], "argp": [], "nu": []}
    for k in range(solution.y.shape[1]):
        rv_elements = rv_to_coe(solution.y[0:3, k], solution.y[3:6, k])
        for key in elems:
            elems[key].append(rv_elements[key])

    elements_np = {key: np.asarray(values, dtype=float) for key, values in elems.items()}
    return PropagationResult(time_s=solution.t, state=solution.y, elements=elements_np)


def fit_secular_rate(time_s: np.ndarray, angle_rad: np.ndarray) -> Tuple[float, np.ndarray]:
    unwrapped = np.unwrap(angle_rad)
    coeff = np.polyfit(time_s, unwrapped, deg=1)
    slope = float(coeff[0])  # rad/s
    return slope, unwrapped


def analytical_j2_secular_rates(config: OrbitConfig) -> Dict[str, float]:
    a = config.semi_major_axis_km
    e = config.eccentricity
    inc = deg2rad(config.inclination_deg)

    p = a * (1.0 - e * e)
    n = np.sqrt(MU_EARTH / (a**3))
    factor = J2_EARTH * (R_EARTH / p) ** 2

    raan_dot = -1.5 * n * factor * np.cos(inc)
    argp_dot = 0.75 * n * factor * (5.0 * np.cos(inc) ** 2 - 1.0)
    return {"raan_dot": float(raan_dot), "argp_dot": float(argp_dot)}


def build_summary_table(
    time_s: np.ndarray,
    raan_kepler: np.ndarray,
    raan_j2: np.ndarray,
    argp_kepler: np.ndarray,
    argp_j2: np.ndarray,
) -> pd.DataFrame:
    # Show several representative rows to keep output compact.
    idx = np.linspace(0, len(time_s) - 1, 7).astype(int)
    table = pd.DataFrame(
        {
            "time_hr": time_s[idx] / 3600.0,
            "RAAN_kepler_deg": rad2deg(raan_kepler[idx]),
            "RAAN_j2_deg": rad2deg(raan_j2[idx]),
            "argp_kepler_deg": rad2deg(argp_kepler[idx]),
            "argp_j2_deg": rad2deg(argp_j2[idx]),
        }
    )
    return table


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v: .6f}")

    config = OrbitConfig()
    validate_config(config)

    result_kepler = propagate_orbit(config=config, with_j2=False)
    result_j2 = propagate_orbit(config=config, with_j2=True)

    raan_rate_kepler, raan_kepler_unwrap = fit_secular_rate(result_kepler.time_s, result_kepler.elements["raan"])
    argp_rate_kepler, argp_kepler_unwrap = fit_secular_rate(result_kepler.time_s, result_kepler.elements["argp"])
    raan_rate_j2, raan_j2_unwrap = fit_secular_rate(result_j2.time_s, result_j2.elements["raan"])
    argp_rate_j2, argp_j2_unwrap = fit_secular_rate(result_j2.time_s, result_j2.elements["argp"])

    theory = analytical_j2_secular_rates(config)

    metrics = {
        "raan_rate_kepler_deg_day": float(np.rad2deg(raan_rate_kepler) * 86400.0),
        "argp_rate_kepler_deg_day": float(np.rad2deg(argp_rate_kepler) * 86400.0),
        "raan_rate_j2_deg_day": float(np.rad2deg(raan_rate_j2) * 86400.0),
        "argp_rate_j2_deg_day": float(np.rad2deg(argp_rate_j2) * 86400.0),
        "raan_rate_theory_deg_day": float(np.rad2deg(theory["raan_dot"]) * 86400.0),
        "argp_rate_theory_deg_day": float(np.rad2deg(theory["argp_dot"]) * 86400.0),
        "mean_a_j2_km": float(np.mean(result_j2.elements["a"])),
        "std_a_j2_km": float(np.std(result_j2.elements["a"])),
        "mean_e_j2": float(np.mean(result_j2.elements["e"])),
        "std_e_j2": float(np.std(result_j2.elements["e"])),
    }

    raan_rel_err = abs((raan_rate_j2 - theory["raan_dot"]) / theory["raan_dot"])
    argp_rel_err = abs((argp_rate_j2 - theory["argp_dot"]) / theory["argp_dot"])
    metrics["raan_rel_err"] = float(raan_rel_err)
    metrics["argp_rel_err"] = float(argp_rel_err)

    summary = build_summary_table(
        time_s=result_j2.time_s,
        raan_kepler=raan_kepler_unwrap,
        raan_j2=raan_j2_unwrap,
        argp_kepler=argp_kepler_unwrap,
        argp_j2=argp_j2_unwrap,
    )

    print("=== Orbital Perturbation Theory MVP (Earth J2) ===")
    print(
        "Initial COE: "
        f"a={config.semi_major_axis_km:.1f} km, e={config.eccentricity:.4f}, "
        f"i={config.inclination_deg:.2f} deg, RAAN={config.raan_deg:.2f} deg, "
        f"argp={config.argp_deg:.2f} deg, nu={config.true_anomaly_deg:.2f} deg"
    )
    print(f"Propagation span: {config.duration_hours:.1f} hours, samples={config.samples}")
    print("\nRepresentative secular trend samples:")
    print(summary.to_string(index=False))

    print("\nRate comparison (deg/day):")
    print(f"  Kepler baseline   : RAAN={metrics['raan_rate_kepler_deg_day']:.6f}, argp={metrics['argp_rate_kepler_deg_day']:.6f}")
    print(f"  Numerical J2      : RAAN={metrics['raan_rate_j2_deg_day']:.6f}, argp={metrics['argp_rate_j2_deg_day']:.6f}")
    print(
        f"  Analytical J2     : RAAN={metrics['raan_rate_theory_deg_day']:.6f}, "
        f"argp={metrics['argp_rate_theory_deg_day']:.6f}"
    )
    print(f"  Relative errors   : RAAN={metrics['raan_rel_err']:.4%}, argp={metrics['argp_rel_err']:.4%}")

    print("\nOsculating element stability under J2 (short-period oscillations expected):")
    print(f"  mean(a)={metrics['mean_a_j2_km']:.6f} km, std(a)={metrics['std_a_j2_km']:.6e} km")
    print(f"  mean(e)={metrics['mean_e_j2']:.8f}, std(e)={metrics['std_e_j2']:.6e}")

    # Acceptance checks for a deterministic CI-friendly MVP.
    # Kepler-only baseline should have near-zero secular precession.
    if abs(metrics["raan_rate_kepler_deg_day"]) > 0.03 or abs(metrics["argp_rate_kepler_deg_day"]) > 0.03:
        raise RuntimeError("Kepler baseline precession is unexpectedly large.")

    # First-order perturbation theory should be close to numerical trend.
    if metrics["raan_rel_err"] > 0.08 or metrics["argp_rel_err"] > 0.12:
        raise RuntimeError("Numerical-vs-theory J2 rate mismatch exceeds tolerance.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
