"""Minimal runnable MVP for the pinch effect (0D radial Z-pinch model)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class PinchConfig:
    """Configuration for a damped cylindrical pinch toy model."""

    current_a: float = 3.0e4
    initial_radius_m: float = 2.0e-2
    line_mass_density_kg_m: float = 2.0e-4
    initial_thermal_pressure_pa: float = 2.5e4
    adiabatic_index: float = 5.0 / 3.0
    damping_rate_s: float = 4.0e4
    time_step_s: float = 5.0e-8
    t_end_s: float = 1.5e-4
    radius_floor_m: float = 2.0e-3
    max_steps: int = 1_000_000


def magnetic_field_theta(radius_m: float, current_a: float) -> float:
    return MU0 * current_a / (2.0 * math.pi * radius_m)


def magnetic_pressure(radius_m: float, current_a: float) -> float:
    b_theta = magnetic_field_theta(radius_m, current_a)
    return 0.5 * b_theta * b_theta / MU0


def thermal_pressure(radius_m: float, cfg: PinchConfig) -> float:
    return cfg.initial_thermal_pressure_pa * (cfg.initial_radius_m / radius_m) ** (2.0 * cfg.adiabatic_index)


def radial_acceleration(radius_m: float, radial_velocity_m_s: float, cfg: PinchConfig) -> float:
    p_th = thermal_pressure(radius_m, cfg)
    p_b = magnetic_pressure(radius_m, cfg.current_a)
    pressure_force_per_length = (p_th - p_b) * 2.0 * math.pi * radius_m
    damping_term = cfg.damping_rate_s * radial_velocity_m_s
    return pressure_force_per_length / cfg.line_mass_density_kg_m - damping_term


def equilibrium_radius(cfg: PinchConfig) -> float:
    magnetic_factor = MU0 * cfg.current_a * cfg.current_a / (8.0 * math.pi * math.pi)
    numerator = cfg.initial_thermal_pressure_pa * cfg.initial_radius_m ** (2.0 * cfg.adiabatic_index)
    exponent = 1.0 / (2.0 * cfg.adiabatic_index - 2.0)
    return (numerator / magnetic_factor) ** exponent


def validate_config(cfg: PinchConfig) -> None:
    if cfg.current_a <= 0.0:
        raise ValueError("current_a must be positive.")
    if cfg.initial_radius_m <= 0.0:
        raise ValueError("initial_radius_m must be positive.")
    if cfg.line_mass_density_kg_m <= 0.0:
        raise ValueError("line_mass_density_kg_m must be positive.")
    if cfg.initial_thermal_pressure_pa <= 0.0:
        raise ValueError("initial_thermal_pressure_pa must be positive.")
    if cfg.adiabatic_index <= 1.0:
        raise ValueError("adiabatic_index must be > 1.0 for this model.")
    if cfg.damping_rate_s < 0.0:
        raise ValueError("damping_rate_s must be >= 0.")
    if cfg.time_step_s <= 0.0 or cfg.t_end_s <= 0.0:
        raise ValueError("time_step_s and t_end_s must be positive.")
    if cfg.radius_floor_m <= 0.0 or cfg.radius_floor_m >= cfg.initial_radius_m:
        raise ValueError("radius_floor_m must be in (0, initial_radius_m).")


def simulate_pinch(cfg: PinchConfig) -> dict[str, float | int | pd.DataFrame]:
    validate_config(cfg)

    n_steps = int(math.ceil(cfg.t_end_s / cfg.time_step_s))
    if n_steps > cfg.max_steps:
        raise RuntimeError(f"required steps {n_steps} exceed max_steps {cfg.max_steps}")

    dt = cfg.t_end_s / n_steps

    times = np.empty(n_steps + 1, dtype=float)
    radius = np.empty(n_steps + 1, dtype=float)
    velocity = np.empty(n_steps + 1, dtype=float)
    p_th_arr = np.empty(n_steps + 1, dtype=float)
    p_b_arr = np.empty(n_steps + 1, dtype=float)
    net_p_arr = np.empty(n_steps + 1, dtype=float)
    accel = np.empty(n_steps + 1, dtype=float)
    b_theta_arr = np.empty(n_steps + 1, dtype=float)

    times[0] = 0.0
    radius[0] = cfg.initial_radius_m
    velocity[0] = 0.0
    p_th_arr[0] = thermal_pressure(radius[0], cfg)
    p_b_arr[0] = magnetic_pressure(radius[0], cfg.current_a)
    net_p_arr[0] = p_th_arr[0] - p_b_arr[0]
    accel[0] = radial_acceleration(radius[0], velocity[0], cfg)
    b_theta_arr[0] = magnetic_field_theta(radius[0], cfg.current_a)

    for i in range(1, n_steps + 1):
        a_now = radial_acceleration(radius[i - 1], velocity[i - 1], cfg)

        # Semi-implicit Euler: update velocity first, then position.
        v_new = velocity[i - 1] + a_now * dt
        r_new = radius[i - 1] + v_new * dt

        if r_new <= cfg.radius_floor_m:
            raise RuntimeError(
                f"radius dropped below floor at step {i}: r={r_new:.6e} <= {cfg.radius_floor_m:.6e}"
            )

        times[i] = i * dt
        radius[i] = r_new
        velocity[i] = v_new

        p_th_arr[i] = thermal_pressure(r_new, cfg)
        p_b_arr[i] = magnetic_pressure(r_new, cfg.current_a)
        net_p_arr[i] = p_th_arr[i] - p_b_arr[i]
        accel[i] = radial_acceleration(r_new, v_new, cfg)
        b_theta_arr[i] = magnetic_field_theta(r_new, cfg.current_a)

    history = pd.DataFrame(
        {
            "time_s": times,
            "radius_m": radius,
            "radial_velocity_m_s": velocity,
            "thermal_pressure_pa": p_th_arr,
            "magnetic_pressure_pa": p_b_arr,
            "net_pressure_pa": net_p_arr,
            "acceleration_m_s2": accel,
            "b_theta_t": b_theta_arr,
        }
    )

    return {
        "history": history,
        "n_steps": n_steps,
        "dt": dt,
        "equilibrium_radius_m": equilibrium_radius(cfg),
    }


def main() -> None:
    cfg = PinchConfig()
    result = simulate_pinch(cfg)
    history: pd.DataFrame = result["history"]

    r0 = float(history["radius_m"].iloc[0])
    r_final = float(history["radius_m"].iloc[-1])
    r_eq = float(result["equilibrium_radius_m"])

    p_th_final = float(history["thermal_pressure_pa"].iloc[-1])
    p_b_final = float(history["magnetic_pressure_pa"].iloc[-1])
    net_p_final = float(history["net_pressure_pa"].iloc[-1])

    compression_ratio = r_final / r0
    eq_radius_rel_err = abs(r_final - r_eq) / max(abs(r_eq), 1e-14)
    force_balance_rel = abs(net_p_final) / max(abs(p_b_final), 1e-14)
    finite_ok = bool(np.isfinite(history.to_numpy()).all())

    max_b_theta = float(np.max(np.abs(history["b_theta_t"])))
    max_speed = float(np.max(np.abs(history["radial_velocity_m_s"])))

    checks = {
        "finite_output": finite_ok,
        "radius_positive": bool((history["radius_m"] > cfg.radius_floor_m).all()),
        "pinch_compression": compression_ratio < 0.98,
        "equilibrium_radius_error<0.10": eq_radius_rel_err < 0.10,
        "force_balance_residual<0.12": force_balance_rel < 0.12,
    }
    passed = all(checks.values())

    print("=== Pinch Effect MVP (Damped 0D Z-pinch Radius Model) ===")
    print(
        "Params: "
        f"I={cfg.current_a:.3e} A, R0={cfg.initial_radius_m:.3e} m, "
        f"m_l={cfg.line_mass_density_kg_m:.3e} kg/m, p0={cfg.initial_thermal_pressure_pa:.3e} Pa"
    )
    print(
        "Numerics: "
        f"gamma={cfg.adiabatic_index:.5f}, nu={cfg.damping_rate_s:.3e} 1/s, "
        f"dt={result['dt']:.3e} s, steps={result['n_steps']}, t_end={cfg.t_end_s:.3e} s"
    )

    print(f"R_eq(theory) = {r_eq:.6e} m")
    print(f"R_final(num) = {r_final:.6e} m")
    print(f"compression_ratio = {compression_ratio:.6f}")
    print(f"equilibrium_radius_relative_error = {eq_radius_rel_err:.6e}")
    print(f"final thermal/magnetic pressure = {p_th_final:.6e} / {p_b_final:.6e} Pa")
    print(f"final net pressure = {net_p_final:.6e} Pa")
    print(f"force_balance_relative_residual = {force_balance_rel:.6e}")
    print(f"max |B_theta| = {max_b_theta:.6e} T")
    print(f"max |v_r| = {max_speed:.6e} m/s")

    print("\nChecks:")
    for name, ok in checks.items():
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")

    sample_count = min(10, len(history))
    sample_idx = np.linspace(0, len(history) - 1, sample_count, dtype=int)
    sample_df = history.iloc[sample_idx][
        [
            "time_s",
            "radius_m",
            "radial_velocity_m_s",
            "thermal_pressure_pa",
            "magnetic_pressure_pa",
            "net_pressure_pa",
        ]
    ].copy()

    print("\nSampled trajectory:")
    print(sample_df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
