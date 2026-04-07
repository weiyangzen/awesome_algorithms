"""Minimal runnable MVP for Clapeyron Equation.

We model a liquid-vapor coexistence line with the Clapeyron relation:
    dP/dT = L(T) / (T * (v_g - v_l))
where:
- v_g is approximated by ideal vapor specific volume R*T/P,
- v_l is treated as weakly compressible constant,
- L(T) is a simple linear latent-heat model.

The script integrates P(T) from one anchor point using an explicit RK4 scheme,
then compares against the simplified Clausius-Clapeyron approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClapeyronConfig:
    """Configuration for a transparent liquid-vapor coexistence MVP."""

    t_ref_k: float = 373.15
    p_ref_pa: float = 101_325.0
    t_min_k: float = 300.0
    t_max_k: float = 420.0
    n_points: int = 49  # use odd number so the reference point sits at center

    # Water-like constants for a didactic MVP.
    r_specific_vapor: float = 461.5  # J/(kg*K)
    v_liquid_m3_per_kg: float = 1.0e-3

    # Simple latent heat model: L(T) = L_ref + slope*(T - T_ref)
    latent_heat_ref_j_per_kg: float = 2.257e6
    latent_heat_slope_j_per_kgk: float = -2400.0
    latent_heat_floor_j_per_kg: float = 5.0e5


def latent_heat_j_per_kg(temperature_k: float, cfg: ClapeyronConfig) -> float:
    """Temperature-dependent latent heat with a positive floor."""
    value = cfg.latent_heat_ref_j_per_kg + cfg.latent_heat_slope_j_per_kgk * (temperature_k - cfg.t_ref_k)
    return float(max(cfg.latent_heat_floor_j_per_kg, value))


def vapor_specific_volume_m3_per_kg(temperature_k: float, pressure_pa: float, cfg: ClapeyronConfig) -> float:
    """Ideal-vapor specific volume approximation v_g = R*T/P."""
    if pressure_pa <= 0.0:
        raise ValueError(f"Pressure must be positive, got {pressure_pa}")
    return float(cfg.r_specific_vapor * temperature_k / pressure_pa)


def clapeyron_rhs_pa_per_k(temperature_k: float, pressure_pa: float, cfg: ClapeyronConfig) -> float:
    """Right-hand side of dP/dT from the Clapeyron equation."""
    if temperature_k <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature_k}")
    if pressure_pa <= 0.0:
        raise ValueError(f"Pressure must be positive, got {pressure_pa}")

    v_g = vapor_specific_volume_m3_per_kg(temperature_k, pressure_pa, cfg)
    delta_v = v_g - cfg.v_liquid_m3_per_kg
    if delta_v <= 1e-12:
        raise ValueError(
            "Invalid state: vapor and liquid specific volumes are too close "
            f"(delta_v={delta_v})"
        )

    latent_heat = latent_heat_j_per_kg(temperature_k, cfg)
    return float(latent_heat / (temperature_k * delta_v))


def rk4_step_pressure(
    temperature_k: float,
    pressure_pa: float,
    dt_k: float,
    cfg: ClapeyronConfig,
) -> float:
    """One explicit RK4 step for dP/dT."""
    k1 = clapeyron_rhs_pa_per_k(temperature_k, pressure_pa, cfg)
    k2 = clapeyron_rhs_pa_per_k(
        temperature_k + 0.5 * dt_k,
        pressure_pa + 0.5 * dt_k * k1,
        cfg,
    )
    k3 = clapeyron_rhs_pa_per_k(
        temperature_k + 0.5 * dt_k,
        pressure_pa + 0.5 * dt_k * k2,
        cfg,
    )
    k4 = clapeyron_rhs_pa_per_k(
        temperature_k + dt_k,
        pressure_pa + dt_k * k3,
        cfg,
    )

    next_pressure = pressure_pa + (dt_k / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if next_pressure <= 0.0:
        raise ValueError(f"Non-physical negative pressure encountered: {next_pressure}")
    return float(next_pressure)


def integrate_monotonic_temperature_grid(
    temperatures_k: np.ndarray,
    pressure_start_pa: float,
    cfg: ClapeyronConfig,
) -> np.ndarray:
    """Integrate P(T) along a monotonic temperature array."""
    if len(temperatures_k) < 2:
        raise ValueError("Need at least two temperature points for integration.")

    pressures_pa = np.empty_like(temperatures_k, dtype=float)
    pressures_pa[0] = pressure_start_pa

    for i in range(len(temperatures_k) - 1):
        t_now = float(temperatures_k[i])
        dt = float(temperatures_k[i + 1] - temperatures_k[i])
        pressures_pa[i + 1] = rk4_step_pressure(t_now, float(pressures_pa[i]), dt, cfg)

    return pressures_pa


def build_temperature_grid(cfg: ClapeyronConfig) -> np.ndarray:
    """Build an ascending grid that contains exactly the reference temperature."""
    if cfg.n_points < 5:
        raise ValueError("n_points must be >= 5")
    if cfg.n_points % 2 == 0:
        raise ValueError("n_points must be odd so reference temperature is centered")
    if not (cfg.t_min_k < cfg.t_ref_k < cfg.t_max_k):
        raise ValueError("Require t_min < t_ref < t_max")

    half = cfg.n_points // 2
    lower = np.linspace(cfg.t_min_k, cfg.t_ref_k, half + 1)
    upper = np.linspace(cfg.t_ref_k, cfg.t_max_k, half + 1)
    return np.concatenate([lower[:-1], upper])


def integrate_clapeyron_curve(cfg: ClapeyronConfig) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the coexistence pressure curve on both sides of reference point."""
    temperatures = build_temperature_grid(cfg)
    half = cfg.n_points // 2

    lower = temperatures[: half + 1]   # ascending t_min -> t_ref
    upper = temperatures[half:]        # ascending t_ref -> t_max

    # Upward branch: straightforward integration from reference.
    p_upper = integrate_monotonic_temperature_grid(upper, cfg.p_ref_pa, cfg)

    # Downward branch: integrate on descending grid from reference, then reverse.
    lower_desc = lower[::-1]
    p_lower_desc = integrate_monotonic_temperature_grid(lower_desc, cfg.p_ref_pa, cfg)
    p_lower = p_lower_desc[::-1]

    pressures = np.concatenate([p_lower[:-1], p_upper])
    return temperatures, pressures


def clausius_clapeyron_pressure_pa(temperatures_k: np.ndarray, cfg: ClapeyronConfig) -> np.ndarray:
    """Simplified Clausius-Clapeyron curve (constant L, ideal vapor, v_l << v_g)."""
    exponent = -cfg.latent_heat_ref_j_per_kg / cfg.r_specific_vapor * (1.0 / temperatures_k - 1.0 / cfg.t_ref_k)
    return cfg.p_ref_pa * np.exp(exponent)


def run_mvp(cfg: ClapeyronConfig) -> pd.DataFrame:
    temperatures_k, pressure_clapeyron_pa = integrate_clapeyron_curve(cfg)
    pressure_clausius_pa = clausius_clapeyron_pressure_pa(temperatures_k, cfg)

    dpressure_dtemp = np.gradient(pressure_clapeyron_pa, temperatures_k)
    v_vapor = cfg.r_specific_vapor * temperatures_k / pressure_clapeyron_pa
    delta_v = v_vapor - cfg.v_liquid_m3_per_kg

    latent_heat_backcalc = temperatures_k * delta_v * dpressure_dtemp
    latent_heat_model = np.array([latent_heat_j_per_kg(float(t), cfg) for t in temperatures_k], dtype=float)

    relative_error_cc = (pressure_clausius_pa - pressure_clapeyron_pa) / pressure_clapeyron_pa

    return pd.DataFrame(
        {
            "temperature_k": temperatures_k,
            "p_clapeyron_pa": pressure_clapeyron_pa,
            "p_clausius_pa": pressure_clausius_pa,
            "dp_dT_pa_per_k": dpressure_dtemp,
            "v_vapor_m3_per_kg": v_vapor,
            "v_liquid_m3_per_kg": cfg.v_liquid_m3_per_kg,
            "latent_heat_model_j_per_kg": latent_heat_model,
            "latent_heat_backcalc_j_per_kg": latent_heat_backcalc,
            "relative_error_cc_vs_clapeyron": relative_error_cc,
        }
    )


def main() -> None:
    cfg = ClapeyronConfig()
    df = run_mvp(cfg)

    p_values = df["p_clapeyron_pa"].to_numpy(dtype=float)

    p_min = float(p_values[0])
    p_ref = float(df.loc[np.isclose(df["temperature_k"], cfg.t_ref_k), "p_clapeyron_pa"].iloc[0])
    p_max = float(p_values[-1])

    rel_err_near_ref = np.abs(
        df.loc[np.abs(df["temperature_k"] - cfg.t_ref_k) <= 10.0, "relative_error_cc_vs_clapeyron"].to_numpy(dtype=float)
    )
    latent_rel_err = np.abs(
        (
            df["latent_heat_backcalc_j_per_kg"].to_numpy(dtype=float)
            - df["latent_heat_model_j_per_kg"].to_numpy(dtype=float)
        )
        / df["latent_heat_model_j_per_kg"].to_numpy(dtype=float)
    )

    print("=== Clapeyron Equation MVP ===")
    print(
        "config:",
        f"T_ref={cfg.t_ref_k} K, P_ref={cfg.p_ref_pa} Pa, ",
        f"T_range=[{cfg.t_min_k}, {cfg.t_max_k}] K, points={cfg.n_points}",
    )
    print(
        "model:",
        f"R_v={cfg.r_specific_vapor} J/(kg*K), v_l={cfg.v_liquid_m3_per_kg} m^3/kg, ",
        f"L_ref={cfg.latent_heat_ref_j_per_kg:.1f} J/kg",
    )
    print()

    with pd.option_context("display.width", 220, "display.precision", 6):
        print(df.to_string(index=False))

    print()
    print(f"Pressure at {cfg.t_min_k:.2f} K: {p_min:.3f} Pa")
    print(f"Pressure at reference {cfg.t_ref_k:.2f} K: {p_ref:.3f} Pa")
    print(f"Pressure at {cfg.t_max_k:.2f} K: {p_max:.3f} Pa")
    print(f"Max |relative error| of Clausius approx within +/-10 K of reference: {float(np.max(rel_err_near_ref)):.4%}")
    print(f"Median latent-heat back-calc relative error (interior points): {float(np.median(latent_rel_err[1:-1])):.4%}")

    # Deterministic checks for expected thermodynamic behavior.
    assert np.all(np.diff(p_values) > 0.0), "Saturation pressure should increase with temperature."
    assert abs(p_ref - cfg.p_ref_pa) <= 1e-6, "Reference anchor point drifted unexpectedly."
    assert 2_000.0 < p_min < 8_000.0, "Low-temperature pressure is outside expected physical range."
    assert np.max(rel_err_near_ref) < 0.08, "Clausius approximation should be accurate near reference point."
    assert np.median(latent_rel_err[1:-1]) < 0.05, "Back-calculated latent heat should match model in interior points."

    print("All checks passed.")


if __name__ == "__main__":
    main()
