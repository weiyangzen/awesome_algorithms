"""Minimal runnable MVP for Tidal Theory in celestial mechanics.

This script implements a compact equilibrium-tide model driven by
Moon/Sun tidal potentials, then performs:
1) parameter inversion (estimate effective Love-number scale), and
2) harmonic decomposition (M2/S2 constituents).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


G = 6.67430e-11
EARTH_RADIUS_M = 6_371_000.0
GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class TideBody:
    name: str
    mass_kg: float
    distance_m: float
    declination_amp_deg: float
    declination_period_h: float
    declination_phase_rad: float
    hour_angle_period_h: float
    hour_angle_phase_rad: float


def legendre_p2(x: np.ndarray) -> np.ndarray:
    """Second-order Legendre polynomial P2(x)."""
    return 0.5 * (3.0 * x * x - 1.0)


def body_declination_rad(t_s: np.ndarray, body: TideBody) -> np.ndarray:
    amp = np.deg2rad(body.declination_amp_deg)
    omega = 2.0 * np.pi / (body.declination_period_h * 3600.0)
    return amp * np.sin(omega * t_s + body.declination_phase_rad)


def body_hour_angle_rad(t_s: np.ndarray, body: TideBody) -> np.ndarray:
    omega = 2.0 * np.pi / (body.hour_angle_period_h * 3600.0)
    return omega * t_s + body.hour_angle_phase_rad


def tidal_potential_body(t_s: np.ndarray, latitude_rad: float, body: TideBody) -> np.ndarray:
    """Compute degree-2 body tide potential at Earth's surface."""
    delta = body_declination_rad(t_s, body)
    hour_angle = body_hour_angle_rad(t_s, body)

    cos_psi = (
        np.sin(latitude_rad) * np.sin(delta)
        + np.cos(latitude_rad) * np.cos(delta) * np.cos(hour_angle)
    )

    scale = G * body.mass_kg / (body.distance_m**3) * (EARTH_RADIUS_M**2)
    return scale * legendre_p2(cos_psi)


def equilibrium_tide_height_m(tidal_potential: np.ndarray, love_k2: float) -> np.ndarray:
    """Convert tidal potential (m^2/s^2) to equilibrium tide height (m)."""
    return (1.0 + love_k2) * tidal_potential / GRAVITY_M_S2


def estimate_linear_scale(u: np.ndarray, eta_obs: np.ndarray) -> tuple[float, float]:
    """Estimate eta = alpha * u by nonlinear least squares (1 parameter)."""

    def model(x: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * x

    popt, _ = curve_fit(model, u, eta_obs, p0=np.array([1.0 / GRAVITY_M_S2]))
    alpha = float(popt[0])
    k2_est = alpha * GRAVITY_M_S2 - 1.0
    return alpha, k2_est


def harmonic_decompose(
    t_s: np.ndarray, y: np.ndarray, omegas: list[float], names: list[str]
) -> pd.DataFrame:
    """Fit y ~= c0 + sum_i (a_i sin(w_i t) + b_i cos(w_i t))."""
    cols = [np.ones_like(t_s)]
    for omega in omegas:
        cols.append(np.sin(omega * t_s))
        cols.append(np.cos(omega * t_s))

    x = np.column_stack(cols)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x, y)
    coef = reg.coef_

    rows: list[dict[str, float | str]] = []
    for i, (name, omega) in enumerate(zip(names, omegas)):
        a = float(coef[1 + 2 * i])
        b = float(coef[2 + 2 * i])
        amplitude = float(np.hypot(a, b))
        phase = float(np.arctan2(b, a))
        rows.append(
            {
                "constituent": name,
                "omega_rad_s": omega,
                "sin_coeff": a,
                "cos_coeff": b,
                "amplitude_m": amplitude,
                "phase_rad": phase,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(20260407)

    latitude_deg = 30.0
    latitude_rad = np.deg2rad(latitude_deg)

    moon = TideBody(
        name="Moon",
        mass_kg=7.34767309e22,
        distance_m=384_400_000.0,
        declination_amp_deg=28.6,
        declination_period_h=27.321661 * 24.0,
        declination_phase_rad=0.8,
        hour_angle_period_h=24.8412,
        hour_angle_phase_rad=0.2,
    )
    sun = TideBody(
        name="Sun",
        mass_kg=1.9885e30,
        distance_m=149_597_870_700.0,
        declination_amp_deg=23.44,
        declination_period_h=365.2422 * 24.0,
        declination_phase_rad=-0.5,
        hour_angle_period_h=24.0,
        hour_angle_phase_rad=-0.1,
    )

    dt_s = 600.0
    sim_hours = 72.0
    t_s = np.arange(0.0, sim_hours * 3600.0 + dt_s, dt_s)

    u_moon = tidal_potential_body(t_s, latitude_rad, moon)
    u_sun = tidal_potential_body(t_s, latitude_rad, sun)
    u_total = u_moon + u_sun

    k2_true = 0.28
    eta_clean = equilibrium_tide_height_m(u_total, love_k2=k2_true)

    eta_obs = eta_clean + rng.normal(0.0, 0.025, size=eta_clean.shape)

    alpha_est, k2_est = estimate_linear_scale(u_total, eta_obs)
    eta_fit = alpha_est * u_total

    residual = eta_obs - eta_fit
    rmse_m = float(np.sqrt(np.mean(residual**2)))
    mae_m = float(np.mean(np.abs(residual)))

    # Classical semidiurnal constituents for a compact decomposition demo.
    omega_m2 = 2.0 * np.pi / (12.4206 * 3600.0)
    omega_s2 = 2.0 * np.pi / (12.0 * 3600.0)
    harmonic_df = harmonic_decompose(
        t_s=t_s,
        y=eta_fit,
        omegas=[omega_m2, omega_s2],
        names=["M2", "S2"],
    )

    amp_m2 = float(harmonic_df.loc[harmonic_df["constituent"] == "M2", "amplitude_m"].iloc[0])
    amp_s2 = float(harmonic_df.loc[harmonic_df["constituent"] == "S2", "amplitude_m"].iloc[0])
    moon_to_sun_std_ratio = float(np.std(u_moon) / np.std(u_sun))

    preview = pd.DataFrame(
        {
            "time_h": t_s / 3600.0,
            "u_total_m2_s2": u_total,
            "eta_obs_m": eta_obs,
            "eta_fit_m": eta_fit,
            "residual_m": residual,
        }
    ).head(8)

    print("=== Tidal Theory MVP (Equilibrium Tide + Inversion + Harmonics) ===")
    print(f"latitude_deg={latitude_deg:.1f}, duration_h={sim_hours:.1f}, dt_s={dt_s:.0f}")
    print(f"true_k2={k2_true:.4f}, estimated_k2={k2_est:.4f}")
    print(f"fit_rmse_m={rmse_m:.4f}, fit_mae_m={mae_m:.4f}")
    print(f"M2_amplitude_m={amp_m2:.4f}, S2_amplitude_m={amp_s2:.4f}")
    print(f"moon_to_sun_potential_std_ratio={moon_to_sun_std_ratio:.3f}")
    print("\nHarmonic decomposition:")
    print(harmonic_df.to_string(index=False))
    print("\nSample time-series rows:")
    print(preview.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    assert rmse_m < 0.05, f"RMSE too large: {rmse_m:.4f} m"
    assert abs(k2_est - k2_true) < 0.06, f"k2 estimate drift too large: {k2_est:.4f}"
    assert moon_to_sun_std_ratio > 1.5, "Moon tidal forcing should dominate Sun in this setup"
    assert amp_m2 > 0.01 and amp_s2 > 0.01, "Harmonic amplitudes should be physically non-trivial"


if __name__ == "__main__":
    main()
