"""Foucault pendulum MVP using a linear small-angle horizontal model.

State in local EN (east-north) coordinates:
    s = [x, y, vx, vy]

Dynamics:
    x_ddot + gamma * x_dot + w0^2 * x - 2 * sigma * y_dot = 0
    y_ddot + gamma * y_dot + w0^2 * y + 2 * sigma * x_dot = 0

where:
    w0 = sqrt(g / L)
    sigma = Omega_earth * sin(latitude)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

EARTH_ANGULAR_SPEED = 7.2921159e-5  # rad/s


@dataclass(frozen=True)
class FoucaultConfig:
    """Physical and numerical parameters for the MVP."""

    latitude_deg: float = 45.0
    length_m: float = 20.0
    gravity_mps2: float = 9.81
    damping_per_s: float = 0.0
    x0_m: float = 0.18
    y0_m: float = 0.0
    vx0_mps: float = 0.0
    vy0_mps: float = 0.0
    t_start_s: float = 0.0
    t_end_s: float = 6.0 * 3600.0
    num_points: int = 43200
    rtol: float = 1e-8
    atol: float = 1e-10


def natural_frequency(cfg: FoucaultConfig) -> float:
    """Return small-angle pendulum natural frequency."""

    return math.sqrt(cfg.gravity_mps2 / cfg.length_m)


def coriolis_coupling_rate(cfg: FoucaultConfig) -> float:
    """Return sigma = Omega * sin(latitude)."""

    lat_rad = math.radians(cfg.latitude_deg)
    return EARTH_ANGULAR_SPEED * math.sin(lat_rad)


def foucault_rhs(_t: float, state: np.ndarray, cfg: FoucaultConfig) -> np.ndarray:
    """RHS of linearized Foucault pendulum in local EN plane."""

    x, y, vx, vy = state
    w0 = natural_frequency(cfg)
    sigma = coriolis_coupling_rate(cfg)
    gamma = cfg.damping_per_s

    ax = -(w0 * w0) * x + 2.0 * sigma * vy - gamma * vx
    ay = -(w0 * w0) * y - 2.0 * sigma * vx - gamma * vy
    return np.array([vx, vy, ax, ay], dtype=float)


def mechanical_energy_like(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w0: float) -> np.ndarray:
    """Return an energy-like invariant for the undamped linear model."""

    kinetic = 0.5 * (vx * vx + vy * vy)
    potential = 0.5 * (w0 * w0) * (x * x + y * y)
    return kinetic + potential


def _pick_local_maxima(values: np.ndarray, min_gap: int) -> np.ndarray:
    """Pick local maxima with a minimum index gap."""

    if values.size < 3:
        return np.array([], dtype=int)

    raw = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    if raw.size == 0:
        return raw

    chosen = [int(raw[0])]
    for idx in raw[1:]:
        if int(idx) - chosen[-1] >= min_gap:
            chosen.append(int(idx))
    return np.array(chosen, dtype=int)


def estimate_precession_rate(t: np.ndarray, x: np.ndarray, y: np.ndarray, w0: float) -> tuple[float, int]:
    """Estimate precession from turning-point orientation drift.

    The turning-point angle is taken modulo pi (same oscillation line).
    We unwrap 2*angle and fit a linear slope.
    """

    if t.size < 10:
        return float("nan"), 0
    dt = float(t[1] - t[0])
    if not math.isfinite(dt) or dt <= 0.0:
        return float("nan"), 0

    r = np.hypot(x, y)
    half_period = math.pi / w0
    min_gap = max(1, int(0.35 * half_period / dt))
    peaks = _pick_local_maxima(r, min_gap=min_gap)
    if peaks.size < 20:
        return float("nan"), int(peaks.size)

    angles = np.arctan2(y[peaks], x[peaks])
    doubled_unwrapped = np.unwrap(2.0 * angles)
    line_orientation = 0.5 * doubled_unwrapped
    slope, _intercept = np.polyfit(t[peaks], line_orientation, deg=1)
    return float(slope), int(peaks.size)


def simulate(cfg: FoucaultConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Integrate and return trajectory + summary metrics."""

    if not (-90.0 <= cfg.latitude_deg <= 90.0):
        raise ValueError("latitude_deg must be in [-90, 90].")
    if cfg.length_m <= 0.0 or cfg.gravity_mps2 <= 0.0:
        raise ValueError("length_m and gravity_mps2 must be positive.")
    if cfg.num_points < 200:
        raise ValueError("num_points must be >= 200.")

    t_eval = np.linspace(cfg.t_start_s, cfg.t_end_s, cfg.num_points)
    y0 = np.array([cfg.x0_m, cfg.y0_m, cfg.vx0_mps, cfg.vy0_mps], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: foucault_rhs(t, y, cfg),
        t_span=(cfg.t_start_s, cfg.t_end_s),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    x, y, vx, vy = sol.y
    w0 = natural_frequency(cfg)
    sigma = coriolis_coupling_rate(cfg)

    energy = mechanical_energy_like(x, y, vx, vy, w0=w0)
    e0 = float(energy[0])
    max_rel_energy_drift = float(np.max(np.abs((energy - e0) / max(1e-12, abs(e0)))))

    rate_est, used_peaks = estimate_precession_rate(sol.t, x, y, w0=w0)

    abs_sigma = abs(sigma)
    abs_est = abs(rate_est)
    rel_error = float(abs(abs_est - abs_sigma) / max(1e-15, abs_sigma))
    period_theory_h = float(2.0 * math.pi / max(1e-15, abs_sigma) / 3600.0)
    period_est_h = float(2.0 * math.pi / max(1e-15, abs_est) / 3600.0)

    df = pd.DataFrame(
        {
            "t_s": sol.t,
            "x_east_m": x,
            "y_north_m": y,
            "vx_east_mps": vx,
            "vy_north_mps": vy,
            "radius_m": np.hypot(x, y),
            "energy_like": energy,
        }
    )

    summary = {
        "natural_period_s": float(2.0 * math.pi / w0),
        "latitude_deg": float(cfg.latitude_deg),
        "sigma_theory_rad_s": float(sigma),
        "precession_rate_est_rad_s": float(rate_est),
        "precession_period_theory_h": period_theory_h,
        "precession_period_est_h": period_est_h,
        "precession_rate_abs_rel_error": rel_error,
        "max_rel_energy_drift": max_rel_energy_drift,
        "turning_points_used": float(used_peaks),
    }
    return df, summary


def main() -> None:
    cfg = FoucaultConfig()
    traj, summary = simulate(cfg)

    print("Foucault Pendulum MVP (linear small-angle EN model)")
    print(
        f"latitude={cfg.latitude_deg:.2f} deg, length={cfg.length_m:.2f} m, "
        f"g={cfg.gravity_mps2:.4f} m/s^2, damping={cfg.damping_per_s:.3e} 1/s"
    )
    print(
        f"time=[{cfg.t_start_s:.1f}, {cfg.t_end_s:.1f}] s, num_points={cfg.num_points}, "
        f"rtol={cfg.rtol:.1e}, atol={cfg.atol:.1e}"
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "natural_period_s", "value": f"{summary['natural_period_s']:.6f}"},
            {"metric": "sigma_theory_rad_s", "value": f"{summary['sigma_theory_rad_s']:.8e}"},
            {"metric": "precession_rate_est_rad_s", "value": f"{summary['precession_rate_est_rad_s']:.8e}"},
            {"metric": "precession_period_theory_h", "value": f"{summary['precession_period_theory_h']:.4f}"},
            {"metric": "precession_period_est_h", "value": f"{summary['precession_period_est_h']:.4f}"},
            {
                "metric": "precession_rate_abs_rel_error",
                "value": f"{summary['precession_rate_abs_rel_error']:.3e}",
            },
            {"metric": "max_rel_energy_drift", "value": f"{summary['max_rel_energy_drift']:.3e}"},
            {"metric": "turning_points_used", "value": f"{int(summary['turning_points_used'])}"},
        ]
    )
    print("\nsummary:")
    print(summary_df.to_string(index=False))

    print("\ntrajectory_head:")
    print(traj.head(6).to_string(index=False))
    print("\ntrajectory_tail:")
    print(traj.tail(6).to_string(index=False))

    if not np.isfinite(summary["precession_rate_est_rad_s"]):
        raise AssertionError("Failed to estimate precession rate from turning points.")
    if summary["precession_rate_abs_rel_error"] > 0.08:
        raise AssertionError("Estimated precession rate deviates too much from |Omega*sin(latitude)|.")
    if summary["max_rel_energy_drift"] > 1e-3:
        raise AssertionError("Energy-like invariant drift too large; verify integration settings.")


if __name__ == "__main__":
    main()
