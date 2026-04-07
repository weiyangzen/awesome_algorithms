"""Minimal runnable MVP for Kerr metric (Boyer-Lindquist coordinates).

This script works in geometric units G=c=M=1 and provides:
- Kerr covariant / contravariant metric components
- horizon and ergosphere radii
- ISCO radii (prograde / retrograde)
- numerical checks: g*g^{-1}=I, determinant identity, Schwarzschild limit
- root-solving check for static limit surface via g_tt = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# SI constants for optional physical-scale reporting.
G = 6.674_30e-11
C = 299_792_458.0
M_SUN = 1.988_47e30


@dataclass(frozen=True)
class KerrConfig:
    mass_solar: float = 10.0
    spins: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99)
    radii_for_checks: tuple[float, ...] = (3.0, 6.0, 10.0)
    theta_for_checks: tuple[float, ...] = (np.pi / 6.0, np.pi / 2.0)
    static_limit_thetas: tuple[float, ...] = (np.pi / 4.0, np.pi / 2.0)
    tolerance_identity: float = 1e-10
    tolerance_det_rel: float = 1e-10
    tolerance_static_root: float = 1e-11
    tolerance_schwarzschild: float = 1e-12


def gravitational_radius_m(mass_kg: float) -> float:
    """Return GM/c^2 in meters."""
    return G * mass_kg / (C * C)


def horizon_radii(a: float) -> tuple[float, float]:
    """Outer/inner horizons r_± for Kerr in units of M."""
    if abs(a) > 1.0:
        raise ValueError("Kerr spin a must satisfy |a| <= 1.")
    root = np.sqrt(1.0 - a * a)
    return 1.0 + root, 1.0 - root


def sigma(r: float, theta: float, a: float) -> float:
    return r * r + a * a * (np.cos(theta) ** 2)


def delta(r: float, a: float) -> float:
    return r * r - 2.0 * r + a * a


def kerr_metric_covariant(r: float, theta: float, a: float) -> np.ndarray:
    """Covariant metric g_{mu nu} in Boyer-Lindquist coordinates.

    Coordinate order: (t, r, theta, phi).
    """
    sig = sigma(r, theta, a)
    dlt = delta(r, a)
    s2 = np.sin(theta) ** 2

    if sig <= 0.0:
        raise ValueError("Sigma must be positive.")
    if dlt <= 0.0:
        raise ValueError("Point must satisfy r > r_+ (outside outer horizon).")

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -(1.0 - 2.0 * r / sig)
    g[1, 1] = sig / dlt
    g[2, 2] = sig
    g[0, 3] = g[3, 0] = -2.0 * a * r * s2 / sig
    g[3, 3] = (r * r + a * a + 2.0 * a * a * r * s2 / sig) * s2
    return g


def kerr_metric_contravariant(r: float, theta: float, a: float) -> np.ndarray:
    """Contravariant metric g^{mu nu} in Boyer-Lindquist coordinates."""
    sig = sigma(r, theta, a)
    dlt = delta(r, a)
    s2 = np.sin(theta) ** 2

    if sig <= 0.0:
        raise ValueError("Sigma must be positive.")
    if dlt <= 0.0:
        raise ValueError("Point must satisfy r > r_+ (outside outer horizon).")
    if s2 <= 0.0:
        raise ValueError("theta must avoid poles for this explicit g^{phi phi} form.")

    big_a = (r * r + a * a) ** 2 - a * a * dlt * s2

    g_inv = np.zeros((4, 4), dtype=float)
    g_inv[0, 0] = -big_a / (sig * dlt)
    g_inv[1, 1] = dlt / sig
    g_inv[2, 2] = 1.0 / sig
    g_inv[0, 3] = g_inv[3, 0] = -2.0 * a * r / (sig * dlt)
    g_inv[3, 3] = (dlt - a * a * s2) / (sig * dlt * s2)
    return g_inv


def metric_identity_error(r: float, theta: float, a: float) -> float:
    g = kerr_metric_covariant(r, theta, a)
    g_inv = kerr_metric_contravariant(r, theta, a)
    ident = g @ g_inv
    return float(np.max(np.abs(ident - np.eye(4))))


def determinant_relative_error(r: float, theta: float, a: float) -> float:
    """Compare numeric det(g) with analytic det(g) = -Sigma^2 sin^2(theta)."""
    g = kerr_metric_covariant(r, theta, a)
    det_numeric = float(np.linalg.det(g))
    det_analytic = -(sigma(r, theta, a) ** 2) * (np.sin(theta) ** 2)
    denom = max(1.0, abs(det_analytic))
    return abs(det_numeric - det_analytic) / denom


def static_limit_radius_analytic(a: float, theta: float) -> float:
    """Static limit (ergosphere boundary): g_tt = 0."""
    return 1.0 + np.sqrt(1.0 - a * a * (np.cos(theta) ** 2))


def static_limit_radius_numeric(a: float, theta: float) -> float:
    """Solve g_tt(r,theta)=0 by bracketing root outside horizon."""
    r_plus, _ = horizon_radii(a)
    target = static_limit_radius_analytic(a, theta)

    # Pole limit can make the root coincide with horizon exactly.
    if abs(target - r_plus) < 1e-13:
        return r_plus

    def g_tt_of_r(r: float) -> float:
        sig = sigma(r, theta, a)
        return -(1.0 - 2.0 * r / sig)

    low = r_plus + 1e-8
    high = max(6.0, target + 2.0)
    f_low = g_tt_of_r(low)
    f_high = g_tt_of_r(high)

    while f_low * f_high > 0.0 and high < 1e6:
        high *= 2.0
        f_high = g_tt_of_r(high)

    if f_low * f_high > 0.0:
        raise RuntimeError("Failed to bracket static-limit root.")

    return float(brentq(g_tt_of_r, low, high, xtol=1e-13, rtol=1e-11, maxiter=200))


def isco_radius(a: float, prograde: bool) -> float:
    """ISCO radius from Bardeen-Press-Teukolsky formula (units of M)."""
    if abs(a) > 1.0:
        raise ValueError("|a| must be <= 1.")

    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    radical = np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    if prograde:
        return float(3.0 + z2 - radical)
    return float(3.0 + z2 + radical)


def zamo_omega(r: float, theta: float, a: float) -> float:
    """Frame-dragging angular velocity Omega = -g_tphi/g_phiphi."""
    g = kerr_metric_covariant(r, theta, a)
    return float(-g[0, 3] / g[3, 3])


def schwarzschild_limit_error(r: float = 8.0, theta: float = np.pi / 3.0) -> float:
    """Check Kerr metric at a=0 against Schwarzschild BL components."""
    g = kerr_metric_covariant(r, theta, a=0.0)
    s2 = np.sin(theta) ** 2

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 0] = -(1.0 - 2.0 / r)
    expected[1, 1] = 1.0 / (1.0 - 2.0 / r)
    expected[2, 2] = r * r
    expected[3, 3] = r * r * s2

    return float(np.max(np.abs(g - expected)))


def build_spin_summary(cfg: KerrConfig, m_geo_m: float) -> pd.DataFrame:
    """Build per-spin physical summary at equator."""
    rows: list[dict[str, float]] = []
    theta_eq = np.pi / 2.0

    for a in cfg.spins:
        r_plus, r_minus = horizon_radii(a)
        r_static_eq = static_limit_radius_analytic(a, theta_eq)
        rows.append(
            {
                "a": a,
                "r_plus_M": r_plus,
                "r_minus_M": r_minus,
                "r_static_eq_M": r_static_eq,
                "ergosphere_width_eq_M": r_static_eq - r_plus,
                "r_isco_pro_M": isco_radius(a, prograde=True),
                "r_isco_retro_M": isco_radius(a, prograde=False),
                "omega_zamo_r6_eq": zamo_omega(6.0, theta_eq, a),
                "r_plus_m": r_plus * m_geo_m,
                "r_isco_pro_m": isco_radius(a, prograde=True) * m_geo_m,
            }
        )

    return pd.DataFrame(rows)


def build_metric_checks(cfg: KerrConfig) -> pd.DataFrame:
    """Evaluate inverse/determinant consistency on grid points."""
    rows: list[dict[str, float]] = []

    for a in cfg.spins:
        r_plus, _ = horizon_radii(a)
        for r in cfg.radii_for_checks:
            if r <= r_plus + 1e-8:
                continue
            for theta in cfg.theta_for_checks:
                rows.append(
                    {
                        "a": a,
                        "r_M": r,
                        "theta_rad": theta,
                        "identity_max_abs_error": metric_identity_error(r, theta, a),
                        "det_relative_error": determinant_relative_error(r, theta, a),
                    }
                )

    return pd.DataFrame(rows)


def build_static_limit_checks(cfg: KerrConfig) -> pd.DataFrame:
    """Compare analytic static-limit radius with numerical root solving."""
    rows: list[dict[str, float]] = []

    for a in cfg.spins:
        for theta in cfg.static_limit_thetas:
            ana = static_limit_radius_analytic(a, theta)
            num = static_limit_radius_numeric(a, theta)
            rows.append(
                {
                    "a": a,
                    "theta_rad": theta,
                    "r_static_analytic_M": ana,
                    "r_static_numeric_M": num,
                    "abs_error": abs(ana - num),
                }
            )

    return pd.DataFrame(rows)


def validate(
    cfg: KerrConfig,
    spin_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    static_df: pd.DataFrame,
    schwarz_err: float,
) -> dict[str, bool]:
    """Threshold checks for this MVP."""
    checks = {
        "r_plus >= r_minus >= 0": bool(
            np.all(spin_df["r_plus_M"].to_numpy() >= spin_df["r_minus_M"].to_numpy())
            and np.all(spin_df["r_minus_M"].to_numpy() >= 0.0)
        ),
        "prograde ISCO <= retrograde ISCO": bool(
            np.all(spin_df["r_isco_pro_M"].to_numpy() <= spin_df["r_isco_retro_M"].to_numpy())
        ),
        "max identity error < tol": bool(
            float(metric_df["identity_max_abs_error"].max()) < cfg.tolerance_identity
        ),
        "max det relative error < tol": bool(
            float(metric_df["det_relative_error"].max()) < cfg.tolerance_det_rel
        ),
        "max static-limit root error < tol": bool(
            float(static_df["abs_error"].max()) < cfg.tolerance_static_root
        ),
        "Schwarzschild limit error < tol": bool(schwarz_err < cfg.tolerance_schwarzschild),
    }
    return checks


def main() -> None:
    cfg = KerrConfig()
    mass_kg = cfg.mass_solar * M_SUN
    m_geo_m = gravitational_radius_m(mass_kg)

    spin_df = build_spin_summary(cfg, m_geo_m)
    metric_df = build_metric_checks(cfg)
    static_df = build_static_limit_checks(cfg)
    schwarz_err = schwarzschild_limit_error()

    checks = validate(cfg, spin_df, metric_df, static_df, schwarz_err)

    print("=== Kerr Metric MVP (PHYS-0375) ===")
    print("Coordinates: Boyer-Lindquist (t, r, theta, phi)")
    print("Units: geometric G=c=M=1, plus SI conversion for scale")
    print(f"Mass = {cfg.mass_solar:.3f} M_sun ({mass_kg:.6e} kg)")
    print(f"Geometric mass length GM/c^2 = {m_geo_m:.6f} m")

    print("\n[Spin summary]")
    print(spin_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Metric consistency checks]")
    print(metric_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\n[Static-limit root checks: g_tt=0]")
    print(static_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Schwarzschild limit]")
    print(f"max |g_kerr(a=0)-g_schw| = {schwarz_err:.6e}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
