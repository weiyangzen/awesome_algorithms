"""Minimal runnable MVP for Kerr-Newman metric (Boyer-Lindquist coordinates).

This script works in geometric units G=c=M=1 and provides:
- Kerr-Newman covariant / contravariant metric components
- horizon and static-limit (ergosphere boundary) radii
- numerical checks: g*g^{-1}=I, determinant identity
- root-solving check for static limit via g_tt = 0
- limit checks to Kerr (Q=0), Reissner-Nordstrom (a=0), Schwarzschild (a=Q=0)
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
class KerrNewmanConfig:
    mass_solar: float = 10.0
    # Geometric-unit parameters: M=1, spin a=J/M^2, charge q=Q/M.
    parameter_sets: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.6, 0.2),
        (0.8, 0.3),
        (0.3, 0.7),
    )
    radii_for_checks: tuple[float, ...] = (3.0, 6.0, 10.0)
    theta_for_checks: tuple[float, ...] = (np.pi / 6.0, np.pi / 2.0)
    static_limit_thetas: tuple[float, ...] = (np.pi / 4.0, np.pi / 2.0)
    tolerance_identity: float = 1e-10
    tolerance_det_rel: float = 1e-10
    tolerance_static_root: float = 1e-11
    tolerance_kerr_limit: float = 1e-12
    tolerance_rn_limit: float = 1e-12
    tolerance_schwarzschild: float = 1e-12


def gravitational_radius_m(mass_kg: float) -> float:
    """Return GM/c^2 in meters."""
    return G * mass_kg / (C * C)


def sigma(r: float, theta: float, a: float) -> float:
    return r * r + a * a * (np.cos(theta) ** 2)


def delta(r: float, a: float, q: float) -> float:
    return r * r - 2.0 * r + a * a + q * q


def extremality_bound_ok(a: float, q: float) -> bool:
    return a * a + q * q <= 1.0 + 1e-15


def horizon_radii(a: float, q: float) -> tuple[float, float]:
    """Outer/inner horizons r_± for Kerr-Newman in units of M."""
    if not extremality_bound_ok(a, q):
        raise ValueError("Kerr-Newman parameters must satisfy a^2 + q^2 <= 1.")
    root = np.sqrt(max(0.0, 1.0 - a * a - q * q))
    return 1.0 + root, 1.0 - root


def kerr_newman_metric_covariant(r: float, theta: float, a: float, q: float) -> np.ndarray:
    """Covariant metric g_{mu nu} in Boyer-Lindquist coordinates.

    Coordinate order: (t, r, theta, phi).
    """
    if not extremality_bound_ok(a, q):
        raise ValueError("Kerr-Newman parameters must satisfy a^2 + q^2 <= 1.")

    sig = sigma(r, theta, a)
    dlt = delta(r, a, q)
    s2 = np.sin(theta) ** 2

    if sig <= 0.0:
        raise ValueError("Sigma must be positive.")
    if dlt <= 0.0:
        raise ValueError("Point must satisfy r > r_+ (outside outer horizon).")

    charge_term = 2.0 * r - q * q

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -(1.0 - charge_term / sig)
    g[1, 1] = sig / dlt
    g[2, 2] = sig
    g[0, 3] = g[3, 0] = -a * charge_term * s2 / sig
    g[3, 3] = (r * r + a * a + a * a * charge_term * s2 / sig) * s2
    return g


def kerr_newman_metric_contravariant(r: float, theta: float, a: float, q: float) -> np.ndarray:
    """Contravariant metric g^{mu nu} in Boyer-Lindquist coordinates."""
    if not extremality_bound_ok(a, q):
        raise ValueError("Kerr-Newman parameters must satisfy a^2 + q^2 <= 1.")

    sig = sigma(r, theta, a)
    dlt = delta(r, a, q)
    s2 = np.sin(theta) ** 2

    if sig <= 0.0:
        raise ValueError("Sigma must be positive.")
    if dlt <= 0.0:
        raise ValueError("Point must satisfy r > r_+ (outside outer horizon).")
    if s2 <= 0.0:
        raise ValueError("theta must avoid poles for this explicit g^{phi phi} form.")

    big_a = (r * r + a * a) ** 2 - a * a * dlt * s2
    charge_term = 2.0 * r - q * q

    g_inv = np.zeros((4, 4), dtype=float)
    g_inv[0, 0] = -big_a / (sig * dlt)
    g_inv[1, 1] = dlt / sig
    g_inv[2, 2] = 1.0 / sig
    g_inv[0, 3] = g_inv[3, 0] = -a * charge_term / (sig * dlt)
    g_inv[3, 3] = (dlt - a * a * s2) / (sig * dlt * s2)
    return g_inv


def metric_identity_error(r: float, theta: float, a: float, q: float) -> float:
    g = kerr_newman_metric_covariant(r, theta, a, q)
    g_inv = kerr_newman_metric_contravariant(r, theta, a, q)
    ident = g @ g_inv
    return float(np.max(np.abs(ident - np.eye(4))))


def determinant_relative_error(r: float, theta: float, a: float, q: float) -> float:
    """Compare numeric det(g) with analytic det(g) = -Sigma^2 sin^2(theta)."""
    g = kerr_newman_metric_covariant(r, theta, a, q)
    det_numeric = float(np.linalg.det(g))
    det_analytic = -(sigma(r, theta, a) ** 2) * (np.sin(theta) ** 2)
    denom = max(1.0, abs(det_analytic))
    return abs(det_numeric - det_analytic) / denom


def static_limit_radius_analytic(a: float, q: float, theta: float) -> float:
    """Outer static limit (ergosphere boundary): g_tt = 0."""
    inside = 1.0 - q * q - a * a * (np.cos(theta) ** 2)
    if inside < -1e-14:
        raise ValueError("No real static-limit root for given parameters/theta.")
    return 1.0 + np.sqrt(max(0.0, inside))


def static_limit_radius_numeric(a: float, q: float, theta: float) -> float:
    """Solve g_tt(r,theta)=0 by bracketing root outside horizon."""
    r_plus, _ = horizon_radii(a, q)
    target = static_limit_radius_analytic(a, q, theta)

    # Pole/near-pole limit can make root coincide with horizon.
    if abs(target - r_plus) < 1e-13:
        return r_plus

    def g_tt_of_r(r: float) -> float:
        sig = sigma(r, theta, a)
        return -(1.0 - (2.0 * r - q * q) / sig)

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


def electromagnetic_potential_covariant(r: float, theta: float, a: float, q: float) -> np.ndarray:
    """Electromagnetic 4-potential A_mu in Boyer-Lindquist coordinates.

    Gauge choice: A_mu dx^mu = -(q r / Sigma) (dt - a sin^2(theta) dphi).
    """
    sig = sigma(r, theta, a)
    s2 = np.sin(theta) ** 2
    a_mu = np.zeros(4, dtype=float)
    a_mu[0] = -q * r / sig
    a_mu[3] = q * r * a * s2 / sig
    return a_mu


def zamo_omega(r: float, theta: float, a: float, q: float) -> float:
    """Frame-dragging angular velocity Omega = -g_tphi/g_phiphi."""
    g = kerr_newman_metric_covariant(r, theta, a, q)
    return float(-g[0, 3] / g[3, 3])


def kerr_limit_error(r: float = 7.0, theta: float = np.pi / 3.0, a: float = 0.65) -> float:
    """At q=0, recover Kerr metric components."""
    g = kerr_newman_metric_covariant(r, theta, a=a, q=0.0)
    sig = r * r + a * a * (np.cos(theta) ** 2)
    dlt = r * r - 2.0 * r + a * a
    s2 = np.sin(theta) ** 2

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 0] = -(1.0 - 2.0 * r / sig)
    expected[1, 1] = sig / dlt
    expected[2, 2] = sig
    expected[0, 3] = expected[3, 0] = -2.0 * a * r * s2 / sig
    expected[3, 3] = (r * r + a * a + 2.0 * a * a * r * s2 / sig) * s2

    return float(np.max(np.abs(g - expected)))


def rn_limit_error(r: float = 7.0, theta: float = np.pi / 3.0, q: float = 0.6) -> float:
    """At a=0, recover Reissner-Nordstrom metric components."""
    g = kerr_newman_metric_covariant(r, theta, a=0.0, q=q)
    f = 1.0 - 2.0 / r + (q * q) / (r * r)
    s2 = np.sin(theta) ** 2

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 0] = -f
    expected[1, 1] = 1.0 / f
    expected[2, 2] = r * r
    expected[3, 3] = r * r * s2

    return float(np.max(np.abs(g - expected)))


def schwarzschild_limit_error(r: float = 8.0, theta: float = np.pi / 3.0) -> float:
    """At a=q=0, recover Schwarzschild metric components."""
    g = kerr_newman_metric_covariant(r, theta, a=0.0, q=0.0)
    s2 = np.sin(theta) ** 2

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 0] = -(1.0 - 2.0 / r)
    expected[1, 1] = 1.0 / (1.0 - 2.0 / r)
    expected[2, 2] = r * r
    expected[3, 3] = r * r * s2

    return float(np.max(np.abs(g - expected)))


def build_parameter_summary(cfg: KerrNewmanConfig, m_geo_m: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    theta_eq = np.pi / 2.0

    for a, q in cfg.parameter_sets:
        r_plus, r_minus = horizon_radii(a, q)
        r_static_eq = static_limit_radius_analytic(a, q, theta_eq)
        a_mu = electromagnetic_potential_covariant(6.0, theta_eq, a, q)
        rows.append(
            {
                "a": a,
                "q": q,
                "a2_plus_q2": a * a + q * q,
                "r_plus_M": r_plus,
                "r_minus_M": r_minus,
                "r_static_eq_M": r_static_eq,
                "ergosphere_width_eq_M": r_static_eq - r_plus,
                "omega_zamo_r6_eq": zamo_omega(6.0, theta_eq, a, q),
                "A_t_r6_eq": a_mu[0],
                "A_phi_r6_eq": a_mu[3],
                "r_plus_m": r_plus * m_geo_m,
            }
        )

    return pd.DataFrame(rows)


def build_metric_checks(cfg: KerrNewmanConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for a, q in cfg.parameter_sets:
        r_plus, _ = horizon_radii(a, q)
        for r in cfg.radii_for_checks:
            if r <= r_plus + 1e-8:
                continue
            for theta in cfg.theta_for_checks:
                rows.append(
                    {
                        "a": a,
                        "q": q,
                        "r_M": r,
                        "theta_rad": theta,
                        "identity_max_abs_error": metric_identity_error(r, theta, a, q),
                        "det_relative_error": determinant_relative_error(r, theta, a, q),
                    }
                )

    return pd.DataFrame(rows)


def build_static_limit_checks(cfg: KerrNewmanConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for a, q in cfg.parameter_sets:
        for theta in cfg.static_limit_thetas:
            ana = static_limit_radius_analytic(a, q, theta)
            num = static_limit_radius_numeric(a, q, theta)
            rows.append(
                {
                    "a": a,
                    "q": q,
                    "theta_rad": theta,
                    "r_static_analytic_M": ana,
                    "r_static_numeric_M": num,
                    "abs_error": abs(ana - num),
                }
            )

    return pd.DataFrame(rows)


def validate(
    cfg: KerrNewmanConfig,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    static_df: pd.DataFrame,
    kerr_err: float,
    rn_err: float,
    schwarz_err: float,
) -> dict[str, bool]:
    checks = {
        "r_plus >= r_minus >= 0": bool(
            np.all(summary_df["r_plus_M"].to_numpy() >= summary_df["r_minus_M"].to_numpy())
            and np.all(summary_df["r_minus_M"].to_numpy() >= 0.0)
        ),
        "a^2 + q^2 <= 1": bool(np.all(summary_df["a2_plus_q2"].to_numpy() <= 1.0 + 1e-12)),
        "r_static_eq >= r_plus": bool(
            np.all(summary_df["r_static_eq_M"].to_numpy() >= summary_df["r_plus_M"].to_numpy() - 1e-13)
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
        "Kerr limit error < tol": bool(kerr_err < cfg.tolerance_kerr_limit),
        "RN limit error < tol": bool(rn_err < cfg.tolerance_rn_limit),
        "Schwarzschild limit error < tol": bool(schwarz_err < cfg.tolerance_schwarzschild),
    }
    return checks


def main() -> None:
    cfg = KerrNewmanConfig()
    mass_kg = cfg.mass_solar * M_SUN
    m_geo_m = gravitational_radius_m(mass_kg)

    summary_df = build_parameter_summary(cfg, m_geo_m)
    metric_df = build_metric_checks(cfg)
    static_df = build_static_limit_checks(cfg)
    kerr_err = kerr_limit_error()
    rn_err = rn_limit_error()
    schwarz_err = schwarzschild_limit_error()

    checks = validate(cfg, summary_df, metric_df, static_df, kerr_err, rn_err, schwarz_err)

    print("=== Kerr-Newman Metric MVP (PHYS-0376) ===")
    print("Coordinates: Boyer-Lindquist (t, r, theta, phi)")
    print("Units: geometric G=c=M=1, plus SI conversion for scale")
    print(f"Mass = {cfg.mass_solar:.3f} M_sun ({mass_kg:.6e} kg)")
    print(f"Geometric mass length GM/c^2 = {m_geo_m:.6f} m")

    print("\n[Parameter summary]")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Metric consistency checks]")
    print(metric_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\n[Static-limit root checks: g_tt=0]")
    print(static_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Limit checks]")
    print(f"max |g_KN(q=0)-g_Kerr| = {kerr_err:.6e}")
    print(f"max |g_KN(a=0)-g_RN| = {rn_err:.6e}")
    print(f"max |g_KN(a=q=0)-g_Schw| = {schwarz_err:.6e}")

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
