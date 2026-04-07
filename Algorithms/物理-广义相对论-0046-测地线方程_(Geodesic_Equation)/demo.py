"""Minimal runnable MVP for the Geodesic Equation (PHYS-0046).

This demo integrates timelike geodesics in Schwarzschild spacetime
(on the equatorial plane, theta = pi/2) with explicit Christoffel terms.

Units: geometric units (G = c = 1), central mass parameter M = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


EPS = 1e-12


@dataclass(frozen=True)
class GeodesicConfig:
    mass_m: float = 1.0
    r0: float = 10.0
    tau_end: float = 600.0
    sample_points: int = 4000
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float = 0.2
    angular_scale_cases: tuple[float, ...] = (1.0, 0.97)


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    angular_scale: float
    success: bool
    message: str
    r_min: float
    r_max: float
    phi_final: float
    energy_rel_drift: float
    angular_momentum_rel_drift: float
    norm_abs_drift: float
    geodesic_residual_rms: float


def circular_orbit_rates(r: float, m: float) -> tuple[float, float]:
    """Return (u^t, u^phi) for an exact circular timelike orbit in Schwarzschild.

    Valid for r > 3m. Stable circular orbits require r > 6m.
    """
    if r <= 3.0 * m:
        raise ValueError("circular timelike orbit requires r > 3M")
    denom = np.sqrt(1.0 - 3.0 * m / r)
    u_t = 1.0 / denom
    u_phi = np.sqrt(m / (r**3)) / denom
    return float(u_t), float(u_phi)


def solve_ut_from_normalization(r: float, ur: float, uphi: float, m: float) -> float:
    """Solve u^t from timelike normalization g_{mu nu} u^mu u^nu = -1."""
    f = 1.0 - 2.0 * m / r
    if f <= 0.0:
        raise ValueError("r must satisfy r > 2M outside horizon")

    numerator = 1.0 + (ur * ur) / f + (r * r) * (uphi * uphi)
    ut_sq = numerator / f
    if ut_sq <= 0.0:
        raise ValueError("computed u^t^2 <= 0, invalid initial state")
    return float(np.sqrt(ut_sq))


def christoffel_equatorial(r: float, m: float) -> dict[str, float]:
    """Non-zero Christoffel symbols used in equatorial-plane (theta=pi/2) dynamics."""
    f = 1.0 - 2.0 * m / r
    if f <= 0.0:
        raise ValueError("r <= 2M encountered in Christoffel evaluation")

    gamma_t_tr = m / (r * (r - 2.0 * m))
    gamma_r_tt = f * m / (r * r)
    gamma_r_rr = -m / (r * (r - 2.0 * m))
    gamma_r_pp = -(r - 2.0 * m)  # phi-phi component in equatorial plane
    gamma_p_rp = 1.0 / r

    return {
        "t_tr": gamma_t_tr,
        "r_tt": gamma_r_tt,
        "r_rr": gamma_r_rr,
        "r_pp": gamma_r_pp,
        "p_rp": gamma_p_rp,
    }


def geodesic_rhs(tau: float, y: np.ndarray, m: float) -> np.ndarray:
    """First-order geodesic system for y=[t, r, phi, ut, ur, uphi]."""
    del tau
    t, r, phi, ut, ur, uphi = y
    del t, phi

    g = christoffel_equatorial(r=float(r), m=m)

    dt = ut
    dr = ur
    dphi = uphi

    dut = -2.0 * g["t_tr"] * ut * ur
    dur = -(g["r_tt"] * ut * ut + g["r_rr"] * ur * ur + g["r_pp"] * uphi * uphi)
    duphi = -2.0 * g["p_rp"] * ur * uphi

    return np.array([dt, dr, dphi, dut, dur, duphi], dtype=float)


def build_horizon_event(m: float):
    """Stop integration if trajectory touches the horizon r=2M."""

    def event(tau: float, y: np.ndarray) -> float:
        del tau
        return float(y[1] - (2.0 * m + 1e-6))

    event.terminal = True
    event.direction = -1.0
    return event


def compute_invariants(state: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return energy E, angular momentum L, and timelike norm kappa."""
    r = state[1, :]
    ut = state[3, :]
    ur = state[4, :]
    uphi = state[5, :]

    f = 1.0 - 2.0 * m / r
    e = f * ut
    l = (r * r) * uphi
    kappa = -f * (ut * ut) + (ur * ur) / f + (r * r) * (uphi * uphi)
    return e, l, kappa


def geodesic_residual_rms(tau_grid: np.ndarray, state: np.ndarray, m: float) -> float:
    """Finite-difference RMS of acceleration mismatch in (ut, ur, uphi)."""
    ut = state[3, :]
    ur = state[4, :]
    uphi = state[5, :]
    r = state[1, :]

    dut_fd = np.gradient(ut, tau_grid, edge_order=2)
    dur_fd = np.gradient(ur, tau_grid, edge_order=2)
    duphi_fd = np.gradient(uphi, tau_grid, edge_order=2)

    g_t_tr = m / (r * (r - 2.0 * m))
    g_r_tt = (1.0 - 2.0 * m / r) * m / (r * r)
    g_r_rr = -m / (r * (r - 2.0 * m))
    g_r_pp = -(r - 2.0 * m)
    g_p_rp = 1.0 / r

    dut_model = -2.0 * g_t_tr * ut * ur
    dur_model = -(g_r_tt * ut * ut + g_r_rr * ur * ur + g_r_pp * uphi * uphi)
    duphi_model = -2.0 * g_p_rp * ur * uphi

    # Ignore boundary artifacts from finite differences.
    sl = slice(2, -2)
    res2 = (dut_fd[sl] - dut_model[sl]) ** 2
    res2 += (dur_fd[sl] - dur_model[sl]) ** 2
    res2 += (duphi_fd[sl] - duphi_model[sl]) ** 2
    return float(np.sqrt(np.mean(res2)))


def integrate_case(cfg: GeodesicConfig, angular_scale: float) -> CaseResult:
    """Integrate one orbit case and compute diagnostics."""
    m = cfg.mass_m
    r0 = cfg.r0

    _, uphi_circ = circular_orbit_rates(r=r0, m=m)
    ur0 = 0.0
    uphi0 = angular_scale * uphi_circ
    ut0 = solve_ut_from_normalization(r=r0, ur=ur0, uphi=uphi0, m=m)

    y0 = np.array([0.0, r0, 0.0, ut0, ur0, uphi0], dtype=float)
    event = build_horizon_event(m)

    sol = solve_ivp(
        fun=lambda tau, y: geodesic_rhs(tau, y, m),
        t_span=(0.0, cfg.tau_end),
        y0=y0,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
        dense_output=True,
        events=event,
    )

    if not sol.success:
        return CaseResult(
            case_name=f"scale={angular_scale:.3f}",
            angular_scale=angular_scale,
            success=False,
            message=sol.message,
            r_min=np.nan,
            r_max=np.nan,
            phi_final=np.nan,
            energy_rel_drift=np.inf,
            angular_momentum_rel_drift=np.inf,
            norm_abs_drift=np.inf,
            geodesic_residual_rms=np.inf,
        )

    if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
        return CaseResult(
            case_name=f"scale={angular_scale:.3f}",
            angular_scale=angular_scale,
            success=False,
            message="trajectory hit horizon event",
            r_min=np.nan,
            r_max=np.nan,
            phi_final=np.nan,
            energy_rel_drift=np.inf,
            angular_momentum_rel_drift=np.inf,
            norm_abs_drift=np.inf,
            geodesic_residual_rms=np.inf,
        )

    tau_grid = np.linspace(0.0, cfg.tau_end, cfg.sample_points)
    state = sol.sol(tau_grid)

    e, l, kappa = compute_invariants(state=state, m=m)

    e0 = max(abs(e[0]), EPS)
    l0 = max(abs(l[0]), EPS)
    energy_rel_drift = float(np.max(np.abs(e - e[0])) / e0)
    angular_momentum_rel_drift = float(np.max(np.abs(l - l[0])) / l0)
    norm_abs_drift = float(np.max(np.abs(kappa + 1.0)))

    residual = geodesic_residual_rms(tau_grid=tau_grid, state=state, m=m)

    r_track = state[1, :]
    phi_track = state[2, :]

    return CaseResult(
        case_name=f"scale={angular_scale:.3f}",
        angular_scale=angular_scale,
        success=True,
        message="ok",
        r_min=float(np.min(r_track)),
        r_max=float(np.max(r_track)),
        phi_final=float(phi_track[-1]),
        energy_rel_drift=energy_rel_drift,
        angular_momentum_rel_drift=angular_momentum_rel_drift,
        norm_abs_drift=norm_abs_drift,
        geodesic_residual_rms=residual,
    )


def validate_results(results: list[CaseResult]) -> dict[str, bool]:
    """Threshold checks for physics consistency and numerical quality."""
    ok_success = all(r.success for r in results)

    energy_ok = all(r.energy_rel_drift < 1e-9 for r in results)
    l_ok = all(r.angular_momentum_rel_drift < 1e-9 for r in results)
    norm_ok = all(r.norm_abs_drift < 1e-9 for r in results)
    residual_ok = all(r.geodesic_residual_rms < 2e-7 for r in results)

    circular = min(results, key=lambda r: abs(r.angular_scale - 1.0))
    circular_span = abs(circular.r_max - circular.r_min)
    circular_ok = circular.success and circular_span < 2e-6

    return {
        "all integrations success": ok_success,
        "relative energy drift < 1e-9": energy_ok,
        "relative angular momentum drift < 1e-9": l_ok,
        "|u.u + 1| max < 1e-9": norm_ok,
        "geodesic residual RMS < 2e-7": residual_ok,
        "circular case radial span < 2e-6": circular_ok,
    }


def main() -> None:
    cfg = GeodesicConfig()
    results = [integrate_case(cfg, scale) for scale in cfg.angular_scale_cases]

    table = pd.DataFrame(
        [
            {
                "case": r.case_name,
                "success": r.success,
                "message": r.message,
                "r_min": r.r_min,
                "r_max": r.r_max,
                "phi_final": r.phi_final,
                "energy_rel_drift": r.energy_rel_drift,
                "ang_mom_rel_drift": r.angular_momentum_rel_drift,
                "norm_abs_drift": r.norm_abs_drift,
                "geodesic_residual_rms": r.geodesic_residual_rms,
            }
            for r in results
        ]
    )

    checks = validate_results(results)

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Geodesic Equation MVP (PHYS-0046) ===")
    print("Model: Timelike geodesics on Schwarzschild equatorial plane (G=c=1, M=1)")
    print(f"Initial radius r0 = {cfg.r0:.3f} M, tau_end = {cfg.tau_end:.1f}")
    print(f"Angular velocity scales: {cfg.angular_scale_cases}")
    print()
    print(table.to_string(index=False))

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
