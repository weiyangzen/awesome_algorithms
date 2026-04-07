"""Schwarzschild Solution MVP for PHYS-0047.

This script demonstrates three core pieces of the Schwarzschild solution:
1) Metric and invariant quantities outside the event horizon.
2) Static-observer redshift and key radii (horizon, photon sphere, ISCO).
3) Radial timelike geodesic (L=0) integrated with explicit RK4 in proper time.

Units: geometric units with G = c = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class SchwarzschildConfig:
    """Configuration for the radial infall demonstration."""

    mass: float = 1.0
    energy: float = 1.0
    r0: float = 20.0
    r_end: float = 2.05
    h: float = 1e-2
    max_steps: int = 50_000


def schwarzschild_radius(mass: float) -> float:
    """Return event horizon radius r_s = 2M in geometric units."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    return 2.0 * mass


def lapse(r: float | np.ndarray, mass: float) -> float | np.ndarray:
    """Return f(r)=1-2M/r for Schwarzschild metric."""
    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr <= 0):
        raise ValueError("r must be > 0")
    return 1.0 - (2.0 * mass / r_arr)


def metric_tensor(r: float, theta: float, mass: float) -> np.ndarray:
    """Covariant Schwarzschild metric g_{mu nu} in (t, r, theta, phi)."""
    if r <= schwarzschild_radius(mass):
        raise ValueError("this MVP evaluates metric outside horizon only (r > 2M)")

    f = float(lapse(r, mass))
    s2 = float(np.sin(theta) ** 2)

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r * r
    g[3, 3] = r * r * s2
    return g


def kretschmann_scalar(r: float | np.ndarray, mass: float) -> float | np.ndarray:
    """Kretschmann invariant K = 48 M^2 / r^6."""
    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr <= 0):
        raise ValueError("r must be > 0")
    return 48.0 * (mass**2) / (r_arr**6)


def redshift_to_infinity(r: float | np.ndarray, mass: float) -> float | np.ndarray:
    """Static gravitational redshift z for signal emitted at r and received at infinity."""
    f = np.asarray(lapse(r, mass), dtype=float)
    if np.any(f <= 0):
        raise ValueError("redshift_to_infinity requires r > 2M")
    return 1.0 / np.sqrt(f) - 1.0


def radial_rhs(state: np.ndarray, mass: float, energy: float) -> np.ndarray:
    """Proper-time RHS for radial timelike geodesic with L=0.

    state = [r, t]
      dr/dtau = -sqrt(E^2 - f)
      dt/dtau = E/f
    with f = 1 - 2M/r.
    """
    r, _t = state
    f = float(lapse(r, mass))
    if f <= 0:
        raise ValueError("integration crossed horizon; increase r_end")

    radicand = max(energy * energy - f, 0.0)
    dr_dtau = -np.sqrt(radicand)
    dt_dtau = energy / f
    return np.array([dr_dtau, dt_dtau], dtype=float)


def rk4_step(state: np.ndarray, h: float, mass: float, energy: float) -> np.ndarray:
    """One explicit RK4 step for the radial system."""
    k1 = radial_rhs(state, mass, energy)
    k2 = radial_rhs(state + 0.5 * h * k1, mass, energy)
    k3 = radial_rhs(state + 0.5 * h * k2, mass, energy)
    k4 = radial_rhs(state + h * k3, mass, energy)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_radial_infall(cfg: SchwarzschildConfig) -> dict[str, np.ndarray]:
    """Integrate radial infall from r0 to r_end in proper time."""
    rs = schwarzschild_radius(cfg.mass)
    if cfg.r0 <= rs or cfg.r_end <= rs:
        raise ValueError("require r0 > r_end > 2M")
    if cfg.r0 <= cfg.r_end:
        raise ValueError("require r0 > r_end")
    if cfg.h <= 0:
        raise ValueError("time step h must be positive")

    state = np.array([cfg.r0, 0.0], dtype=float)

    tau_values = [0.0]
    r_values = [cfg.r0]
    t_values = [0.0]
    dr_dtau_values = [radial_rhs(state, cfg.mass, cfg.energy)[0]]
    dt_dtau_values = [radial_rhs(state, cfg.mass, cfg.energy)[1]]

    tau = 0.0
    for _ in range(cfg.max_steps):
        if state[0] <= cfg.r_end:
            break

        state_next = rk4_step(state, cfg.h, cfg.mass, cfg.energy)
        tau += cfg.h

        state = state_next
        rhs = radial_rhs(state, cfg.mass, cfg.energy)

        tau_values.append(tau)
        r_values.append(float(state[0]))
        t_values.append(float(state[1]))
        dr_dtau_values.append(float(rhs[0]))
        dt_dtau_values.append(float(rhs[1]))

        if state[0] <= cfg.r_end:
            break
    else:
        raise RuntimeError("max_steps reached before hitting r_end")

    return {
        "tau": np.asarray(tau_values),
        "r": np.asarray(r_values),
        "t": np.asarray(t_values),
        "dr_dtau": np.asarray(dr_dtau_values),
        "dt_dtau": np.asarray(dt_dtau_values),
    }


def analytic_tau_for_e1(r0: float, r: float, mass: float) -> float:
    """Exact proper-time drop for E=1 radial infall from r0 to r."""
    prefactor = 2.0 / (3.0 * np.sqrt(2.0 * mass))
    return float(prefactor * (r0 ** 1.5 - r ** 1.5))


def build_metric_table(mass: float, radii: np.ndarray) -> pd.DataFrame:
    """Tabulate metric factors and invariants at selected radii."""
    f = lapse(radii, mass)
    table = pd.DataFrame(
        {
            "r": radii,
            "f(r)=1-2M/r": f,
            "g_tt": -f,
            "g_rr": 1.0 / f,
            "z_to_infinity": redshift_to_infinity(radii, mass),
            "K=48M^2/r^6": kretschmann_scalar(radii, mass),
        }
    )
    return table


def main() -> None:
    cfg = SchwarzschildConfig()
    rs = schwarzschild_radius(cfg.mass)
    photon_sphere = 3.0 * cfg.mass
    isco = 6.0 * cfg.mass

    sample_r = np.array([20.0, 10.0, 6.0, 3.0, 2.5, 2.1], dtype=float) * cfg.mass
    metric_df = build_metric_table(cfg.mass, sample_r)

    traj = integrate_radial_infall(cfg)
    tau_num = float(traj["tau"][-1])
    r_final = float(traj["r"][-1])
    t_num = float(traj["t"][-1])

    tau_exact = analytic_tau_for_e1(cfg.r0, r_final, cfg.mass)
    rel_tau_err = abs(tau_num - tau_exact) / max(abs(tau_exact), EPS)

    metric_at_r0 = metric_tensor(cfg.r0, theta=np.pi / 2.0, mass=cfg.mass)

    checks = {
        "lapse(2M) == 0": abs(float(lapse(rs, cfg.mass))) < 1e-14,
        "metric tensor symmetric": np.allclose(metric_at_r0, metric_at_r0.T, atol=1e-14),
        "r strictly decreases": bool(np.all(np.diff(traj["r"]) < 0.0)),
        "all sampled r are outside horizon": bool(np.all(traj["r"] > rs)),
        "numerical tau matches analytic tau (rel<5e-4)": rel_tau_err < 5e-4,
        "dt/dtau grows near horizon": float(traj["dt_dtau"][-1] / traj["dt_dtau"][0]) > 8.0,
        "Kretschmann increases inward": bool(metric_df["K=48M^2/r^6"].is_monotonic_increasing),
        "all trajectory values finite": bool(
            np.all(np.isfinite(traj["tau"]))
            and np.all(np.isfinite(traj["r"]))
            and np.all(np.isfinite(traj["t"]))
        ),
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Schwarzschild Solution MVP (PHYS-0047) ===")
    print(f"Mass M = {cfg.mass:.3f} (geometric units, G=c=1)")
    print(f"Event horizon r_s = 2M = {rs:.6f}")
    print(f"Photon sphere r = 3M = {photon_sphere:.6f}")
    print(f"ISCO (timelike) r = 6M = {isco:.6f}")

    print("\n[Metric / Invariant Table]")
    print(metric_df.to_string(index=False))

    print("\n[Radial Infall: L=0, E=1]")
    print(f"r0 = {cfg.r0:.6f}, r_final = {r_final:.6f}, r_end(target) = {cfg.r_end:.6f}")
    print(f"proper time tau_num   = {tau_num:.8f}")
    print(f"proper time tau_exact = {tau_exact:.8f}")
    print(f"relative tau error    = {rel_tau_err:.3e}")
    print(f"coordinate time delta t = {t_num:.8f}")

    print("\n[Checks]")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
