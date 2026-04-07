"""Minimal MVP for Inflation Theory (single-field slow-roll).

This script builds a small, transparent numerical pipeline for a family of
monomial inflation potentials:
    V(phi) = lambda * phi^p

For each exponent p, we:
1) Solve for inflation end field phi_end from epsilon_V(phi_end)=1.
2) Solve for pivot field phi_star such that N(phi_star -> phi_end)=N_target.
3) Calibrate lambda so scalar amplitude A_s matches an observed target.
4) Report slow-roll observables (n_s, r, A_s) and a rough plausibility flag.
5) Cross-check e-fold count with an ODE integration in N-space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class InflationConfig:
    n_target: float = 55.0
    target_as: float = 2.1e-9
    p_values: tuple[float, ...] = (2.0 / 3.0, 1.0, 2.0, 3.0, 4.0)
    ns_band: tuple[float, float] = (0.955, 0.975)
    r_upper: float = 0.07


def potential(phi: float, p: float, lam: float) -> float:
    return lam * phi**p


def dpotential(phi: float, p: float, lam: float) -> float:
    return lam * p * phi ** (p - 1.0)


def epsilon_v(phi: float, p: float) -> float:
    return 0.5 * (p / phi) ** 2


def eta_v(phi: float, p: float) -> float:
    return p * (p - 1.0) / (phi**2)


def find_phi_end(p: float) -> float:
    """Solve epsilon_V(phi_end) = 1 for phi_end > 0."""
    if p <= 0.0:
        raise ValueError("Exponent p must be positive.")
    return float(optimize.brentq(lambda phi: epsilon_v(phi, p) - 1.0, 1e-8, 100.0))


def efolds_between(phi_start: float, phi_end: float, p: float) -> float:
    """Compute N = integral(V/V') dphi from phi_end to phi_start."""
    if phi_start <= phi_end:
        raise ValueError("phi_start must be larger than phi_end for inflation.")
    integrand = lambda phi: potential(phi, p, 1.0) / dpotential(phi, p, 1.0)
    val, _ = integrate.quad(integrand, phi_end, phi_start, limit=200)
    return float(val)


def find_phi_star(p: float, n_target: float) -> tuple[float, float]:
    """Solve for phi_star that yields a target number of e-folds."""
    phi_end = find_phi_end(p)

    def objective(phi: float) -> float:
        return efolds_between(phi, phi_end, p) - n_target

    lower = phi_end * (1.0 + 1e-8)
    upper = max(phi_end + 1.0, math.sqrt(max(1e-12, 2.0 * p * (n_target + 5.0))) + phi_end)

    while objective(upper) < 0.0:
        upper *= 1.5
        if upper > 1e6:
            raise RuntimeError("Failed to bracket phi_star root.")

    phi_star = optimize.brentq(objective, lower, upper, xtol=1e-12, rtol=1e-10)
    return phi_end, float(phi_star)


def calibrate_lambda(phi_star: float, p: float, target_as: float) -> float:
    """Choose lambda so A_s = V / (24 pi^2 epsilon_V) matches target_as."""
    eps = epsilon_v(phi_star, p)
    if eps <= 0.0:
        raise ValueError("epsilon_V must be positive.")
    return target_as * 24.0 * math.pi**2 * eps / (phi_star**p)


def predict_observables(phi_star: float, p: float, lam: float) -> dict[str, float]:
    eps = epsilon_v(phi_star, p)
    eta = eta_v(phi_star, p)
    as_pred = potential(phi_star, p, lam) / (24.0 * math.pi**2 * eps)
    ns = 1.0 - 6.0 * eps + 2.0 * eta
    r = 16.0 * eps
    return {"epsilon_v": eps, "eta_v": eta, "as_pred": as_pred, "ns": ns, "r": r}


def efolds_by_ode(phi_star: float, p: float) -> float:
    """Cross-check N via slow-roll ODE dphi/dN = -V'/V = -p/phi."""

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        phi = max(float(y[0]), 1e-12)
        return np.array([-p / phi], dtype=float)

    def end_event(_: float, y: np.ndarray) -> float:
        phi = max(float(y[0]), 1e-12)
        return epsilon_v(phi, p) - 1.0

    end_event.terminal = True
    end_event.direction = 1.0

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, 400.0),
        y0=np.array([phi_star], dtype=float),
        events=end_event,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2,
    )
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("ODE integration did not reach inflation end.")
    return float(sol.t_events[0][0])


def run_grid(cfg: InflationConfig) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for p in cfg.p_values:
        phi_end, phi_star = find_phi_star(p, cfg.n_target)
        n_from_quad = efolds_between(phi_star, phi_end, p)
        lam = calibrate_lambda(phi_star, p, cfg.target_as)
        obs = predict_observables(phi_star, p, lam)
        n_from_ode = efolds_by_ode(phi_star, p)

        plausible = (
            cfg.ns_band[0] <= obs["ns"] <= cfg.ns_band[1]
            and obs["r"] <= cfg.r_upper
        )

        rows.append(
            {
                "p": float(p),
                "phi_end": phi_end,
                "phi_star": phi_star,
                "lambda": lam,
                "n_from_quad": n_from_quad,
                "n_from_ode": n_from_ode,
                "epsilon_v": obs["epsilon_v"],
                "eta_v": obs["eta_v"],
                "as_pred": obs["as_pred"],
                "ns": obs["ns"],
                "r": obs["r"],
                "plausible_under_simple_cut": bool(plausible),
            }
        )

    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def print_summary(df: pd.DataFrame, cfg: InflationConfig) -> None:
    with pd.option_context("display.precision", 6, "display.width", 170):
        print(df.to_string(index=False))
    print()
    print(f"Target settings: N_target={cfg.n_target}, A_s={cfg.target_as:.3e}")
    print(f"Simple plausibility cut: n_s in [{cfg.ns_band[0]}, {cfg.ns_band[1]}], r <= {cfg.r_upper}")


def run_self_checks(df: pd.DataFrame, cfg: InflationConfig) -> None:
    assert len(df) == len(cfg.p_values)
    assert np.all(df["phi_star"].to_numpy() > df["phi_end"].to_numpy())
    assert np.max(np.abs(df["n_from_quad"].to_numpy() - cfg.n_target)) < 1e-7
    assert np.max(np.abs(df["n_from_ode"].to_numpy() - cfg.n_target)) < 1e-2
    assert np.max(np.abs(df["as_pred"].to_numpy() - cfg.target_as)) < 1e-14

    r_vals = df.sort_values("p")["r"].to_numpy()
    assert np.all(np.diff(r_vals) > 0.0), "Expected r to grow with p at fixed N."


def main() -> None:
    cfg = InflationConfig()
    print("Inflation Theory MVP: single-field slow-roll with monomial potentials")
    df = run_grid(cfg)
    print_summary(df, cfg)

    best = (
        df.assign(
            score=lambda x: np.abs(x["ns"] - 0.965) + np.maximum(x["r"] - cfg.r_upper, 0.0)
        )
        .sort_values("score")
        .iloc[0]
    )
    print(
        "Heuristic best candidate in this tiny grid: "
        f"p={best['p']:.6g}, n_s={best['ns']:.6f}, r={best['r']:.6f}"
    )

    run_self_checks(df, cfg)
    print("Checks passed: root-finding, integration, and ODE cross-check are consistent.")


if __name__ == "__main__":
    main()
