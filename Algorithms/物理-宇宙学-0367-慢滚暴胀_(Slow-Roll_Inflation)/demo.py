"""Minimal runnable MVP for Slow-Roll Inflation.

This script implements a transparent single-field slow-roll pipeline for
monomial potentials V(phi)=lambda*phi^p (M_pl=1), and reports background/
observable quantities with cross-checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import integrate, optimize
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class SlowRollConfig:
    n_target: float = 55.0
    target_as: float = 2.1e-9
    p_values: tuple[float, ...] = (2.0 / 3.0, 1.0, 2.0, 3.0, 4.0)
    ns_band: tuple[float, float] = (0.955, 0.975)
    r_upper: float = 0.07


def potential(phi: float, p: float, lam: float) -> float:
    return lam * phi**p


def dpotential(phi: float, p: float, lam: float) -> float:
    return lam * p * phi ** (p - 1.0)


def d2potential(phi: float, p: float, lam: float) -> float:
    return lam * p * (p - 1.0) * phi ** (p - 2.0)


def epsilon_v(phi: float, p: float, lam: float) -> float:
    v = potential(phi, p, lam)
    dv = dpotential(phi, p, lam)
    return 0.5 * (dv / v) ** 2


def eta_v(phi: float, p: float, lam: float) -> float:
    v = potential(phi, p, lam)
    d2v = d2potential(phi, p, lam)
    return d2v / v


def find_phi_end(p: float) -> float:
    """Solve epsilon_V(phi_end)=1 for phi_end>0."""
    return float(
        optimize.brentq(
            lambda x: epsilon_v(x, p, 1.0) - 1.0,
            1e-10,
            1e3,
        )
    )


def efolds_between(phi_start: float, phi_end: float, p: float) -> float:
    """N(phi_start -> phi_end)=∫(V/V') dphi under slow-roll."""
    if phi_start <= phi_end:
        raise ValueError("phi_start must be > phi_end")
    integrand = lambda x: potential(x, p, 1.0) / dpotential(x, p, 1.0)
    val, _ = integrate.quad(integrand, phi_end, phi_start, limit=200)
    return float(val)


def find_phi_star(p: float, n_target: float) -> tuple[float, float]:
    """Find phi_star such that N(phi_star -> phi_end)=n_target."""
    phi_end = find_phi_end(p)

    def objective(phi: float) -> float:
        return efolds_between(phi, phi_end, p) - n_target

    lower = phi_end * (1.0 + 1e-9)
    upper = max(phi_end + 1.0, math.sqrt(2.0 * p * (n_target + 6.0)) + phi_end)
    while objective(upper) < 0.0:
        upper *= 1.5
        if upper > 1e7:
            raise RuntimeError("Cannot bracket phi_star")

    phi_star = optimize.brentq(objective, lower, upper, xtol=1e-12, rtol=1e-10)
    return phi_end, float(phi_star)


def calibrate_lambda(phi_star: float, p: float, target_as: float) -> float:
    eps = epsilon_v(phi_star, p, 1.0)
    return target_as * 24.0 * math.pi**2 * eps / (phi_star**p)


def ns_r(phi_star: float, p: float, lam: float) -> tuple[float, float, float, float]:
    eps = epsilon_v(phi_star, p, lam)
    eta = eta_v(phi_star, p, lam)
    ns = 1.0 - 6.0 * eps + 2.0 * eta
    r = 16.0 * eps
    as_pred = potential(phi_star, p, lam) / (24.0 * math.pi**2 * eps)
    return ns, r, as_pred, eps


def efolds_by_ode(phi_star: float, p: float) -> float:
    """Cross-check N with dphi/dN = -V'/V = -p/phi."""

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        phi = max(float(y[0]), 1e-12)
        return np.array([-p / phi], dtype=float)

    def stop_event(_: float, y: np.ndarray) -> float:
        phi = max(float(y[0]), 1e-12)
        return epsilon_v(phi, p, 1.0) - 1.0

    stop_event.terminal = True
    stop_event.direction = 1.0

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, 400.0),
        y0=np.array([phi_star], dtype=float),
        events=stop_event,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2,
    )
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("ODE failed to reach epsilon=1")
    return float(sol.t_events[0][0])


def torch_slowroll_at_star(phi_star: float, p: float, lam: float) -> tuple[float, float]:
    """Use PyTorch autograd to independently recover epsilon_V and eta_V."""
    phi = torch.tensor([phi_star], dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(float(p), dtype=torch.float64)
    lam_t = torch.tensor(float(lam), dtype=torch.float64)

    v = lam_t * torch.pow(phi, p_t)
    dv = torch.autograd.grad(v, phi, create_graph=True)[0]
    d2v = torch.autograd.grad(dv, phi)[0]

    eps = 0.5 * torch.pow(dv / v, 2)
    eta = d2v / v
    return float(eps.item()), float(eta.item())


def estimate_ns_with_regression(phi_star: float, p: float, lam: float) -> float:
    """Estimate n_s by fitting ln P_R vs ln(k/k*) with sklearn linear regression.

    We use ln(k/k*)≈ΔN around the pivot and solve dphi/dN=-p/phi analytically.
    """
    dn = np.linspace(-3.0, 3.0, 25)
    # phi(N) from integrating dphi/dN=-p/phi with N=0 at pivot
    phi_sq = np.maximum(phi_star**2 - 2.0 * p * dn, 1e-12)
    phi_vals = np.sqrt(phi_sq)

    eps_vals = np.array([epsilon_v(x, p, lam) for x in phi_vals])
    v_vals = np.array([potential(x, p, lam) for x in phi_vals])
    p_r = v_vals / (24.0 * math.pi**2 * eps_vals)

    x = dn.reshape(-1, 1)
    y = np.log(p_r)
    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    return 1.0 + slope


def run_grid(cfg: SlowRollConfig) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []

    for p in cfg.p_values:
        phi_end, phi_star = find_phi_star(p, cfg.n_target)
        n_quad = efolds_between(phi_star, phi_end, p)
        lam = calibrate_lambda(phi_star, p, cfg.target_as)

        ns_analytic, r, as_pred, eps = ns_r(phi_star, p, lam)
        n_ode = efolds_by_ode(phi_star, p)

        eps_torch, eta_torch = torch_slowroll_at_star(phi_star, p, lam)
        ns_reg = estimate_ns_with_regression(phi_star, p, lam)

        plausible = (cfg.ns_band[0] <= ns_analytic <= cfg.ns_band[1]) and (r <= cfg.r_upper)

        rows.append(
            {
                "p": float(p),
                "phi_end": phi_end,
                "phi_star": phi_star,
                "lambda": lam,
                "N_quad": n_quad,
                "N_ode": n_ode,
                "epsilon_v": eps,
                "epsilon_torch": eps_torch,
                "eta_torch": eta_torch,
                "A_s": as_pred,
                "n_s_analytic": ns_analytic,
                "n_s_regression": ns_reg,
                "r": r,
                "plausible_under_simple_cut": bool(plausible),
            }
        )

    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def run_checks(df: pd.DataFrame, cfg: SlowRollConfig) -> None:
    assert len(df) == len(cfg.p_values)
    assert np.all(df["phi_star"].to_numpy() > df["phi_end"].to_numpy())
    assert np.max(np.abs(df["N_quad"].to_numpy() - cfg.n_target)) < 1e-7
    assert np.max(np.abs(df["N_ode"].to_numpy() - cfg.n_target)) < 1e-2
    assert np.max(np.abs(df["A_s"].to_numpy() - cfg.target_as)) < 1e-14
    assert np.max(np.abs(df["epsilon_v"].to_numpy() - df["epsilon_torch"].to_numpy())) < 1e-10
    assert np.max(np.abs(df["n_s_analytic"].to_numpy() - df["n_s_regression"].to_numpy())) < 1e-2

    sorted_r = df.sort_values("p")["r"].to_numpy()
    assert np.all(np.diff(sorted_r) > 0.0), "Expected r to increase with p at fixed N."


def main() -> None:
    cfg = SlowRollConfig()
    print("Slow-Roll Inflation MVP (single-field, monomial potentials, M_pl=1)")

    df = run_grid(cfg)
    with pd.option_context("display.precision", 6, "display.width", 180):
        print(df.to_string(index=False))

    print()
    print(f"Target: N={cfg.n_target}, A_s={cfg.target_as:.3e}")
    print(f"Simple cut: n_s in [{cfg.ns_band[0]}, {cfg.ns_band[1]}], r <= {cfg.r_upper}")

    best = (
        df.assign(
            score=lambda x: np.abs(x["n_s_analytic"] - 0.965)
            + np.maximum(x["r"] - cfg.r_upper, 0.0)
        )
        .sort_values("score")
        .iloc[0]
    )
    print(
        "Heuristic best p in this tiny grid: "
        f"p={best['p']:.6g}, n_s={best['n_s_analytic']:.6f}, r={best['r']:.6f}"
    )

    run_checks(df, cfg)
    print("Checks passed: root/integral/ODE/autograd/regression are mutually consistent.")


if __name__ == "__main__":
    main()
