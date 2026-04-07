"""Minimal runnable MVP for Boundary Layer Theory (Blasius flat-plate solution)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class BoundaryLayerConfig:
    """Configuration for laminar flat-plate boundary layer (zero pressure gradient)."""

    u_inf: float = 15.0  # free-stream velocity [m/s]
    nu: float = 1.5e-5  # kinematic viscosity [m^2/s]
    rho: float = 1.225  # density [kg/m^3]
    eta_max: float = 12.0  # finite truncation for eta->infinity
    n_eta: int = 2400
    x_stations: tuple[float, ...] = (0.03, 0.06, 0.10, 0.20, 0.30)


@dataclass
class BlasiusProfile:
    eta: np.ndarray
    f: np.ndarray
    fp: np.ndarray
    fpp: np.ndarray
    fpp0: float


@dataclass(frozen=True)
class SimilarityMetrics:
    eta_99: float
    displacement_eta: float
    momentum_eta: float
    shape_factor: float


def blasius_rhs(_: float, y: np.ndarray) -> np.ndarray:
    """RHS of Blasius ODE system.

    Let y = [f, f', f''], then:
    f'   = y1
    f''  = y2
    f''' = -0.5 * f * f''
    """

    f_val, fp_val, fpp_val = y
    return np.array([fp_val, fpp_val, -0.5 * f_val * fpp_val], dtype=float)


def integrate_blasius(fpp0: float, cfg: BoundaryLayerConfig) -> BlasiusProfile:
    """Integrate Blasius ODE for a guessed wall curvature f''(0)."""

    eta_grid = np.linspace(0.0, cfg.eta_max, cfg.n_eta)
    sol = solve_ivp(
        blasius_rhs,
        t_span=(0.0, cfg.eta_max),
        y0=np.array([0.0, 0.0, fpp0], dtype=float),
        t_eval=eta_grid,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Blasius ODE integration failed: {sol.message}")

    return BlasiusProfile(
        eta=eta_grid,
        f=sol.y[0],
        fp=sol.y[1],
        fpp=sol.y[2],
        fpp0=float(fpp0),
    )


def shooting_residual(fpp0: float, cfg: BoundaryLayerConfig) -> float:
    """Residual for shooting method: enforce f'(eta_max) -> 1."""

    profile = integrate_blasius(fpp0=fpp0, cfg=cfg)
    return float(profile.fp[-1] - 1.0)


def solve_blasius(cfg: BoundaryLayerConfig) -> BlasiusProfile:
    """Solve Blasius equation via shooting + Brent root finder."""

    lower, upper = 0.2, 0.5
    res_lower = shooting_residual(lower, cfg)
    res_upper = shooting_residual(upper, cfg)
    if res_lower * res_upper > 0.0:
        raise RuntimeError(
            "Unable to bracket Blasius shooting root. "
            f"res(lower)={res_lower:.3e}, res(upper)={res_upper:.3e}"
        )

    fpp0_opt = float(brentq(lambda s: shooting_residual(s, cfg), lower, upper, xtol=1.0e-13, rtol=1.0e-11))
    return integrate_blasius(fpp0=fpp0_opt, cfg=cfg)


def compute_similarity_metrics(profile: BlasiusProfile) -> SimilarityMetrics:
    """Compute eta-based integral metrics from similarity profile."""

    u_ratio = profile.fp

    if np.max(u_ratio) < 0.99:
        raise AssertionError("f'(eta) never reaches 0.99; eta_max may be too small.")

    # eta_99 from interpolation on monotonic profile.
    eta_99 = float(np.interp(0.99, u_ratio, profile.eta))

    displacement_eta = float(np.trapezoid(1.0 - u_ratio, profile.eta))
    momentum_eta = float(np.trapezoid(u_ratio * (1.0 - u_ratio), profile.eta))
    shape_factor = displacement_eta / momentum_eta

    return SimilarityMetrics(
        eta_99=eta_99,
        displacement_eta=displacement_eta,
        momentum_eta=momentum_eta,
        shape_factor=float(shape_factor),
    )


def build_station_table(
    cfg: BoundaryLayerConfig,
    profile: BlasiusProfile,
    metrics: SimilarityMetrics,
) -> pd.DataFrame:
    """Project similarity solution to dimensional quantities at x stations."""

    x = np.array(cfg.x_stations, dtype=float)
    re_x = cfg.u_inf * x / cfg.nu
    length_scale = np.sqrt(cfg.nu * x / cfg.u_inf)

    cf_similarity = (2.0 * profile.fpp0) / np.sqrt(re_x)
    cf_classic = 0.664 / np.sqrt(re_x)
    cf_rel_error = np.abs(cf_similarity - cf_classic) / cf_classic

    tau_w = 0.5 * cfg.rho * (cfg.u_inf**2) * cf_similarity
    delta_99 = metrics.eta_99 * length_scale
    delta_star = metrics.displacement_eta * length_scale
    theta = metrics.momentum_eta * length_scale
    shape_factor = delta_star / theta

    return pd.DataFrame(
        {
            "x_m": x,
            "Re_x": re_x,
            "Cf_similarity": cf_similarity,
            "Cf_classic": cf_classic,
            "Cf_rel_error": cf_rel_error,
            "tau_w_Pa": tau_w,
            "delta99_mm": 1.0e3 * delta_99,
            "delta_star_mm": 1.0e3 * delta_star,
            "theta_mm": 1.0e3 * theta,
            "shape_factor_H": shape_factor,
        }
    )


def run_checks(cfg: BoundaryLayerConfig, profile: BlasiusProfile, df: pd.DataFrame, metrics: SimilarityMetrics) -> None:
    """Physical and numerical sanity checks for automated validation."""

    if cfg.u_inf <= 0.0 or cfg.nu <= 0.0 or cfg.rho <= 0.0:
        raise AssertionError("u_inf, nu, rho must be positive.")

    if not np.all(np.isfinite(profile.fp)):
        raise AssertionError("Profile contains non-finite values.")

    # Boundary condition at infinity (truncated eta_max).
    if abs(profile.fp[-1] - 1.0) > 3.0e-6:
        raise AssertionError(f"f'(eta_max) not close to 1: {profile.fp[-1]:.8f}")

    # f' should be nonnegative and mostly monotonic increasing for Blasius.
    if np.min(profile.fp) < -1.0e-10:
        raise AssertionError("Velocity ratio f' became negative.")

    fp_diff = np.diff(profile.fp)
    if np.min(fp_diff) < -1.0e-5:
        raise AssertionError("Blasius profile lost monotonicity beyond tolerance.")

    # Known canonical value is f''(0) ~= 0.332057.
    if not (0.331 < profile.fpp0 < 0.3335):
        raise AssertionError(f"Unexpected f''(0) estimate: {profile.fpp0:.8f}")

    # Classic engineering correlation consistency.
    if float(df["Cf_rel_error"].max()) > 0.005:
        raise AssertionError("Cf similarity and classic correlation diverge too much.")

    # Shape factor for Blasius should be around 2.59.
    if not (2.55 < metrics.shape_factor < 2.63):
        raise AssertionError(f"Unexpected Blasius shape factor: H={metrics.shape_factor:.6f}")

    # Ensure laminar assumption remains reasonable (Re_x below common transition threshold).
    if float(df["Re_x"].max()) >= 5.0e5:
        raise AssertionError("x_stations include turbulent-prone Re_x >= 5e5.")


def main() -> None:
    cfg = BoundaryLayerConfig()
    profile = solve_blasius(cfg)
    metrics = compute_similarity_metrics(profile)
    df = build_station_table(cfg, profile, metrics)
    run_checks(cfg, profile, df, metrics)

    print("Boundary Layer Theory MVP (Blasius Flat Plate)")
    print(
        "config:",
        {
            "u_inf_m_s": cfg.u_inf,
            "nu_m2_s": cfg.nu,
            "rho_kg_m3": cfg.rho,
            "eta_max": cfg.eta_max,
            "n_eta": cfg.n_eta,
        },
    )
    print(
        "similarity_metrics:",
        {
            "fpp0": profile.fpp0,
            "eta_99": metrics.eta_99,
            "displacement_eta": metrics.displacement_eta,
            "momentum_eta": metrics.momentum_eta,
            "shape_factor_H": metrics.shape_factor,
        },
    )

    print("\nstation_table:")
    print(
        df.to_string(
            index=False,
            formatters={
                "x_m": lambda v: f"{v:.3f}",
                "Re_x": lambda v: f"{v:.1f}",
                "Cf_similarity": lambda v: f"{v:.6f}",
                "Cf_classic": lambda v: f"{v:.6f}",
                "Cf_rel_error": lambda v: f"{v:.3e}",
                "tau_w_Pa": lambda v: f"{v:.5f}",
                "delta99_mm": lambda v: f"{v:.4f}",
                "delta_star_mm": lambda v: f"{v:.4f}",
                "theta_mm": lambda v: f"{v:.4f}",
                "shape_factor_H": lambda v: f"{v:.5f}",
            },
        )
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
