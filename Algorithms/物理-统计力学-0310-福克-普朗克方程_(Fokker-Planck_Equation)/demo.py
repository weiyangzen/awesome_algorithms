r"""Minimal runnable MVP for the 1D Fokker-Planck equation.

Model:
    dX_t = -k X_t dt + sqrt(2D) dW_t
which corresponds to the Fokker-Planck PDE
    \partial_t p = -\partial_x(A p) + D \partial_{xx} p,  A(x) = -k x.

The script solves the PDE with a conservative finite-volume scheme and
compares density/moments against the analytical Ornstein-Uhlenbeck solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate


@dataclass(frozen=True)
class FPConfig:
    """Configuration for the Fokker-Planck MVP."""

    x_min: float = -6.0
    x_max: float = 6.0
    nx: int = 501

    drift_rate_k: float = 1.0
    diffusion_d: float = 0.55

    mean0: float = 1.5
    var0: float = 0.18

    t_final: float = 1.2
    dt: float = 2.0e-4
    snapshot_times: tuple[float, ...] = (0.0, 0.3, 0.7, 1.2)

    mass_tol: float = 6.0e-3
    mean_tol: float = 4.5e-2
    var_tol: float = 5.5e-2
    final_l1_tol: float = 8.0e-2
    negativity_tol: float = 2.0e-6


def build_grid(cfg: FPConfig) -> tuple[np.ndarray, float]:
    """Create uniform spatial grid and return (x, dx)."""
    if cfg.nx < 5:
        raise ValueError("nx must be >= 5")
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float64)
    dx = float(x[1] - x[0])
    return x, dx


def gaussian_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Return Gaussian PDF evaluated on x."""
    if var <= 0.0:
        raise ValueError("variance must be positive")
    inv_std = 1.0 / np.sqrt(var)
    z = (x - mean) * inv_std
    return np.exp(-0.5 * z * z) * inv_std / np.sqrt(2.0 * np.pi)


def normalize_density(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize p(x) so integral is 1."""
    mass = float(integrate.trapezoid(p, x))
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("density mass must be finite and positive")
    return p / mass


def analytic_ou_moments(t: float, cfg: FPConfig) -> tuple[float, float]:
    """Return analytical mean and variance for OU process."""
    k = cfg.drift_rate_k
    d = cfg.diffusion_d
    mean_t = cfg.mean0 * np.exp(-k * t)
    var_inf = d / k
    var_t = var_inf + (cfg.var0 - var_inf) * np.exp(-2.0 * k * t)
    return float(mean_t), float(var_t)


def analytic_ou_pdf(x: np.ndarray, t: float, cfg: FPConfig) -> np.ndarray:
    """Return analytical OU density p(x,t)."""
    mean_t, var_t = analytic_ou_moments(t, cfg)
    return gaussian_pdf(x, mean_t, var_t)


def explicit_stability_bounds(x: np.ndarray, dx: float, cfg: FPConfig) -> dict[str, float]:
    """Return conservative explicit-step limits for advection+diffusion terms."""
    a = -cfg.drift_rate_k * x
    max_speed = float(np.max(np.abs(a)))
    dt_adv = np.inf if max_speed == 0.0 else dx / max_speed
    dt_diff = np.inf if cfg.diffusion_d == 0.0 else dx * dx / (2.0 * cfg.diffusion_d)
    dt_max = min(dt_adv, dt_diff)
    return {
        "dx": dx,
        "max_speed": max_speed,
        "dt_adv": float(dt_adv),
        "dt_diff": float(dt_diff),
        "dt_max": float(dt_max),
    }


def compute_flux(p: np.ndarray, x: np.ndarray, dx: float, cfg: FPConfig) -> np.ndarray:
    """Compute face flux F = A p - D * dp/dx with no-flux boundaries.

    We use upwind for A p and centered gradient for diffusion.
    """
    a = -cfg.drift_rate_k * x

    a_face = 0.5 * (a[:-1] + a[1:])
    p_left = p[:-1]
    p_right = p[1:]
    p_upwind = np.where(a_face >= 0.0, p_left, p_right)
    adv_flux = a_face * p_upwind

    grad_p = (p_right - p_left) / dx
    diff_flux = -cfg.diffusion_d * grad_p

    interior_flux = adv_flux + diff_flux

    flux = np.zeros(p.shape[0] + 1, dtype=np.float64)
    flux[1:-1] = interior_flux
    # No-flux boundaries: F(x_min)=F(x_max)=0
    flux[0] = 0.0
    flux[-1] = 0.0
    return flux


def step_fokker_planck(
    p: np.ndarray,
    x: np.ndarray,
    dx: float,
    dt: float,
    cfg: FPConfig,
) -> np.ndarray:
    """Advance one explicit finite-volume step."""
    flux = compute_flux(p, x, dx, cfg)
    div_flux = (flux[1:] - flux[:-1]) / dx
    return p - dt * div_flux


def density_moments(p: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Return mass, mean, variance, and min density."""
    mass = float(integrate.trapezoid(p, x))
    mean = float(integrate.trapezoid(x * p, x))
    centered = x - mean
    var = float(integrate.trapezoid(centered * centered * p, x))
    min_density = float(np.min(p))
    return mass, mean, var, min_density


def evaluate_snapshot(p: np.ndarray, x: np.ndarray, t: float, cfg: FPConfig) -> dict[str, float]:
    """Compute diagnostics against analytical OU solution at time t."""
    p_ref = analytic_ou_pdf(x, t, cfg)

    mass, mean, var, min_density = density_moments(p, x)
    mean_ref, var_ref = analytic_ou_moments(t, cfg)

    l1_err = float(integrate.trapezoid(np.abs(p - p_ref), x))
    l2_err = float(np.sqrt(integrate.trapezoid((p - p_ref) ** 2, x)))
    linf_err = float(np.max(np.abs(p - p_ref)))

    return {
        "t": float(t),
        "mass": mass,
        "mean_num": mean,
        "mean_ref": mean_ref,
        "mean_abs_err": abs(mean - mean_ref),
        "var_num": var,
        "var_ref": var_ref,
        "var_abs_err": abs(var - var_ref),
        "l1_err": l1_err,
        "l2_err": l2_err,
        "linf_err": linf_err,
        "min_density": min_density,
    }


def solve_fokker_planck(cfg: FPConfig) -> tuple[np.ndarray, float, dict[float, np.ndarray], pd.DataFrame, dict[str, float]]:
    """Solve PDE and return snapshots + diagnostics table."""
    x, dx = build_grid(cfg)

    stability = explicit_stability_bounds(x, dx, cfg)
    if cfg.dt >= 0.95 * stability["dt_max"]:
        raise ValueError(
            "Configured dt is too large for explicit stability margin: "
            f"dt={cfg.dt:.3e}, suggested < {0.95 * stability['dt_max']:.3e}"
        )

    p = gaussian_pdf(x, cfg.mean0, cfg.var0)
    p = normalize_density(p, x)

    n_steps = int(np.ceil(cfg.t_final / cfg.dt))
    dt = cfg.t_final / n_steps

    target_times = sorted(set(float(t) for t in cfg.snapshot_times if 0.0 <= t <= cfg.t_final + 1e-12))
    if not target_times or target_times[0] > 0.0:
        target_times.insert(0, 0.0)

    snapshots: dict[float, np.ndarray] = {}
    next_index = 0

    if abs(target_times[0] - 0.0) <= 1e-14:
        snapshots[target_times[0]] = p.copy()
        next_index = 1

    for step in range(1, n_steps + 1):
        p = step_fokker_planck(p, x, dx, dt, cfg)

        # Capture one or multiple target times crossed by current step.
        t_now = step * dt
        while next_index < len(target_times) and t_now + 0.5 * dt >= target_times[next_index]:
            snapshots[target_times[next_index]] = p.copy()
            next_index += 1

    # Ensure final state is captured.
    if cfg.t_final not in snapshots:
        snapshots[cfg.t_final] = p.copy()

    records = [evaluate_snapshot(snapshots[t], x, t, cfg) for t in sorted(snapshots.keys())]
    report_df = pd.DataFrame.from_records(records)
    return x, dt, snapshots, report_df, stability


def run_consistency_checks(report_df: pd.DataFrame, cfg: FPConfig) -> None:
    """Run sanity checks on conservation and analytical agreement."""
    max_mass_dev = float(np.max(np.abs(report_df["mass"] - 1.0)))
    max_mean_err = float(np.max(report_df["mean_abs_err"]))
    max_var_err = float(np.max(report_df["var_abs_err"]))
    min_density = float(np.min(report_df["min_density"]))

    final_row = report_df.loc[np.isclose(report_df["t"], cfg.t_final)].iloc[0]
    final_l1 = float(final_row["l1_err"])

    assert max_mass_dev < cfg.mass_tol, f"Mass drift too large: {max_mass_dev:.3e}"
    assert max_mean_err < cfg.mean_tol, f"Mean error too large: {max_mean_err:.3e}"
    assert max_var_err < cfg.var_tol, f"Variance error too large: {max_var_err:.3e}"
    assert final_l1 < cfg.final_l1_tol, f"Final L1 density error too large: {final_l1:.3e}"
    assert min_density > -cfg.negativity_tol, f"Density became too negative: min={min_density:.3e}"


def main() -> None:
    cfg = FPConfig()
    x, dt, snapshots, report_df, stability = solve_fokker_planck(cfg)
    run_consistency_checks(report_df, cfg)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("=== Fokker-Planck Equation MVP (1D Ornstein-Uhlenbeck) ===")
    print(f"grid: nx={cfg.nx}, x in [{cfg.x_min:.2f}, {cfg.x_max:.2f}], dx={stability['dx']:.5f}")
    print(f"time: t_final={cfg.t_final:.3f}, dt_used={dt:.3e}, snapshots={sorted(snapshots.keys())}")
    print(
        "stability bounds: "
        f"dt_adv={stability['dt_adv']:.3e}, dt_diff={stability['dt_diff']:.3e}, dt_max={stability['dt_max']:.3e}"
    )
    print()
    print(report_df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    final_row = report_df.loc[np.isclose(report_df["t"], cfg.t_final)].iloc[0]
    print("\nFinal snapshot summary:")
    print(f"mass        = {float(final_row['mass']):.6f}")
    print(f"mean error  = {float(final_row['mean_abs_err']):.6e}")
    print(f"var error   = {float(final_row['var_abs_err']):.6e}")
    print(f"L1 density  = {float(final_row['l1_err']):.6e}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
