"""Minimal MVP for cosmological Linear Perturbation Theory.

This script solves the linear growth equation for matter density contrast in
an FRW background:

    d^2D/d(ln a)^2 + [2 + d ln H/d ln a] dD/d(ln a) - 3/2 * Omega_m(a) * D = 0

where D(a) is the linear growth factor (delta ~ D).

We implement:
1) transparent ODE construction from background cosmology;
2) numerical integration with SciPy's solve_ivp;
3) growth-rate diagnostics f = d ln D / d ln a;
4) gamma-fit check f ~ Omega_m(a)^gamma in LambdaCDM;
5) EdS analytic sanity check D(a)=a.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CosmologyParams:
    """Background cosmology parameters at z=0."""

    name: str
    h0_km_s_mpc: float
    omega_m0: float
    omega_r0: float
    omega_lambda0: float
    omega_k0: float | None = None

    def with_derived_curvature(self) -> "CosmologyParams":
        if self.omega_k0 is None:
            omega_k0 = 1.0 - self.omega_m0 - self.omega_r0 - self.omega_lambda0
        else:
            omega_k0 = self.omega_k0
        return CosmologyParams(
            name=self.name,
            h0_km_s_mpc=self.h0_km_s_mpc,
            omega_m0=self.omega_m0,
            omega_r0=self.omega_r0,
            omega_lambda0=self.omega_lambda0,
            omega_k0=omega_k0,
        )


@dataclass(frozen=True)
class GrowthSolverConfig:
    """Numerical controls for integrating the growth ODE."""

    a_init: float = 1.0e-3
    n_eval: int = 800
    rtol: float = 1.0e-9
    atol: float = 1.0e-11


def validate_cosmology(p: CosmologyParams) -> None:
    if p.h0_km_s_mpc <= 0.0:
        raise ValueError("H0 must be positive.")
    for key, value in {
        "omega_m0": p.omega_m0,
        "omega_r0": p.omega_r0,
        "omega_lambda0": p.omega_lambda0,
        "omega_k0": p.omega_k0,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite.")

    a_test = np.geomspace(1.0e-4, 1.0, 400)
    e2 = e2_of_a(a_test, p)
    if np.any(e2 <= 0.0):
        raise ValueError("Encountered non-positive E(a)^2; parameters are not physical.")


def e2_of_a(a: np.ndarray | float, p: CosmologyParams) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    return (
        p.omega_r0 / a_arr**4
        + p.omega_m0 / a_arr**3
        + p.omega_k0 / a_arr**2
        + p.omega_lambda0
    )


def omega_m_of_a(a: np.ndarray | float, p: CosmologyParams) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    return (p.omega_m0 / a_arr**3) / e2_of_a(a_arr, p)


def dlnh_dlna(a: np.ndarray | float, p: CosmologyParams) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    numerator = (
        -4.0 * p.omega_r0 / a_arr**4
        -3.0 * p.omega_m0 / a_arr**3
        -2.0 * p.omega_k0 / a_arr**2
    )
    return 0.5 * numerator / e2_of_a(a_arr, p)


def growth_rhs(log_a: float, y: np.ndarray, p: CosmologyParams) -> np.ndarray:
    """Growth ODE in x = ln(a), with state y=[D, dD/dx]."""
    a = float(np.exp(log_a))
    d_val = float(y[0])
    d_prime = float(y[1])

    friction = 2.0 + float(dlnh_dlna(a, p))
    source = 1.5 * float(omega_m_of_a(a, p)) * d_val

    d2_val = -friction * d_prime + source
    return np.array([d_prime, d2_val], dtype=float)


def solve_linear_growth(
    p_raw: CosmologyParams,
    cfg: GrowthSolverConfig,
) -> pd.DataFrame:
    p = p_raw.with_derived_curvature()
    validate_cosmology(p)

    if not (0.0 < cfg.a_init < 1.0):
        raise ValueError("a_init must be in (0, 1).")
    if cfg.n_eval < 50:
        raise ValueError("n_eval must be >= 50.")

    log_a_start = float(np.log(cfg.a_init))
    log_a_end = 0.0
    log_a_grid = np.linspace(log_a_start, log_a_end, cfg.n_eval)

    # Matter-era initial condition: D ~ a and dD/dln(a) ~ D.
    y0 = np.array([cfg.a_init, cfg.a_init], dtype=float)

    sol = solve_ivp(
        fun=growth_rhs,
        t_span=(log_a_start, log_a_end),
        y0=y0,
        args=(p,),
        t_eval=log_a_grid,
        rtol=cfg.rtol,
        atol=cfg.atol,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Growth ODE integration failed: {sol.message}")

    a_grid = np.exp(log_a_grid)
    z_grid = 1.0 / a_grid - 1.0

    d_raw = sol.y[0]
    d_prime_raw = sol.y[1]

    # Normalize to D(a=1)=1 for convenient cosmology comparisons.
    norm = float(d_raw[-1])
    d_norm = d_raw / norm
    d_prime_norm = d_prime_raw / norm

    # f = d ln D / d ln a = (dD/dln a)/D.
    f_growth = d_prime_norm / d_norm
    omega_m_a = omega_m_of_a(a_grid, p)

    df = pd.DataFrame(
        {
            "z": z_grid,
            "a": a_grid,
            "D": d_norm,
            "dD_dln_a": d_prime_norm,
            "f": f_growth,
            "omega_m_a": omega_m_a,
        }
    ).sort_values("z").reset_index(drop=True)

    return df


def fit_growth_index_gamma(df: pd.DataFrame, z_max: float = 3.0) -> tuple[float, float]:
    """Fit f ~ Omega_m(a)^gamma in log space over 0 <= z <= z_max."""
    mask = (
        (df["z"] >= 0.0)
        & (df["z"] <= z_max)
        & (df["f"] > 0.0)
        & (df["omega_m_a"] > 0.0)
        & (df["omega_m_a"] < 0.995)
    )
    sub = df.loc[mask]
    if len(sub) < 20:
        raise RuntimeError("Not enough points for gamma fit.")

    x = np.log(sub["omega_m_a"].to_numpy())
    y = np.log(sub["f"].to_numpy())

    gamma = float(np.dot(x, y) / np.dot(x, x))
    f_fit = sub["omega_m_a"].to_numpy() ** gamma
    rms = float(np.sqrt(np.mean((sub["f"].to_numpy() - f_fit) ** 2)))
    return gamma, rms


def interp_from_df(df: pd.DataFrame, x_col: str, y_col: str, x_val: float) -> float:
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    return float(np.interp(x_val, x, y))


def run_eds_checks(df_eds: pd.DataFrame) -> dict[str, float]:
    """Einstein-de Sitter analytic benchmark: D(a)=a and f=1."""
    err_d = float(np.max(np.abs(df_eds["D"].to_numpy() - df_eds["a"].to_numpy())))
    err_f = float(np.max(np.abs(df_eds["f"].to_numpy() - 1.0)))

    assert err_d < 5.0e-4, f"EdS growth factor mismatch too large: {err_d}"
    assert err_f < 5.0e-4, f"EdS growth-rate mismatch too large: {err_f}"

    return {"eds_max_abs_D_error": err_d, "eds_max_abs_f_error": err_f}


def run_lcdm_checks(df_lcdm: pd.DataFrame, gamma: float, gamma_rms: float) -> dict[str, float]:
    df_by_a = df_lcdm.sort_values("a")
    d_values = df_by_a["D"].to_numpy()
    assert np.all(np.diff(d_values) >= 0.0), "D(a) must grow with scale factor a."

    f0 = interp_from_df(df_lcdm, "z", "f", 0.0)
    f1 = interp_from_df(df_lcdm, "z", "f", 1.0)
    f3 = interp_from_df(df_lcdm, "z", "f", 3.0)

    assert 0.45 < f0 < 0.65, f"Unexpected LambdaCDM f(z=0): {f0}"
    assert f3 > f1 > f0, "Growth rate should approach matter-era value at high z."
    assert 0.50 < gamma < 0.62, f"Fitted gamma out of expected range: {gamma}"
    assert gamma_rms < 0.03, f"Gamma fit RMS too large: {gamma_rms}"

    return {
        "f_z0": f0,
        "f_z1": f1,
        "f_z3": f3,
        "gamma_fit": gamma,
        "gamma_fit_rms": gamma_rms,
    }


def make_report_table(df_lcdm: pd.DataFrame, gamma: float) -> pd.DataFrame:
    z_nodes = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    rows: list[dict[str, float]] = []
    for z in z_nodes:
        d_val = interp_from_df(df_lcdm, "z", "D", z)
        f_val = interp_from_df(df_lcdm, "z", "f", z)
        om_val = interp_from_df(df_lcdm, "z", "omega_m_a", z)
        rows.append(
            {
                "z": z,
                "a": 1.0 / (1.0 + z),
                "D(z)": d_val,
                "f(z)": f_val,
                "Omega_m(a)": om_val,
                "Omega_m(a)^gamma": om_val**gamma,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = GrowthSolverConfig()

    lcdm = CosmologyParams(
        name="Flat-LambdaCDM",
        h0_km_s_mpc=67.4,
        omega_m0=0.315,
        omega_r0=9.0e-5,
        omega_lambda0=1.0 - 0.315 - 9.0e-5,
        omega_k0=0.0,
    )

    eds = CosmologyParams(
        name="Einstein-de Sitter",
        h0_km_s_mpc=67.4,
        omega_m0=1.0,
        omega_r0=0.0,
        omega_lambda0=0.0,
        omega_k0=0.0,
    )

    df_lcdm = solve_linear_growth(lcdm, cfg)
    df_eds = solve_linear_growth(eds, cfg)

    gamma, gamma_rms = fit_growth_index_gamma(df_lcdm, z_max=3.0)

    eds_metrics = run_eds_checks(df_eds)
    lcdm_metrics = run_lcdm_checks(df_lcdm, gamma, gamma_rms)
    report = make_report_table(df_lcdm, gamma)

    print("Linear Perturbation Theory MVP: scalar growth in background cosmology")
    print()
    print("Model: Flat LambdaCDM")
    with pd.option_context("display.precision", 6, "display.width", 180):
        print(report.to_string(index=False))

    print()
    print("Summary diagnostics:")
    print(
        f"  gamma fit (0<=z<=3): {lcdm_metrics['gamma_fit']:.6f} "
        f"(RMS={lcdm_metrics['gamma_fit_rms']:.6e})"
    )
    print(
        f"  f(z=0,1,3): {lcdm_metrics['f_z0']:.6f}, "
        f"{lcdm_metrics['f_z1']:.6f}, {lcdm_metrics['f_z3']:.6f}"
    )
    print(
        f"  EdS max errors: D(a)-a={eds_metrics['eds_max_abs_D_error']:.6e}, "
        f"f-1={eds_metrics['eds_max_abs_f_error']:.6e}"
    )
    print("Checks passed: ODE evolution, EdS benchmark, and LambdaCDM growth-index fit.")


if __name__ == "__main__":
    main()
