"""Population Inversion MVP (Three-level laser rate equations).

This script models a pumped three-level medium with explicit rate equations:
- Level 1: ground / lower laser level
- Level 2: upper laser level
- Level 3: pump band

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ThreeLevelLaserParams:
    """Physical parameters for the three-level rate model."""

    n_total: float = 1.0
    tau32: float = 5e-7
    tau21: float = 1e-3
    stim_rate: float = 450.0
    pump_rate: float = 1200.0


def validate_params(params: ThreeLevelLaserParams) -> None:
    if params.n_total <= 0.0:
        raise ValueError("n_total must be > 0.")
    if params.tau32 <= 0.0 or params.tau21 <= 0.0:
        raise ValueError("tau32 and tau21 must be > 0.")
    if params.stim_rate < 0.0:
        raise ValueError("stim_rate must be >= 0.")
    if params.pump_rate < 0.0:
        raise ValueError("pump_rate must be >= 0.")


def rate_equations(t: float, y: np.ndarray, params: ThreeLevelLaserParams) -> np.ndarray:
    """Three-level rate equations.

    y = [N1, N2, N3]
    N1: lower laser level (ground)
    N2: upper laser level
    N3: pump level
    """

    del t  # autonomous ODE
    n1, n2, n3 = y

    if not np.all(np.isfinite(y)):
        raise ValueError("State vector contains non-finite values.")

    w_p = params.pump_rate
    w_32 = 1.0 / params.tau32
    w_21 = 1.0 / params.tau21
    w_st = params.stim_rate

    inversion = n2 - n1

    dn3 = w_p * n1 - w_32 * n3
    dn2 = w_32 * n3 - w_21 * n2 - w_st * inversion
    dn1 = -w_p * n1 + w_21 * n2 + w_st * inversion

    return np.array([dn1, dn2, dn3], dtype=float)


def simulate_population(
    params: ThreeLevelLaserParams,
    t_end: float = 0.02,
    num_points: int = 1200,
) -> pd.DataFrame:
    validate_params(params)
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0.")
    if num_points < 20:
        raise ValueError("num_points must be >= 20.")

    y0 = np.array([params.n_total, 0.0, 0.0], dtype=float)
    t_eval = np.linspace(0.0, t_end, num_points)

    solution = solve_ivp(
        fun=lambda t, y: rate_equations(t, y, params),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")

    n1 = solution.y[0]
    n2 = solution.y[1]
    n3 = solution.y[2]
    total = n1 + n2 + n3
    inversion = n2 - n1

    return pd.DataFrame(
        {
            "t": solution.t,
            "N1": n1,
            "N2": n2,
            "N3": n3,
            "N_total": total,
            "inversion": inversion,
        }
    )


def summarize_trajectory(df: pd.DataFrame, steady_fraction: float = 0.2) -> Dict[str, float]:
    if df.empty:
        raise ValueError("Trajectory DataFrame is empty.")
    if not (0.0 < steady_fraction <= 0.5):
        raise ValueError("steady_fraction must be in (0, 0.5].")

    start = int((1.0 - steady_fraction) * len(df))
    tail = df.iloc[start:]

    total_ref = float(df["N_total"].iloc[0])
    conservation_err = float(np.max(np.abs(df["N_total"].to_numpy() - total_ref)))

    return {
        "inversion_final": float(df["inversion"].iloc[-1]),
        "inversion_max": float(df["inversion"].max()),
        "inversion_ss": float(tail["inversion"].mean()),
        "n1_ss": float(tail["N1"].mean()),
        "n2_ss": float(tail["N2"].mean()),
        "n3_ss": float(tail["N3"].mean()),
        "conservation_max_error": conservation_err,
    }


def run_pump_sweep(
    base_params: ThreeLevelLaserParams,
    pump_rates: np.ndarray,
    t_end: float = 0.02,
    num_points: int = 800,
) -> pd.DataFrame:
    rows = []
    for pump in pump_rates:
        p = replace(base_params, pump_rate=float(pump))
        df = simulate_population(p, t_end=t_end, num_points=num_points)
        stats = summarize_trajectory(df)
        rows.append(
            {
                "pump_rate": float(pump),
                "inversion_ss": stats["inversion_ss"],
                "inversion_max": stats["inversion_max"],
                "conservation_max_error": stats["conservation_max_error"],
                "is_inverted": bool(stats["inversion_ss"] > 0.0),
            }
        )
    return pd.DataFrame(rows)


def estimate_threshold_linear(sweep_df: pd.DataFrame) -> float:
    """Estimate threshold pump by local linear regression around inversion ~ 0."""

    if sweep_df.empty:
        return float("nan")

    near = sweep_df.iloc[(sweep_df["inversion_ss"].abs().argsort())[:5]].copy()
    x = near[["pump_rate"]].to_numpy(dtype=float)
    y = near["inversion_ss"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if abs(slope) < 1e-15:
        return float("nan")
    return -intercept / slope


def first_positive_threshold(sweep_df: pd.DataFrame) -> float:
    positive = sweep_df[sweep_df["inversion_ss"] > 0.0]
    if positive.empty:
        return float("nan")
    return float(positive["pump_rate"].iloc[0])


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    base = ThreeLevelLaserParams(
        n_total=1.0,
        tau32=5e-7,
        tau21=1e-3,
        stim_rate=450.0,
        pump_rate=1400.0,
    )

    # Representative trajectory
    traj = simulate_population(base, t_end=0.02, num_points=1200)
    summary = summarize_trajectory(traj)

    # Pump scan for inversion threshold
    pump_rates = np.linspace(0.0, 2200.0, 12)
    sweep = run_pump_sweep(base, pump_rates=pump_rates, t_end=0.02, num_points=900)

    analytic_threshold = 1.0 / base.tau21
    coarse_threshold = first_positive_threshold(sweep)
    linear_threshold = estimate_threshold_linear(sweep)

    sample_idx = np.linspace(0, len(traj) - 1, 12, dtype=int)
    traj_sample = traj.iloc[sample_idx].copy()

    print("Population Inversion MVP (Three-level Laser)")
    print(
        "params:"
        f" tau32={base.tau32:.2e}s,"
        f" tau21={base.tau21:.2e}s,"
        f" stim_rate={base.stim_rate:.1f}s^-1,"
        f" pump_rate={base.pump_rate:.1f}s^-1"
    )
    print()

    print("Representative trajectory (sampled rows):")
    print(traj_sample.round(6).to_string(index=False))
    print()

    print("Representative steady-state summary:")
    for k, v in summary.items():
        print(f"- {k}={v:.6e}")
    print()

    print("Pump sweep:")
    print(sweep.round(6).to_string(index=False))
    print()

    print("Threshold estimates:")
    print(f"analytic_threshold(1/tau21)={analytic_threshold:.6f} s^-1")
    print(f"coarse_numerical_threshold={coarse_threshold:.6f} s^-1")
    print(f"linearized_threshold={linear_threshold:.6f} s^-1")
    print()

    step = float(pump_rates[1] - pump_rates[0])
    checks = {
        "trajectory_finite": bool(np.all(np.isfinite(traj.to_numpy()))),
        "population_nonnegative": bool(traj[["N1", "N2", "N3"]].to_numpy().min() >= -1e-9),
        "population_conserved": bool(summary["conservation_max_error"] < 1e-8),
        "low_pump_not_inverted": bool(float(sweep.iloc[0]["inversion_ss"]) < 0.0),
        "high_pump_inverted": bool(float(sweep.iloc[-1]["inversion_ss"]) > 0.0),
        "coarse_threshold_close_to_theory": bool(abs(coarse_threshold - analytic_threshold) <= step),
        "linear_threshold_close_to_theory": bool(abs(linear_threshold - analytic_threshold) <= 0.35 * step),
        "representative_run_is_inverted": bool(summary["inversion_ss"] > 0.0),
    }

    print("Checks:")
    for key, value in checks.items():
        print(f"- {key}={value}")
    print(f"all_core_checks_pass={all(checks.values())}")


if __name__ == "__main__":
    main()
