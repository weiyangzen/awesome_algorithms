"""Minimal runnable MVP for bifurcation theory.

This script demonstrates three canonical bifurcation workflows:
1) Saddle-node bifurcation in a 1D ODE.
2) Supercritical pitchfork bifurcation in a 1D ODE.
3) First period-doubling onset in the logistic map.

It is intentionally small, deterministic, and non-interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class ContinuousModel:
    """Container for a 1D continuous-time normal-form model."""

    name: str
    vector_field: Callable[[float, float], float]
    equilibria: Callable[[float], Array]
    dfdx: Callable[[float, float], float]


def stability_label(dfdx_value: float, tol: float = 1e-10) -> str:
    """Classify fixed-point stability from the local linearization slope."""
    if dfdx_value < -tol:
        return "stable"
    if dfdx_value > tol:
        return "unstable"
    return "marginal"


def saddle_node_vector_field(x: float, mu: float) -> float:
    """Saddle-node normal form: x' = mu - x^2."""
    return mu - x * x


def saddle_node_equilibria(mu: float) -> Array:
    if mu < 0.0:
        return np.array([], dtype=float)
    root = np.sqrt(mu)
    if np.isclose(root, 0.0):
        return np.array([0.0], dtype=float)
    return np.array([-root, root], dtype=float)


def saddle_node_dfdx(x: float, mu: float) -> float:
    del mu
    return -2.0 * x


def pitchfork_vector_field(x: float, mu: float) -> float:
    """Supercritical pitchfork normal form: x' = mu*x - x^3."""
    return mu * x - x**3


def pitchfork_equilibria(mu: float) -> Array:
    if mu < 0.0:
        return np.array([0.0], dtype=float)
    root = np.sqrt(mu)
    if np.isclose(root, 0.0):
        return np.array([0.0], dtype=float)
    return np.array([-root, 0.0, root], dtype=float)


def pitchfork_dfdx(x: float, mu: float) -> float:
    return mu - 3.0 * x * x


def sweep_equilibria(model: ContinuousModel, mu_values: Array) -> pd.DataFrame:
    """Enumerate fixed points and local stability over a parameter grid."""
    rows: list[dict[str, float | str]] = []
    for mu in mu_values:
        eqs = model.equilibria(float(mu))
        if eqs.size == 0:
            rows.append(
                {
                    "model": model.name,
                    "mu": float(mu),
                    "x_eq": np.nan,
                    "dfdx": np.nan,
                    "stability": "none",
                }
            )
            continue

        for x_eq in eqs:
            slope = model.dfdx(float(x_eq), float(mu))
            rows.append(
                {
                    "model": model.name,
                    "mu": float(mu),
                    "x_eq": float(x_eq),
                    "dfdx": float(slope),
                    "stability": stability_label(float(slope)),
                }
            )

    return pd.DataFrame(rows)


def detect_count_change_bifurcation(df: pd.DataFrame, mu_values: Array) -> float:
    """Estimate bifurcation location from the largest equilibrium-count jump."""
    with_eq = df[df["stability"] != "none"]
    counts = with_eq.groupby("mu").size().reindex(mu_values, fill_value=0)
    diffs = counts.diff().abs().fillna(0)
    idx = int(np.argmax(diffs.to_numpy()))
    return float(mu_values[idx])


def simulate_1d_trajectory(
    vector_field: Callable[[float, float], float],
    mu: float,
    x0: float,
    dt: float,
    steps: int,
) -> Array:
    """Explicit-Euler integration for a scalar autonomous ODE."""
    traj = np.empty(steps + 1, dtype=float)
    traj[0] = x0
    x = x0
    for i in range(steps):
        x = x + dt * vector_field(x, mu)
        traj[i + 1] = x
    return traj


def logistic_map_step(r: float, x: float) -> float:
    return r * x * (1.0 - x)


def orbit_period_proxy(r: float, transient: int = 1500, keep: int = 256, tol: float = 1e-6) -> int:
    """Approximate attractor period by counting unique tail points with tolerance."""
    x = 0.123456789
    for _ in range(transient):
        x = logistic_map_step(r, x)

    tail = np.empty(keep, dtype=float)
    for i in range(keep):
        x = logistic_map_step(r, x)
        tail[i] = x

    sorted_tail = np.sort(tail)
    unique_count = 1
    last = sorted_tail[0]
    for value in sorted_tail[1:]:
        if abs(value - last) > tol:
            unique_count += 1
            last = value
    return unique_count


def detect_logistic_first_period_doubling(r_values: Array) -> tuple[float, pd.DataFrame]:
    """Find first r where orbit proxy leaves period-1 and reaches period-2."""
    period_counts = [orbit_period_proxy(float(r)) for r in r_values]
    df = pd.DataFrame({"r": r_values, "period_proxy": period_counts})

    bif_r = float("nan")
    for i in range(1, len(period_counts)):
        if period_counts[i - 1] <= 1 and period_counts[i] >= 2:
            bif_r = float(r_values[i])
            break

    return bif_r, df


def main() -> None:
    mu_values = np.linspace(-1.0, 1.0, 401)

    saddle_model = ContinuousModel(
        name="saddle-node",
        vector_field=saddle_node_vector_field,
        equilibria=saddle_node_equilibria,
        dfdx=saddle_node_dfdx,
    )
    pitchfork_model = ContinuousModel(
        name="pitchfork-supercritical",
        vector_field=pitchfork_vector_field,
        equilibria=pitchfork_equilibria,
        dfdx=pitchfork_dfdx,
    )

    saddle_df = sweep_equilibria(saddle_model, mu_values)
    pitchfork_df = sweep_equilibria(pitchfork_model, mu_values)

    saddle_mu_star = detect_count_change_bifurcation(saddle_df, mu_values)
    pitchfork_mu_star = detect_count_change_bifurcation(pitchfork_df, mu_values)

    saddle_traj = simulate_1d_trajectory(
        saddle_model.vector_field,
        mu=0.25,
        x0=0.9,
        dt=0.01,
        steps=6000,
    )
    pitchfork_traj_pos = simulate_1d_trajectory(
        pitchfork_model.vector_field,
        mu=0.16,
        x0=0.2,
        dt=0.01,
        steps=6000,
    )
    pitchfork_traj_neg = simulate_1d_trajectory(
        pitchfork_model.vector_field,
        mu=0.16,
        x0=-0.2,
        dt=0.01,
        steps=6000,
    )

    r_values = np.linspace(2.8, 3.2, 801)
    logistic_bif_r, logistic_df = detect_logistic_first_period_doubling(r_values)

    # --- Deterministic validation checks ---
    expected_saddle_fixed = 0.5
    expected_pitchfork_fixed = 0.4

    assert abs(saddle_mu_star) <= 0.01, f"Saddle-node bifurcation estimate too far: {saddle_mu_star}"
    assert abs(pitchfork_mu_star) <= 0.01, f"Pitchfork bifurcation estimate too far: {pitchfork_mu_star}"

    assert abs(saddle_traj[-1] - expected_saddle_fixed) < 0.03
    assert abs(pitchfork_traj_pos[-1] - expected_pitchfork_fixed) < 0.03
    assert abs(pitchfork_traj_neg[-1] + expected_pitchfork_fixed) < 0.03

    assert stability_label(pitchfork_dfdx(0.0, -0.2)) == "stable"
    assert stability_label(pitchfork_dfdx(0.0, 0.2)) == "unstable"

    assert 2.95 <= logistic_bif_r <= 3.05, f"Unexpected logistic first bifurcation: {logistic_bif_r}"

    # --- Compact report ---
    print("=== Bifurcation Theory MVP ===")
    print(f"Saddle-node bifurcation estimate mu*: {saddle_mu_star:.4f}")
    print(f"Pitchfork bifurcation estimate mu*:   {pitchfork_mu_star:.4f}")
    print(f"Saddle trajectory final x:            {saddle_traj[-1]:.6f}")
    print(f"Pitchfork (+) final x:               {pitchfork_traj_pos[-1]:.6f}")
    print(f"Pitchfork (-) final x:               {pitchfork_traj_neg[-1]:.6f}")
    print(f"Logistic first period-doubling r*:   {logistic_bif_r:.4f}")

    def eq_count_at(df: pd.DataFrame, mu_target: float) -> int:
        mask = (df["mu"] == mu_target) & (df["stability"] != "none")
        return int(mask.sum())

    summary_rows = [
        {
            "model": "saddle-node",
            "mu=-0.5 eq_count": eq_count_at(saddle_df, -0.5),
            "mu=+0.5 eq_count": eq_count_at(saddle_df, 0.5),
        },
        {
            "model": "pitchfork-supercritical",
            "mu=-0.5 eq_count": eq_count_at(pitchfork_df, -0.5),
            "mu=+0.5 eq_count": eq_count_at(pitchfork_df, 0.5),
        },
    ]
    print("\nEquilibrium count snapshot:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nLogistic period proxy near r=3:")
    near_three = logistic_df[(logistic_df["r"] >= 2.98) & (logistic_df["r"] <= 3.02)]
    print(near_three.iloc[::20].to_string(index=False))


if __name__ == "__main__":
    main()
