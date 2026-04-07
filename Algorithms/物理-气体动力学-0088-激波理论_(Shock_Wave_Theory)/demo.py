"""Shock Wave Theory MVP: normal-shock relations for a perfect gas.

This script is intentionally small and explicit:
- Closed-form Rankine-Hugoniot jump relations for a normal shock
- Optional inverse problem solved with scipy.root_scalar
- Batched table generation with numpy + pandas
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


def _validate_inputs(mach_1: np.ndarray | float, gamma: float) -> None:
    """Validate physical domain: gamma > 1 and upstream Mach > 1 for shocks."""
    if gamma <= 1.0:
        raise ValueError(f"gamma must be > 1, got {gamma}")

    m = np.asarray(mach_1, dtype=float)
    if not np.all(np.isfinite(m)):
        raise ValueError("mach_1 contains non-finite values")
    if np.any(m <= 1.0):
        raise ValueError("normal shock requires mach_1 > 1")


def _p0_over_p(mach: np.ndarray | float, gamma: float) -> np.ndarray:
    """Isentropic stagnation-to-static pressure ratio p0/p."""
    a = 0.5 * (gamma - 1.0)
    mach_arr = np.asarray(mach, dtype=float)
    return np.power(1.0 + a * mach_arr * mach_arr, gamma / (gamma - 1.0))


def normal_shock_relations(mach_1: np.ndarray | float, gamma: float = 1.4) -> dict[str, np.ndarray]:
    """Compute classical normal-shock jump relations for a perfect gas.

    Returns a dictionary of numpy arrays (or scalars promoted to arrays).
    """
    _validate_inputs(mach_1, gamma)

    m1 = np.asarray(mach_1, dtype=float)
    m1_sq = m1 * m1

    p2_p1 = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (m1_sq - 1.0)
    rho2_rho1 = ((gamma + 1.0) * m1_sq) / ((gamma - 1.0) * m1_sq + 2.0)
    t2_t1 = p2_p1 / rho2_rho1

    m2_sq_num = 1.0 + 0.5 * (gamma - 1.0) * m1_sq
    m2_sq_den = gamma * m1_sq - 0.5 * (gamma - 1.0)
    m2 = np.sqrt(m2_sq_num / m2_sq_den)

    # For a stationary 1D normal shock in constant area, u2/u1 = rho1/rho2.
    u2_u1 = 1.0 / rho2_rho1

    p02_p01 = p2_p1 * (_p0_over_p(m2, gamma) / _p0_over_p(m1, gamma))
    ds_over_R = np.log(p2_p1) - gamma * np.log(rho2_rho1)

    return {
        "mach_1": m1,
        "mach_2": m2,
        "p2_p1": p2_p1,
        "rho2_rho1": rho2_rho1,
        "t2_t1": t2_t1,
        "u2_u1": u2_u1,
        "p02_p01": p02_p01,
        "ds_over_R": ds_over_R,
    }


def infer_mach_from_pressure_ratio(p2_p1_target: float, gamma: float = 1.4) -> float:
    """Recover upstream Mach number from target static pressure jump via root finding."""
    if p2_p1_target <= 1.0:
        raise ValueError("p2/p1 must be > 1 for a compression shock")

    def objective(m: float) -> float:
        relation = normal_shock_relations(mach_1=np.array([m]), gamma=gamma)
        return float(relation["p2_p1"][0] - p2_p1_target)

    sol = root_scalar(objective, bracket=(1.000001, 20.0), method="brentq")
    if not sol.converged:
        raise RuntimeError("Mach inversion failed to converge")
    return float(sol.root)


def build_table(mach_grid: np.ndarray, gamma: float = 1.4) -> pd.DataFrame:
    """Build a tabular sweep over upstream Mach numbers."""
    rel = normal_shock_relations(mach_grid, gamma=gamma)
    return pd.DataFrame(
        {
            "Mach1": rel["mach_1"],
            "Mach2": rel["mach_2"],
            "p2/p1": rel["p2_p1"],
            "rho2/rho1": rel["rho2_rho1"],
            "T2/T1": rel["t2_t1"],
            "u2/u1": rel["u2_u1"],
            "p02/p01": rel["p02_p01"],
            "ds/R": rel["ds_over_R"],
        }
    )


def summarize(table: pd.DataFrame) -> dict[str, float]:
    """Compute lightweight diagnostics to verify physically consistent trends."""
    diagnostics = {
        "all_m2_subsonic": float((table["Mach2"] < 1.0).all()),
        "all_pressure_increase": float((table["p2/p1"] > 1.0).all()),
        "all_density_increase": float((table["rho2/rho1"] > 1.0).all()),
        "all_total_pressure_drop": float((table["p02/p01"] < 1.0).all()),
        "min_entropy_jump": float(table["ds/R"].min()),
        "max_entropy_jump": float(table["ds/R"].max()),
    }
    return diagnostics


def main() -> None:
    gamma = 1.4
    mach_grid = np.linspace(1.05, 5.0, 40)

    table = build_table(mach_grid, gamma=gamma)

    # Inverse check: pick one case and recover M1 from p2/p1.
    sample_row = table.iloc[20]
    recovered_m1 = infer_mach_from_pressure_ratio(float(sample_row["p2/p1"]), gamma=gamma)

    out_path = Path(__file__).with_name("shock_table.csv")
    table.to_csv(out_path, index=False)

    diag = summarize(table)
    print("=== Shock Wave Theory MVP (Normal Shock) ===")
    print(f"gamma = {gamma}")
    print(f"Mach range = [{mach_grid.min():.2f}, {mach_grid.max():.2f}], samples = {len(mach_grid)}")
    print(f"saved table: {out_path.name}")
    print(
        "inverse check: "
        f"target p2/p1={sample_row['p2/p1']:.6f}, "
        f"Mach1_true={sample_row['Mach1']:.6f}, "
        f"Mach1_recovered={recovered_m1:.6f}"
    )
    print("diagnostics:")
    for key, value in diag.items():
        if key.startswith("all_"):
            print(f"- {key}: {bool(value)}")
        else:
            print(f"- {key}: {value:.6e}")


if __name__ == "__main__":
    main()
