"""Minimal runnable MVP for Bernoulli's Equation (incompressible flow).

The script models a horizontal/vertical two-section pipe (Venturi-style):
1) Use continuity A1*v1 = A2*v2.
2) Use generalized Bernoulli with a lumped local-loss coefficient K.
3) Solve downstream velocity and volumetric flow rate from pressure drop.
4) Sweep downstream pressure and verify physically expected trends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BernoulliConfig:
    rho: float = 998.2  # kg/m^3 (water near 20 C)
    g: float = 9.81  # m/s^2
    area_upstream: float = 0.020  # m^2
    area_downstream: float = 0.005  # m^2
    z_upstream: float = 1.20  # m
    z_downstream: float = 0.90  # m
    loss_k: float = 0.18  # dimensionless local loss term: h_loss = K*v2^2/(2g)
    p_upstream: float = 220_000.0  # Pa
    p_downstream_ref: float = 180_000.0  # Pa
    p_downstream_min: float = 150_000.0  # Pa
    p_downstream_max: float = 210_000.0  # Pa
    n_sweep_points: int = 9


def solve_from_pressure_pair(cfg: BernoulliConfig, p_downstream: float) -> tuple[float, float, float]:
    """Solve (v1, v2, Q) from p1, p2 using Bernoulli + continuity.

    Derived formula:
      v2 = sqrt( 2 * [ (p1-p2) + rho*g*(z1-z2) ] / (rho * [1 - alpha^2 + K]) )
      alpha = A2/A1, v1 = alpha*v2, Q = A2*v2
    """
    alpha = cfg.area_downstream / cfg.area_upstream
    driving_term = (cfg.p_upstream - p_downstream) + cfg.rho * cfg.g * (cfg.z_upstream - cfg.z_downstream)
    denominator = cfg.rho * (1.0 - alpha * alpha + cfg.loss_k)

    if denominator <= 0.0:
        raise ValueError("Non-positive denominator; check area ratio or loss_k.")
    if driving_term <= 0.0:
        raise ValueError("Non-positive driving term; pressure/head is insufficient for forward flow.")

    v_down = math.sqrt(2.0 * driving_term / denominator)
    v_up = alpha * v_down
    flow_rate = cfg.area_downstream * v_down
    return v_up, v_down, flow_rate


def bernoulli_residual_pa(cfg: BernoulliConfig, p_downstream: float, v_up: float, v_down: float) -> float:
    """Energy residual in pressure unit (Pa). Ideally near zero."""
    lhs = cfg.p_upstream + 0.5 * cfg.rho * v_up * v_up + cfg.rho * cfg.g * cfg.z_upstream
    rhs = (
        p_downstream
        + 0.5 * cfg.rho * v_down * v_down
        + cfg.rho * cfg.g * cfg.z_downstream
        + 0.5 * cfg.rho * cfg.loss_k * v_down * v_down
    )
    return lhs - rhs


def build_pressure_sweep(cfg: BernoulliConfig) -> pd.DataFrame:
    """Sweep p2 and return solved hydraulic states."""
    p2_values = np.linspace(cfg.p_downstream_max, cfg.p_downstream_min, cfg.n_sweep_points)
    rows: list[dict[str, float]] = []

    for p2 in p2_values:
        p2_float = float(p2)
        v1, v2, q = solve_from_pressure_pair(cfg, p2_float)
        residual = bernoulli_residual_pa(cfg, p2_float, v1, v2)
        continuity_error = cfg.area_upstream * v1 - cfg.area_downstream * v2

        rows.append(
            {
                "p_downstream_pa": p2_float,
                "delta_p_pa": cfg.p_upstream - p2_float,
                "v_upstream_mps": v1,
                "v_downstream_mps": v2,
                "flow_rate_m3s": q,
                "flow_rate_Lps": 1000.0 * q,
                "continuity_error_m3s": continuity_error,
                "energy_residual_pa": residual,
            }
        )

    return pd.DataFrame(rows).sort_values("delta_p_pa", ignore_index=True)


def main() -> None:
    cfg = BernoulliConfig()
    v1_ref, v2_ref, q_ref = solve_from_pressure_pair(cfg, cfg.p_downstream_ref)
    residual_ref = bernoulli_residual_pa(cfg, cfg.p_downstream_ref, v1_ref, v2_ref)

    sweep_df = build_pressure_sweep(cfg)

    cfg_lossless = replace(cfg, loss_k=0.0)
    _, _, q_ref_lossless = solve_from_pressure_pair(cfg_lossless, cfg.p_downstream_ref)

    print("=== Bernoulli Equation MVP (Incompressible Venturi-style Flow) ===")
    print(
        "config:",
        f"rho={cfg.rho:.1f} kg/m^3, g={cfg.g:.2f} m/s^2, "
        f"A1={cfg.area_upstream:.4f} m^2, A2={cfg.area_downstream:.4f} m^2, K={cfg.loss_k:.3f}"
    )
    print(
        "reference case:",
        f"p1={cfg.p_upstream:.1f} Pa, p2={cfg.p_downstream_ref:.1f} Pa, "
        f"v1={v1_ref:.5f} m/s, v2={v2_ref:.5f} m/s, Q={q_ref:.6f} m^3/s ({q_ref * 1000.0:.3f} L/s)"
    )
    print(f"reference energy residual = {residual_ref:.6e} Pa")
    print()

    with pd.option_context("display.width", 220, "display.precision", 6):
        print(sweep_df.to_string(index=False))
    print()

    assert abs(residual_ref) < 1e-6
    assert float(np.max(np.abs(sweep_df["energy_residual_pa"].to_numpy(dtype=float)))) < 1e-6
    assert float(np.max(np.abs(sweep_df["continuity_error_m3s"].to_numpy(dtype=float)))) < 1e-12

    q_values = sweep_df["flow_rate_m3s"].to_numpy(dtype=float)
    assert np.all(np.diff(q_values) > 0.0)
    assert q_ref < q_ref_lossless

    print("Checks passed: continuity, Bernoulli energy balance, monotonic pressure-flow relation, and loss effect.")


if __name__ == "__main__":
    main()
