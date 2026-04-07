"""Maxwell relations: minimal runnable numerical verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


ScalarField2D = Callable[[float, float], float]


@dataclass(frozen=True)
class MaxwellConfig:
    """Configuration for finite-difference verification."""

    h: float = 1e-6
    tolerance: float = 1e-7
    sample_points: int = 7


def partial_x(func: ScalarField2D, x: float, y: float, h: float) -> float:
    """First partial derivative with respect to the first argument."""
    return (func(x + h, y) - func(x - h, y)) / (2.0 * h)


def partial_y(func: ScalarField2D, x: float, y: float, h: float) -> float:
    """First partial derivative with respect to the second argument."""
    return (func(x, y + h) - func(x, y - h)) / (2.0 * h)


# ---------- Thermodynamic potentials ----------
# U(S, V)
def internal_energy(s: float, v: float) -> float:
    return 1.30 * s**2 + 0.90 * v**2 + 0.42 * s * v + 0.70 * np.log(v)


def temperature_from_u(s: float, v: float) -> float:
    # T = (∂U/∂S)_V
    return 2.60 * s + 0.42 * v


def pressure_from_u(s: float, v: float) -> float:
    # P = -(∂U/∂V)_S
    return -(1.80 * v + 0.42 * s + 0.70 / v)


# H(S, P)
def enthalpy(s: float, p: float) -> float:
    return 1.10 * s**2 + 0.55 * p**2 + 0.33 * s * p + 0.80 * np.log(p)


def temperature_from_h(s: float, p: float) -> float:
    # T = (∂H/∂S)_P
    return 2.20 * s + 0.33 * p


def volume_from_h(s: float, p: float) -> float:
    # V = (∂H/∂P)_S
    return 1.10 * p + 0.33 * s + 0.80 / p


# F(T, V)
def helmholtz_free_energy(t: float, v: float) -> float:
    return 0.06 * t**2 + 0.75 * v**2 + 0.31 * t * v + 0.65 * np.log(v)


def entropy_from_f(t: float, v: float) -> float:
    # S = -(∂F/∂T)_V
    return -(0.12 * t + 0.31 * v)


def pressure_from_f(t: float, v: float) -> float:
    # P = -(∂F/∂V)_T
    return -(1.50 * v + 0.31 * t + 0.65 / v)


# G(T, P)
def gibbs_free_energy(t: float, p: float) -> float:
    return 0.05 * t**2 + 0.48 * p**2 + 0.29 * t * p + 0.72 * np.log(p)


def entropy_from_g(t: float, p: float) -> float:
    # S = -(∂G/∂T)_P
    return -(0.10 * t + 0.29 * p)


def volume_from_g(t: float, p: float) -> float:
    # V = (∂G/∂P)_T
    return 0.96 * p + 0.29 * t + 0.72 / p


def evaluate_relation(
    name: str,
    x_name: str,
    y_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    lhs: ScalarField2D,
    rhs: ScalarField2D,
    cfg: MaxwellConfig,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    """Evaluate one Maxwell relation on a 2D grid."""
    rows: list[dict[str, float | str]] = []
    residuals: list[float] = []

    for x in x_values:
        for y in y_values:
            lhs_val = lhs(float(x), float(y))
            rhs_val = rhs(float(x), float(y))
            residual = lhs_val - rhs_val
            residuals.append(residual)
            rows.append(
                {
                    "relation": name,
                    "x_name": x_name,
                    "x_value": float(x),
                    "y_name": y_name,
                    "y_value": float(y),
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "residual": residual,
                }
            )

    residual_arr = np.asarray(residuals, dtype=np.float64)
    summary = {
        "relation": name,
        "max_abs_residual": float(np.max(np.abs(residual_arr))),
        "mean_abs_residual": float(np.mean(np.abs(residual_arr))),
        "passed": bool(np.max(np.abs(residual_arr)) <= cfg.tolerance),
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    cfg = MaxwellConfig(h=1e-6, tolerance=1e-7, sample_points=7)

    s_values = np.linspace(0.8, 2.0, cfg.sample_points)
    v_values = np.linspace(1.1, 2.3, cfg.sample_points)
    t_values = np.linspace(280.0, 360.0, cfg.sample_points)
    p_values = np.linspace(1.0, 2.2, cfg.sample_points)

    relation_tasks = [
        (
            "(∂T/∂V)_S = -(∂P/∂S)_V  [from U(S,V)]",
            "S",
            "V",
            s_values,
            v_values,
            lambda s, v: partial_y(temperature_from_u, s, v, cfg.h),
            lambda s, v: -partial_x(pressure_from_u, s, v, cfg.h),
        ),
        (
            "(∂T/∂P)_S = (∂V/∂S)_P  [from H(S,P)]",
            "S",
            "P",
            s_values,
            p_values,
            lambda s, p: partial_y(temperature_from_h, s, p, cfg.h),
            lambda s, p: partial_x(volume_from_h, s, p, cfg.h),
        ),
        (
            "(∂S/∂V)_T = (∂P/∂T)_V  [from F(T,V)]",
            "T",
            "V",
            t_values,
            v_values,
            lambda t, v: partial_y(entropy_from_f, t, v, cfg.h),
            lambda t, v: partial_x(pressure_from_f, t, v, cfg.h),
        ),
        (
            "(∂S/∂P)_T = -(∂V/∂T)_P  [from G(T,P)]",
            "T",
            "P",
            t_values,
            p_values,
            lambda t, p: partial_y(entropy_from_g, t, p, cfg.h),
            lambda t, p: -partial_x(volume_from_g, t, p, cfg.h),
        ),
    ]

    summaries: list[dict[str, float | str | bool]] = []
    all_details: list[pd.DataFrame] = []

    for task in relation_tasks:
        summary, details = evaluate_relation(*task, cfg=cfg)
        summaries.append(summary)
        all_details.append(details)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(all_details, ignore_index=True)

    print("Maxwell relation verification summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\nSample point checks (first 12 rows):")
    print(detail_df.head(12).to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    assert summary_df["passed"].all(), "At least one Maxwell relation check failed."
    assert (summary_df["max_abs_residual"] < cfg.tolerance).all()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
