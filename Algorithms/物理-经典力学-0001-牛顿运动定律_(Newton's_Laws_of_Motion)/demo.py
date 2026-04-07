"""Minimal runnable MVP for Newton's Laws of Motion.

This script provides three deterministic numerical experiments:
1) First law   : zero net force -> constant velocity.
2) Second law  : measured acceleration matches F / m.
3) Third law   : interaction forces are equal/opposite and total momentum is conserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


def simulate_first_law(v0: float, dt: float, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single particle with zero net force in 1D."""
    t = np.linspace(0.0, steps * dt, steps + 1)
    x = np.zeros(steps + 1, dtype=np.float64)
    v = np.zeros(steps + 1, dtype=np.float64)
    v[0] = float(v0)

    for i in range(steps):
        a = 0.0
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt

    return t, x, v


def evaluate_first_law(v: np.ndarray) -> dict[str, float]:
    v0 = float(v[0])
    max_velocity_deviation = float(np.max(np.abs(v - v0)))
    return {
        "v0": v0,
        "max_velocity_deviation": max_velocity_deviation,
    }


def simulate_second_law(
    mass: float,
    force: float,
    v0: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 1D motion under constant force."""
    t = np.linspace(0.0, steps * dt, steps + 1)
    x = np.zeros(steps + 1, dtype=np.float64)
    v = np.zeros(steps + 1, dtype=np.float64)
    v[0] = float(v0)

    for i in range(steps):
        a = force / mass
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt

    return t, x, v


def evaluate_second_law(t: np.ndarray, v: np.ndarray, mass: float, force: float) -> dict[str, float]:
    regression = linregress(t, v)
    a_hat = float(regression.slope)
    a_theory = float(force / mass)
    return {
        "a_theory": a_theory,
        "a_estimated": a_hat,
        "a_abs_error": float(abs(a_hat - a_theory)),
        "fit_rvalue": float(regression.rvalue),
    }


def simulate_third_law(
    m1: float,
    m2: float,
    k: float,
    rest_length: float,
    x1_0: float,
    x2_0: float,
    v1_0: float,
    v2_0: float,
    dt: float,
    steps: int,
) -> dict[str, np.ndarray]:
    """Two-body spring interaction in 1D.

    force_12 acts on body 1 due to body 2;
    force_21 is exactly -force_12 by construction.
    """
    x1 = np.zeros(steps + 1, dtype=np.float64)
    x2 = np.zeros(steps + 1, dtype=np.float64)
    v1 = np.zeros(steps + 1, dtype=np.float64)
    v2 = np.zeros(steps + 1, dtype=np.float64)
    f12 = np.zeros(steps + 1, dtype=np.float64)
    f21 = np.zeros(steps + 1, dtype=np.float64)
    p = np.zeros(steps + 1, dtype=np.float64)

    x1[0], x2[0] = float(x1_0), float(x2_0)
    v1[0], v2[0] = float(v1_0), float(v2_0)
    p[0] = m1 * v1[0] + m2 * v2[0]

    for i in range(steps):
        extension = (x2[i] - x1[i]) - rest_length
        force_12 = k * extension
        force_21 = -force_12

        a1 = force_12 / m1
        a2 = force_21 / m2

        v1[i + 1] = v1[i] + a1 * dt
        v2[i + 1] = v2[i] + a2 * dt
        x1[i + 1] = x1[i] + v1[i + 1] * dt
        x2[i + 1] = x2[i] + v2[i + 1] * dt

        f12[i + 1] = force_12
        f21[i + 1] = force_21
        p[i + 1] = m1 * v1[i + 1] + m2 * v2[i + 1]

    t = np.linspace(0.0, steps * dt, steps + 1)
    return {
        "t": t,
        "x1": x1,
        "x2": x2,
        "v1": v1,
        "v2": v2,
        "f12": f12,
        "f21": f21,
        "p": p,
    }


def evaluate_third_law(result: dict[str, np.ndarray]) -> dict[str, float]:
    force_pair_residual = result["f12"] + result["f21"]
    momentum = result["p"]

    return {
        "max_force_pair_residual": float(np.max(np.abs(force_pair_residual))),
        "max_momentum_drift": float(np.max(np.abs(momentum - momentum[0]))),
    }


def build_report(
    law1: dict[str, float],
    law2: dict[str, float],
    law3: dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "law": "First law",
            "metric": "max_velocity_deviation",
            "value": law1["max_velocity_deviation"],
            "target": 1e-12,
        },
        {
            "law": "Second law",
            "metric": "a_abs_error",
            "value": law2["a_abs_error"],
            "target": 1e-10,
        },
        {
            "law": "Second law",
            "metric": "fit_rvalue",
            "value": law2["fit_rvalue"],
            "target": 0.999999,
        },
        {
            "law": "Third law",
            "metric": "max_force_pair_residual",
            "value": law3["max_force_pair_residual"],
            "target": 1e-12,
        },
        {
            "law": "Third law",
            "metric": "max_momentum_drift",
            "value": law3["max_momentum_drift"],
            "target": 1e-12,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    # Experiment 1: First law
    _, _, v_law1 = simulate_first_law(v0=3.5, dt=0.01, steps=500)
    law1 = evaluate_first_law(v_law1)

    # Experiment 2: Second law
    t_law2, _, v_law2 = simulate_second_law(mass=2.0, force=6.0, v0=1.0, dt=0.01, steps=600)
    law2 = evaluate_second_law(t_law2, v_law2, mass=2.0, force=6.0)

    # Experiment 3: Third law
    third_result = simulate_third_law(
        m1=1.5,
        m2=2.0,
        k=8.0,
        rest_length=1.0,
        x1_0=0.0,
        x2_0=1.4,
        v1_0=0.3,
        v2_0=-0.1,
        dt=0.002,
        steps=4000,
    )
    law3 = evaluate_third_law(third_result)

    report = build_report(law1, law2, law3)

    print("Newton's Laws MVP report")
    print("=" * 72)
    print(f"First law  : v0={law1['v0']:.6f}, max |v-v0|={law1['max_velocity_deviation']:.3e}")
    print(
        "Second law : "
        f"a_theory={law2['a_theory']:.6f}, "
        f"a_estimated={law2['a_estimated']:.6f}, "
        f"|error|={law2['a_abs_error']:.3e}, "
        f"r={law2['fit_rvalue']:.6f}"
    )
    print(
        "Third law  : "
        f"max |F12+F21|={law3['max_force_pair_residual']:.3e}, "
        f"max momentum drift={law3['max_momentum_drift']:.3e}"
    )
    print("-" * 72)
    print(report.to_string(index=False))

    # Deterministic validation gates
    assert law1["max_velocity_deviation"] < 1e-12
    assert law2["a_abs_error"] < 1e-10
    assert law2["fit_rvalue"] > 0.999999
    assert law3["max_force_pair_residual"] < 1e-12
    assert law3["max_momentum_drift"] < 1e-12


if __name__ == "__main__":
    main()
