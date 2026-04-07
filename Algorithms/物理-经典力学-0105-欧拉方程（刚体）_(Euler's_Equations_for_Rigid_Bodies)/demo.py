"""Minimal runnable MVP for Euler's equations of torque-free rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


TorqueFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for integrating Euler's rigid-body equations."""

    inertia: tuple[float, float, float] = (1.0, 2.0, 3.0)
    omega0: tuple[float, float, float] = (0.02, 2.0, 0.02)
    t_start: float = 0.0
    t_end: float = 40.0
    num_points: int = 2000
    rtol: float = 1e-9
    atol: float = 1e-11


def zero_torque(_: float, __: np.ndarray) -> np.ndarray:
    """External torque set to zero for a free rigid body."""

    return np.zeros(3, dtype=float)


def euler_rhs(
    t: float,
    omega: np.ndarray,
    inertia: tuple[float, float, float],
    torque_fn: TorqueFn,
) -> np.ndarray:
    """Right-hand side of Euler's equations in body principal axes."""

    i1, i2, i3 = inertia
    w1, w2, w3 = omega
    tau1, tau2, tau3 = torque_fn(t, omega)

    dw1 = ((i2 - i3) * w2 * w3 + tau1) / i1
    dw2 = ((i3 - i1) * w3 * w1 + tau2) / i2
    dw3 = ((i1 - i2) * w1 * w2 + tau3) / i3
    return np.array([dw1, dw2, dw3], dtype=float)


def simulate(
    cfg: SimulationConfig,
    torque_fn: TorqueFn = zero_torque,
) -> pd.DataFrame:
    """Integrate the Euler ODE and return a time-series dataframe."""

    t_eval = np.linspace(cfg.t_start, cfg.t_end, cfg.num_points)
    solution = solve_ivp(
        fun=lambda t, y: euler_rhs(t, y, cfg.inertia, torque_fn),
        t_span=(cfg.t_start, cfg.t_end),
        y0=np.array(cfg.omega0, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    omega = solution.y.T
    inertia_vec = np.array(cfg.inertia, dtype=float)
    angular_momentum_body = omega * inertia_vec[None, :]
    kinetic_energy = 0.5 * np.sum(inertia_vec[None, :] * omega**2, axis=1)
    l_norm = np.linalg.norm(angular_momentum_body, axis=1)

    return pd.DataFrame(
        {
            "t": solution.t,
            "omega1": omega[:, 0],
            "omega2": omega[:, 1],
            "omega3": omega[:, 2],
            "energy": kinetic_energy,
            "L_norm": l_norm,
        }
    )


def summarize_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    """Compute conservation and motion diagnostics."""

    e = df["energy"].to_numpy()
    l = df["L_norm"].to_numpy()
    w2 = df["omega2"].to_numpy()

    e0 = float(e[0])
    l0 = float(l[0])
    max_rel_energy_drift = float(np.max(np.abs((e - e0) / e0)))
    max_rel_l_drift = float(np.max(np.abs((l - l0) / l0)))
    w2_sign_changes = int(np.count_nonzero(np.diff(np.signbit(w2))))
    max_abs_omega = float(np.max(np.linalg.norm(df[["omega1", "omega2", "omega3"]], axis=1)))

    return {
        "max_rel_energy_drift": max_rel_energy_drift,
        "max_rel_l_drift": max_rel_l_drift,
        "w2_sign_changes": float(w2_sign_changes),
        "max_abs_omega": max_abs_omega,
    }


def main() -> None:
    cfg = SimulationConfig()
    series = simulate(cfg)
    diag = summarize_diagnostics(series)

    print("Euler equations MVP (torque-free rigid body)")
    print(f"inertia={cfg.inertia}, omega0={cfg.omega0}")
    print(
        f"time_span=[{cfg.t_start}, {cfg.t_end}], "
        f"num_points={cfg.num_points}, rtol={cfg.rtol}, atol={cfg.atol}"
    )
    print("diagnostics:")
    print(
        f"  max_rel_energy_drift={diag['max_rel_energy_drift']:.3e}, "
        f"max_rel_L_drift={diag['max_rel_l_drift']:.3e}"
    )
    print(
        f"  w2_sign_changes={int(diag['w2_sign_changes'])}, "
        f"max_abs_omega={diag['max_abs_omega']:.6f}"
    )

    print("\ntrajectory_head:")
    print(series.head(5).to_string(index=False))
    print("\ntrajectory_tail:")
    print(series.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
