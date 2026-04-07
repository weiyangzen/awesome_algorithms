"""Minimal runnable MVP for the law of conservation of angular momentum.

The script implements three deterministic experiments:
1) Closed central-force orbit: angular momentum should be conserved.
2) Constant external torque: dL/dt should match the applied torque.
3) Variable inertia with zero external torque: L = I*omega should stay constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class OrbitConfig:
    """Configuration for a 2D central-force orbit experiment."""

    mu: float = 1.0
    mass: float = 2.0
    dt: float = 0.01
    steps: int = 6000
    r0_x: float = 1.0
    r0_y: float = 0.0
    v0_x: float = 0.0
    v0_y: float = 1.0


@dataclass(frozen=True)
class TorqueConfig:
    """Configuration for constant-torque rigid-rotor experiment."""

    inertia: float = 2.5
    torque: float = 0.15
    omega0: float = 1.1
    dt: float = 0.02
    steps: int = 2500


@dataclass(frozen=True)
class VariableInertiaConfig:
    """Configuration for zero-torque variable-inertia rotor."""

    inertia0: float = 3.0
    alpha: float = 0.45
    freq_hz: float = 0.2
    omega0: float = 1.3
    dt: float = 0.002
    steps: int = 8000


def central_acceleration(r: np.ndarray, mu: float) -> np.ndarray:
    """Acceleration under a central inverse-square force field."""
    dist = float(np.linalg.norm(r))
    if dist <= 0.0:
        raise ValueError("Radius must be positive in central-force model.")
    return -mu * r / (dist**3)


def simulate_central_orbit(cfg: OrbitConfig) -> dict[str, np.ndarray]:
    """Velocity-Verlet simulation for planar orbit in central force."""
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")

    t = np.arange(cfg.steps + 1, dtype=np.float64) * cfg.dt
    r = np.zeros((cfg.steps + 1, 2), dtype=np.float64)
    v = np.zeros((cfg.steps + 1, 2), dtype=np.float64)

    r[0] = np.array([cfg.r0_x, cfg.r0_y], dtype=np.float64)
    v[0] = np.array([cfg.v0_x, cfg.v0_y], dtype=np.float64)

    acc = central_acceleration(r[0], cfg.mu)

    for k in range(cfg.steps):
        r_next = r[k] + v[k] * cfg.dt + 0.5 * acc * cfg.dt**2
        acc_next = central_acceleration(r_next, cfg.mu)
        v_next = v[k] + 0.5 * (acc + acc_next) * cfg.dt

        r[k + 1] = r_next
        v[k + 1] = v_next
        acc = acc_next

    radius = np.linalg.norm(r, axis=1)
    speed2 = np.sum(v * v, axis=1)
    lz = cfg.mass * (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])
    energy = 0.5 * cfg.mass * speed2 - cfg.mass * cfg.mu / radius

    area_segments = 0.5 * (r[:-1, 0] * r[1:, 1] - r[:-1, 1] * r[1:, 0])
    cumulative_area = np.concatenate(([0.0], np.cumsum(area_segments)))

    return {
        "t": t,
        "r": r,
        "v": v,
        "radius": radius,
        "lz": lz,
        "energy": energy,
        "cumulative_area": cumulative_area,
    }


def evaluate_central_orbit(result: dict[str, np.ndarray], cfg: OrbitConfig) -> dict[str, float]:
    """Compute conservation and areal-velocity metrics."""
    t = result["t"]
    lz = result["lz"]
    energy = result["energy"]
    cum_area = result["cumulative_area"]
    r = result["r"]

    l0 = float(lz[0])
    e0 = float(energy[0])

    l_rel_drift = float(np.max(np.abs(lz - l0)) / max(abs(l0), 1e-15))
    e_rel_drift = float(np.max(np.abs(energy - e0)) / max(abs(e0), 1e-15))

    area_fit = linregress(t, cum_area)
    area_rate_theory = float(np.mean(0.5 * lz / cfg.mass))
    area_rate_error = abs(float(area_fit.slope) - area_rate_theory)

    angles = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    omega_est = float((angles[-1] - angles[0]) / (t[-1] - t[0]))
    period_numeric = float(2.0 * math.pi / omega_est)

    r0 = float(np.linalg.norm(r[0]))
    period_theory = float(2.0 * math.pi * math.sqrt(r0**3 / cfg.mu))
    period_rel_error = abs(period_numeric - period_theory) / period_theory

    return {
        "L0": l0,
        "L_rel_drift": l_rel_drift,
        "E_rel_drift": e_rel_drift,
        "area_rate_theory": area_rate_theory,
        "area_rate_fit": float(area_fit.slope),
        "area_rate_abs_error": area_rate_error,
        "area_fit_rvalue": float(area_fit.rvalue),
        "period_numeric": period_numeric,
        "period_theory": period_theory,
        "period_rel_error": float(period_rel_error),
    }


def simulate_constant_torque(cfg: TorqueConfig) -> dict[str, np.ndarray]:
    """Integrate dL/dt=tau for a rigid rotor with constant inertia."""
    if cfg.inertia <= 0.0:
        raise ValueError("inertia must be positive.")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")

    t = np.arange(cfg.steps + 1, dtype=np.float64) * cfg.dt
    L = np.zeros(cfg.steps + 1, dtype=np.float64)
    omega = np.zeros(cfg.steps + 1, dtype=np.float64)

    L[0] = cfg.inertia * cfg.omega0
    omega[0] = cfg.omega0

    for k in range(cfg.steps):
        L[k + 1] = L[k] + cfg.torque * cfg.dt
        omega[k + 1] = L[k + 1] / cfg.inertia

    return {
        "t": t,
        "L": L,
        "omega": omega,
    }


def evaluate_constant_torque(result: dict[str, np.ndarray], cfg: TorqueConfig) -> dict[str, float]:
    """Estimate torque from L(t) slope and compare with configured torque."""
    t = result["t"]
    L = result["L"]

    model = LinearRegression()
    model.fit(t.reshape(-1, 1), L)
    tau_hat = float(model.coef_[0])

    fit = linregress(t, L)
    L_expected = L[0] + cfg.torque * t
    max_linear_residual = float(np.max(np.abs(L - L_expected)))

    return {
        "tau_config": float(cfg.torque),
        "tau_estimated": tau_hat,
        "tau_abs_error": abs(tau_hat - cfg.torque),
        "fit_rvalue": float(fit.rvalue),
        "max_linear_residual": max_linear_residual,
    }


def inertia_profile(t: np.ndarray | float, cfg: VariableInertiaConfig) -> np.ndarray | float:
    """Time-varying rotational inertia profile I(t)."""
    return cfg.inertia0 * (1.0 + cfg.alpha * np.sin(2.0 * math.pi * cfg.freq_hz * t))


def inertia_dot(t: float, cfg: VariableInertiaConfig) -> float:
    """Time derivative of I(t)."""
    return cfg.inertia0 * cfg.alpha * 2.0 * math.pi * cfg.freq_hz * math.cos(
        2.0 * math.pi * cfg.freq_hz * t
    )


def simulate_zero_torque_variable_inertia(cfg: VariableInertiaConfig) -> dict[str, np.ndarray]:
    """Solve I*omega' + I'*omega = 0 with RK4 under zero external torque."""
    if cfg.inertia0 <= 0.0:
        raise ValueError("inertia0 must be positive.")
    if not (0.0 <= cfg.alpha < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1 to keep I(t) positive.")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")

    t = np.arange(cfg.steps + 1, dtype=np.float64) * cfg.dt
    omega = np.zeros(cfg.steps + 1, dtype=np.float64)
    omega[0] = cfg.omega0

    def domega_dt(t_scalar: float, omega_scalar: float) -> float:
        I_val = float(inertia_profile(t_scalar, cfg))
        return -(inertia_dot(t_scalar, cfg) / I_val) * omega_scalar

    for k in range(cfg.steps):
        tk = float(t[k])
        wk = float(omega[k])

        k1 = domega_dt(tk, wk)
        k2 = domega_dt(tk + 0.5 * cfg.dt, wk + 0.5 * cfg.dt * k1)
        k3 = domega_dt(tk + 0.5 * cfg.dt, wk + 0.5 * cfg.dt * k2)
        k4 = domega_dt(tk + cfg.dt, wk + cfg.dt * k3)

        omega[k + 1] = wk + (cfg.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    inertia = np.asarray(inertia_profile(t, cfg), dtype=np.float64)
    L = inertia * omega

    return {
        "t": t,
        "inertia": inertia,
        "omega": omega,
        "L": L,
    }


def evaluate_zero_torque_variable_inertia(
    result: dict[str, np.ndarray],
) -> dict[str, float]:
    """Check L conservation under variable inertia and zero external torque."""
    inertia = result["inertia"]
    omega = result["omega"]
    L = result["L"]

    L0 = float(L[0])
    L_rel_drift = float(np.max(np.abs(L - L0)) / max(abs(L0), 1e-15))

    omega_ratio_numeric = float(np.max(omega) / np.min(omega))
    omega_ratio_theory = float(np.max(inertia) / np.min(inertia))

    return {
        "L0": L0,
        "L_rel_drift": L_rel_drift,
        "omega_ratio_numeric": omega_ratio_numeric,
        "omega_ratio_theory": omega_ratio_theory,
        "omega_ratio_abs_error": abs(omega_ratio_numeric - omega_ratio_theory),
    }


def torch_consistency_lz(result: dict[str, np.ndarray], mass: float) -> float:
    """Cross-check angular momentum computed by NumPy vs PyTorch."""
    r = result["r"]
    v = result["v"]

    lz_np = mass * (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])

    t_r = torch.tensor(r, dtype=torch.float64)
    t_v = torch.tensor(v, dtype=torch.float64)
    lz_t = mass * (t_r[:, 0] * t_v[:, 1] - t_r[:, 1] * t_v[:, 0])

    return float(np.max(np.abs(lz_np - lz_t.numpy())))


def build_summary(
    orbit_eval: dict[str, float],
    torque_eval: dict[str, float],
    variable_eval: dict[str, float],
    torch_diff: float,
) -> pd.DataFrame:
    """Build a compact metric table for deterministic validation."""
    rows = [
        ("central_orbit", "L_rel_drift", orbit_eval["L_rel_drift"], 5e-4),
        ("central_orbit", "E_rel_drift", orbit_eval["E_rel_drift"], 2e-3),
        ("central_orbit", "area_rate_abs_error", orbit_eval["area_rate_abs_error"], 5e-4),
        ("central_orbit", "period_rel_error", orbit_eval["period_rel_error"], 5e-3),
        ("constant_torque", "tau_abs_error", torque_eval["tau_abs_error"], 1e-12),
        ("constant_torque", "max_linear_residual", torque_eval["max_linear_residual"], 1e-12),
        ("zero_torque_variable_I", "L_rel_drift", variable_eval["L_rel_drift"], 2e-6),
        (
            "zero_torque_variable_I",
            "omega_ratio_abs_error",
            variable_eval["omega_ratio_abs_error"],
            2e-3,
        ),
        ("cross_check", "torch_numpy_lz_max_diff", torch_diff, 1e-12),
    ]
    return pd.DataFrame(rows, columns=["experiment", "metric", "value", "target"])


def main() -> None:
    orbit_cfg = OrbitConfig()
    torque_cfg = TorqueConfig()
    variable_cfg = VariableInertiaConfig()

    orbit_result = simulate_central_orbit(orbit_cfg)
    orbit_eval = evaluate_central_orbit(orbit_result, orbit_cfg)

    torque_result = simulate_constant_torque(torque_cfg)
    torque_eval = evaluate_constant_torque(torque_result, torque_cfg)

    variable_result = simulate_zero_torque_variable_inertia(variable_cfg)
    variable_eval = evaluate_zero_torque_variable_inertia(variable_result)

    torch_diff = torch_consistency_lz(orbit_result, orbit_cfg.mass)

    summary = build_summary(orbit_eval, torque_eval, variable_eval, torch_diff)

    snapshot_idx = np.linspace(0, orbit_cfg.steps, 8, dtype=int)
    orbit_snapshot = pd.DataFrame(
        {
            "t": orbit_result["t"][snapshot_idx],
            "x": orbit_result["r"][snapshot_idx, 0],
            "y": orbit_result["r"][snapshot_idx, 1],
            "Lz": orbit_result["lz"][snapshot_idx],
            "E": orbit_result["energy"][snapshot_idx],
        }
    )

    print("=== Angular Momentum Conservation MVP ===")
    print(
        "central orbit: "
        f"L_rel_drift={orbit_eval['L_rel_drift']:.3e}, "
        f"E_rel_drift={orbit_eval['E_rel_drift']:.3e}, "
        f"period_rel_error={orbit_eval['period_rel_error']:.3e}, "
        f"area_fit_r={orbit_eval['area_fit_rvalue']:.8f}"
    )
    print(
        "constant torque: "
        f"tau_config={torque_eval['tau_config']:.6f}, "
        f"tau_estimated={torque_eval['tau_estimated']:.6f}, "
        f"|error|={torque_eval['tau_abs_error']:.3e}"
    )
    print(
        "zero torque + variable inertia: "
        f"L_rel_drift={variable_eval['L_rel_drift']:.3e}, "
        f"omega_ratio_abs_error={variable_eval['omega_ratio_abs_error']:.3e}"
    )
    print(f"torch vs numpy Lz max diff: {torch_diff:.3e}")
    print("-" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print("\n--- Orbit Snapshot ---")
    print(orbit_snapshot.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    if orbit_eval["L_rel_drift"] > 5e-4:
        raise RuntimeError("Central-orbit angular momentum drift exceeds threshold.")
    if orbit_eval["E_rel_drift"] > 2e-3:
        raise RuntimeError("Central-orbit energy drift exceeds threshold.")
    if orbit_eval["area_rate_abs_error"] > 5e-4:
        raise RuntimeError("Areal-velocity consistency check failed.")
    if orbit_eval["period_rel_error"] > 5e-3:
        raise RuntimeError("Numerical period is too far from theoretical period.")

    if torque_eval["tau_abs_error"] > 1e-12:
        raise RuntimeError("Torque slope estimate is inconsistent with configured torque.")
    if torque_eval["max_linear_residual"] > 1e-12:
        raise RuntimeError("L(t) residual under constant torque is unexpectedly large.")

    if variable_eval["L_rel_drift"] > 2e-6:
        raise RuntimeError("Zero-torque variable-inertia angular momentum drift too large.")
    if variable_eval["omega_ratio_abs_error"] > 2e-3:
        raise RuntimeError("Omega/inertia ratio consistency check failed.")

    if torch_diff > 1e-12:
        raise RuntimeError("Torch and NumPy angular momentum values diverge unexpectedly.")


if __name__ == "__main__":
    main()
