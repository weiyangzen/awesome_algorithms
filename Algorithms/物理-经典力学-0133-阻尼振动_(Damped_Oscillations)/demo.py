"""Minimal runnable MVP for single-DOF damped oscillations.

Model:
    m x'' + c x' + k x = 0

The script compares:
1) closed-form analytical solution (under/critical/over damping), and
2) numerical integration via scipy.solve_ivp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


@dataclass(frozen=True)
class DampedOscillationConfig:
    m: float
    c: float
    k: float
    x0: float
    v0: float
    t_end: float
    num_points: int
    rtol: float
    atol: float


def validate_config(cfg: DampedOscillationConfig) -> None:
    values = {
        "m": cfg.m,
        "c": cfg.c,
        "k": cfg.k,
        "x0": cfg.x0,
        "v0": cfg.v0,
        "t_end": cfg.t_end,
        "rtol": cfg.rtol,
        "atol": cfg.atol,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if cfg.m <= 0.0:
        raise ValueError("m must be positive")
    if cfg.k <= 0.0:
        raise ValueError("k must be positive")
    if cfg.c < 0.0:
        raise ValueError("c must be non-negative")
    if cfg.t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if cfg.num_points < 3:
        raise ValueError("num_points must be >= 3")
    if cfg.rtol <= 0.0 or cfg.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")


def system_characteristics(cfg: DampedOscillationConfig) -> dict[str, float | str]:
    wn = math.sqrt(cfg.k / cfg.m)
    alpha = cfg.c / (2.0 * cfg.m)
    zeta = cfg.c / (2.0 * math.sqrt(cfg.k * cfg.m))

    eps = 1e-12
    if zeta < 1.0 - eps:
        regime = "underdamped"
        wd = wn * math.sqrt(1.0 - zeta * zeta)
    elif abs(zeta - 1.0) <= eps:
        regime = "critical"
        wd = 0.0
    else:
        regime = "overdamped"
        wd = float("nan")

    return {
        "wn": wn,
        "alpha": alpha,
        "zeta": zeta,
        "wd": wd,
        "regime": regime,
    }


def analytical_solution(
    times: np.ndarray,
    cfg: DampedOscillationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    chars = system_characteristics(cfg)
    wn = float(chars["wn"])
    alpha = float(chars["alpha"])
    zeta = float(chars["zeta"])

    x0 = cfg.x0
    v0 = cfg.v0
    eps = 1e-12

    if zeta < 1.0 - eps:
        wd = float(chars["wd"])
        a = x0
        b = (v0 + alpha * x0) / wd
        envelope = np.exp(-alpha * times)
        c = np.cos(wd * times)
        s = np.sin(wd * times)

        x = envelope * (a * c + b * s)
        v = envelope * ((-alpha) * (a * c + b * s) + (-a * wd * s + b * wd * c))
        return x, v, chars

    if abs(zeta - 1.0) <= eps:
        a = x0
        b = v0 + wn * x0
        envelope = np.exp(-wn * times)
        x = (a + b * times) * envelope
        v = envelope * (b - wn * (a + b * times))
        return x, v, chars

    # Overdamped: two distinct real roots.
    root = math.sqrt(zeta * zeta - 1.0)
    s1 = -wn * (zeta - root)
    s2 = -wn * (zeta + root)
    c1 = (v0 - s2 * x0) / (s1 - s2)
    c2 = x0 - c1

    e1 = np.exp(s1 * times)
    e2 = np.exp(s2 * times)
    x = c1 * e1 + c2 * e2
    v = c1 * s1 * e1 + c2 * s2 * e2
    return x, v, chars


def integrate_numerically(
    times: np.ndarray,
    cfg: DampedOscillationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x, v = y
        a = -(cfg.c / cfg.m) * v - (cfg.k / cfg.m) * x
        return np.array([v, a], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=(float(times[0]), float(times[-1])),
        y0=np.array([cfg.x0, cfg.v0], dtype=float),
        t_eval=times,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    x_num = sol.y[0]
    v_num = sol.y[1]
    if not np.all(np.isfinite(x_num)) or not np.all(np.isfinite(v_num)):
        raise RuntimeError("non-finite values detected in numerical solution")

    return x_num, v_num


def total_energy(x: np.ndarray, v: np.ndarray, cfg: DampedOscillationConfig) -> np.ndarray:
    return 0.5 * cfg.m * (v * v) + 0.5 * cfg.k * (x * x)


def estimate_log_decrement(times: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    peak_idx, _ = find_peaks(x)
    if peak_idx.size < 2:
        return float("nan"), float("nan")

    amplitudes = np.abs(x[peak_idx])
    valid = amplitudes > 1e-12
    amplitudes = amplitudes[valid]
    peak_idx = peak_idx[valid]
    if amplitudes.size < 2:
        return float("nan"), float("nan")

    delta = math.log(float(amplitudes[0] / amplitudes[-1])) / float(amplitudes.size - 1)
    if delta <= 0.0 or not math.isfinite(delta):
        return float("nan"), float("nan")

    zeta_est = delta / math.sqrt((2.0 * math.pi) ** 2 + delta * delta)

    period = float(np.mean(np.diff(times[peak_idx])))
    if period <= 0.0 or not math.isfinite(period):
        return zeta_est, float("nan")

    wd_est = 2.0 * math.pi / period
    return zeta_est, wd_est


def simulate(cfg: DampedOscillationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    validate_config(cfg)
    times = np.linspace(0.0, cfg.t_end, cfg.num_points, dtype=float)

    x_num, v_num = integrate_numerically(times, cfg)
    x_ana, v_ana, chars = analytical_solution(times, cfg)

    x_abs_err = np.abs(x_num - x_ana)
    v_abs_err = np.abs(v_num - v_ana)

    energy = total_energy(x_num, v_num, cfg)
    rel_energy_end_change = float((energy[-1] - energy[0]) / max(abs(energy[0]), 1e-15))
    positive_energy_jumps = int(np.sum(np.diff(energy) > 1e-10))

    zeta_est, wd_est = estimate_log_decrement(times, x_num)

    traj = pd.DataFrame(
        {
            "t_s": times,
            "x_num_m": x_num,
            "v_num_mps": v_num,
            "x_ana_m": x_ana,
            "v_ana_mps": v_ana,
            "abs_x_err": x_abs_err,
            "abs_v_err": v_abs_err,
            "energy_J": energy,
        }
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "regime",
                "omega_n_rad_s",
                "zeta",
                "alpha_1_over_s",
                "omega_d_rad_s",
                "max_abs_x_err",
                "max_abs_v_err",
                "energy_end_relative_change",
                "positive_energy_jumps",
                "zeta_est_log_dec",
                "omega_d_est_from_peaks",
            ],
            "value": [
                chars["regime"],
                float(chars["wn"]),
                float(chars["zeta"]),
                float(chars["alpha"]),
                float(chars["wd"]),
                float(np.max(x_abs_err)),
                float(np.max(v_abs_err)),
                rel_energy_end_change,
                float(positive_energy_jumps),
                float(zeta_est),
                float(wd_est),
            ],
        }
    )

    diagnostics = {
        "max_abs_x_err": float(np.max(x_abs_err)),
        "max_abs_v_err": float(np.max(v_abs_err)),
        "rel_energy_end_change": rel_energy_end_change,
        "positive_energy_jumps": float(positive_energy_jumps),
        "zeta": float(chars["zeta"]),
        "zeta_est": float(zeta_est),
        "regime": str(chars["regime"]),
    }
    return traj, summary, diagnostics


def format_sample_rows(df: pd.DataFrame, n_head: int = 6, n_tail: int = 6) -> pd.DataFrame:
    head = df.head(n_head)
    tail = df.tail(n_tail)
    return pd.concat([head, tail], axis=0)


def main() -> None:
    cfg = DampedOscillationConfig(
        m=1.0,
        c=0.6,
        k=16.0,
        x0=0.10,
        v0=-0.05,
        t_end=14.0,
        num_points=2200,
        rtol=1e-10,
        atol=1e-12,
    )

    traj, summary, diagnostics = simulate(cfg)

    print("Damped oscillation MVP (single DOF)")
    print("Config:")
    print(cfg)
    print("\nSummary metrics:")
    print(summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.8e}"))
    print("\nTrajectory sample (head + tail):")
    print(format_sample_rows(traj).to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    # Reproducible automatic checks.
    assert diagnostics["max_abs_x_err"] < 5e-7, (
        f"position error too large: {diagnostics['max_abs_x_err']:.6e}"
    )
    assert diagnostics["max_abs_v_err"] < 2e-6, (
        f"velocity error too large: {diagnostics['max_abs_v_err']:.6e}"
    )
    assert diagnostics["rel_energy_end_change"] < 0.0, "damped system should lose energy"

    if diagnostics["regime"] == "underdamped" and math.isfinite(diagnostics["zeta_est"]):
        assert abs(diagnostics["zeta_est"] - diagnostics["zeta"]) < 0.02, (
            "log-decrement damping ratio estimate deviates too much"
        )


if __name__ == "__main__":
    main()
