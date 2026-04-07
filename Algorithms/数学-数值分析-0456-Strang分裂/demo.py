"""Minimal runnable MVP for Strang splitting on a 1D reaction-diffusion equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the Strang-splitting demo."""

    length: float = 2.0 * np.pi
    nx: int = 256
    diffusion: float = 0.08
    reaction_rate: float = 2.0
    final_time: float = 0.5
    dt_candidates: Tuple[float, ...] = (0.05, 0.025, 0.0125)
    reference_dt: float = 0.003125

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("length must be a positive finite number")
        if not isinstance(self.nx, int) or self.nx < 8:
            raise ValueError("nx must be an integer >= 8")
        if not np.isfinite(self.diffusion) or self.diffusion <= 0:
            raise ValueError("diffusion must be a positive finite number")
        if not np.isfinite(self.reaction_rate) or self.reaction_rate <= 0:
            raise ValueError("reaction_rate must be a positive finite number")
        if not np.isfinite(self.final_time) or self.final_time <= 0:
            raise ValueError("final_time must be a positive finite number")
        if not self.dt_candidates:
            raise ValueError("dt_candidates must not be empty")
        for dt in self.dt_candidates:
            _validate_time_step(self.final_time, dt)
        _validate_time_step(self.final_time, self.reference_dt)
        if self.reference_dt >= min(self.dt_candidates):
            raise ValueError("reference_dt must be smaller than all dt_candidates")


def _validate_time_step(final_time: float, dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number")
    steps_float = final_time / dt
    steps = int(round(steps_float))
    if steps <= 0 or abs(steps * dt - final_time) > 1e-12:
        raise ValueError(f"dt={dt} does not evenly divide final_time={final_time}")


def _time_steps(final_time: float, dt: float) -> int:
    return int(round(final_time / dt))


def build_grid(length: float, nx: int) -> Tuple[np.ndarray, float]:
    x = np.linspace(0.0, length, nx, endpoint=False)
    dx = length / nx
    return x, dx


def initial_condition(x: np.ndarray) -> np.ndarray:
    u0 = 0.35 + 0.20 * np.sin(x) + 0.10 * np.cos(2.0 * x)
    return np.clip(u0, 1e-6, 1.0 - 1e-6)


def reaction_flow_logistic(u: np.ndarray, tau: float, rate: float) -> np.ndarray:
    exp_term = np.exp(rate * tau)
    denominator = 1.0 + u * (exp_term - 1.0)
    return (u * exp_term) / denominator


def diffusion_flow_fft(u: np.ndarray, tau: float, diffusion: float, k2: np.ndarray) -> np.ndarray:
    u_hat = np.fft.fft(u)
    damping = np.exp(-diffusion * k2 * tau)
    updated_hat = u_hat * damping
    return np.fft.ifft(updated_hat).real


def strang_step(
    u: np.ndarray,
    dt: float,
    diffusion: float,
    reaction_rate: float,
    k2: np.ndarray,
) -> np.ndarray:
    u_half = reaction_flow_logistic(u, 0.5 * dt, reaction_rate)
    u_full = diffusion_flow_fft(u_half, dt, diffusion, k2)
    return reaction_flow_logistic(u_full, 0.5 * dt, reaction_rate)


def lie_step(
    u: np.ndarray,
    dt: float,
    diffusion: float,
    reaction_rate: float,
    k2: np.ndarray,
) -> np.ndarray:
    u_react = reaction_flow_logistic(u, dt, reaction_rate)
    return diffusion_flow_fft(u_react, dt, diffusion, k2)


def simulate(cfg: SimulationConfig, dt: float, scheme: str = "strang") -> Tuple[np.ndarray, np.ndarray]:
    x, dx = build_grid(cfg.length, cfg.nx)
    k = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    k2 = k * k
    u = initial_condition(x)

    steps = _time_steps(cfg.final_time, dt)
    for _ in range(steps):
        if scheme == "strang":
            u = strang_step(u, dt, cfg.diffusion, cfg.reaction_rate, k2)
        elif scheme == "lie":
            u = lie_step(u, dt, cfg.diffusion, cfg.reaction_rate, k2)
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        if not np.isfinite(u).all():
            raise RuntimeError("solution contains NaN/Inf; unstable configuration")

    return x, u


def l2_error(u: np.ndarray, v: np.ndarray) -> float:
    diff = u - v
    return float(np.sqrt(np.mean(diff * diff)))


def build_report(cfg: SimulationConfig) -> List[Dict[str, float]]:
    _, u_ref = simulate(cfg, cfg.reference_dt, scheme="strang")
    rows: List[Dict[str, float]] = []

    for dt in cfg.dt_candidates:
        _, u_strang = simulate(cfg, dt, scheme="strang")
        _, u_lie = simulate(cfg, dt, scheme="lie")

        err_strang = l2_error(u_strang, u_ref)
        err_lie = l2_error(u_lie, u_ref)

        rows.append(
            {
                "dt": dt,
                "strang_error": err_strang,
                "lie_error": err_lie,
                "max_u": float(np.max(u_strang)),
                "min_u": float(np.min(u_strang)),
            }
        )

    return rows


def print_report(rows: List[Dict[str, float]]) -> None:
    print("Strang splitting on u_t = D*u_xx + r*u*(1-u)")
    print("reference: Strang with much smaller dt")
    print(
        "{:<10} {:<16} {:<16} {:<10} {:<10}".format(
            "dt", "strang_L2_error", "lie_L2_error", "min(u)", "max(u)"
        )
    )
    for row in rows:
        print(
            "{:<10.6f} {:<16.8e} {:<16.8e} {:<10.6f} {:<10.6f}".format(
                row["dt"],
                row["strang_error"],
                row["lie_error"],
                row["min_u"],
                row["max_u"],
            )
        )


def run_checks(rows: List[Dict[str, float]]) -> None:
    strang_errors = [row["strang_error"] for row in rows]

    for idx in range(1, len(strang_errors)):
        if not strang_errors[idx] < strang_errors[idx - 1]:
            raise AssertionError("Strang errors are not decreasing when dt is refined")

    rates = [strang_errors[i - 1] / strang_errors[i] for i in range(1, len(strang_errors))]
    if not all(rate > 2.5 for rate in rates):
        raise AssertionError(f"Convergence factors are too weak: {rates}")

    for row in rows:
        if not row["strang_error"] < row["lie_error"]:
            raise AssertionError("Strang should outperform Lie splitting at the same dt")
        if row["min_u"] < -1e-3 or row["max_u"] > 1.0 + 1e-3:
            raise AssertionError("solution left expected logistic range [0,1] by too much")


def main() -> None:
    cfg = SimulationConfig()
    rows = build_report(cfg)
    print_report(rows)
    run_checks(rows)
    print("All checks passed.")


if __name__ == "__main__":
    main()
