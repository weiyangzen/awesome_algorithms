"""Minimal runnable MVP for Linear Response Theory.

This demo uses a 1D Ornstein-Uhlenbeck velocity process and compares:
1) direct response under a weak constant force,
2) linear-response prediction from equilibrium correlation via FDT,
3) closed-form analytic response.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearResponseConfig:
    mass: float = 1.0
    gamma: float = 1.5
    kbt: float = 1.0
    dt: float = 0.002
    n_steps: int = 2500
    n_traj: int = 5000
    force: float = 0.05
    seed: int = 20260407


def _validate_config(cfg: LinearResponseConfig) -> None:
    if cfg.mass <= 0.0:
        raise ValueError("mass must be positive")
    if cfg.gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if cfg.kbt <= 0.0:
        raise ValueError("kbt must be positive")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")
    if cfg.n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if cfg.n_traj < 10:
        raise ValueError("n_traj must be >= 10 for stable statistics")


def _simulate_ensemble_response(cfg: LinearResponseConfig) -> dict[str, np.ndarray]:
    """Simulate unperturbed and perturbed ensembles with shared noise.

    Returns time series for:
    - mean velocity in unperturbed / perturbed ensembles,
    - equilibrium correlation Cvv(t) = <v(t) v(0)> from the unperturbed ensemble.
    """

    rng = np.random.default_rng(cfg.seed)

    n_points = cfg.n_steps + 1
    time = np.arange(n_points, dtype=float) * cfg.dt

    eq_sigma = np.sqrt(cfg.kbt / cfg.mass)
    noise_sigma = np.sqrt(2.0 * cfg.gamma * cfg.kbt / cfg.mass * cfg.dt)

    v_eq = rng.normal(loc=0.0, scale=eq_sigma, size=cfg.n_traj)
    v_pert = v_eq.copy()
    v0 = v_eq.copy()

    mean_eq = np.empty(n_points, dtype=float)
    mean_pert = np.empty(n_points, dtype=float)
    corr_vv = np.empty(n_points, dtype=float)

    mean_eq[0] = float(np.mean(v_eq))
    mean_pert[0] = float(np.mean(v_pert))
    corr_vv[0] = float(np.mean(v_eq * v0))

    for t_idx in range(1, n_points):
        xi = rng.normal(loc=0.0, scale=1.0, size=cfg.n_traj)
        noise = noise_sigma * xi

        v_eq = v_eq + (-cfg.gamma * v_eq) * cfg.dt + noise
        v_pert = v_pert + (-cfg.gamma * v_pert + cfg.force / cfg.mass) * cfg.dt + noise

        mean_eq[t_idx] = float(np.mean(v_eq))
        mean_pert[t_idx] = float(np.mean(v_pert))
        corr_vv[t_idx] = float(np.mean(v_eq * v0))

    return {
        "time": time,
        "mean_eq": mean_eq,
        "mean_pert": mean_pert,
        "corr_vv": corr_vv,
    }


def _cumulative_trapezoid(y: np.ndarray, dt: float) -> np.ndarray:
    """Return cumulative integral with trapezoidal rule and integral(0)=0."""

    out = np.zeros_like(y)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out


def run_linear_response_demo(cfg: LinearResponseConfig) -> dict[str, object]:
    _validate_config(cfg)

    sim = _simulate_ensemble_response(cfg)
    time = sim["time"]
    mean_eq = sim["mean_eq"]
    mean_pert = sim["mean_pert"]
    corr_vv = sim["corr_vv"]

    direct_delta = mean_pert - mean_eq
    predicted_delta = (cfg.force / cfg.kbt) * _cumulative_trapezoid(corr_vv, cfg.dt)
    analytic_delta = (cfg.force / (cfg.mass * cfg.gamma)) * (
        1.0 - np.exp(-cfg.gamma * time)
    )

    abs_err_pred = np.abs(direct_delta - predicted_delta)
    abs_err_analytic = np.abs(direct_delta - analytic_delta)

    metrics = {
        "corr0_expected": cfg.kbt / cfg.mass,
        "corr0_measured": float(corr_vv[0]),
        "final_direct_delta": float(direct_delta[-1]),
        "final_lrt_prediction": float(predicted_delta[-1]),
        "final_analytic": float(analytic_delta[-1]),
        "mae_direct_vs_lrt": float(np.mean(abs_err_pred)),
        "max_err_direct_vs_lrt": float(np.max(abs_err_pred)),
        "mae_direct_vs_analytic": float(np.mean(abs_err_analytic)),
        "final_rel_err_lrt": float(
            abs(direct_delta[-1] - predicted_delta[-1])
            / max(abs(predicted_delta[-1]), 1e-12)
        ),
    }

    sample_idx = np.linspace(0, cfg.n_steps, num=8, dtype=int)
    table = pd.DataFrame(
        {
            "t": time[sample_idx],
            "delta_direct": direct_delta[sample_idx],
            "delta_lrt": predicted_delta[sample_idx],
            "delta_analytic": analytic_delta[sample_idx],
        }
    )

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "sample_table": table,
    }


def main() -> None:
    cfg = LinearResponseConfig()
    result = run_linear_response_demo(cfg)

    # Basic statistical sanity checks for this specific MVP setup.
    corr0_expected = result["metrics"]["corr0_expected"]
    corr0_measured = result["metrics"]["corr0_measured"]
    if abs(corr0_measured - corr0_expected) / corr0_expected > 0.1:
        raise RuntimeError("equilibrium variance sanity check failed")

    final_rel_err_lrt = result["metrics"]["final_rel_err_lrt"]
    if final_rel_err_lrt > 0.25:
        raise RuntimeError("linear-response prediction too far from direct response")

    print("=== Linear Response Sample Table ===")
    print(result["sample_table"].to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    payload = {
        "config": result["config"],
        "metrics": result["metrics"],
    }
    print("\n=== JSON Summary ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
