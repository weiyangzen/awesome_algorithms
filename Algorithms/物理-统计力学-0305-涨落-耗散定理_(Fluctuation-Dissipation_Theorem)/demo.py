"""Fluctuation-Dissipation Theorem (FDT) MVP.

Model:
    Overdamped Langevin dynamics in harmonic potential
        dx/dt = -mu*k*x + mu*f(t) + sqrt(2*mu*kBT) * xi(t)

Validation target:
    chi(t) = (C(0) - C(t)) / kBT
where
    C(t) = <x(0) x(t)> (equilibrium correlation)
    chi(t) = d<x(t)>/df0 under a small step force f(t)=f0*Theta(t).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FDTConfig:
    mu: float = 1.2
    k: float = 2.0
    kbt: float = 1.0
    dt: float = 0.002
    n_steps_eq: int = 120_000
    n_steps_resp: int = 3_000
    n_ensembles: int = 4_000
    step_force: float = 0.05
    seed: int = 2026


def validate_config(cfg: FDTConfig) -> None:
    if cfg.mu <= 0.0:
        raise ValueError("mu must be positive.")
    if cfg.k <= 0.0:
        raise ValueError("k must be positive.")
    if cfg.kbt <= 0.0:
        raise ValueError("kbt must be positive.")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.n_steps_eq < 4:
        raise ValueError("n_steps_eq must be >= 4.")
    if cfg.n_steps_resp < 2:
        raise ValueError("n_steps_resp must be >= 2.")
    if cfg.n_ensembles < 2:
        raise ValueError("n_ensembles must be >= 2.")
    if cfg.step_force == 0.0:
        raise ValueError("step_force must be non-zero.")


def compute_discrete_coefficients(
    mu: float, k: float, kbt: float, dt: float, step_force: float
) -> dict[str, float]:
    """Exact discrete-time coefficients for OU process with constant step force."""
    lambd = mu * k
    a = float(np.exp(-lambd * dt))
    var_eq = kbt / k
    sigma = float(np.sqrt(var_eq * (1.0 - a * a)))
    force_drift = float((step_force / k) * (1.0 - a))
    return {
        "lambda": lambd,
        "a": a,
        "sigma": sigma,
        "var_eq": var_eq,
        "force_drift": force_drift,
    }


def simulate_equilibrium_trajectory(cfg: FDTConfig, coeff: dict[str, float]) -> np.ndarray:
    """Generate a long equilibrium trajectory x_t for correlation estimation."""
    rng = np.random.default_rng(cfg.seed)
    x = np.empty(cfg.n_steps_eq, dtype=np.float64)
    x[0] = np.sqrt(coeff["var_eq"]) * rng.standard_normal()
    a = coeff["a"]
    sigma = coeff["sigma"]

    for i in range(1, cfg.n_steps_eq):
        x[i] = a * x[i - 1] + sigma * rng.standard_normal()
    return x


def autocorr_unbiased(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation using FFT, C[tau] = mean x_t x_{t+tau}."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples for autocorrelation.")
    x = x - np.mean(x)

    fft_size = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=fft_size)
    power = fx * np.conj(fx)
    corr_full = np.fft.irfft(power, n=fft_size)[:n]
    norm = np.arange(n, 0, -1, dtype=np.float64)
    return corr_full / norm


def simulate_step_response_susceptibility(
    cfg: FDTConfig, coeff: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate chi(t) from paired trajectories with common noise.

    We evolve two ensembles with identical noise:
    - reference: no external force
    - driven: step force f0 for t >= 0
    Then chi_est(t) = (mean(x_driven) - mean(x_ref)) / f0
    """
    rng = np.random.default_rng(cfg.seed + 1)
    m = cfg.n_ensembles
    n = cfg.n_steps_resp
    a = coeff["a"]
    sigma = coeff["sigma"]
    drift = coeff["force_drift"]

    x0 = np.sqrt(coeff["var_eq"]) * rng.standard_normal(m)
    x_ref = x0.copy()
    x_drv = x0.copy()

    chi = np.zeros(n, dtype=np.float64)
    times = np.arange(n, dtype=np.float64) * cfg.dt

    for i in range(1, n):
        noise = rng.standard_normal(m)
        x_ref = a * x_ref + sigma * noise
        x_drv = a * x_drv + drift + sigma * noise
        chi[i] = np.mean(x_drv - x_ref) / cfg.step_force
    return times, chi


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.linalg.norm(y_true))
    if denom == 0.0:
        return float("inf")
    return float(np.linalg.norm(y_true - y_pred) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_demo(cfg: FDTConfig) -> dict[str, object]:
    validate_config(cfg)
    coeff = compute_discrete_coefficients(
        mu=cfg.mu, k=cfg.k, kbt=cfg.kbt, dt=cfg.dt, step_force=cfg.step_force
    )

    x_eq = simulate_equilibrium_trajectory(cfg, coeff)
    c = autocorr_unbiased(x_eq)
    max_lag = min(cfg.n_steps_resp, c.size)
    c_cut = c[:max_lag]
    chi_fdt = (c_cut[0] - c_cut) / cfg.kbt

    times, chi_est = simulate_step_response_susceptibility(cfg, coeff)
    times = times[:max_lag]
    chi_est = chi_est[:max_lag]

    lambd = coeff["lambda"]
    chi_theory = (1.0 / cfg.k) * (1.0 - np.exp(-lambd * times))

    fdt_rmse = rmse(chi_fdt, chi_est)
    fdt_max_abs = float(np.max(np.abs(chi_fdt - chi_est)))
    fdt_rel_l2 = relative_l2(chi_fdt, chi_est)

    sample_indices = np.linspace(0, max_lag - 1, num=6, dtype=int)
    samples = [
        {
            "step": int(i),
            "t": float(times[i]),
            "chi_est": float(chi_est[i]),
            "chi_fdt": float(chi_fdt[i]),
            "chi_theory": float(chi_theory[i]),
        }
        for i in sample_indices
    ]

    result = {
        "coefficients": coeff,
        "var_eq_theory": coeff["var_eq"],
        "var_eq_est": float(np.var(x_eq)),
        "C0_est": float(c_cut[0]),
        "fdt_rmse": fdt_rmse,
        "fdt_max_abs": fdt_max_abs,
        "fdt_rel_l2": fdt_rel_l2,
        "chi_est_final": float(chi_est[-1]),
        "chi_fdt_final": float(chi_fdt[-1]),
        "chi_theory_final": float(chi_theory[-1]),
        "samples": samples,
    }
    return result


def main() -> None:
    cfg = FDTConfig()
    result = run_demo(cfg)
    payload = {
        "algorithm": "Fluctuation-Dissipation Theorem",
        "config": asdict(cfg),
        "result": result,
        "note": "Finite sampling causes residual error; increase n_steps_eq or n_ensembles for tighter agreement.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
