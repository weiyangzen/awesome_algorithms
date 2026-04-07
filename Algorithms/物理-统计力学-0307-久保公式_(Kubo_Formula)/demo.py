"""Kubo Formula MVP.

This demo estimates transport coefficients from equilibrium fluctuations
using a 1D Ornstein-Uhlenbeck (Langevin) velocity process:

    D = integral_0^inf <v(0) v(t)> dt                    (Green-Kubo)
    sigma = beta * integral_0^inf <J(0) J(t)> dt         (Kubo)

with J = q v, beta = 1 / (k_B T), and unit volume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class KuboConfig:
    n_steps: int = 200_000
    dt: float = 0.002
    gamma: float = 1.5
    mass: float = 1.0
    kbt: float = 1.0
    charge: float = 1.0
    seed: int = 7
    max_lag_fraction: float = 0.2
    min_lag_for_zero_crossing: int = 10


def simulate_ou_velocity(cfg: KuboConfig) -> np.ndarray:
    """Simulate equilibrium OU velocity trajectory."""
    rng = np.random.default_rng(cfg.seed)
    a = np.exp(-cfg.gamma * cfg.dt)
    thermal_var = cfg.kbt / cfg.mass
    noise_scale = np.sqrt(thermal_var * (1.0 - a * a))

    v = np.empty(cfg.n_steps, dtype=np.float64)
    v[0] = np.sqrt(thermal_var) * rng.standard_normal()
    for i in range(1, cfg.n_steps):
        v[i] = a * v[i - 1] + noise_scale * rng.standard_normal()
    return v


def autocorr_unbiased(x: np.ndarray) -> np.ndarray:
    """Return unbiased autocorrelation C[tau] = <x(0) x(tau)>."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples for autocorrelation.")
    # Demeaning suppresses finite-sample bias in the long-lag tail.
    x = x - np.mean(x)

    fft_size = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=fft_size)
    power = fx * np.conj(fx)
    corr_full = np.fft.irfft(power, n=fft_size)[:n]
    norm = np.arange(n, 0, -1, dtype=np.float64)
    return corr_full / norm


def integrate_trapezoid(y: np.ndarray, dt: float, max_lag: int) -> float:
    if max_lag < 2:
        raise ValueError("max_lag must be >= 2.")
    return float(np.trapezoid(y[:max_lag], dx=dt))


def choose_integration_lag(
    corr: np.ndarray, cap_lag: int, min_lag_for_zero_crossing: int
) -> int:
    """Pick a robust truncation lag for Green-Kubo integration."""
    if cap_lag < 2:
        return 2
    search_start = max(1, min_lag_for_zero_crossing)
    search_end = min(cap_lag, corr.size - 1)
    if search_start >= search_end:
        return cap_lag

    negative = np.flatnonzero(corr[search_start : search_end + 1] <= 0.0)
    if negative.size == 0:
        return cap_lag
    return int(search_start + negative[0])


def relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / abs(truth)


def run_kubo_demo(cfg: KuboConfig) -> dict[str, float | int]:
    v = simulate_ou_velocity(cfg)
    c_vv = autocorr_unbiased(v)
    cap_lag = max(2, int(cfg.max_lag_fraction * cfg.n_steps))
    max_lag = choose_integration_lag(
        c_vv, cap_lag=cap_lag, min_lag_for_zero_crossing=cfg.min_lag_for_zero_crossing
    )

    d_est = integrate_trapezoid(c_vv, cfg.dt, max_lag=max_lag)
    d_theory = cfg.kbt / (cfg.mass * cfg.gamma)

    beta = 1.0 / cfg.kbt
    j = cfg.charge * v
    c_jj = autocorr_unbiased(j)
    sigma_est = beta * integrate_trapezoid(c_jj, cfg.dt, max_lag=max_lag)
    sigma_theory = (cfg.charge * cfg.charge) / (cfg.mass * cfg.gamma)

    return {
        "n_steps": cfg.n_steps,
        "dt": cfg.dt,
        "gamma": cfg.gamma,
        "mass": cfg.mass,
        "kbt": cfg.kbt,
        "charge": cfg.charge,
        "cap_lag": cap_lag,
        "max_lag": max_lag,
        "D_estimate": d_est,
        "D_theory": d_theory,
        "D_rel_error": relative_error(d_est, d_theory),
        "sigma_estimate": sigma_est,
        "sigma_theory": sigma_theory,
        "sigma_rel_error": relative_error(sigma_est, sigma_theory),
        "Cvv_0": float(c_vv[0]),
        "Cjj_0": float(c_jj[0]),
    }


def main() -> None:
    cfg = KuboConfig()
    result = run_kubo_demo(cfg)
    payload = {
        "config": asdict(cfg),
        "result": result,
        "note": "Finite trajectory causes statistical error; increase n_steps for tighter agreement.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
