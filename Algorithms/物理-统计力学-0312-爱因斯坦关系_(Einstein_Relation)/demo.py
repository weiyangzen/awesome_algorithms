"""Minimal MVP for Einstein relation in 1D overdamped Brownian dynamics.

We verify the fluctuation-dissipation Einstein relation
    D = mu * k_B * T
(using reduced units k_B = 1, so D = mu * kBT)
by estimating:
1) diffusion coefficient D from zero-force MSD slope;
2) mobility mu from drift velocity under a constant small force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EinsteinConfig:
    kbt: float = 1.20
    mobility_true: float = 0.80
    external_force: float = 0.75
    dt: float = 0.002
    n_steps: int = 5000
    record_stride: int = 50
    n_particles: int = 6000
    seed: int = 20260407


@dataclass(frozen=True)
class TrajectoryStats:
    times: np.ndarray
    mean_displacement: np.ndarray
    msd: np.ndarray


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r2: float


def simulate_overdamped_ensemble(
    *,
    n_particles: int,
    n_steps: int,
    dt: float,
    mobility: float,
    diffusion: float,
    force: float,
    record_stride: int,
    rng: np.random.Generator,
) -> TrajectoryStats:
    """Simulate x(t) via Euler-Maruyama for overdamped Langevin dynamics.

    Dynamics:
        x_{n+1} = x_n + mobility * force * dt + sqrt(2*diffusion*dt) * eta_n
    where eta_n ~ N(0,1).
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if record_stride <= 0:
        raise ValueError("record_stride must be positive.")

    x = np.zeros(n_particles, dtype=float)
    drift = mobility * force * dt
    noise_scale = np.sqrt(2.0 * diffusion * dt)

    sample_times: list[float] = [0.0]
    sample_mean: list[float] = [0.0]
    sample_msd: list[float] = [0.0]

    for step in range(1, n_steps + 1):
        x += drift + noise_scale * rng.standard_normal(n_particles)
        if step % record_stride == 0 or step == n_steps:
            sample_times.append(step * dt)
            sample_mean.append(float(np.mean(x)))
            sample_msd.append(float(np.mean(x * x)))

    return TrajectoryStats(
        times=np.asarray(sample_times, dtype=float),
        mean_displacement=np.asarray(sample_mean, dtype=float),
        msd=np.asarray(sample_msd, dtype=float),
    )


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFit:
    """Fit y ~= slope * x + intercept with least squares and return R^2."""
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length.")
    if x.size < 2:
        raise ValueError("Need at least 2 points for linear fit.")

    design = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-16 else 1.0 - ss_res / ss_tot
    return LinearFit(slope=slope, intercept=intercept, r2=r2)


def relative_error(estimate: float, truth: float) -> float:
    denom = max(abs(truth), 1e-12)
    return abs(estimate - truth) / denom


def run_einstein_relation_demo(config: EinsteinConfig) -> dict[str, object]:
    if config.kbt <= 0.0:
        raise ValueError("kbt must be positive.")
    if config.mobility_true <= 0.0:
        raise ValueError("mobility_true must be positive.")
    if config.external_force == 0.0:
        raise ValueError("external_force must be non-zero.")

    diffusion_true = config.mobility_true * config.kbt

    rng_unbiased = np.random.default_rng(config.seed)
    rng_biased = np.random.default_rng(config.seed + 1)

    stats_unbiased = simulate_overdamped_ensemble(
        n_particles=config.n_particles,
        n_steps=config.n_steps,
        dt=config.dt,
        mobility=config.mobility_true,
        diffusion=diffusion_true,
        force=0.0,
        record_stride=config.record_stride,
        rng=rng_unbiased,
    )

    stats_biased = simulate_overdamped_ensemble(
        n_particles=config.n_particles,
        n_steps=config.n_steps,
        dt=config.dt,
        mobility=config.mobility_true,
        diffusion=diffusion_true,
        force=config.external_force,
        record_stride=config.record_stride,
        rng=rng_biased,
    )

    # Skip t=0 for regression to avoid overweighting the trivial origin point.
    t = stats_unbiased.times[1:]
    fit_msd = fit_linear(t, stats_unbiased.msd[1:])
    fit_drift = fit_linear(t, stats_biased.mean_displacement[1:])

    diffusion_from_msd = 0.5 * fit_msd.slope
    drift_velocity = fit_drift.slope
    mobility_from_drift = drift_velocity / config.external_force
    diffusion_from_einstein = mobility_from_drift * config.kbt

    ratio = diffusion_from_msd / max(diffusion_from_einstein, 1e-12)

    summary = pd.DataFrame(
        {
            "quantity": [
                "D_true",
                "D_from_MSD",
                "mu_true",
                "mu_from_drift",
                "D_from_mu_kBT",
                "einstein_ratio_D_over_mu_kBT",
                "R2_MSD_linear_fit",
                "R2_drift_linear_fit",
            ],
            "value": [
                diffusion_true,
                diffusion_from_msd,
                config.mobility_true,
                mobility_from_drift,
                diffusion_from_einstein,
                ratio,
                fit_msd.r2,
                fit_drift.r2,
            ],
        }
    )

    trace = pd.DataFrame(
        {
            "time": stats_unbiased.times,
            "mean_disp_unbiased": stats_unbiased.mean_displacement,
            "msd_unbiased": stats_unbiased.msd,
            "mean_disp_biased": stats_biased.mean_displacement,
            "msd_biased": stats_biased.msd,
        }
    )

    checks = {
        "rel_err_D": relative_error(diffusion_from_msd, diffusion_true),
        "rel_err_mu": relative_error(mobility_from_drift, config.mobility_true),
        "einstein_ratio_abs_err": abs(ratio - 1.0),
        "r2_msd": fit_msd.r2,
        "r2_drift": fit_drift.r2,
    }

    return {
        "config": config,
        "summary": summary,
        "trace": trace,
        "checks": checks,
        "fit_msd": fit_msd,
        "fit_drift": fit_drift,
    }


def main() -> None:
    config = EinsteinConfig()
    result = run_einstein_relation_demo(config)

    summary = result["summary"]
    trace = result["trace"]
    checks = result["checks"]

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)

    print("Einstein relation demo (1D overdamped Brownian dynamics)")
    print("=" * 72)
    print(config)
    print("\n--- Summary ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n--- Time-series head (recorded observables) ---")
    print(trace.head(8).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n--- Time-series tail (recorded observables) ---")
    print(trace.tail(8).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Sanity checks for reproducible MVP validation.
    assert checks["rel_err_D"] < 0.10, f"Diffusion estimate too noisy: {checks['rel_err_D']:.4f}"
    assert checks["rel_err_mu"] < 0.10, f"Mobility estimate too noisy: {checks['rel_err_mu']:.4f}"
    assert checks["einstein_ratio_abs_err"] < 0.08, (
        f"Einstein ratio deviates too much: {checks['einstein_ratio_abs_err']:.4f}"
    )
    assert checks["r2_msd"] > 0.995, f"MSD linearity too weak: {checks['r2_msd']:.6f}"
    assert checks["r2_drift"] > 0.995, f"Drift linearity too weak: {checks['r2_drift']:.6f}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
