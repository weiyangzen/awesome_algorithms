"""MVP for the 1D Smoluchowski equation in a harmonic potential.

This script solves
    d_t p = -d_x J,
    J = -D d_x p + mu * F(x) * p,
with reflecting boundaries J(-L)=J(L)=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    domain_half_width: float = 4.0
    grid_size: int = 241
    total_time: float = 2.0
    k_spring: float = 2.0
    mobility: float = 1.0
    kbt: float = 1.0
    initial_mean: float = -1.2
    initial_std: float = 0.35
    cfl_safety: float = 0.45
    checkpoint_times: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0, 2.0)


@dataclass
class SimulationResult:
    x: np.ndarray
    p_final: np.ndarray
    p_equilibrium: np.ndarray
    checkpoints: pd.DataFrame
    dt: float
    num_steps: int
    total_clipped_negative_mass: float


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def normalize_density(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    mass = trapz_integral(p, x)
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("Density normalization failed: non-positive or non-finite mass.")
    return p / mass


def harmonic_potential(x: np.ndarray, k_spring: float) -> np.ndarray:
    return 0.5 * k_spring * x * x


def harmonic_force(x: np.ndarray, k_spring: float) -> np.ndarray:
    return -k_spring * x


def gaussian_density(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    p = np.exp(-0.5 * z * z)
    p /= std * np.sqrt(2.0 * np.pi)
    return p


def equilibrium_density(x: np.ndarray, k_spring: float, kbt: float) -> np.ndarray:
    u = harmonic_potential(x, k_spring)
    unnormalized = np.exp(-u / kbt)
    return normalize_density(unnormalized, x)


def choose_stable_dt(
    x: np.ndarray,
    diffusion: float,
    mobility: float,
    k_spring: float,
    total_time: float,
    cfl_safety: float,
) -> tuple[float, int]:
    dx = float(x[1] - x[0])
    vmax = float(np.max(np.abs(harmonic_force(x, k_spring))) * mobility)

    dt_diffusion = cfl_safety * dx * dx / max(diffusion, 1e-12)
    dt_drift = cfl_safety * dx / max(vmax, 1e-12)

    dt_raw = min(dt_diffusion, dt_drift)
    if not np.isfinite(dt_raw) or dt_raw <= 0.0:
        raise RuntimeError("Failed to choose a stable time step.")

    num_steps = int(np.ceil(total_time / dt_raw))
    num_steps = max(num_steps, 1)
    dt = total_time / num_steps
    return dt, num_steps


def compute_flux_faces(
    p: np.ndarray,
    x: np.ndarray,
    diffusion: float,
    mobility: float,
    k_spring: float,
) -> np.ndarray:
    """Return face flux J on a staggered grid, with reflecting boundaries."""
    n = p.size
    dx = float(x[1] - x[0])
    j_face = np.zeros(n + 1, dtype=float)

    x_left = x[:-1]
    x_right = x[1:]
    x_face = 0.5 * (x_left + x_right)

    p_left = p[:-1]
    p_right = p[1:]

    diff_flux = -diffusion * (p_right - p_left) / dx
    velocity = mobility * harmonic_force(x_face, k_spring)

    # Upwind convection term keeps the MVP numerically robust.
    conv_flux = np.where(velocity >= 0.0, velocity * p_left, velocity * p_right)

    j_face[1:-1] = diff_flux + conv_flux
    j_face[0] = 0.0
    j_face[-1] = 0.0
    return j_face


def smoluchowski_step(
    p: np.ndarray,
    x: np.ndarray,
    dt: float,
    diffusion: float,
    mobility: float,
    k_spring: float,
) -> tuple[np.ndarray, float]:
    dx = float(x[1] - x[0])
    j_face = compute_flux_faces(
        p=p,
        x=x,
        diffusion=diffusion,
        mobility=mobility,
        k_spring=k_spring,
    )

    divergence = (j_face[1:] - j_face[:-1]) / dx
    p_next = p - dt * divergence

    clipped_negative_mass = trapz_integral(np.maximum(-p_next, 0.0), x)
    p_next = np.maximum(p_next, 0.0)
    p_next = normalize_density(p_next, x)
    return p_next, clipped_negative_mass


def summarize_state(
    time_value: float,
    x: np.ndarray,
    p: np.ndarray,
    p_eq: np.ndarray,
) -> dict[str, float]:
    mass = trapz_integral(p, x)
    mean_x = trapz_integral(x * p, x)
    variance_x = trapz_integral((x - mean_x) ** 2 * p, x)
    l1_error = trapz_integral(np.abs(p - p_eq), x)
    eps = 1e-14
    kl_div = trapz_integral(p * np.log((p + eps) / (p_eq + eps)), x)

    return {
        "time": float(time_value),
        "mass": float(mass),
        "mean_x": float(mean_x),
        "var_x": float(variance_x),
        "l1_to_equilibrium": float(l1_error),
        "kl_to_equilibrium": float(kl_div),
    }


def run_simulation(config: SimulationConfig) -> SimulationResult:
    if config.grid_size < 5:
        raise ValueError("grid_size must be >= 5.")
    if config.total_time <= 0.0:
        raise ValueError("total_time must be positive.")
    if config.k_spring <= 0.0 or config.mobility <= 0.0 or config.kbt <= 0.0:
        raise ValueError("k_spring, mobility, and kbt must be positive.")

    x = np.linspace(-config.domain_half_width, config.domain_half_width, config.grid_size)
    diffusion = config.mobility * config.kbt
    dt, num_steps = choose_stable_dt(
        x=x,
        diffusion=diffusion,
        mobility=config.mobility,
        k_spring=config.k_spring,
        total_time=config.total_time,
        cfl_safety=config.cfl_safety,
    )

    p = gaussian_density(x, mean=config.initial_mean, std=config.initial_std)
    p = normalize_density(p, x)
    p_eq = equilibrium_density(x, k_spring=config.k_spring, kbt=config.kbt)

    checkpoint_times = sorted(set(float(t) for t in config.checkpoint_times))
    checkpoint_times = [t for t in checkpoint_times if 0.0 <= t <= config.total_time]
    if 0.0 not in checkpoint_times:
        checkpoint_times = [0.0] + checkpoint_times
    if config.total_time not in checkpoint_times:
        checkpoint_times.append(config.total_time)

    records: list[dict[str, float]] = []
    records.append(summarize_state(0.0, x, p, p_eq))

    total_clipped = 0.0
    checkpoint_idx = 1
    for step in range(1, num_steps + 1):
        p, clipped = smoluchowski_step(
            p=p,
            x=x,
            dt=dt,
            diffusion=diffusion,
            mobility=config.mobility,
            k_spring=config.k_spring,
        )
        total_clipped += clipped
        current_time = step * dt

        while checkpoint_idx < len(checkpoint_times) and current_time >= checkpoint_times[checkpoint_idx] - 0.5 * dt:
            records.append(summarize_state(checkpoint_times[checkpoint_idx], x, p, p_eq))
            checkpoint_idx += 1

    checkpoints_df = pd.DataFrame(records)
    return SimulationResult(
        x=x,
        p_final=p,
        p_equilibrium=p_eq,
        checkpoints=checkpoints_df,
        dt=dt,
        num_steps=num_steps,
        total_clipped_negative_mass=float(total_clipped),
    )


def main() -> None:
    config = SimulationConfig()
    result = run_simulation(config)

    print("=== Smoluchowski Equation MVP (1D, harmonic potential) ===")
    print(
        "config:",
        f"L={config.domain_half_width}, grid_size={config.grid_size}, total_time={config.total_time},",
        f"k={config.k_spring}, mu={config.mobility}, kBT={config.kbt},",
        f"initial_mean={config.initial_mean}, initial_std={config.initial_std}",
    )
    print(f"derived diffusion D = mu*kBT = {config.mobility * config.kbt:.6f}")
    print(f"time step dt = {result.dt:.6e}, steps = {result.num_steps}")
    print(f"total clipped negative mass (safeguard metric) = {result.total_clipped_negative_mass:.6e}")
    print()
    print(result.checkpoints.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
    print()

    initial_l1 = float(result.checkpoints.iloc[0]["l1_to_equilibrium"])
    final_l1 = float(result.checkpoints.iloc[-1]["l1_to_equilibrium"])
    mass_err = abs(float(result.checkpoints.iloc[-1]["mass"]) - 1.0)

    print(f"initial L1 distance to equilibrium = {initial_l1:.6f}")
    print(f"final   L1 distance to equilibrium = {final_l1:.6f}")
    print(f"final mass conservation error       = {mass_err:.3e}")
    print(f"equilibrium approach check          = {'PASS' if final_l1 < initial_l1 else 'CHECK_MANUALLY'}")


if __name__ == "__main__":
    main()
