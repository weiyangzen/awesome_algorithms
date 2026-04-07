"""Minimal runnable MVP for Lorenz Attractor (PHYS-0118).

This script explicitly implements:
1) Lorenz ODE right-hand side
2) Fixed-step RK4 integration
3) Largest Lyapunov exponent estimation via perturbation renormalization
4) Short-horizon cross-check against scipy.solve_ivp
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def validate_inputs(initial_state: np.ndarray, dt: float, steps: int, params: LorenzParams) -> None:
    if initial_state.shape != (3,):
        raise ValueError("initial_state must be a 3D vector.")
    if not np.isfinite(initial_state).all():
        raise ValueError("initial_state must contain finite values.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    p = np.array([params.sigma, params.rho, params.beta], dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Lorenz parameters must be finite.")


def lorenz_rhs(state: np.ndarray, params: LorenzParams) -> np.ndarray:
    x, y, z = state
    dx = params.sigma * (y - x)
    dy = x * (params.rho - z) - y
    dz = x * y - params.beta * z
    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state: np.ndarray, dt: float, params: LorenzParams) -> np.ndarray:
    k1 = lorenz_rhs(state, params)
    k2 = lorenz_rhs(state + 0.5 * dt * k1, params)
    k3 = lorenz_rhs(state + 0.5 * dt * k2, params)
    k4 = lorenz_rhs(state + dt * k3, params)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state


def integrate_lorenz(
    initial_state: np.ndarray,
    params: LorenzParams,
    dt: float,
    steps: int,
) -> np.ndarray:
    validate_inputs(initial_state=initial_state, dt=dt, steps=steps, params=params)
    traj = np.empty((steps + 1, 3), dtype=float)
    traj[0] = initial_state

    for i in range(steps):
        traj[i + 1] = rk4_step(traj[i], dt, params)
        if not np.isfinite(traj[i + 1]).all():
            raise FloatingPointError(f"Numerical overflow/NaN detected at step {i + 1}.")

    return traj


def summarize_trajectory(traj: np.ndarray, burn_in: int) -> dict[str, np.ndarray | float | int]:
    if burn_in < 0 or burn_in >= traj.shape[0]:
        raise ValueError("burn_in must satisfy 0 <= burn_in < len(trajectory).")

    post = traj[burn_in:]
    x_sign = np.sign(post[:, 0])
    switches = int(np.count_nonzero(x_sign[1:] * x_sign[:-1] < 0.0))

    return {
        "final_state": traj[-1],
        "mean_state": np.mean(post, axis=0),
        "std_state": np.std(post, axis=0),
        "bbox_min": np.min(post, axis=0),
        "bbox_max": np.max(post, axis=0),
        "wing_switches": switches,
    }


def estimate_largest_lyapunov(
    base_state: np.ndarray,
    params: LorenzParams,
    dt: float,
    steps: int,
    d0: float = 1e-8,
) -> float:
    if d0 <= 0.0:
        raise ValueError("d0 must be positive.")

    s1 = np.array(base_state, dtype=float)
    s2 = s1 + np.array([d0, 0.0, 0.0], dtype=float)
    sum_log = 0.0

    for i in range(steps):
        s1 = rk4_step(s1, dt, params)
        s2 = rk4_step(s2, dt, params)

        delta = s2 - s1
        dist = float(np.linalg.norm(delta))
        if not np.isfinite(dist) or dist < 1e-20:
            raise FloatingPointError(
                f"Invalid perturbation distance at step {i + 1}: {dist}"
            )

        sum_log += np.log(dist / d0)
        s2 = s1 + (d0 / dist) * delta

    return float(sum_log / (steps * dt))


def reference_error_against_scipy(
    initial_state: np.ndarray,
    params: LorenzParams,
    dt: float,
    steps: int,
) -> float:
    rk4_traj = integrate_lorenz(initial_state, params, dt=dt, steps=steps)
    t_eval = np.linspace(0.0, steps * dt, steps + 1)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return lorenz_rhs(y, params)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, steps * dt),
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    ref = sol.y.T
    rmse = float(np.sqrt(np.mean((rk4_traj - ref) ** 2)))
    return rmse


def run_checks(
    traj: np.ndarray,
    summary: dict[str, np.ndarray | float | int],
    lyapunov: float,
    ref_rmse: float,
    params: LorenzParams,
    dt: float,
    steps: int,
    initial_state: np.ndarray,
) -> None:
    if not np.isfinite(traj).all():
        raise AssertionError("Trajectory contains non-finite values.")

    max_abs = float(np.max(np.abs(traj)))
    if max_abs > 100.0:
        raise AssertionError(f"Trajectory seems divergent: max_abs={max_abs:.3f}")

    if lyapunov <= 0.1:
        raise AssertionError(f"Largest Lyapunov exponent too small: {lyapunov:.6f}")

    if ref_rmse > 2e-2:
        raise AssertionError(f"Short-horizon RK4 vs solve_ivp RMSE too large: {ref_rmse:.6e}")

    # Determinism check: identical configuration should produce the same trajectory.
    traj2 = integrate_lorenz(initial_state, params, dt=dt, steps=steps)
    if not np.array_equal(traj, traj2):
        diff = float(np.max(np.abs(traj - traj2)))
        raise AssertionError(f"Determinism check failed, max diff = {diff:.3e}")

    switches = int(summary["wing_switches"])
    if switches < 10:
        raise AssertionError(f"Too few wing switches ({switches}); unexpected orbit behavior.")


def main() -> None:
    params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    dt = 0.01
    steps = 10_000
    burn_in = 3_000
    initial_state = np.array([1.0, 1.0, 1.0], dtype=float)

    traj = integrate_lorenz(initial_state, params, dt=dt, steps=steps)
    summary = summarize_trajectory(traj, burn_in=burn_in)

    base_for_lyapunov = traj[burn_in]
    lyapunov = estimate_largest_lyapunov(
        base_state=base_for_lyapunov,
        params=params,
        dt=dt,
        steps=steps - burn_in,
        d0=1e-8,
    )

    ref_rmse = reference_error_against_scipy(
        initial_state=initial_state,
        params=params,
        dt=dt,
        steps=300,
    )

    run_checks(
        traj=traj,
        summary=summary,
        lyapunov=lyapunov,
        ref_rmse=ref_rmse,
        params=params,
        dt=dt,
        steps=steps,
        initial_state=initial_state,
    )

    print("Lorenz Attractor MVP (RK4 + Lyapunov + reference cross-check)")
    print(
        "params="
        f"(sigma={params.sigma}, rho={params.rho}, beta={params.beta})"
    )
    print(f"dt={dt}, steps={steps}, burn_in={burn_in}")
    print(f"final_state={summary['final_state']}")
    print(f"mean_state_after_burnin={summary['mean_state']}")
    print(f"std_state_after_burnin={summary['std_state']}")
    print(f"bbox_min={summary['bbox_min']}")
    print(f"bbox_max={summary['bbox_max']}")
    print(f"wing_switches={summary['wing_switches']}")
    print(f"largest_lyapunov={lyapunov:.6f}")
    print(f"reference_short_horizon_rmse={ref_rmse:.6e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
