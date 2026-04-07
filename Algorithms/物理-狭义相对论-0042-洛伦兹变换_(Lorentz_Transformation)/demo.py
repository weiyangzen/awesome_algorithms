"""Minimal runnable MVP for Lorentz Transformation in 1+1 dimensions.

Conventions:
- Natural units c = 1.
- Event vectors are [t, x].
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def lorentz_gamma(beta: float) -> float:
    """Return gamma = 1/sqrt(1-beta^2) for |beta|<1."""
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    if abs(beta) >= 1.0:
        raise ValueError("|beta| must be < 1.")
    return float(1.0 / np.sqrt(1.0 - beta * beta))


def lorentz_matrix(beta: float) -> np.ndarray:
    """Return the 1+1 Lorentz boost matrix along x."""
    g = lorentz_gamma(beta)
    return np.array([[g, -g * beta], [-g * beta, g]], dtype=np.float64)


def transform_events(events: np.ndarray, beta: float) -> np.ndarray:
    """Transform event rows [t, x] from frame S to S' moving at +beta."""
    ev = np.asarray(events, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[1] != 2:
        raise ValueError("events must have shape (N, 2) with rows [t, x].")
    return ev @ lorentz_matrix(beta).T


def minkowski_interval_sq(events: np.ndarray) -> np.ndarray:
    """Return s^2 = t^2 - x^2 for each event row [t, x]."""
    ev = np.asarray(events, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[1] != 2:
        raise ValueError("events must have shape (N, 2).")
    return ev[:, 0] ** 2 - ev[:, 1] ** 2


def velocity_addition(u: float, beta: float) -> float:
    """Return transformed velocity u' = (u-beta)/(1-u*beta)."""
    u = float(u)
    beta = float(beta)
    if abs(u) >= 1.0:
        raise ValueError("|u| must be < 1.")
    if abs(beta) >= 1.0:
        raise ValueError("|beta| must be < 1.")
    denom = 1.0 - u * beta
    if abs(denom) < 1e-14:
        raise ValueError("denominator too close to zero for stable evaluation.")
    return float((u - beta) / denom)


def build_uniform_worldline(u: float, t_start: float, t_end: float, n_points: int, x0: float) -> np.ndarray:
    """Create rows [t, x] for uniform motion x = x0 + u*(t-t_start)."""
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    t = np.linspace(float(t_start), float(t_end), int(n_points), dtype=np.float64)
    x = float(x0) + float(u) * (t - float(t_start))
    return np.column_stack([t, x])


def estimate_velocity_from_worldline(worldline: np.ndarray) -> float:
    """Estimate velocity by averaging dx/dt over a sampled worldline."""
    wl = np.asarray(worldline, dtype=np.float64)
    if wl.ndim != 2 or wl.shape[1] != 2:
        raise ValueError("worldline must have shape (N, 2).")

    dt = np.diff(wl[:, 0])
    dx = np.diff(wl[:, 1])
    if np.any(dt <= 0.0):
        raise ValueError("worldline time must be strictly increasing.")
    return float(np.mean(dx / dt))


def run_interval_invariance_demo() -> Dict[str, float]:
    """Check interval invariance and inverse transform closure."""
    beta = 0.60
    events = np.array(
        [
            [0.0, 0.0],
            [1.2, 0.3],
            [2.4, -1.1],
            [3.5, 2.0],
            [5.0, 0.8],
            [7.2, -3.0],
        ],
        dtype=np.float64,
    )

    events_prime = transform_events(events, beta=beta)
    s2 = minkowski_interval_sq(events)
    s2_prime = minkowski_interval_sq(events_prime)

    max_abs_err = float(np.max(np.abs(s2_prime - s2)))
    denom = max(1e-12, float(np.max(np.abs(s2))))
    max_rel_err = max_abs_err / denom

    recovered = transform_events(events_prime, beta=-beta)
    recovery_max_abs_err = float(np.max(np.abs(recovered - events)))

    identity_err = float(np.max(np.abs(lorentz_matrix(-beta) @ lorentz_matrix(beta) - np.eye(2, dtype=np.float64))))

    assert max_abs_err < 1e-12, f"Interval invariance absolute error too large: {max_abs_err:.3e}"
    assert recovery_max_abs_err < 1e-12, f"Inverse transform recovery error too large: {recovery_max_abs_err:.3e}"
    assert identity_err < 1e-12, f"Lambda(-beta)Lambda(beta) closure error too large: {identity_err:.3e}"

    return {
        "beta": beta,
        "max_interval_abs_err": max_abs_err,
        "max_interval_rel_err": max_rel_err,
        "recovery_max_abs_err": recovery_max_abs_err,
        "matrix_closure_err": identity_err,
    }


def run_velocity_addition_demo() -> Dict[str, float]:
    """Check worldline slope in S' against analytic relativistic velocity addition."""
    beta = 0.45
    u = 0.70

    worldline_s = build_uniform_worldline(u=u, t_start=0.0, t_end=20.0, n_points=20_001, x0=1.2)
    worldline_sp = transform_events(worldline_s, beta=beta)

    u_num = estimate_velocity_from_worldline(worldline_sp)
    u_ref = velocity_addition(u=u, beta=beta)
    abs_err = abs(u_num - u_ref)
    rel_err = abs_err / max(1e-12, abs(u_ref))

    assert abs(u_num) < 1.0, "Transformed speed must remain subluminal."
    assert rel_err < 1e-10, f"Velocity-addition mismatch too large: {rel_err:.3e}"

    return {
        "u_in_S": u,
        "beta_frame": beta,
        "u_prime_num": u_num,
        "u_prime_ref": u_ref,
        "u_prime_abs_err": abs_err,
        "u_prime_rel_err": rel_err,
    }


def run_relativity_of_simultaneity_demo() -> Dict[str, float]:
    """Show that events simultaneous in S are not simultaneous in S'."""
    beta = 0.80
    g = lorentz_gamma(beta)

    # Two events at the same S-time but different positions.
    t0 = 10.0
    x1 = 0.5
    x2 = 3.5
    events = np.array([[t0, x1], [t0, x2]], dtype=np.float64)

    events_prime = transform_events(events, beta=beta)
    delta_t_prime_num = float(events_prime[1, 0] - events_prime[0, 0])

    delta_x = x2 - x1
    delta_t_prime_ref = float(-g * beta * delta_x)
    abs_err = abs(delta_t_prime_num - delta_t_prime_ref)

    assert abs_err < 1e-12, f"Relativity-of-simultaneity formula mismatch: {abs_err:.3e}"
    assert abs(delta_t_prime_num) > 1e-6, "Expected non-zero simultaneity shift in S'."

    return {
        "beta": beta,
        "delta_x_in_S": delta_x,
        "delta_t_prime_num": delta_t_prime_num,
        "delta_t_prime_ref": delta_t_prime_ref,
        "delta_t_prime_abs_err": abs_err,
    }


def main() -> None:
    print("Lorentz Transformation MVP (1+1D, c=1)")
    print()

    print("=== Demo A: Interval invariance and inverse transform ===")
    report_a = run_interval_invariance_demo()
    for key, value in report_a.items():
        print(f"{key:>24s}: {value:.12e}")

    print("\n=== Demo B: Relativistic velocity addition ===")
    report_b = run_velocity_addition_demo()
    for key, value in report_b.items():
        print(f"{key:>24s}: {value:.12e}")

    print("\n=== Demo C: Relativity of simultaneity ===")
    report_c = run_relativity_of_simultaneity_demo()
    for key, value in report_c.items():
        print(f"{key:>24s}: {value:.12e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
