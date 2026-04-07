"""Special Relativity MVP: 1D Lorentz transform and core numerical checks."""

from __future__ import annotations

from typing import Any

import numpy as np

C = 299_792_458.0  # m/s
EPS = 1e-12


def validate_beta(beta: float) -> float:
    """Validate beta=v/c for physical and numerical correctness."""
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    if abs(beta) >= 1.0:
        raise ValueError("beta must satisfy |beta| < 1")
    return beta


def validate_events(events: np.ndarray) -> np.ndarray:
    """Validate event array shape (n,2) where each row is [ct, x]."""
    arr = np.asarray(events, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("events must have shape (n, 2) with columns [ct, x]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("events contain non-finite values")
    return arr


def gamma_from_beta(beta: float) -> float:
    """Lorentz factor gamma = 1 / sqrt(1 - beta^2)."""
    beta = validate_beta(beta)
    return 1.0 / np.sqrt(1.0 - beta * beta)


def lorentz_transform_1d(events: np.ndarray, beta: float) -> np.ndarray:
    """Apply 1D Lorentz transform along x-axis for events [ct, x]."""
    events = validate_events(events)
    beta = validate_beta(beta)
    gamma = gamma_from_beta(beta)

    ct = events[:, 0]
    x = events[:, 1]

    ct_prime = gamma * (ct - beta * x)
    x_prime = gamma * (x - beta * ct)
    return np.column_stack((ct_prime, x_prime))


def minkowski_interval_sq(events: np.ndarray) -> np.ndarray:
    """Compute s^2=(ct)^2-x^2 for each event."""
    events = validate_events(events)
    ct = events[:, 0]
    x = events[:, 1]
    return ct * ct - x * x


def velocity_addition(u: float, v: float, c: float = C) -> float:
    """Relativistic velocity addition: (u+v)/(1+uv/c^2)."""
    u = float(u)
    v = float(v)
    c = float(c)

    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(c)):
        raise ValueError("u, v, c must be finite")
    if c <= 0.0:
        raise ValueError("c must be positive")
    if abs(u) >= c or abs(v) >= c:
        raise ValueError("u and v must satisfy |u|<c and |v|<c")

    denom = 1.0 + (u * v) / (c * c)
    if abs(denom) < EPS:
        raise ZeroDivisionError("velocity addition denominator is too close to zero")

    return (u + v) / denom


def max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Stable max relative error metric."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return float(np.max(np.abs(a - b) / denom))


def run_interval_invariance_demo(
    rng: np.random.Generator,
    n_events: int,
    betas: list[float],
) -> list[dict[str, Any]]:
    """Generate random events and check interval invariance after transform."""
    t = rng.uniform(-5.0, 5.0, size=n_events)  # seconds
    x = rng.uniform(-1.0e9, 1.0e9, size=n_events)  # meters
    events = np.column_stack((C * t, x))

    base_interval = minkowski_interval_sq(events)
    reports: list[dict[str, Any]] = []

    for beta in betas:
        transformed = lorentz_transform_1d(events, beta)
        transformed_interval = minkowski_interval_sq(transformed)
        abs_err = np.abs(transformed_interval - base_interval)

        reports.append(
            {
                "beta": float(beta),
                "gamma": float(gamma_from_beta(beta)),
                "max_abs_error": float(np.max(abs_err)),
                "max_rel_error": max_relative_error(transformed_interval, base_interval),
            }
        )

    return reports


def run_time_dilation_demo(beta: float) -> dict[str, Any]:
    """Verify Delta t = gamma * Delta tau on fixed examples."""
    gamma = gamma_from_beta(beta)
    proper_time = np.array([1.0, 10.0, 60.0], dtype=float)
    frame_time = gamma * proper_time
    recovered_proper_time = frame_time / gamma

    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "proper_time_s": proper_time,
        "frame_time_s": frame_time,
        "recover_error_max": float(np.max(np.abs(recovered_proper_time - proper_time))),
    }


def run_relativity_of_simultaneity_demo(beta: float) -> dict[str, Any]:
    """Two events simultaneous in one frame are generally not simultaneous in another."""
    beta = validate_beta(beta)
    gamma = gamma_from_beta(beta)

    t0 = 2.0e-6  # s
    x1 = 0.0
    x2 = 900.0  # m

    events = np.array(
        [
            [C * t0, x1],
            [C * t0, x2],
        ],
        dtype=float,
    )

    transformed = lorentz_transform_1d(events, beta)
    delta_t_prime = (transformed[1, 0] - transformed[0, 0]) / C
    expected = -gamma * beta * (x2 - x1) / C

    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "delta_x_m": float(x2 - x1),
        "delta_t_in_lab_s": 0.0,
        "delta_t_prime_s": float(delta_t_prime),
        "expected_delta_t_prime_s": float(expected),
        "abs_error": float(abs(delta_t_prime - expected)),
    }


def run_velocity_addition_demo(rng: np.random.Generator, n_pairs: int) -> dict[str, Any]:
    """Compare classical and relativistic composition and test subluminal bound."""
    u = 0.8 * C
    v = 0.7 * C

    classical = u + v
    relativistic = velocity_addition(u, v)

    uv = rng.uniform(-0.99 * C, 0.99 * C, size=(n_pairs, 2))
    composed = np.array([velocity_addition(ui, vi) for ui, vi in uv], dtype=float)

    return {
        "u_over_c": float(u / C),
        "v_over_c": float(v / C),
        "classical_sum_over_c": float(classical / C),
        "relativistic_sum_over_c": float(relativistic / C),
        "all_sub_luminal": bool(np.all(np.abs(composed) < C)),
        "max_composed_speed_over_c": float(np.max(np.abs(composed) / C)),
    }


def main() -> None:
    rng = np.random.default_rng(41)

    betas = [0.1, -0.3, 0.75, 0.92]
    interval_reports = run_interval_invariance_demo(rng=rng, n_events=4000, betas=betas)
    dilation_report = run_time_dilation_demo(beta=0.8)
    simultaneity_report = run_relativity_of_simultaneity_demo(beta=0.6)
    velocity_report = run_velocity_addition_demo(rng=rng, n_pairs=2000)

    interval_ok = all(r["max_rel_error"] < 1e-10 for r in interval_reports)
    dilation_ok = dilation_report["recover_error_max"] < 1e-12
    simultaneity_ok = (
        abs(simultaneity_report["delta_t_prime_s"]) > 0.0
        and simultaneity_report["abs_error"] < 1e-15
    )
    velocity_ok = velocity_report["all_sub_luminal"]
    all_ok = interval_ok and dilation_ok and simultaneity_ok and velocity_ok

    print("=== Special Relativity MVP (1D) ===")
    print("[Interval Invariance]")
    for r in interval_reports:
        print(
            f"beta={r['beta']:+.2f}, gamma={r['gamma']:.6f}, "
            f"max_abs_error={r['max_abs_error']:.6e}, "
            f"max_rel_error={r['max_rel_error']:.6e}"
        )

    print("\n[Time Dilation]")
    print(f"beta={dilation_report['beta']:.2f}, gamma={dilation_report['gamma']:.6f}")
    print(f"proper_time_s={dilation_report['proper_time_s']}")
    print(f"frame_time_s ={dilation_report['frame_time_s']}")
    print(f"recover_error_max={dilation_report['recover_error_max']:.6e}")

    print("\n[Relativity of Simultaneity]")
    print(
        f"delta_t_in_lab_s={simultaneity_report['delta_t_in_lab_s']:.6e}, "
        f"delta_t_prime_s={simultaneity_report['delta_t_prime_s']:.6e}, "
        f"expected={simultaneity_report['expected_delta_t_prime_s']:.6e}, "
        f"abs_error={simultaneity_report['abs_error']:.6e}"
    )

    print("\n[Velocity Addition]")
    print(
        f"u/c={velocity_report['u_over_c']:.3f}, "
        f"v/c={velocity_report['v_over_c']:.3f}, "
        f"classical_sum/c={velocity_report['classical_sum_over_c']:.6f}, "
        f"relativistic_sum/c={velocity_report['relativistic_sum_over_c']:.6f}"
    )
    print(
        f"all_sub_luminal={velocity_report['all_sub_luminal']}, "
        f"max_composed_speed_over_c={velocity_report['max_composed_speed_over_c']:.6f}"
    )

    print("\n[Checks]")
    print(
        f"interval_ok={interval_ok}, dilation_ok={dilation_ok}, "
        f"simultaneity_ok={simultaneity_ok}, velocity_ok={velocity_ok}"
    )
    print(f"OVERALL={'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
