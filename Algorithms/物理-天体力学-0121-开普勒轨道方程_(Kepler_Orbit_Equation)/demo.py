"""Kepler Orbit Equation MVP.

Solve the elliptical Kepler equation
    M = E - e * sin(E)
for eccentric anomaly E, then derive orbital quantities:
- true anomaly nu
- orbital radius r
- perifocal coordinates (x, y)

The implementation is intentionally explicit:
- Newton iterations as the main solver
- bisection fallback for hard/high-eccentricity points
- SciPy solver only used as a numeric cross-check

The script is deterministic and needs no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import newton


TWO_PI = 2.0 * np.pi


@dataclass
class KeplerSolveResult:
    eccentric_anomaly: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray


def validate_inputs(mean_anomaly: np.ndarray, eccentricity: float) -> None:
    if not np.all(np.isfinite(mean_anomaly)):
        raise ValueError("mean_anomaly contains non-finite values.")
    if not np.isfinite(eccentricity):
        raise ValueError("eccentricity must be finite.")
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError("This MVP handles elliptical orbits only: 0 <= e < 1.")


def normalize_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Map angles to [0, 2*pi)."""
    return np.mod(angle, TWO_PI)


def kepler_residual(eccentric_anomaly: np.ndarray, mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly


def solve_kepler_bisection_scalar(
    mean_anomaly: float,
    eccentricity: float,
    tol: float = 1e-13,
    max_iter: int = 200,
) -> Tuple[float, int]:
    """Robust scalar fallback on [0, 2*pi]."""
    lower = 0.0
    upper = TWO_PI

    f_lower = lower - eccentricity * np.sin(lower) - mean_anomaly
    f_upper = upper - eccentricity * np.sin(upper) - mean_anomaly

    if abs(f_lower) <= tol:
        return lower, 0
    if abs(f_upper) <= tol:
        return upper, 0
    if f_lower > 0.0 or f_upper < 0.0:
        raise RuntimeError("Bisection bracket failed. Inputs may be invalid.")

    for k in range(1, max_iter + 1):
        mid = 0.5 * (lower + upper)
        f_mid = mid - eccentricity * np.sin(mid) - mean_anomaly

        if abs(f_mid) <= tol or (upper - lower) <= tol:
            return mid, k

        if f_mid > 0.0:
            upper = mid
        else:
            lower = mid

    return 0.5 * (lower + upper), max_iter


def solve_kepler_newton(
    mean_anomaly: np.ndarray,
    eccentricity: float,
    tol: float = 1e-13,
    max_iter: int = 50,
) -> KeplerSolveResult:
    """Vectorized Newton solver with bisection fallback for non-converged entries."""
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    validate_inputs(mean_anomaly, eccentricity)

    m = normalize_angle_rad(mean_anomaly)

    if eccentricity < 0.8:
        e_anomaly = m.copy()
    else:
        e_anomaly = np.where(m < np.pi, np.pi, np.pi)

    converged = np.zeros_like(m, dtype=bool)
    iterations = np.zeros_like(m, dtype=int)

    for step in range(1, max_iter + 1):
        residual = kepler_residual(e_anomaly, m, eccentricity)
        derivative = 1.0 - eccentricity * np.cos(e_anomaly)
        delta = residual / derivative
        e_anomaly = e_anomaly - delta

        done = np.abs(delta) <= tol
        new_done = (~converged) & done
        iterations[new_done] = step
        converged |= done

        if np.all(converged):
            break

    if not np.all(converged):
        missing = np.argwhere(~converged)
        for idx in missing:
            idx_tuple = tuple(idx)
            solved, extra_iter = solve_kepler_bisection_scalar(float(m[idx_tuple]), eccentricity, tol=tol)
            e_anomaly[idx_tuple] = solved
            iterations[idx_tuple] = max_iter + extra_iter
            converged[idx_tuple] = True

    return KeplerSolveResult(
        eccentric_anomaly=e_anomaly,
        iterations=iterations,
        converged=converged,
    )


def eccentric_to_true_anomaly(eccentric_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    denom = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    sin_nu = np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly) / denom
    cos_nu = (np.cos(eccentric_anomaly) - eccentricity) / denom
    true_anomaly = np.arctan2(sin_nu, cos_nu)
    return normalize_angle_rad(true_anomaly)


def orbital_radius(semi_major_axis: float, eccentricity: float, eccentric_anomaly: np.ndarray) -> np.ndarray:
    return semi_major_axis * (1.0 - eccentricity * np.cos(eccentric_anomaly))


def perifocal_coordinates(
    semi_major_axis: float,
    eccentricity: float,
    eccentric_anomaly: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = semi_major_axis * (np.cos(eccentric_anomaly) - eccentricity)
    y = semi_major_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly)
    return x, y


def solve_kepler_scipy_reference(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    """Reference solution from scipy.optimize.newton for verification only."""
    m = normalize_angle_rad(np.asarray(mean_anomaly, dtype=float))
    out = np.empty_like(m)

    flat_m = m.reshape(-1)
    flat_out = out.reshape(-1)

    for i, m_i in enumerate(flat_m):
        x0 = float(m_i) if eccentricity < 0.8 else np.pi
        flat_out[i] = newton(
            func=lambda x: x - eccentricity * np.sin(x) - float(m_i),
            x0=x0,
            fprime=lambda x: 1.0 - eccentricity * np.cos(x),
            tol=1e-13,
            maxiter=100,
        )

    return out


def build_orbit_table(
    eccentricity: float,
    semi_major_axis: float,
    n_samples: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mean_anomaly = np.linspace(0.0, TWO_PI, n_samples, endpoint=False)
    solution = solve_kepler_newton(mean_anomaly, eccentricity=eccentricity)

    eccentric_anomaly = solution.eccentric_anomaly
    true_anomaly = eccentric_to_true_anomaly(eccentric_anomaly, eccentricity=eccentricity)
    radius = orbital_radius(semi_major_axis, eccentricity=eccentricity, eccentric_anomaly=eccentric_anomaly)
    x, y = perifocal_coordinates(semi_major_axis, eccentricity=eccentricity, eccentric_anomaly=eccentric_anomaly)

    residual = np.abs(kepler_residual(eccentric_anomaly, normalize_angle_rad(mean_anomaly), eccentricity))

    scipy_ref = solve_kepler_scipy_reference(mean_anomaly, eccentricity=eccentricity)
    diff_vs_ref = np.abs(normalize_angle_rad(eccentric_anomaly - scipy_ref))
    diff_vs_ref = np.minimum(diff_vs_ref, TWO_PI - diff_vs_ref)

    table = pd.DataFrame(
        {
            "M_rad": mean_anomaly,
            "E_rad": eccentric_anomaly,
            "nu_rad": true_anomaly,
            "r": radius,
            "x": x,
            "y": y,
            "residual_abs": residual,
            "iter": solution.iterations,
        }
    )

    metrics = {
        "max_residual": float(np.max(residual)),
        "max_abs_diff_vs_scipy": float(np.max(diff_vs_ref)),
        "max_iterations": int(np.max(solution.iterations)),
        "all_converged": float(np.all(solution.converged)),
        "r_min": float(np.min(radius)),
        "r_max": float(np.max(radius)),
    }
    return table, metrics


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 140)

    cases = [
        {"name": "Earth-like", "eccentricity": 0.0167, "semi_major_axis": 1.0, "n_samples": 16},
        {"name": "High-e ellipse", "eccentricity": 0.90, "semi_major_axis": 2.0, "n_samples": 16},
    ]

    print("Kepler Orbit Equation MVP")
    print("Equation: M = E - e*sin(E), with 0 <= e < 1")

    for case in cases:
        table, metrics = build_orbit_table(
            eccentricity=case["eccentricity"],
            semi_major_axis=case["semi_major_axis"],
            n_samples=case["n_samples"],
        )

        print("\n" + "=" * 88)
        print(
            f"Case={case['name']}, e={case['eccentricity']:.4f}, "
            f"a={case['semi_major_axis']:.3f}, samples={case['n_samples']}"
        )
        print(table.head(8).to_string(index=False, float_format=lambda v: f"{v: .6e}"))
        print(
            "Checks: "
            f"max_residual={metrics['max_residual']:.3e}, "
            f"max_abs_diff_vs_scipy={metrics['max_abs_diff_vs_scipy']:.3e}, "
            f"max_iterations={int(metrics['max_iterations'])}, "
            f"all_converged={bool(metrics['all_converged'])}, "
            f"r_range=[{metrics['r_min']:.6f}, {metrics['r_max']:.6f}]"
        )

        if metrics["max_residual"] > 1e-10:
            raise RuntimeError("Residual check failed: solution is not accurate enough.")
        if metrics["max_abs_diff_vs_scipy"] > 1e-10:
            raise RuntimeError("Cross-check failed: mismatch with SciPy reference is too large.")
        if not bool(metrics["all_converged"]):
            raise RuntimeError("Convergence check failed.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
