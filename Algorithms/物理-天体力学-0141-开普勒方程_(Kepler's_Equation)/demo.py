"""Kepler's Equation MVP.

Solve M = E - e sin(E) for elliptical orbits (0 <= e < 1) with:
1) an explicit Newton/Halley iterative solver (main algorithm),
2) a SciPy brentq solver as a reference baseline.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import brentq

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class KeplerParams:
    e_values: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 0.9, 0.99)
    n_mean_anomalies: int = 360
    tol: float = 1e-12
    max_iter: int = 20
    random_seed: int = 42
    semi_major_axis: float = 1.0


def validate_params(params: KeplerParams) -> None:
    if params.n_mean_anomalies < 64:
        raise ValueError("n_mean_anomalies must be >= 64 for stable diagnostics.")
    if params.tol <= 0.0:
        raise ValueError("tol must be positive.")
    if params.max_iter < 3:
        raise ValueError("max_iter must be >= 3.")
    if params.semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis must be positive.")
    for e in params.e_values:
        if not (0.0 <= e < 1.0):
            raise ValueError(f"eccentricity must satisfy 0 <= e < 1, got {e}.")


def normalize_mean_anomaly(mean_anomaly: np.ndarray) -> np.ndarray:
    return np.mod(mean_anomaly, TWO_PI)


def initial_guess(mean_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    # Third-order Fourier-like starter, then a bias for high-e cases.
    e = ecc
    m = mean_anomaly
    guess = m + e * np.sin(m) + 0.5 * e * e * np.sin(2.0 * m)
    if e >= 0.8:
        bias = 0.85 * e * np.sign(np.sin(m))
        guess = m + bias
    return np.clip(guess, 0.0, TWO_PI)


def kepler_residual(ecc_anomaly: np.ndarray, mean_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    return ecc_anomaly - ecc * np.sin(ecc_anomaly) - mean_anomaly


def solve_kepler_newton_halley(
    mean_anomaly: np.ndarray, ecc: float, tol: float, max_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Newton/Halley solve.

    Returns:
        ecc_anomaly: solved E values
        n_iters: iterations used per sample
        converged: convergence mask
    """
    e = ecc
    m = normalize_mean_anomaly(mean_anomaly)
    e_anom = initial_guess(m, e)
    n_iters = np.zeros_like(m, dtype=np.int32)
    converged = np.zeros_like(m, dtype=bool)

    for k in range(1, max_iter + 1):
        f = kepler_residual(e_anom, m, e)
        fp = 1.0 - e * np.cos(e_anom)
        fpp = e * np.sin(e_anom)

        # Halley correction: E_{n+1} = E_n - f/(f' - 0.5 f f''/f')
        denom = fp - 0.5 * f * fpp / np.maximum(np.abs(fp), 1e-15)
        use_newton = np.abs(denom) < 1e-15
        step = np.empty_like(e_anom)
        step[~use_newton] = f[~use_newton] / denom[~use_newton]
        step[use_newton] = f[use_newton] / np.maximum(np.abs(fp[use_newton]), 1e-15)

        e_anom = e_anom - step
        e_anom = np.clip(e_anom, 0.0, TWO_PI)

        now_converged = np.abs(step) <= tol
        just_converged = (~converged) & now_converged
        n_iters[just_converged] = k
        converged |= now_converged
        if bool(np.all(converged)):
            break

    n_iters[~converged] = max_iter
    return e_anom, n_iters, converged


def solve_kepler_reference_brentq(mean_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    m = normalize_mean_anomaly(mean_anomaly)
    out = np.empty_like(m)

    for i, mi in enumerate(m):
        if np.isclose(mi, 0.0, atol=1e-15) or np.isclose(mi, TWO_PI, atol=1e-15):
            out[i] = 0.0
            continue

        def scalar_f(x: float) -> float:
            return x - ecc * np.sin(x) - mi

        out[i] = brentq(scalar_f, 0.0, TWO_PI, xtol=1e-14, rtol=1e-14, maxiter=200)

    return out


def true_anomaly_from_eccentric(ecc_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    s = np.sqrt(1.0 + ecc) * np.sin(ecc_anomaly / 2.0)
    c = np.sqrt(1.0 - ecc) * np.cos(ecc_anomaly / 2.0)
    return 2.0 * np.arctan2(s, c)


def orbital_radius(semi_major_axis: float, ecc_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    return semi_major_axis * (1.0 - ecc * np.cos(ecc_anomaly))


def build_report(params: KeplerParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(params.random_seed)
    base_m = np.linspace(0.0, TWO_PI, params.n_mean_anomalies, endpoint=False)
    jitter = rng.uniform(-1e-4, 1e-4, size=base_m.shape)
    mean_anomaly = normalize_mean_anomaly(base_m + jitter)

    rows: list[dict[str, float]] = []
    global_worst_residual = 0.0
    global_worst_error = 0.0
    global_worst_iter = 0
    global_convergence = 1.0

    for ecc in params.e_values:
        t0 = perf_counter()
        e_fast, n_iters, converged = solve_kepler_newton_halley(
            mean_anomaly=mean_anomaly, ecc=ecc, tol=params.tol, max_iter=params.max_iter
        )
        t_fast = perf_counter() - t0

        t1 = perf_counter()
        e_ref = solve_kepler_reference_brentq(mean_anomaly=mean_anomaly, ecc=ecc)
        t_ref = perf_counter() - t1

        residual = np.abs(kepler_residual(e_fast, mean_anomaly, ecc))
        abs_error = np.abs(e_fast - e_ref)

        # Orbit geometry diagnostics for one representative eccentricity.
        if np.isclose(ecc, 0.8):
            nu = true_anomaly_from_eccentric(e_fast, ecc)
            radius = orbital_radius(params.semi_major_axis, e_fast, ecc)
            nu_span = float(np.max(nu) - np.min(nu))
            r_min, r_max = float(np.min(radius)), float(np.max(radius))
        else:
            nu_span = np.nan
            r_min, r_max = np.nan, np.nan

        row = {
            "eccentricity_e": ecc,
            "n_samples": float(mean_anomaly.size),
            "converged_ratio": float(np.mean(converged)),
            "mean_iterations": float(np.mean(n_iters)),
            "max_iterations": float(np.max(n_iters)),
            "max_abs_residual": float(np.max(residual)),
            "mean_abs_residual": float(np.mean(residual)),
            "max_abs_error_vs_brentq": float(np.max(abs_error)),
            "mean_abs_error_vs_brentq": float(np.mean(abs_error)),
            "solver_time_ms": float(t_fast * 1e3),
            "reference_time_ms": float(t_ref * 1e3),
            "speedup_vs_ref": float((t_ref / t_fast) if t_fast > 0 else np.inf),
            "nu_span_for_e=0.8": nu_span,
            "radius_min_for_e=0.8": r_min,
            "radius_max_for_e=0.8": r_max,
        }
        rows.append(row)

        global_worst_residual = max(global_worst_residual, row["max_abs_residual"])
        global_worst_error = max(global_worst_error, row["max_abs_error_vs_brentq"])
        global_worst_iter = max(global_worst_iter, int(row["max_iterations"]))
        global_convergence = min(global_convergence, row["converged_ratio"])

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "global_worst_abs_residual", "value": global_worst_residual},
            {"metric": "global_worst_abs_error_vs_brentq", "value": global_worst_error},
            {"metric": "global_max_iterations", "value": float(global_worst_iter)},
            {"metric": "global_min_converged_ratio", "value": global_convergence},
            {"metric": "n_eccentricity_cases", "value": float(len(params.e_values))},
            {"metric": "n_samples_per_case", "value": float(params.n_mean_anomalies)},
        ]
    )
    return detail, summary


def main() -> None:
    params = KeplerParams()
    validate_params(params)

    detail, summary = build_report(params)

    print("=== Kepler Equation MVP (Elliptical Orbit, 0 <= e < 1) ===")
    print("params:", params)
    print("\n[Per-e report]")
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\n[Global summary]")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    worst_residual = float(summary.loc[summary["metric"] == "global_worst_abs_residual", "value"].iloc[0])
    worst_error = float(
        summary.loc[summary["metric"] == "global_worst_abs_error_vs_brentq", "value"].iloc[0]
    )
    min_conv = float(summary.loc[summary["metric"] == "global_min_converged_ratio", "value"].iloc[0])

    assert min_conv >= 1.0, "Not all samples converged."
    assert worst_residual <= 1e-10, f"Residual too large: {worst_residual}"
    assert worst_error <= 1e-10, f"Error vs reference too large: {worst_error}"


if __name__ == "__main__":
    main()
