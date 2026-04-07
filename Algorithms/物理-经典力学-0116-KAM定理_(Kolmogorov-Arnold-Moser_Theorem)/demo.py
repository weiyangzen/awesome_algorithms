"""KAM theorem numerical proxy via the Chirikov standard map.

This MVP does not prove the KAM theorem. It demonstrates a common
computational proxy: as perturbation strength K increases, invariant-torus-like
regular behavior decreases and chaotic indicators increase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TWOPI = 2.0 * np.pi
GOLDEN_RATIO_CONJ = (np.sqrt(5.0) - 1.0) * 0.5


@dataclass(frozen=True)
class KAMConfig:
    """Configuration for the standard-map KAM proxy experiment."""

    k_values: tuple[float, ...] = (0.20, 0.80, 1.40)
    n_initial_conditions: int = 72
    n_steps: int = 4000
    burn_in: int = 800
    x0_phase_shift: float = 0.13
    p0_min: float = -np.pi
    p0_max: float = np.pi
    regular_lyapunov_threshold: float = 0.02


def standard_map_step(x: float, p: float, k: float) -> tuple[float, float]:
    """One step of the kicked-rotor/standard map on cylinder phase space.

    Equations:
        p_{n+1} = p_n + K sin(x_n)
        x_{n+1} = (x_n + p_{n+1}) mod 2pi
    """

    p_next = p + k * np.sin(x)
    x_next = (x + p_next) % TWOPI
    return float(x_next), float(p_next)


def generate_initial_conditions(cfg: KAMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic initial-condition ensemble."""

    if cfg.n_initial_conditions < 8:
        raise ValueError("n_initial_conditions must be >= 8.")

    idx = np.arange(cfg.n_initial_conditions, dtype=float)
    x0 = TWOPI * ((idx * GOLDEN_RATIO_CONJ + cfg.x0_phase_shift) % 1.0)
    p0 = np.linspace(cfg.p0_min, cfg.p0_max, cfg.n_initial_conditions, dtype=float)
    return x0.astype(float), p0.astype(float)


def estimate_rotation_numbers(k: float, x0: np.ndarray, p0: np.ndarray, cfg: KAMConfig) -> np.ndarray:
    """Estimate rotation number omega/(2pi) by time-averaging lifted increments."""

    x = x0.copy()
    p = p0.copy()
    lifted_x_increment_sum = np.zeros_like(x)
    kept_steps = 0

    for step_idx in range(cfg.n_steps):
        p = p + k * np.sin(x)
        x = (x + p) % TWOPI
        if step_idx >= cfg.burn_in:
            lifted_x_increment_sum += p
            kept_steps += 1

    if kept_steps <= 0:
        raise ValueError("burn_in must be smaller than n_steps.")

    return lifted_x_increment_sum / (kept_steps * TWOPI)


def finite_time_lyapunov(k: float, x0: float, p0: float, n_steps: int) -> float:
    """Compute finite-time Lyapunov exponent for one trajectory.

    The Jacobian at x_n is:
        [[1 + K cos(x_n), 1],
         [K cos(x_n),     1]]
    """

    x = float(x0)
    p = float(p0)
    tangent = np.array([1.0, 0.0], dtype=float)
    log_growth_sum = 0.0

    for _ in range(n_steps):
        c = k * np.cos(x)
        j11 = 1.0 + c
        j12 = 1.0
        j21 = c
        j22 = 1.0

        v0 = j11 * tangent[0] + j12 * tangent[1]
        v1 = j21 * tangent[0] + j22 * tangent[1]
        norm_v = float(np.hypot(v0, v1))
        if norm_v < 1e-15:
            norm_v = 1e-15

        log_growth_sum += float(np.log(norm_v))
        tangent[0] = v0 / norm_v
        tangent[1] = v1 / norm_v

        x, p = standard_map_step(x, p, k)

    return float(log_growth_sum / n_steps)


def rotation_curve_roughness(rotation_sorted: np.ndarray) -> float:
    """Second-difference roughness: smooth quasi-periodic curves have lower values."""

    if rotation_sorted.size < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(rotation_sorted, n=2))))


def plateau_fraction(rotation_sorted: np.ndarray, slope_tol: float = 5e-3) -> float:
    """Fraction of near-flat local slopes in rotation curve."""

    if rotation_sorted.size < 2:
        return float("nan")
    local_slope = np.abs(np.diff(rotation_sorted))
    return float(np.mean(local_slope < slope_tol))


def analyze_single_k(k: float, cfg: KAMConfig, x0: np.ndarray, p0: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Analyze one perturbation level K."""

    rotation = estimate_rotation_numbers(k, x0, p0, cfg)
    lyapunov = np.array([finite_time_lyapunov(k, xi, pi, cfg.n_steps) for xi, pi in zip(x0, p0)], dtype=float)

    order = np.argsort(p0)
    p_sorted = p0[order]
    rotation_sorted = rotation[order]
    lyapunov_sorted = lyapunov[order]

    regular_mask = lyapunov_sorted < cfg.regular_lyapunov_threshold

    per_ic = pd.DataFrame(
        {
            "K": float(k),
            "p0": p_sorted,
            "x0": x0[order],
            "rotation_number": rotation_sorted,
            "lyapunov": lyapunov_sorted,
            "is_regular": regular_mask.astype(int),
        }
    )

    summary = {
        "K": float(k),
        "median_lyapunov": float(np.median(lyapunov_sorted)),
        "p90_lyapunov": float(np.percentile(lyapunov_sorted, 90)),
        "regular_fraction": float(np.mean(regular_mask)),
        "rotation_roughness": rotation_curve_roughness(rotation_sorted),
        "rotation_plateau_fraction": plateau_fraction(rotation_sorted),
        "rotation_span": float(np.max(rotation_sorted) - np.min(rotation_sorted)),
    }
    return per_ic, summary


def run_kam_proxy(cfg: KAMConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run KAM proxy experiment for all configured K values."""

    if cfg.n_steps <= 50:
        raise ValueError("n_steps must be > 50.")
    if cfg.burn_in >= cfg.n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")
    if cfg.p0_min >= cfg.p0_max:
        raise ValueError("p0_min must be < p0_max.")

    x0, p0 = generate_initial_conditions(cfg)
    per_ic_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float]] = []

    for k in cfg.k_values:
        per_ic, summary = analyze_single_k(float(k), cfg, x0, p0)
        per_ic_tables.append(per_ic)
        summary_rows.append(summary)

    all_per_ic = pd.concat(per_ic_tables, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("K").reset_index(drop=True)
    return all_per_ic, summary_df


def main() -> None:
    cfg = KAMConfig()
    per_ic, summary = run_kam_proxy(cfg)

    print("KAM Theorem Numerical Proxy (Standard Map)")
    print(
        f"K={cfg.k_values}, n_ic={cfg.n_initial_conditions}, n_steps={cfg.n_steps}, "
        f"burn_in={cfg.burn_in}, regular_threshold={cfg.regular_lyapunov_threshold}"
    )

    print("\nsummary_by_K:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    sample_rows = (
        per_ic.groupby("K", group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )
    print("\nper_ic_sample(first 4 rows for each K):")
    print(
        sample_rows.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    low = summary.iloc[0]
    high = summary.iloc[-1]
    if high["median_lyapunov"] <= low["median_lyapunov"]:
        raise AssertionError("Expected larger K to have larger median Lyapunov exponent.")
    if high["regular_fraction"] >= low["regular_fraction"]:
        raise AssertionError("Expected larger K to have lower regular fraction.")
    if high["rotation_roughness"] <= low["rotation_roughness"]:
        raise AssertionError("Expected larger K to have rougher rotation-number curve.")


if __name__ == "__main__":
    main()
