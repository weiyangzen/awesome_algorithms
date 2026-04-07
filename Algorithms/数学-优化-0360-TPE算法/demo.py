"""Minimal runnable MVP for Tree-structured Parzen Estimator (TPE).

This script demonstrates a transparent 1D TPE optimizer for black-box minimization:
- no interactive input
- deterministic via fixed random seed
- fully traceable source code (no optimizer black box)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class TrialRecord:
    iter_id: int
    source: str
    x: float
    y: float
    best_x: float
    best_y: float
    y_star: float
    score: float


@dataclass
class TPEResult:
    best_x: float
    best_y: float
    x_obs: Array
    y_obs: Array
    records: List[TrialRecord]


def rugged_objective(x: Array) -> Array:
    """Demo objective: 1D multi-modal function for minimization."""
    x = np.asarray(x, dtype=float)
    return 0.20 * (x - 1.8) ** 2 + np.sin(3.2 * x) + 0.30 * np.cos(5.0 * x + 0.2)


def validate_bounds(bounds: Sequence[float]) -> Tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two values: (lower, upper)")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("bounds values must be finite")
    if not lower < upper:
        raise ValueError("bounds must satisfy lower < upper")
    return lower, upper


def silverman_bandwidth(samples: Array, lower: float, upper: float) -> float:
    """Robust Silverman-like bandwidth with lower/upper clipping."""
    values = np.asarray(samples, dtype=float).reshape(-1)
    n = values.size
    span = upper - lower
    if n <= 1:
        return max(0.05 * span, 1e-3)

    std = float(np.std(values, ddof=1))
    q75, q25 = np.percentile(values, [75.0, 25.0])
    iqr = float(q75 - q25)
    robust_sigma = min(std, iqr / 1.349) if iqr > 0.0 else std
    if not np.isfinite(robust_sigma) or robust_sigma < 1e-12:
        robust_sigma = max(span / 20.0, 1e-3)

    bw = 0.9 * robust_sigma * (n ** (-1.0 / 5.0))
    return float(np.clip(bw, span * 1e-3, span * 0.5))


def parzen_density(points: Array, samples: Array, bandwidth: float) -> Array:
    """Evaluate Gaussian Parzen density: mean_i N(points | sample_i, bandwidth^2)."""
    x = np.asarray(points, dtype=float).reshape(-1)
    s = np.asarray(samples, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("samples for Parzen density cannot be empty")
    h = float(bandwidth)
    if h <= 0.0 or not np.isfinite(h):
        raise ValueError("bandwidth must be positive and finite")

    z = (x[:, None] - s[None, :]) / h
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * h)
    return coeff * np.exp(-0.5 * z * z).mean(axis=1)


def sample_from_parzen(
    rng: np.random.Generator,
    samples: Array,
    bandwidth: float,
    n_draws: int,
    lower: float,
    upper: float,
) -> Array:
    """Draw candidates from a Gaussian mixture centered at observed samples."""
    centers = np.asarray(samples, dtype=float).reshape(-1)
    if centers.size == 0:
        raise ValueError("cannot sample from empty Parzen centers")
    idx = rng.integers(0, centers.size, size=n_draws)
    draws = centers[idx] + rng.normal(0.0, bandwidth, size=n_draws)
    return np.clip(draws, lower, upper)


def split_good_bad(x_obs: Array, y_obs: Array, gamma: float) -> Tuple[Array, Array, float]:
    """Split observations by y quantile: good=best gamma fraction, bad=rest."""
    n = y_obs.size
    n_good = max(1, int(np.ceil(gamma * n)))
    order = np.argsort(y_obs)
    good_idx = order[:n_good]
    bad_idx = order[n_good:]
    y_star = float(y_obs[good_idx[-1]])
    return x_obs[good_idx], x_obs[bad_idx], y_star


def tpe_suggest(
    rng: np.random.Generator,
    x_obs: Array,
    y_obs: Array,
    lower: float,
    upper: float,
    gamma: float,
    n_candidates: int,
) -> Tuple[float, float, float]:
    """Suggest next point by maximizing l(x)/g(x) over candidates from l(x)."""
    x_good, x_bad, y_star = split_good_bad(x_obs, y_obs, gamma)
    if x_good.size < 2 or x_bad.size < 2:
        # Not enough data for stable KDE split; fallback to random.
        x_next = float(rng.uniform(lower, upper))
        return x_next, float("nan"), y_star

    bw_good = silverman_bandwidth(x_good, lower, upper)
    bw_bad = silverman_bandwidth(x_bad, lower, upper)

    candidates = sample_from_parzen(rng, x_good, bw_good, n_candidates, lower, upper)
    l_vals = parzen_density(candidates, x_good, bw_good)
    g_vals = parzen_density(candidates, x_bad, bw_bad)
    scores = l_vals / np.maximum(g_vals, 1e-12)

    best_idx = int(np.argmax(scores))
    x_next = float(candidates[best_idx])
    best_score = float(scores[best_idx])
    return x_next, best_score, y_star


def run_tpe(
    objective: Callable[[Array], Array],
    bounds: Sequence[float],
    n_trials: int = 60,
    n_startup: int = 12,
    gamma: float = 0.2,
    n_candidates: int = 256,
    seed: int = 2026,
) -> TPEResult:
    """Run minimal TPE optimization for a 1D black-box function."""
    lower, upper = validate_bounds(bounds)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_startup <= 0:
        raise ValueError("n_startup must be positive")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0,1)")
    if n_candidates < 8:
        raise ValueError("n_candidates must be >= 8")

    rng = np.random.default_rng(seed)

    x_obs: List[float] = []
    y_obs: List[float] = []
    records: List[TrialRecord] = []

    best_x = float("nan")
    best_y = float("inf")

    for t in range(1, n_trials + 1):
        if t <= n_startup or len(x_obs) < 4:
            x_next = float(rng.uniform(lower, upper))
            source = "random"
            y_star = float("nan")
            score = float("nan")
        else:
            x_arr = np.asarray(x_obs, dtype=float)
            y_arr = np.asarray(y_obs, dtype=float)
            x_next, score, y_star = tpe_suggest(
                rng=rng,
                x_obs=x_arr,
                y_obs=y_arr,
                lower=lower,
                upper=upper,
                gamma=gamma,
                n_candidates=n_candidates,
            )
            source = "tpe" if np.isfinite(score) else "random-fallback"
            # Soft de-dup: if candidate is too close to an existing sample, jitter once.
            if np.any(np.isclose(x_next, x_arr, atol=1e-9)):
                x_next = float(np.clip(x_next + rng.normal(0.0, 1e-3 * (upper - lower)), lower, upper))
                if np.any(np.isclose(x_next, x_arr, atol=1e-9)):
                    x_next = float(rng.uniform(lower, upper))
                    source = "random-fallback"

        y_next = float(objective(np.array([x_next], dtype=float))[0])
        if not np.isfinite(y_next):
            raise RuntimeError("objective returned non-finite value")

        x_obs.append(x_next)
        y_obs.append(y_next)

        if y_next < best_y:
            best_y = y_next
            best_x = x_next

        records.append(
            TrialRecord(
                iter_id=t,
                source=source,
                x=x_next,
                y=y_next,
                best_x=best_x,
                best_y=best_y,
                y_star=y_star,
                score=score,
            )
        )

    return TPEResult(
        best_x=best_x,
        best_y=best_y,
        x_obs=np.asarray(x_obs, dtype=float),
        y_obs=np.asarray(y_obs, dtype=float),
        records=records,
    )


def approximate_minimum(
    objective: Callable[[Array], Array],
    bounds: Sequence[float],
    n_grid: int = 40000,
) -> Tuple[float, float]:
    """Dense-grid reference minimum for demo validation."""
    lower, upper = validate_bounds(bounds)
    xs = np.linspace(lower, upper, n_grid, dtype=float)
    ys = objective(xs)
    idx = int(np.argmin(ys))
    return float(xs[idx]), float(ys[idx])


def print_iteration_table(records: Sequence[TrialRecord], max_lines: int = 18) -> None:
    """Print a compact iteration summary."""
    n = len(records)
    print("iter | source           | x         | y         | best_y    | y_star    | l/g")
    print("-----+------------------+-----------+-----------+-----------+-----------+-----------")

    if n <= max_lines:
        selected = list(records)
    else:
        head = records[:10]
        tail = records[-8:]
        selected = list(head) + list(tail)

    for r in selected:
        y_star_str = f"{r.y_star: .5f}" if np.isfinite(r.y_star) else "   nan   "
        score_str = f"{r.score: .4f}" if np.isfinite(r.score) else "  nan   "
        print(
            f"{r.iter_id:>4d} | {r.source:<16} | {r.x: .5f} | {r.y: .5f} |"
            f" {r.best_y: .5f} | {y_star_str} | {score_str}"
        )

    if n > max_lines:
        print("... (table truncated: showing first 10 and last 8 iterations)")


def main() -> None:
    bounds = (-4.0, 4.0)
    n_trials = 70
    n_startup = 12
    gamma = 0.2
    n_candidates = 512
    seed = 2026

    result = run_tpe(
        objective=rugged_objective,
        bounds=bounds,
        n_trials=n_trials,
        n_startup=n_startup,
        gamma=gamma,
        n_candidates=n_candidates,
        seed=seed,
    )

    x_ref, y_ref = approximate_minimum(rugged_objective, bounds=bounds, n_grid=40000)
    x_err = abs(result.best_x - x_ref)
    y_err = abs(result.best_y - y_ref)

    print("TPE demo (1D minimization)")
    print(
        f"bounds={bounds}, n_trials={n_trials}, n_startup={n_startup}, "
        f"gamma={gamma}, n_candidates={n_candidates}, seed={seed}"
    )
    print_iteration_table(result.records)

    print("\nFinal summary")
    print(f"TPE best x={result.best_x:.6f}, y={result.best_y:.6f}")
    print(f"Reference grid best x={x_ref:.6f}, y={y_ref:.6f}")
    print(f"Absolute error: |x-x_ref|={x_err:.6f}, |y-y_ref|={y_err:.6f}")
    print(f"Pass loose check (|y-y_ref| <= 0.20): {y_err <= 0.20}")


if __name__ == "__main__":
    main()
