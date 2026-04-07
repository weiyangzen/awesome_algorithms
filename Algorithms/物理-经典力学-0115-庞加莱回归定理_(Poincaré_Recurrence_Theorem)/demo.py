"""Minimal runnable MVP for Poincare recurrence theorem (PHYS-0115)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CAT_MATRIX = np.array([[1, 1], [1, 2]], dtype=int)


@dataclass
class DiscreteRecurrenceReport:
    grid_size: int
    num_states: int
    num_cycles: int
    min_period: int
    max_period: int
    mean_period: float
    period_sum: int


@dataclass
class ContinuousRecurrenceReport:
    sample_size: int
    epsilon: float
    max_steps: int
    returned_ratio: float
    mean_first_return: float
    median_first_return: float
    min_distance_q50: float
    min_distance_q95: float


def cat_map_mod_n_step(state: tuple[int, int], n: int) -> tuple[int, int]:
    """One step of Arnold cat map on Z_n^2."""
    x, y = state
    return ((x + y) % n, (x + 2 * y) % n)


def is_bijection_mod_n(n: int) -> bool:
    """Check whether T_n maps Z_n^2 bijectively onto itself."""
    mapped = set()
    for x in range(n):
        for y in range(n):
            mapped.add(cat_map_mod_n_step((x, y), n))
    return len(mapped) == n * n


def decompose_cycles_mod_n(n: int) -> np.ndarray:
    """Cycle decomposition lengths for T_n on all points in Z_n^2."""
    visited = np.zeros((n, n), dtype=bool)
    periods: list[int] = []

    for x in range(n):
        for y in range(n):
            if visited[x, y]:
                continue

            start = (x, y)
            current = start
            period = 0

            while True:
                cx, cy = current
                if visited[cx, cy]:
                    raise RuntimeError("Encountered previously visited state before closing cycle.")
                visited[cx, cy] = True
                period += 1
                current = cat_map_mod_n_step(current, n)
                if current == start:
                    break

            periods.append(period)

    return np.array(periods, dtype=int)


def summarize_periods(periods: np.ndarray) -> pd.DataFrame:
    """Frequency table for cycle periods."""
    unique_periods, counts = np.unique(periods, return_counts=True)
    return pd.DataFrame({"period": unique_periods, "count": counts})


def cat_map_continuous(points: np.ndarray) -> np.ndarray:
    """One step of Arnold cat map on the continuous torus [0,1)^2."""
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack(((x + y) % 1.0, (x + 2.0 * y) % 1.0))


def torus_l2_distance(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Shortest Euclidean distance on torus with periodic boundaries."""
    delta = np.abs(points_a - points_b)
    wrapped_delta = np.minimum(delta, 1.0 - delta)
    return np.sqrt(np.sum(wrapped_delta * wrapped_delta, axis=1))


def first_return_times(
    initial_points: np.ndarray,
    epsilon: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-hit times to epsilon-neighborhood and min reached distances."""
    current_points = initial_points.copy()
    first_hits = np.full(initial_points.shape[0], -1, dtype=int)
    min_distances = np.full(initial_points.shape[0], np.inf, dtype=float)

    for t in range(1, max_steps + 1):
        current_points = cat_map_continuous(current_points)
        d = torus_l2_distance(current_points, initial_points)
        min_distances = np.minimum(min_distances, d)

        hit_mask = (first_hits < 0) & (d < epsilon)
        first_hits[hit_mask] = t

    return first_hits, min_distances


def build_discrete_report(n: int, periods: np.ndarray) -> DiscreteRecurrenceReport:
    """Aggregate discrete recurrence statistics."""
    return DiscreteRecurrenceReport(
        grid_size=n,
        num_states=n * n,
        num_cycles=int(periods.size),
        min_period=int(periods.min()),
        max_period=int(periods.max()),
        mean_period=float(periods.mean()),
        period_sum=int(periods.sum()),
    )


def build_continuous_report(
    first_hits: np.ndarray,
    min_distances: np.ndarray,
    epsilon: float,
    max_steps: int,
) -> ContinuousRecurrenceReport:
    """Aggregate continuous recurrence statistics."""
    returned = first_hits > 0
    returned_ratio = float(np.mean(returned))

    if np.any(returned):
        first_return_samples = first_hits[returned].astype(float)
        mean_first_return = float(np.mean(first_return_samples))
        median_first_return = float(np.median(first_return_samples))
    else:
        mean_first_return = float("nan")
        median_first_return = float("nan")

    return ContinuousRecurrenceReport(
        sample_size=int(first_hits.size),
        epsilon=epsilon,
        max_steps=max_steps,
        returned_ratio=returned_ratio,
        mean_first_return=mean_first_return,
        median_first_return=median_first_return,
        min_distance_q50=float(np.quantile(min_distances, 0.50)),
        min_distance_q95=float(np.quantile(min_distances, 0.95)),
    )


def main() -> None:
    # 1) Discrete exact recurrence on finite phase space Z_n^2.
    n = 37
    periods = decompose_cycles_mod_n(n)
    discrete_report = build_discrete_report(n, periods)
    period_table = summarize_periods(periods)

    # 2) Continuous near-recurrence statistics on [0,1)^2.
    seed = 20260407
    rng = np.random.default_rng(seed)
    sample_size = 400
    epsilon = 0.025
    max_steps = 3000
    points0 = rng.random((sample_size, 2))

    first_hits, min_distances = first_return_times(points0, epsilon, max_steps)
    continuous_report = build_continuous_report(first_hits, min_distances, epsilon, max_steps)

    # Theoretical condition: det(A)=1 -> area preserving.
    determinant = int(round(np.linalg.det(CAT_MATRIX)))

    checks = {
        "det(A) == 1": determinant == 1,
        "mod-n map bijective": is_bijection_mod_n(n),
        "cycle coverage equals n^2": discrete_report.period_sum == discrete_report.num_states,
        "continuous return ratio >= 0.99": continuous_report.returned_ratio >= 0.99,
    }

    pd.set_option("display.max_rows", 200)

    print("=== Poincare Recurrence Theorem MVP (PHYS-0115) ===")
    print(f"Cat map matrix A =\n{CAT_MATRIX}")
    print(f"det(A) = {determinant}")

    print("\n[Discrete exact recurrence on Z_n^2]")
    print(
        "n = {n}, states = {states}, cycles = {cycles}, "
        "period min/max/mean = {pmin}/{pmax}/{pmean:.3f}".format(
            n=discrete_report.grid_size,
            states=discrete_report.num_states,
            cycles=discrete_report.num_cycles,
            pmin=discrete_report.min_period,
            pmax=discrete_report.max_period,
            pmean=discrete_report.mean_period,
        )
    )
    print("Cycle length distribution:")
    print(period_table.to_string(index=False))

    print("\n[Continuous near-recurrence on torus]")
    print(
        "samples = {m}, epsilon = {eps}, max_steps = {t}, return_ratio = {rr:.4f}".format(
            m=continuous_report.sample_size,
            eps=continuous_report.epsilon,
            t=continuous_report.max_steps,
            rr=continuous_report.returned_ratio,
        )
    )
    print(
        "first return mean/median = {mean_rt:.2f}/{median_rt:.2f}".format(
            mean_rt=continuous_report.mean_first_return,
            median_rt=continuous_report.median_first_return,
        )
    )
    print(
        "min-distance q50/q95 = {q50:.5f}/{q95:.5f}".format(
            q50=continuous_report.min_distance_q50,
            q95=continuous_report.min_distance_q95,
        )
    )

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
