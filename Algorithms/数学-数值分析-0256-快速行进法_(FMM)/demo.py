"""Fast Marching Method (FMM) MVP for the 2D isotropic Eikonal equation.

We solve arrival time T on a Cartesian grid:
    F(x, y) * |grad(T)| = 1
with source constraint T(source)=0.

Implementation highlights:
- 4-neighbor upwind stencil (first-order)
- Min-heap narrow band
- Frozen/Narrow/Far states
- Explicit local quadratic update (no black-box solver)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

FAR = np.int8(0)
NARROW = np.int8(1)
FROZEN = np.int8(2)

NEIGHBORS_4: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class FMStats:
    popped_nodes: int
    frozen_nodes: int


def solve_local_eikonal_2d(a: float, b: float, speed: float, h: float) -> float:
    """Solve local upwind update with two directional minima.

    Local equation (2D isotropic first-order upwind):
        (max(T-a, 0))^2 + (max(T-b, 0))^2 = (h/F)^2
    where a,b are the best frozen times along x/y directions.
    """
    if speed <= 0.0:
        return float("inf")

    d = h / speed
    a_finite = math.isfinite(a)
    b_finite = math.isfinite(b)

    if not a_finite and not b_finite:
        return float("inf")
    if a_finite and (not b_finite):
        return a + d
    if b_finite and (not a_finite):
        return b + d

    # Both finite.
    diff = abs(a - b)
    if diff >= d:
        return min(a, b) + d

    disc = 2.0 * d * d - diff * diff
    if disc < 0.0:
        # Numerically safe fallback.
        return min(a, b) + d

    t = 0.5 * (a + b + math.sqrt(disc))
    return max(t, a, b)


def axis_min_frozen(
    t: np.ndarray,
    state: np.ndarray,
    i: int,
    j: int,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> float:
    """Return min arrival time among frozen nodes in a coordinate axis pair."""
    nrows, ncols = t.shape
    best = float("inf")

    for di, dj in (p1, p2):
        ni, nj = i + di, j + dj
        if 0 <= ni < nrows and 0 <= nj < ncols and state[ni, nj] == FROZEN:
            val = float(t[ni, nj])
            if val < best:
                best = val

    return best


def local_update(
    t: np.ndarray,
    state: np.ndarray,
    speed: np.ndarray,
    i: int,
    j: int,
    h: float,
) -> float:
    """Compute one tentative value for grid node (i,j)."""
    if speed[i, j] <= 0.0:
        return float("inf")

    a = axis_min_frozen(t, state, i, j, (-1, 0), (1, 0))
    b = axis_min_frozen(t, state, i, j, (0, -1), (0, 1))
    return solve_local_eikonal_2d(a=a, b=b, speed=float(speed[i, j]), h=h)


def fast_marching(
    speed: np.ndarray,
    sources: Sequence[Tuple[int, int]],
    h: float,
) -> Tuple[np.ndarray, np.ndarray, FMStats]:
    """Compute arrival-time field by Fast Marching Method."""
    if speed.ndim != 2:
        raise ValueError("speed must be a 2D array")

    nrows, ncols = speed.shape
    t = np.full((nrows, ncols), np.inf, dtype=float)
    state = np.full((nrows, ncols), FAR, dtype=np.int8)
    heap: List[Tuple[float, int, int]] = []

    for si, sj in sources:
        if not (0 <= si < nrows and 0 <= sj < ncols):
            raise ValueError(f"source {(si, sj)} out of bounds")
        if speed[si, sj] <= 0.0:
            raise ValueError(f"source {(si, sj)} lies on non-traversable cell")
        t[si, sj] = 0.0
        state[si, sj] = FROZEN

    # Initialize narrow band around all sources.
    for si, sj in sources:
        for di, dj in NEIGHBORS_4:
            ni, nj = si + di, sj + dj
            if not (0 <= ni < nrows and 0 <= nj < ncols):
                continue
            if state[ni, nj] == FROZEN or speed[ni, nj] <= 0.0:
                continue
            trial = local_update(t=t, state=state, speed=speed, i=ni, j=nj, h=h)
            if trial < t[ni, nj]:
                t[ni, nj] = trial
                state[ni, nj] = NARROW
                heapq.heappush(heap, (trial, ni, nj))

    popped_nodes = 0

    while heap:
        trial, i, j = heapq.heappop(heap)
        popped_nodes += 1

        # Skip stale heap entries.
        if trial > t[i, j]:
            continue
        if state[i, j] == FROZEN:
            continue

        state[i, j] = FROZEN

        for di, dj in NEIGHBORS_4:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nrows and 0 <= nj < ncols):
                continue
            if state[ni, nj] == FROZEN or speed[ni, nj] <= 0.0:
                continue

            new_trial = local_update(t=t, state=state, speed=speed, i=ni, j=nj, h=h)
            if new_trial < t[ni, nj]:
                t[ni, nj] = new_trial
                state[ni, nj] = NARROW
                heapq.heappush(heap, (new_trial, ni, nj))

    frozen_nodes = int(np.count_nonzero(state == FROZEN))
    return t, state, FMStats(popped_nodes=popped_nodes, frozen_nodes=frozen_nodes)


def build_piecewise_speed(n: int) -> np.ndarray:
    """Create a heterogeneous speed map with a slow ring and an obstacle."""
    if n < 21:
        raise ValueError("n should be >= 21 for this demo")

    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")

    speed = np.ones((n, n), dtype=float)
    radius = np.sqrt(xg * xg + yg * yg)

    slow_band = (radius >= 0.35) & (radius <= 0.60)
    speed[slow_band] = 0.45

    # A hard obstacle (speed = 0) near the top-right quadrant.
    obstacle = (xg > 0.20) & (xg < 0.55) & (yg > 0.25) & (yg < 0.55)
    speed[obstacle] = 0.0

    return speed


def run_constant_speed_check() -> None:
    """Validate FMM against the exact Euclidean distance for F=1."""
    n = 101
    h = 1.0 / (n - 1)
    speed = np.ones((n, n), dtype=float)

    source = (n // 2, n // 2)
    t, state, stats = fast_marching(speed=speed, sources=[source], h=h)

    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    x0 = x[source[0]]
    y0 = y[source[1]]
    exact = np.sqrt((xg - x0) ** 2 + (yg - y0) ** 2)

    finite_mask = np.isfinite(t)
    abs_err = np.abs(t[finite_mask] - exact[finite_mask])
    max_err = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    print("[Case A] Constant speed F=1 (single source)")
    print(f"grid={n}x{n}, h={h:.5f}, source={source}")
    print(
        f"frozen_nodes={stats.frozen_nodes}, popped_nodes={stats.popped_nodes}, "
        f"max_abs_err={max_err:.6e}, mean_abs_err={mean_err:.6e}"
    )

    # Basic correctness checks for this first-order stencil implementation.
    assert state[source] == FROZEN
    assert abs(float(t[source])) < 1e-15
    assert max_err < 2.5 * h
    assert mean_err < 0.8 * h


def run_heterogeneous_demo() -> None:
    """Run FMM on piecewise speed field and print summary diagnostics."""
    n = 121
    h = 2.0 / (n - 1)  # domain is [-1, 1]^2
    speed = build_piecewise_speed(n)

    source = (n // 2, n // 2)
    t, state, stats = fast_marching(speed=speed, sources=[source], h=h)

    finite_mask = np.isfinite(t)
    reachable = int(np.count_nonzero(finite_mask))
    total = int(t.size)
    blocked = int(np.count_nonzero(speed <= 0.0))

    t_finite = t[finite_mask]
    t_min = float(np.min(t_finite))
    t_max = float(np.max(t_finite))
    t_p95 = float(np.quantile(t_finite, 0.95))

    print("[Case B] Heterogeneous speed with obstacle")
    print(
        f"grid={n}x{n}, h={h:.5f}, source={source}, blocked_cells={blocked}, "
        f"reachable={reachable}/{total}"
    )
    print(
        f"frozen_nodes={stats.frozen_nodes}, popped_nodes={stats.popped_nodes}, "
        f"T_min={t_min:.6e}, T_p95={t_p95:.6e}, T_max={t_max:.6e}"
    )

    # Sanity checks for traversal field.
    assert reachable + blocked == total
    assert t_min == 0.0
    assert state[source] == FROZEN


def main() -> None:
    run_constant_speed_check()
    print()
    run_heterogeneous_demo()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
