"""3D / high-dimensional convex hull MVP without external geometry libraries.

This script intentionally uses a small, auditable implementation:
- Enumerate candidate hyperplanes from point combinations.
- Keep only supporting hyperplanes (all points on one side).
- Deduplicate facets by coplanar vertex index sets.

It is suitable for small demo sizes and keeps all logic in source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class HullData:
    facets: List[np.ndarray]  # each facet is a sorted vertex-index array
    equations: np.ndarray  # shape (m, d+1), each row [a_1, ..., a_d, b]
    vertices: np.ndarray  # sorted unique vertex indices
    combinations_tested: int
    combinations_kept: int


@dataclass
class HullReport:
    name: str
    dimension: int
    n_points: int
    n_vertices: int
    n_facets: int
    max_violation: float
    random_inside_rate: float
    combinations_tested: int
    combinations_kept: int


def make_3d_points(seed: int = 7, n_inner: int = 40) -> np.ndarray:
    """Generate stable 3D points: octahedron extremes + interior noise."""
    rng = np.random.default_rng(seed)
    extreme = np.array(
        [
            [1.7, 0.0, 0.0],
            [-1.7, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, -1.5, 0.0],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, -1.3],
        ],
        dtype=float,
    )
    inner = rng.uniform(low=-0.35, high=0.35, size=(n_inner, 3))
    return np.vstack([inner, extreme])


def make_high_dim_points(seed: int = 17, d: int = 6, n_inner: int = 8) -> np.ndarray:
    """Generate 6D points: cross-polytope extremes + interior points."""
    if d < 4:
        raise ValueError("high-dimensional demo expects d >= 4")
    rng = np.random.default_rng(seed)
    extreme = np.vstack([1.4 * np.eye(d), -1.4 * np.eye(d)])
    inner = rng.uniform(low=-0.08, high=0.08, size=(n_inner, d))
    return np.vstack([inner, extreme])


def _hyperplane_from_indices(
    points: np.ndarray, idx: Sequence[int], tol_rank: float
) -> Tuple[np.ndarray, float] | None:
    """Return normalized hyperplane (normal, offset) through d points or None."""
    d = points.shape[1]
    p0 = points[idx[0]]
    mat = points[np.array(idx[1:], dtype=int)] - p0  # (d-1, d)
    _, singular, vh = np.linalg.svd(mat, full_matrices=True)
    rank = int(np.sum(singular > tol_rank))
    if rank < d - 1:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= tol_rank:
        return None
    normal = normal / norm
    offset = -float(normal @ p0)
    return normal, offset


def compute_convex_hull_enumeration(points: np.ndarray) -> HullData:
    """Compute convex hull facets by brute-force supporting hyperplane search."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, dimension)")
    n_points, d = points.shape
    if n_points <= d:
        raise ValueError(f"Need n_points > dimension, got {n_points} <= {d}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contains NaN or Inf.")

    scale = max(1.0, float(np.max(np.abs(points))))
    tol_rank = 1e-10 * scale
    tol_side = 1e-9 * scale
    tol_plane = 5e-8 * scale

    facet_map: dict[Tuple[int, ...], np.ndarray] = {}
    combinations_tested = 0
    combinations_kept = 0

    for idx in combinations(range(n_points), d):
        combinations_tested += 1
        hyper = _hyperplane_from_indices(points, idx, tol_rank=tol_rank)
        if hyper is None:
            continue
        normal, offset = hyper

        signed = points @ normal + offset
        max_signed = float(np.max(signed))
        min_signed = float(np.min(signed))

        if max_signed <= tol_side:
            pass
        elif min_signed >= -tol_side:
            normal = -normal
            offset = -offset
            signed = -signed
        else:
            continue

        active = np.where(np.abs(signed) <= tol_plane)[0]
        if active.size < d:
            continue

        key = tuple(int(i) for i in active.tolist())
        if key in facet_map:
            continue
        facet_map[key] = np.concatenate([normal, np.array([offset])])
        combinations_kept += 1

    if not facet_map:
        raise RuntimeError("No supporting facets found; input may be degenerate.")

    facets = [np.array(k, dtype=int) for k in facet_map.keys()]
    equations = np.vstack(list(facet_map.values()))
    vertices = np.array(sorted({int(i) for f in facets for i in f}), dtype=int)

    return HullData(
        facets=facets,
        equations=equations,
        vertices=vertices,
        combinations_tested=combinations_tested,
        combinations_kept=combinations_kept,
    )


def max_halfspace_violation(points: np.ndarray, equations: np.ndarray) -> float:
    """Return max(Ax+b), which should be <= tolerance for all original points."""
    a = equations[:, :-1]
    b = equations[:, -1]
    signed = np.einsum("nd,md->nm", points, a, optimize=True) + b
    return float(np.max(signed))


def estimate_inside_rate(
    equations: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, seed: int, n_samples: int = 6000
) -> float:
    """Estimate how much of the axis-aligned bounding box lies inside the hull."""
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=lo, high=hi, size=(n_samples, lo.size))
    signed = (
        np.einsum("nd,md->nm", samples, equations[:, :-1], optimize=True)
        + equations[:, -1]
    )
    inside = np.all(signed <= 1e-7, axis=1)
    return float(np.mean(inside))


def run_case(name: str, points: np.ndarray, seed: int) -> HullReport:
    hull = compute_convex_hull_enumeration(points)
    violation = max_halfspace_violation(points, hull.equations)
    inside_rate = estimate_inside_rate(
        hull.equations, points.min(axis=0), points.max(axis=0), seed=seed
    )

    report = HullReport(
        name=name,
        dimension=points.shape[1],
        n_points=points.shape[0],
        n_vertices=int(hull.vertices.size),
        n_facets=int(len(hull.facets)),
        max_violation=violation,
        random_inside_rate=inside_rate,
        combinations_tested=hull.combinations_tested,
        combinations_kept=hull.combinations_kept,
    )

    assert report.n_vertices >= report.dimension + 1, (
        f"{name}: too few vertices {report.n_vertices} for d={report.dimension}"
    )
    assert report.n_facets >= report.dimension + 1, (
        f"{name}: too few facets {report.n_facets} for d={report.dimension}"
    )
    assert report.max_violation <= 1e-7, (
        f"{name}: max violation {report.max_violation:.3e} exceeds tolerance"
    )
    return report


def print_report(report: HullReport) -> None:
    print(f"[{report.name}]")
    print(f"dimension            : {report.dimension}")
    print(f"n_points             : {report.n_points}")
    print(f"n_vertices           : {report.n_vertices}")
    print(f"n_facets             : {report.n_facets}")
    print(f"max_violation        : {report.max_violation:.3e}")
    print(f"random_inside_rate   : {report.random_inside_rate:.4f}")
    print(f"combinations_tested  : {report.combinations_tested}")
    print(f"combinations_kept    : {report.combinations_kept}")
    print("")


def main() -> None:
    points_3d = make_3d_points()
    points_6d = make_high_dim_points()

    total_3d = comb(points_3d.shape[0], points_3d.shape[1])
    total_6d = comb(points_6d.shape[0], points_6d.shape[1])
    print("Convex Hull MVP (Enumeration)")
    print("-----------------------------")
    print(f"3D candidate combinations : C({points_3d.shape[0]}, 3) = {total_3d}")
    print(f"6D candidate combinations : C({points_6d.shape[0]}, 6) = {total_6d}")
    print("")

    report_3d = run_case("3D_case", points_3d, seed=123)
    report_6d = run_case("6D_case", points_6d, seed=456)

    print_report(report_3d)
    print_report(report_6d)
    print("All checks passed.")


if __name__ == "__main__":
    main()
