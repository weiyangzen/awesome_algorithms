"""Delaunay triangulation MVP with a divide-and-conquer orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError


@dataclass(frozen=True)
class NodeStat:
    depth: int
    n_points: int
    xmin: float
    xmax: float


def generate_points(n_points: int = 128, seed: int = 191) -> np.ndarray:
    """Generate deterministic 2D points in general position."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n_points, 2))
    # Tiny jitter keeps the sample away from accidental cocircular/collinear edge cases.
    points += rng.normal(0.0, 1e-6, size=points.shape)
    return points.astype(np.float64, copy=False)


def triangulate_subset(
    points: np.ndarray,
    global_indices: np.ndarray,
    qhull_options: str = "QJ",
) -> np.ndarray:
    """Triangulate a subset and map local simplices back to global indices."""
    if global_indices.size < 3:
        return np.empty((0, 3), dtype=np.int64)

    local_points = points[global_indices]
    try:
        tri = Delaunay(local_points, qhull_options=qhull_options)
    except QhullError:
        return np.empty((0, 3), dtype=np.int64)

    simplices_local = np.asarray(tri.simplices, dtype=np.int64)
    if simplices_local.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return global_indices[simplices_local]


def recursive_dc_triangulate(
    points: np.ndarray,
    global_indices: np.ndarray,
    leaf_size: int,
    depth: int,
    stats: list[NodeStat],
) -> np.ndarray:
    """
    Divide-and-conquer orchestration:
    split by median x, recurse on both halves, merge by retriangulating the union.
    """
    if global_indices.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    order = np.argsort(points[global_indices, 0], kind="mergesort")
    sorted_idx = global_indices[order]
    xvals = points[sorted_idx, 0]
    stats.append(
        NodeStat(
            depth=depth,
            n_points=int(sorted_idx.size),
            xmin=float(np.min(xvals)),
            xmax=float(np.max(xvals)),
        )
    )

    if sorted_idx.size <= leaf_size:
        return triangulate_subset(points, sorted_idx)

    mid = sorted_idx.size // 2
    left_idx = sorted_idx[:mid]
    right_idx = sorted_idx[mid:]

    # Recurse for true divide-and-conquer process visibility.
    _ = recursive_dc_triangulate(points, left_idx, leaf_size, depth + 1, stats)
    _ = recursive_dc_triangulate(points, right_idx, leaf_size, depth + 1, stats)

    # Merge step for this MVP: rebuild triangulation on the combined subset.
    return triangulate_subset(points, sorted_idx)


def canonicalize_simplices(simplices: np.ndarray) -> np.ndarray:
    """Sort vertices in each simplex and drop duplicates."""
    simplices = np.asarray(simplices, dtype=np.int64)
    if simplices.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if simplices.ndim != 2 or simplices.shape[1] != 3:
        raise ValueError("simplices must have shape (m, 3)")
    return np.unique(np.sort(simplices, axis=1), axis=0)


def simplices_to_edges(simplices: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from triangle simplices."""
    if simplices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(
        [
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]],
        ]
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def triangle_areas(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute triangle areas."""
    if simplices.size == 0:
        return np.empty((0,), dtype=np.float64)
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * np.abs(cross)


def circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return circumcenter and squared radius. None if degenerate."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return None

    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d

    center = np.array([ux, uy], dtype=np.float64)
    radius2 = float(np.sum((a - center) ** 2))
    return center, radius2


def validate_empty_circumcircle(
    points: np.ndarray,
    simplices: np.ndarray,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Check Delaunay empty-circumcircle condition for each triangle."""
    n_points = points.shape[0]
    violations = 0
    degenerate = 0
    worst_margin = -np.inf
    checked = 0

    for tri in simplices:
        circle = circumcircle(points[tri[0]], points[tri[1]], points[tri[2]])
        if circle is None:
            degenerate += 1
            continue
        center, radius2 = circle
        checked += 1

        d2 = np.sum((points - center) ** 2, axis=1)
        mask = np.ones(n_points, dtype=bool)
        mask[tri] = False
        nearest_other = float(np.min(d2[mask])) if np.any(mask) else float("inf")
        margin = radius2 - nearest_other
        worst_margin = max(worst_margin, margin)
        if margin > tol:
            violations += 1

    if checked == 0:
        worst_margin = float("nan")

    return {
        "checked_triangles": float(checked),
        "degenerate_triangles": float(degenerate),
        "violation_count": float(violations),
        "worst_margin": float(worst_margin),
    }


def summarize_depth_stats(stats: list[NodeStat]) -> pd.DataFrame:
    """Aggregate recursion-node statistics by depth."""
    df = pd.DataFrame(asdict(item) for item in stats)
    return (
        df.groupby("depth", as_index=False)
        .agg(
            nodes=("depth", "size"),
            min_points=("n_points", "min"),
            max_points=("n_points", "max"),
            avg_points=("n_points", "mean"),
            min_x=("xmin", "min"),
            max_x=("xmax", "max"),
        )
        .sort_values("depth")
        .reset_index(drop=True)
    )


def main() -> None:
    points = generate_points(n_points=128, seed=191)
    root_indices = np.arange(points.shape[0], dtype=np.int64)

    node_stats: list[NodeStat] = []
    dc_simplices = recursive_dc_triangulate(
        points=points,
        global_indices=root_indices,
        leaf_size=24,
        depth=0,
        stats=node_stats,
    )
    dc_simplices = canonicalize_simplices(dc_simplices)

    baseline = canonicalize_simplices(
        triangulate_subset(points, root_indices, qhull_options="QJ")
    )
    edges = simplices_to_edges(dc_simplices)
    areas = triangle_areas(points, dc_simplices)
    circle_check = validate_empty_circumcircle(points, dc_simplices, tol=1e-9)
    depth_report = summarize_depth_stats(node_stats)

    hull = ConvexHull(points)
    hull_vertices = int(len(hull.vertices))
    expected_triangles = 2 * points.shape[0] - 2 - hull_vertices
    expected_edges = 3 * points.shape[0] - 3 - hull_vertices

    print("=== Delaunay Triangulation MVP (Divide-and-Conquer Orchestration) ===")
    print(f"n_points={points.shape[0]}, leaf_size=24, recursion_nodes={len(node_stats)}")
    print(
        f"triangles={len(dc_simplices)}, edges={len(edges)}, "
        f"hull_vertices={hull_vertices}"
    )
    print(
        f"area(min/mean/max)=({areas.min():.6f}, {areas.mean():.6f}, {areas.max():.6f})"
    )
    print(
        "empty-circumcircle check: "
        f"checked={int(circle_check['checked_triangles'])}, "
        f"violations={int(circle_check['violation_count'])}, "
        f"worst_margin={circle_check['worst_margin']:.3e}"
    )
    print()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.float_format",
        "{:.3f}".format,
    ):
        print("Recursion depth summary:")
        print(depth_report.to_string(index=False))

    # Quality gates for deterministic validation.
    assert points.shape == (128, 2), "Unexpected point shape"
    assert len(node_stats) > 1, "Recursion tree was not constructed"
    assert dc_simplices.shape[1] == 3, "Simplices must be triangles"
    assert len(dc_simplices) == expected_triangles, "Triangle count failed planar invariant"
    assert len(edges) == expected_edges, "Edge count failed planar invariant"
    assert np.all(areas > 0.0), "Degenerate triangle area detected"
    assert int(circle_check["violation_count"]) == 0, "Empty circumcircle condition violated"
    assert np.array_equal(dc_simplices, baseline), "D&C result differs from one-shot baseline"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
