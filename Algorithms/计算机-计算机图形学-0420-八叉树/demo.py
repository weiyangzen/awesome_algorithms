"""CS-0261 八叉树：最小可运行 MVP。

实现内容：
1) 手写八叉树构建（点插入 + 节点细分）。
2) 手写 AABB 范围查询，并统计访问代价。
3) 对照暴力扫描，验证查询结果正确性。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box in 3D."""

    min_corner: Array
    max_corner: Array

    def __post_init__(self) -> None:
        mn = np.asarray(self.min_corner, dtype=float)
        mx = np.asarray(self.max_corner, dtype=float)
        if mn.shape != (3,) or mx.shape != (3,):
            raise ValueError("AABB corners must be 3D vectors")
        if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
            raise ValueError("AABB corners must be finite")
        if np.any(mn > mx):
            raise ValueError("AABB min_corner must be <= max_corner component-wise")

        object.__setattr__(self, "min_corner", mn)
        object.__setattr__(self, "max_corner", mx)

    def contains_point(self, point: Array) -> bool:
        p = np.asarray(point, dtype=float)
        return bool(np.all(p >= self.min_corner) and np.all(p <= self.max_corner))

    def intersects(self, other: "AABB") -> bool:
        return bool(np.all(self.max_corner >= other.min_corner) and np.all(other.max_corner >= self.min_corner))

    @property
    def center(self) -> Array:
        return (self.min_corner + self.max_corner) * 0.5


@dataclass
class QueryStats:
    visited_nodes: int = 0
    visited_leaves: int = 0
    point_tests: int = 0


@dataclass
class OctreeNode:
    bounds: AABB
    depth: int
    max_depth: int
    bucket_size: int
    points: list[Array] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    children: list["OctreeNode"] | None = None

    def is_leaf(self) -> bool:
        return self.children is None

    def _child_index(self, point: Array) -> int:
        c = self.bounds.center
        idx = 0
        if point[0] >= c[0]:
            idx |= 1
        if point[1] >= c[1]:
            idx |= 2
        if point[2] >= c[2]:
            idx |= 4
        return idx

    def _child_bounds(self, child_idx: int) -> AABB:
        c = self.bounds.center
        mn = self.bounds.min_corner.copy()
        mx = self.bounds.max_corner.copy()

        for axis, bit in enumerate((1, 2, 4)):
            if child_idx & bit:
                mn[axis] = c[axis]
            else:
                mx[axis] = c[axis]

        return AABB(min_corner=mn, max_corner=mx)

    def _insert_into_child(self, point: Array, index: int) -> None:
        if self.children is None:
            raise RuntimeError("Internal error: children expected after subdivision")

        preferred = self.children[self._child_index(point)]
        if preferred.bounds.contains_point(point):
            if not preferred.insert(point, index):
                raise RuntimeError("Failed to insert point into preferred child")
            return

        for child in self.children:
            if child.bounds.contains_point(point):
                if not child.insert(point, index):
                    raise RuntimeError("Failed to insert point into fallback child")
                return

        raise RuntimeError("Point is inside parent but not inside any child")

    def _subdivide(self) -> None:
        self.children = [
            OctreeNode(
                bounds=self._child_bounds(i),
                depth=self.depth + 1,
                max_depth=self.max_depth,
                bucket_size=self.bucket_size,
            )
            for i in range(8)
        ]

        old_points = self.points
        old_indices = self.indices
        self.points = []
        self.indices = []

        for point, index in zip(old_points, old_indices):
            self._insert_into_child(point, index)

    def insert(self, point: Array, index: int) -> bool:
        if not self.bounds.contains_point(point):
            return False

        if self.is_leaf():
            if len(self.points) < self.bucket_size or self.depth >= self.max_depth:
                self.points.append(np.asarray(point, dtype=float))
                self.indices.append(int(index))
                return True
            self._subdivide()

        self._insert_into_child(point, int(index))
        return True

    def query_range(self, query: AABB, out_indices: list[int], stats: QueryStats) -> None:
        if not self.bounds.intersects(query):
            return

        stats.visited_nodes += 1

        if self.is_leaf():
            stats.visited_leaves += 1
            for point, index in zip(self.points, self.indices):
                stats.point_tests += 1
                if query.contains_point(point):
                    out_indices.append(index)
            return

        if self.children is None:
            raise RuntimeError("Internal error: children missing on non-leaf")

        for child in self.children:
            child.query_range(query, out_indices, stats)

    def count_nodes(self) -> int:
        if self.is_leaf():
            return 1
        if self.children is None:
            raise RuntimeError("Internal error: children missing on non-leaf")
        return 1 + sum(child.count_nodes() for child in self.children)

    def count_leaves(self) -> int:
        if self.is_leaf():
            return 1
        if self.children is None:
            raise RuntimeError("Internal error: children missing on non-leaf")
        return sum(child.count_leaves() for child in self.children)

    def max_depth_reached(self) -> int:
        if self.is_leaf():
            return self.depth
        if self.children is None:
            raise RuntimeError("Internal error: children missing on non-leaf")
        return max(child.max_depth_reached() for child in self.children)

    def collect_leaf_loads(self, out: list[int]) -> None:
        if self.is_leaf():
            out.append(len(self.points))
            return
        if self.children is None:
            raise RuntimeError("Internal error: children missing on non-leaf")
        for child in self.children:
            child.collect_leaf_loads(out)

    def collect_depth_histogram(self, hist: dict[int, int]) -> None:
        hist[self.depth] = hist.get(self.depth, 0) + 1
        if self.children is None:
            return
        for child in self.children:
            child.collect_depth_histogram(hist)


def build_octree(points: Array, max_depth: int, bucket_size: int) -> OctreeNode:
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("points cannot be empty")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite values")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0")

    min_corner = np.min(pts, axis=0)
    max_corner = np.max(pts, axis=0)
    center = (min_corner + max_corner) * 0.5

    half_extent = float(np.max(max_corner - min_corner) * 0.5)
    if half_extent <= 0.0:
        half_extent = 1e-6
    half_extent *= 1.000001

    root_bounds = AABB(
        min_corner=center - half_extent,
        max_corner=center + half_extent,
    )
    root = OctreeNode(bounds=root_bounds, depth=0, max_depth=max_depth, bucket_size=bucket_size)

    for idx, point in enumerate(pts):
        if not root.insert(point, idx):
            raise RuntimeError(f"Point {idx} is outside root bounds; build failed")

    return root


def brute_force_range_query(points: Array, query: AABB) -> list[int]:
    pts = np.asarray(points, dtype=float)
    mask = np.all(pts >= query.min_corner, axis=1) & np.all(pts <= query.max_corner, axis=1)
    return np.flatnonzero(mask).astype(int).tolist()


def build_demo_points(seed: int = 20260407) -> Array:
    rng = np.random.default_rng(seed)

    cluster_centers = np.array(
        [
            [-0.55, -0.35, 0.10],
            [0.50, -0.30, -0.20],
            [-0.15, 0.55, 0.45],
            [0.45, 0.45, 0.55],
        ],
        dtype=float,
    )

    clusters = []
    for center in cluster_centers:
        cloud = rng.normal(loc=center, scale=0.12, size=(400, 3))
        clusters.append(cloud)

    noise = rng.uniform(-1.0, 1.0, size=(400, 3))
    points = np.vstack(clusters + [noise])
    points = np.clip(points, -1.0, 1.0)
    return points


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    points = build_demo_points()
    n_points = int(points.shape[0])

    max_depth = 7
    bucket_size = 24
    root = build_octree(points=points, max_depth=max_depth, bucket_size=bucket_size)

    queries: list[tuple[str, AABB]] = [
        (
            "center_cube",
            AABB(
                min_corner=np.array([-0.25, -0.25, -0.25], dtype=float),
                max_corner=np.array([0.25, 0.25, 0.25], dtype=float),
            ),
        ),
        (
            "x_positive_slab",
            AABB(
                min_corner=np.array([0.20, -1.00, -1.00], dtype=float),
                max_corner=np.array([1.00, 1.00, 1.00], dtype=float),
            ),
        ),
        (
            "upper_cluster_probe",
            AABB(
                min_corner=np.array([0.20, 0.20, 0.30], dtype=float),
                max_corner=np.array([0.80, 0.80, 0.80], dtype=float),
            ),
        ),
        (
            "sparse_corner",
            AABB(
                min_corner=np.array([-1.00, -1.00, -1.00], dtype=float),
                max_corner=np.array([-0.55, -0.55, -0.55], dtype=float),
            ),
        ),
    ]

    query_rows: list[dict[str, float]] = []
    for query_name, query_box in queries:
        octree_hits: list[int] = []
        stats = QueryStats()
        root.query_range(query=query_box, out_indices=octree_hits, stats=stats)

        octree_hits_sorted = sorted(octree_hits)
        brute_hits_sorted = sorted(brute_force_range_query(points, query_box))

        if octree_hits_sorted != brute_hits_sorted:
            raise AssertionError(
                f"Query mismatch for {query_name}: octree={len(octree_hits_sorted)} brute={len(brute_hits_sorted)}"
            )

        octree_point_tests = int(stats.point_tests)
        brute_force_point_tests = n_points
        prune_ratio = 1.0 - float(octree_point_tests / brute_force_point_tests)
        test_reduction = float(brute_force_point_tests / max(octree_point_tests, 1))

        query_rows.append(
            {
                "query": query_name,
                "hit_count": float(len(octree_hits_sorted)),
                "visited_nodes": float(stats.visited_nodes),
                "visited_leaves": float(stats.visited_leaves),
                "point_tests_octree": float(octree_point_tests),
                "point_tests_bruteforce": float(brute_force_point_tests),
                "prune_ratio": prune_ratio,
                "test_reduction_x": test_reduction,
            }
        )

    query_df = pd.DataFrame(query_rows)

    leaf_loads: list[int] = []
    root.collect_leaf_loads(leaf_loads)

    depth_hist: dict[int, int] = {}
    root.collect_depth_histogram(depth_hist)
    depth_hist_df = pd.DataFrame(
        [{"depth": float(d), "node_count": float(c)} for d, c in sorted(depth_hist.items())]
    )

    tree_metrics = pd.DataFrame(
        {
            "metric": [
                "n_points",
                "max_depth_limit",
                "bucket_size",
                "node_count",
                "leaf_count",
                "max_depth_reached",
                "min_leaf_load",
                "avg_leaf_load",
                "max_leaf_load",
                "mean_query_prune_ratio",
                "mean_query_test_reduction_x",
            ],
            "value": [
                float(n_points),
                float(max_depth),
                float(bucket_size),
                float(root.count_nodes()),
                float(root.count_leaves()),
                float(root.max_depth_reached()),
                float(min(leaf_loads)),
                float(np.mean(leaf_loads)),
                float(max(leaf_loads)),
                float(query_df["prune_ratio"].mean()),
                float(query_df["test_reduction_x"].mean()),
            ],
        }
    )

    print("=== Octree MVP (CS-0261) ===")
    print(f"points: {n_points}")
    print(f"max_depth: {max_depth}, bucket_size: {bucket_size}")
    print()

    print("=== Tree Metrics ===")
    print(tree_metrics.to_string(index=False))
    print()

    print("=== Depth Histogram ===")
    print(depth_hist_df.to_string(index=False))
    print()

    print("=== Query Comparison (Octree vs Brute Force) ===")
    print(query_df.to_string(index=False))
    print()

    print("All checks passed.")


if __name__ == "__main__":
    main()
