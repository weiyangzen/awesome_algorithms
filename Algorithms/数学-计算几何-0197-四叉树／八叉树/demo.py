"""Minimal runnable MVP for Quadtree/Octree (MATH-0197)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class AABB2D:
    """Axis-aligned square bounding box in 2D."""

    cx: float
    cy: float
    half: float

    def contains(self, p: np.ndarray) -> bool:
        return (
            self.cx - self.half <= p[0] <= self.cx + self.half
            and self.cy - self.half <= p[1] <= self.cy + self.half
        )

    def intersects(self, other: "AABB2D") -> bool:
        return (
            abs(self.cx - other.cx) <= (self.half + other.half)
            and abs(self.cy - other.cy) <= (self.half + other.half)
        )


class QuadtreeNode:
    """Point quadtree supporting insertion and axis-aligned range query."""

    def __init__(self, boundary: AABB2D, capacity: int, max_depth: int, depth: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.points: List[np.ndarray] = []
        self.children: List[QuadtreeNode] | None = None

    def subdivide(self) -> None:
        if self.children is not None:
            return

        child_half = self.boundary.half / 2.0
        cx = self.boundary.cx
        cy = self.boundary.cy
        offsets = [(-child_half, -child_half), (child_half, -child_half), (-child_half, child_half), (child_half, child_half)]

        self.children = [
            QuadtreeNode(
                boundary=AABB2D(cx + dx, cy + dy, child_half),
                capacity=self.capacity,
                max_depth=self.max_depth,
                depth=self.depth + 1,
            )
            for dx, dy in offsets
        ]

    def insert(self, p: np.ndarray) -> bool:
        if not self.boundary.contains(p):
            return False

        if self.children is None:
            if len(self.points) < self.capacity or self.depth >= self.max_depth:
                self.points.append(p)
                return True

            self.subdivide()
            assert self.children is not None

            old_points = self.points
            self.points = []
            for old in old_points:
                inserted = False
                for child in self.children:
                    if child.insert(old):
                        inserted = True
                        break
                if not inserted:
                    raise RuntimeError("existing point failed to reinsert during quadtree split")

        assert self.children is not None
        for child in self.children:
            if child.insert(p):
                return True

        # Numerical fallback for edge points on split boundary.
        self.points.append(p)
        return True

    def _range_query(self, query: AABB2D, out: List[np.ndarray], counter: List[int]) -> None:
        counter[0] += 1
        if not self.boundary.intersects(query):
            return

        for p in self.points:
            if query.contains(p):
                out.append(p)

        if self.children is None:
            return

        for child in self.children:
            child._range_query(query, out, counter)

    def range_query(self, query: AABB2D) -> Tuple[List[np.ndarray], int]:
        out: List[np.ndarray] = []
        visited = [0]
        self._range_query(query, out, visited)
        return out, visited[0]

    def stats(self) -> Tuple[int, int, int, int]:
        """Return (node_count, leaf_count, max_depth_seen, stored_points)."""
        node_count = 1
        leaf_count = 1 if self.children is None else 0
        max_depth_seen = self.depth
        stored_points = len(self.points)

        if self.children is not None:
            leaf_count = 0
            for child in self.children:
                c_nodes, c_leaf, c_depth, c_points = child.stats()
                node_count += c_nodes
                leaf_count += c_leaf
                max_depth_seen = max(max_depth_seen, c_depth)
                stored_points += c_points

        return node_count, leaf_count, max_depth_seen, stored_points


@dataclass(frozen=True)
class AABB3D:
    """Axis-aligned cube bounding box in 3D."""

    cx: float
    cy: float
    cz: float
    half: float

    def contains(self, p: np.ndarray) -> bool:
        return (
            self.cx - self.half <= p[0] <= self.cx + self.half
            and self.cy - self.half <= p[1] <= self.cy + self.half
            and self.cz - self.half <= p[2] <= self.cz + self.half
        )

    def intersects(self, other: "AABB3D") -> bool:
        return (
            abs(self.cx - other.cx) <= (self.half + other.half)
            and abs(self.cy - other.cy) <= (self.half + other.half)
            and abs(self.cz - other.cz) <= (self.half + other.half)
        )


class OctreeNode:
    """Point octree supporting insertion and axis-aligned range query."""

    def __init__(self, boundary: AABB3D, capacity: int, max_depth: int, depth: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.points: List[np.ndarray] = []
        self.children: List[OctreeNode] | None = None

    def subdivide(self) -> None:
        if self.children is not None:
            return

        child_half = self.boundary.half / 2.0
        cx = self.boundary.cx
        cy = self.boundary.cy
        cz = self.boundary.cz
        offsets = [
            (-child_half, -child_half, -child_half),
            (child_half, -child_half, -child_half),
            (-child_half, child_half, -child_half),
            (child_half, child_half, -child_half),
            (-child_half, -child_half, child_half),
            (child_half, -child_half, child_half),
            (-child_half, child_half, child_half),
            (child_half, child_half, child_half),
        ]

        self.children = [
            OctreeNode(
                boundary=AABB3D(cx + dx, cy + dy, cz + dz, child_half),
                capacity=self.capacity,
                max_depth=self.max_depth,
                depth=self.depth + 1,
            )
            for dx, dy, dz in offsets
        ]

    def insert(self, p: np.ndarray) -> bool:
        if not self.boundary.contains(p):
            return False

        if self.children is None:
            if len(self.points) < self.capacity or self.depth >= self.max_depth:
                self.points.append(p)
                return True

            self.subdivide()
            assert self.children is not None

            old_points = self.points
            self.points = []
            for old in old_points:
                inserted = False
                for child in self.children:
                    if child.insert(old):
                        inserted = True
                        break
                if not inserted:
                    raise RuntimeError("existing point failed to reinsert during octree split")

        assert self.children is not None
        for child in self.children:
            if child.insert(p):
                return True

        self.points.append(p)
        return True

    def _range_query(self, query: AABB3D, out: List[np.ndarray], counter: List[int]) -> None:
        counter[0] += 1
        if not self.boundary.intersects(query):
            return

        for p in self.points:
            if query.contains(p):
                out.append(p)

        if self.children is None:
            return

        for child in self.children:
            child._range_query(query, out, counter)

    def range_query(self, query: AABB3D) -> Tuple[List[np.ndarray], int]:
        out: List[np.ndarray] = []
        visited = [0]
        self._range_query(query, out, visited)
        return out, visited[0]

    def stats(self) -> Tuple[int, int, int, int]:
        """Return (node_count, leaf_count, max_depth_seen, stored_points)."""
        node_count = 1
        leaf_count = 1 if self.children is None else 0
        max_depth_seen = self.depth
        stored_points = len(self.points)

        if self.children is not None:
            leaf_count = 0
            for child in self.children:
                c_nodes, c_leaf, c_depth, c_points = child.stats()
                node_count += c_nodes
                leaf_count += c_leaf
                max_depth_seen = max(max_depth_seen, c_depth)
                stored_points += c_points

        return node_count, leaf_count, max_depth_seen, stored_points


def brute_force_query_2d(points: np.ndarray, query: AABB2D) -> np.ndarray:
    mask = (
        (points[:, 0] >= query.cx - query.half)
        & (points[:, 0] <= query.cx + query.half)
        & (points[:, 1] >= query.cy - query.half)
        & (points[:, 1] <= query.cy + query.half)
    )
    return points[mask]


def brute_force_query_3d(points: np.ndarray, query: AABB3D) -> np.ndarray:
    mask = (
        (points[:, 0] >= query.cx - query.half)
        & (points[:, 0] <= query.cx + query.half)
        & (points[:, 1] >= query.cy - query.half)
        & (points[:, 1] <= query.cy + query.half)
        & (points[:, 2] >= query.cz - query.half)
        & (points[:, 2] <= query.cz + query.half)
    )
    return points[mask]


def canonical_set(points: Sequence[np.ndarray], decimals: int = 12) -> set[Tuple[float, ...]]:
    rounded: set[Tuple[float, ...]] = set()
    for p in points:
        t = tuple(float(x) for x in np.round(np.asarray(p, dtype=np.float64), decimals=decimals))
        rounded.add(t)
    return rounded


def run_quadtree_demo() -> None:
    rng = np.random.default_rng(197)
    points = rng.uniform(-1.0, 1.0, size=(2000, 2)).astype(np.float64)

    root = QuadtreeNode(boundary=AABB2D(0.0, 0.0, 1.0), capacity=16, max_depth=8)
    for p in points:
        if not root.insert(p):
            raise RuntimeError("point outside root boundary in quadtree demo")

    node_count, leaf_count, max_depth_seen, stored_points = root.stats()
    print("\n[Quadtree]")
    print(f"points={len(points)}, nodes={node_count}, leaves={leaf_count}, max_depth={max_depth_seen}, stored_points={stored_points}")

    queries = [
        AABB2D(0.0, 0.0, 0.35),
        AABB2D(-0.5, 0.45, 0.25),
        AABB2D(0.65, -0.2, 0.18),
    ]

    for i, q in enumerate(queries, start=1):
        tree_res, visited = root.range_query(q)
        brute_res = brute_force_query_2d(points, q)

        tree_set = canonical_set(tree_res)
        brute_set = canonical_set(brute_res)
        if tree_set != brute_set:
            raise RuntimeError(f"quadtree query {i} mismatch: tree={len(tree_set)} brute={len(brute_set)}")

        print(
            f"query{i}: hit={len(tree_set):4d}, visited_nodes={visited:4d}/{node_count:4d}, "
            f"center=({q.cx:+.2f},{q.cy:+.2f}), half={q.half:.2f}"
        )



def run_octree_demo() -> None:
    rng = np.random.default_rng(198)
    points = rng.uniform(-1.0, 1.0, size=(4000, 3)).astype(np.float64)

    root = OctreeNode(boundary=AABB3D(0.0, 0.0, 0.0, 1.0), capacity=24, max_depth=7)
    for p in points:
        if not root.insert(p):
            raise RuntimeError("point outside root boundary in octree demo")

    node_count, leaf_count, max_depth_seen, stored_points = root.stats()
    print("\n[Octree]")
    print(f"points={len(points)}, nodes={node_count}, leaves={leaf_count}, max_depth={max_depth_seen}, stored_points={stored_points}")

    queries = [
        AABB3D(0.0, 0.0, 0.0, 0.30),
        AABB3D(-0.45, 0.40, -0.35, 0.24),
        AABB3D(0.62, -0.18, 0.22, 0.20),
    ]

    for i, q in enumerate(queries, start=1):
        tree_res, visited = root.range_query(q)
        brute_res = brute_force_query_3d(points, q)

        tree_set = canonical_set(tree_res)
        brute_set = canonical_set(brute_res)
        if tree_set != brute_set:
            raise RuntimeError(f"octree query {i} mismatch: tree={len(tree_set)} brute={len(brute_set)}")

        print(
            f"query{i}: hit={len(tree_set):4d}, visited_nodes={visited:5d}/{node_count:5d}, "
            f"center=({q.cx:+.2f},{q.cy:+.2f},{q.cz:+.2f}), half={q.half:.2f}"
        )



def main() -> None:
    print("Quadtree/Octree MVP (MATH-0197)")
    print("=" * 72)
    run_quadtree_demo()
    run_octree_demo()
    print("\nAll checks passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
