"""Point location MVP using uniform-grid filtering + ray casting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class LocateResult:
    label: str
    relation: str  # INSIDE / BOUNDARY / OUTSIDE
    polygon_index: int | None


class PointLocator:
    """Locate query points in a planar polygon set.

    Assumption for this MVP:
    - polygons are simple (non-self-intersecting)
    - polygons do not overlap in area
    """

    def __init__(
        self,
        polygons: Sequence[Sequence[Point]],
        labels: Sequence[str],
        grid_resolution: int = 12,
        eps: float = 1e-9,
    ) -> None:
        if len(polygons) != len(labels):
            raise ValueError("polygons and labels must have same length")
        if not polygons:
            raise ValueError("at least one polygon is required")

        self.polygons = [np.asarray(poly, dtype=float) for poly in polygons]
        for poly in self.polygons:
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                raise ValueError("each polygon must be shaped as (n>=3, 2)")

        self.labels = list(labels)
        self.eps = eps
        self.grid_resolution = max(2, int(grid_resolution))

        self.bboxes = np.array(
            [
                [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()]
                for poly in self.polygons
            ],
            dtype=float,
        )
        self.global_bbox = np.array(
            [
                self.bboxes[:, 0].min(),
                self.bboxes[:, 1].min(),
                self.bboxes[:, 2].max(),
                self.bboxes[:, 3].max(),
            ],
            dtype=float,
        )

        self._grid: Dict[Tuple[int, int], List[int]] = {}
        self._build_grid_index()

    def _build_grid_index(self) -> None:
        """Map each grid cell to candidate polygon indices."""
        min_x, min_y, max_x, max_y = self.global_bbox
        width = max(max_x - min_x, self.eps)
        height = max(max_y - min_y, self.eps)

        for idx, (px_min, py_min, px_max, py_max) in enumerate(self.bboxes):
            gx0 = int(np.clip((px_min - min_x) / width * self.grid_resolution, 0, self.grid_resolution - 1))
            gy0 = int(np.clip((py_min - min_y) / height * self.grid_resolution, 0, self.grid_resolution - 1))
            gx1 = int(np.clip((px_max - min_x) / width * self.grid_resolution, 0, self.grid_resolution - 1))
            gy1 = int(np.clip((py_max - min_y) / height * self.grid_resolution, 0, self.grid_resolution - 1))

            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self._grid.setdefault((gx, gy), []).append(idx)

    def _cell_of_point(self, x: float, y: float) -> Tuple[int, int]:
        min_x, min_y, max_x, max_y = self.global_bbox
        width = max(max_x - min_x, self.eps)
        height = max(max_y - min_y, self.eps)

        tx = np.clip((x - min_x) / width, 0.0, 1.0 - 1e-12)
        ty = np.clip((y - min_y) / height, 0.0, 1.0 - 1e-12)

        gx = int(tx * self.grid_resolution)
        gy = int(ty * self.grid_resolution)
        return gx, gy

    @staticmethod
    def _point_on_segment(p: Point, a: np.ndarray, b: np.ndarray, eps: float) -> bool:
        px, py = p
        ax, ay = a
        bx, by = b

        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        if abs(cross) > eps:
            return False

        # p is between a and b iff projection distances have opposite signs or one is zero.
        dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
        return dot <= eps

    def _point_in_polygon(self, p: Point, poly: np.ndarray) -> str:
        x, y = p
        n = poly.shape[0]

        # Boundary check first.
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if self._point_on_segment(p, a, b, self.eps):
                return "BOUNDARY"

        # Ray casting: count crossings of horizontal ray to +x.
        inside = False
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[(i + 1) % n]

            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi + self.eps) + xi
                if x < x_cross:
                    inside = not inside

        return "INSIDE" if inside else "OUTSIDE"

    def _candidate_indices(self, x: float, y: float) -> List[int]:
        min_x, min_y, max_x, max_y = self.global_bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return []
        cell = self._cell_of_point(x, y)
        return self._grid.get(cell, [])

    def locate(self, p: Point) -> LocateResult:
        x, y = p
        candidates = self._candidate_indices(x, y)

        # If a grid cell has no candidates (sparse map), fallback to all polygons.
        scan_order = candidates if candidates else list(range(len(self.polygons)))

        for idx in scan_order:
            relation = self._point_in_polygon((x, y), self.polygons[idx])
            if relation != "OUTSIDE":
                return LocateResult(self.labels[idx], relation, idx)

        return LocateResult("OUTSIDE", "OUTSIDE", None)

    def locate_bruteforce(self, p: Point) -> LocateResult:
        x, y = p
        for idx, poly in enumerate(self.polygons):
            relation = self._point_in_polygon((x, y), poly)
            if relation != "OUTSIDE":
                return LocateResult(self.labels[idx], relation, idx)
        return LocateResult("OUTSIDE", "OUTSIDE", None)


def build_demo_map() -> Tuple[List[List[Point]], List[str]]:
    polygons = [
        [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
        [(5.0, 0.0), (9.0, 0.0), (8.0, 4.0), (6.0, 5.0), (5.0, 3.0)],
        [(1.0, 4.0), (3.0, 8.0), (-1.0, 7.0)],
        [(6.0, 6.0), (10.0, 6.0), (10.0, 10.0), (8.0, 8.0), (6.0, 10.0)],
    ]
    labels = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]
    return polygons, labels


def run_demo() -> None:
    polygons, labels = build_demo_map()
    locator = PointLocator(polygons, labels, grid_resolution=10)

    query_points: List[Point] = [
        (2.0, 1.0),
        (4.0, 2.0),
        (7.2, 1.8),
        (7.9, 8.1),
        (2.0, 6.5),
        (11.0, 11.0),
        (5.0, 3.0),
    ]

    print("=== Point Location Results ===")
    for p in query_points:
        result = locator.locate(p)
        print(f"point={p!r:>14} -> label={result.label:<8} relation={result.relation}")

    # Simple benchmark: indexed locate vs brute force.
    rng = np.random.default_rng(42)
    bench_points = rng.uniform(low=-2.0, high=12.0, size=(20000, 2))

    t0 = perf_counter()
    indexed_results = [locator.locate((float(x), float(y))) for x, y in bench_points]
    t1 = perf_counter()

    brute_results = [locator.locate_bruteforce((float(x), float(y))) for x, y in bench_points]
    t2 = perf_counter()

    # Consistency check.
    mismatch = sum(
        1
        for a, b in zip(indexed_results, brute_results)
        if (a.label, a.relation, a.polygon_index) != (b.label, b.relation, b.polygon_index)
    )

    indexed_ms = (t1 - t0) * 1000
    brute_ms = (t2 - t1) * 1000
    speedup = brute_ms / indexed_ms if indexed_ms > 0 else float("inf")

    print("\n=== Benchmark (20,000 queries) ===")
    print(f"indexed search : {indexed_ms:.2f} ms")
    print(f"brute force    : {brute_ms:.2f} ms")
    print(f"speedup        : {speedup:.2f}x")
    print(f"result mismatch: {mismatch}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
