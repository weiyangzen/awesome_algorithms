"""Minimal runnable MVP for point-in-polygon (MATH-0204)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


EPS = 1e-9


def polygon_signed_area(polygon: np.ndarray) -> float:
    """Return signed area of a polygon (positive for CCW)."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_polygon(polygon: Sequence[Sequence[float]], eps: float = EPS) -> np.ndarray:
    """Normalize polygon representation and remove duplicated consecutive vertices."""
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be an Nx2 array")
    if poly.shape[0] < 3:
        raise ValueError("polygon needs at least 3 points")

    if np.linalg.norm(poly[0] - poly[-1]) <= eps:
        poly = poly[:-1]

    cleaned: List[np.ndarray] = []
    for point in poly:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > eps:
            cleaned.append(point)

    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
        cleaned.pop()

    if len(cleaned) < 3:
        raise ValueError("polygon degenerates after cleanup")

    normalized = np.array(cleaned, dtype=np.float64)
    if abs(polygon_signed_area(normalized)) <= eps:
        raise ValueError("degenerate polygon with near-zero area")
    return normalized


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product value a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = EPS) -> bool:
    """Check whether point lies on the segment [a, b]."""
    ab = b - a
    ap = point - a
    bp = point - b

    if abs(cross2d(ab, ap)) > eps:
        return False

    # Dot product <= 0 means projection is between endpoints.
    if float(np.dot(ap, bp)) > eps:
        return False

    min_x = min(a[0], b[0]) - eps
    max_x = max(a[0], b[0]) + eps
    min_y = min(a[1], b[1]) - eps
    max_y = max(a[1], b[1]) + eps
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def point_in_polygon_ray(point: Sequence[float], polygon: Sequence[Sequence[float]], eps: float = EPS) -> str:
    """Classify a point using ray casting.

    Returns one of: "inside", "outside", "on_boundary".
    """
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (2,):
        raise ValueError("point must be a length-2 vector")

    poly = normalize_polygon(polygon, eps=eps)
    inside = False
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if point_on_segment(p, a, b, eps=eps):
            return "on_boundary"

        ay, by = a[1], b[1]
        crosses_scanline = (ay > p[1]) != (by > p[1])
        if not crosses_scanline:
            continue

        x_intersection = a[0] + (p[1] - ay) * (b[0] - a[0]) / (by - ay)
        if x_intersection > p[0] + eps:
            inside = not inside
        elif abs(x_intersection - p[0]) <= eps:
            return "on_boundary"

    return "inside" if inside else "outside"


def point_in_polygon_winding(point: Sequence[float], polygon: Sequence[Sequence[float]], eps: float = EPS) -> str:
    """Reference implementation using winding number for cross-checking."""
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (2,):
        raise ValueError("point must be a length-2 vector")

    poly = normalize_polygon(polygon, eps=eps)
    winding = 0
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if point_on_segment(p, a, b, eps=eps):
            return "on_boundary"

        if a[1] <= p[1]:
            if b[1] > p[1] and cross2d(b - a, p - a) > eps:
                winding += 1
        else:
            if b[1] <= p[1] and cross2d(b - a, p - a) < -eps:
                winding -= 1

    return "inside" if winding != 0 else "outside"


def classify_points(
    points: Iterable[Sequence[float]],
    polygon: Sequence[Sequence[float]],
    method: str = "ray",
    eps: float = EPS,
) -> List[str]:
    """Batch classify points against one polygon."""
    if method not in {"ray", "winding"}:
        raise ValueError("method must be one of {'ray', 'winding'}")

    classifier = point_in_polygon_ray if method == "ray" else point_in_polygon_winding
    return [classifier(point, polygon, eps=eps) for point in points]


def generate_random_convex_polygon(n_vertices: int, seed: int) -> np.ndarray:
    """Generate a simple random convex polygon for regression testing."""
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3")

    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))
    radii = rng.uniform(1.0, 3.0, size=n_vertices)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.column_stack([x, y])
    return normalize_polygon(polygon)


def run_named_case(case_name: str, polygon: np.ndarray, labeled_points: List[Tuple[str, np.ndarray]]) -> None:
    """Run one deterministic test case and print a compact report."""
    print(f"\n[{case_name}]")
    print("label                point(x, y)                 ray_cast")

    points = [pt for _, pt in labeled_points]
    ray_results = classify_points(points, polygon, method="ray")
    winding_results = classify_points(points, polygon, method="winding")

    for (label, pt), ray_result, winding_result in zip(labeled_points, ray_results, winding_results):
        if ray_result != winding_result:
            raise RuntimeError(
                f"method mismatch on {case_name}/{label}: ray={ray_result}, winding={winding_result}"
            )
        print(f"{label:<20} ({pt[0]:>7.3f}, {pt[1]:>7.3f})   {ray_result}")


def run_random_regression(num_trials: int = 5, points_per_trial: int = 40) -> None:
    """Randomized regression: ray casting must match winding-number reference."""
    for trial in range(num_trials):
        polygon = generate_random_convex_polygon(n_vertices=8, seed=100 + trial)
        rng = np.random.default_rng(1000 + trial)
        test_points = rng.uniform(-3.5, 3.5, size=(points_per_trial, 2))

        ray_results = classify_points(test_points, polygon, method="ray")
        winding_results = classify_points(test_points, polygon, method="winding")

        if ray_results != winding_results:
            for idx, (ray_result, winding_result) in enumerate(zip(ray_results, winding_results)):
                if ray_result != winding_result:
                    raise RuntimeError(
                        "random regression mismatch: "
                        f"trial={trial}, idx={idx}, ray={ray_result}, winding={winding_result}"
                    )


def main() -> None:
    print("Point-in-Polygon MVP (MATH-0204)")
    print("=" * 72)

    concave_polygon = np.array(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 4.0],
            [3.5, 2.2],
            [1.5, 4.8],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )

    labeled_points: List[Tuple[str, np.ndarray]] = [
        ("strictly_inside", np.array([1.0, 1.0], dtype=np.float64)),
        ("outside_right", np.array([6.8, 2.0], dtype=np.float64)),
        ("on_left_edge", np.array([0.0, 2.0], dtype=np.float64)),
        ("on_vertex", np.array([6.0, 4.0], dtype=np.float64)),
        ("in_concavity_outside", np.array([3.8, 3.9], dtype=np.float64)),
        ("near_indent_inside", np.array([3.2, 2.0], dtype=np.float64)),
    ]

    run_named_case("concave polygon", concave_polygon, labeled_points)

    axis_aligned_square = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )
    square_points: List[Tuple[str, np.ndarray]] = [
        ("center", np.array([2.0, 2.0], dtype=np.float64)),
        ("outside", np.array([4.1, 2.0], dtype=np.float64)),
        ("on_bottom", np.array([2.0, 0.0], dtype=np.float64)),
    ]

    run_named_case("axis-aligned square", axis_aligned_square, square_points)

    run_random_regression(num_trials=8, points_per_trial=64)

    print("\nRandom regression passed: ray casting == winding reference")
    print("All test cases passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
