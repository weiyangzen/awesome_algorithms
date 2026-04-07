"""Linear-time triangulation MVP for y-monotone polygons.

This script demonstrates the classic stack-based O(n) triangulation phase
for a simple y-monotone polygon. It keeps the implementation transparent and
avoids black-box geometry libraries.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]

EPS = 1e-12


def signed_area2(vertices: Sequence[Point]) -> float:
    """Return 2x signed polygon area (positive for CCW order)."""
    total = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def orientation(a: Point, b: Point, c: Point) -> float:
    """Cross product sign of vectors AB and AC."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def normalized_edge(i: int, j: int) -> Edge:
    """Store an undirected edge in canonical order."""
    return (i, j) if i < j else (j, i)


def is_boundary_edge(i: int, j: int, n: int) -> bool:
    """Check whether (i, j) is one of the polygon boundary edges."""
    if i == j:
        return False
    return (i + 1) % n == j or (j + 1) % n == i


def ensure_ccw(vertices: Sequence[Point]) -> List[Point]:
    """Return vertices in CCW order."""
    ordered = list(vertices)
    if signed_area2(ordered) < 0:
        ordered.reverse()
    return ordered


def walk_indices(start: int, end: int, step: int, n: int) -> List[int]:
    """Walk cyclic indices from start to end (inclusive)."""
    path = [start]
    cursor = start
    while cursor != end:
        cursor = (cursor + step) % n
        path.append(cursor)
    return path


def assert_nonincreasing_chain(vertices: Sequence[Point], chain: Sequence[int]) -> None:
    """Validate y-monotone direction from top to bottom on a chain."""
    ys = [vertices[idx][1] for idx in chain]
    for k in range(len(ys) - 1):
        if ys[k] < ys[k + 1] - EPS:
            raise ValueError("Input polygon is not y-monotone.")


def x_intersection_on_chain(vertices: Sequence[Point], chain: Sequence[int], y_ref: float) -> float:
    """Compute x where chain intersects the horizontal line y=y_ref."""
    for u, v in zip(chain, chain[1:]):
        y1 = vertices[u][1]
        y2 = vertices[v][1]
        if abs(y1 - y2) < EPS:
            continue
        if (y1 - y_ref) * (y2 - y_ref) <= 0:
            t = (y_ref - y1) / (y2 - y1)
            return vertices[u][0] + t * (vertices[v][0] - vertices[u][0])
    raise ValueError("Failed to classify chains: ambiguous y-reference intersection.")


def classify_y_monotone_chains(vertices: Sequence[Point]) -> Tuple[int, int, List[int], List[int], Dict[int, str]]:
    """Classify polygon vertices into left/right chains.

    Returns:
        top_idx, bottom_idx, left_chain(top->bottom), right_chain(top->bottom), chain_map
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon needs at least 3 vertices.")

    top = max(range(n), key=lambda i: (vertices[i][1], -vertices[i][0]))
    bottom = min(range(n), key=lambda i: (vertices[i][1], vertices[i][0]))

    chain_forward = walk_indices(top, bottom, step=1, n=n)
    chain_backward = walk_indices(top, bottom, step=-1, n=n)

    assert_nonincreasing_chain(vertices, chain_forward)
    assert_nonincreasing_chain(vertices, chain_backward)

    y_ref = 0.5 * (vertices[top][1] + vertices[bottom][1])
    x_fwd = x_intersection_on_chain(vertices, chain_forward, y_ref)
    x_bwd = x_intersection_on_chain(vertices, chain_backward, y_ref)
    if abs(x_fwd - x_bwd) < EPS:
        raise ValueError("Failed to classify chains: left/right overlap at y_ref.")

    if x_fwd < x_bwd:
        left_chain, right_chain = chain_forward, chain_backward
    else:
        left_chain, right_chain = chain_backward, chain_forward

    chain_map: Dict[int, str] = {idx: "left" for idx in left_chain}
    for idx in right_chain:
        chain_map[idx] = "right"
    chain_map[top] = "top"
    chain_map[bottom] = "bottom"

    return top, bottom, left_chain, right_chain, chain_map


def merge_chains_descending_y(
    vertices: Sequence[Point],
    left_chain: Sequence[int],
    right_chain: Sequence[int],
) -> List[int]:
    """Merge two top->bottom chains in O(n) by descending y (tie by x)."""
    order: List[int] = []
    seen = set()

    i = 0
    j = 0
    while i < len(left_chain) and j < len(right_chain):
        li = left_chain[i]
        rj = right_chain[j]
        ly, ry = vertices[li][1], vertices[rj][1]
        if ly > ry + EPS:
            chosen = li
            i += 1
        elif ry > ly + EPS:
            chosen = rj
            j += 1
        else:
            lx = vertices[li][0]
            rx = vertices[rj][0]
            if lx <= rx:
                chosen = li
                i += 1
            else:
                chosen = rj
                j += 1
        if chosen not in seen:
            seen.add(chosen)
            order.append(chosen)

    for chain, start in ((left_chain, i), (right_chain, j)):
        for idx in chain[start:]:
            if idx not in seen:
                seen.add(idx)
                order.append(idx)

    return order


def same_chain(i: int, j: int, chain_map: Dict[int, str]) -> bool:
    """Check whether two vertices are on the same strict side chain."""
    side_i = chain_map[i]
    side_j = chain_map[j]
    return side_i in {"left", "right"} and side_i == side_j


def triangulate_y_monotone_linear(vertices: Sequence[Point]) -> Tuple[List[Point], List[int], Dict[int, str], List[Edge]]:
    """Triangulate a simple y-monotone polygon in linear time.

    The input may be CW or CCW; output vertices are normalized to CCW.
    """
    polygon = ensure_ccw(vertices)
    n = len(polygon)
    if n < 3:
        raise ValueError("A polygon needs at least 3 vertices.")
    if n == 3:
        return polygon, [0, 1, 2], {0: "top", 1: "left", 2: "bottom"}, []

    top, bottom, left_chain, right_chain, chain_map = classify_y_monotone_chains(polygon)
    order = merge_chains_descending_y(polygon, left_chain, right_chain)

    if len(order) != n:
        raise ValueError("Internal error: merged order does not cover all vertices.")
    if order[0] != top or order[-1] != bottom:
        raise ValueError("Input polygon is not y-monotone from top to bottom.")

    stack: List[int] = [order[0], order[1]]
    diagonals = set()

    for pos in range(2, n - 1):
        current = order[pos]
        if not same_chain(current, stack[-1], chain_map):
            while len(stack) > 1:
                a = current
                b = stack[-1]
                if not is_boundary_edge(a, b, n):
                    diagonals.add(normalized_edge(a, b))
                stack.pop()
            stack.pop()
            previous = order[pos - 1]
            stack = [previous, current]
            continue

        last = stack.pop()
        while stack:
            turn = orientation(polygon[current], polygon[last], polygon[stack[-1]])
            if chain_map[current] == "left":
                visible = turn > EPS
            else:
                visible = turn < -EPS
            if not visible:
                break
            a = current
            b = stack[-1]
            if not is_boundary_edge(a, b, n):
                diagonals.add(normalized_edge(a, b))
            last = stack.pop()
        stack.append(last)
        stack.append(current)

    final_vertex = order[-1]
    stack.pop()
    while len(stack) > 1:
        a = final_vertex
        b = stack[-1]
        if not is_boundary_edge(a, b, n):
            diagonals.add(normalized_edge(a, b))
        stack.pop()

    diagonal_list = sorted(diagonals)
    if len(diagonal_list) != n - 3:
        raise ValueError(
            f"Triangulation failed: expected {n - 3} diagonals, got {len(diagonal_list)}."
        )

    return polygon, order, chain_map, diagonal_list


def triangles_from_edges(vertices: Sequence[Point], diagonals: Sequence[Edge]) -> List[Triangle]:
    """Recover triangle faces from boundary edges + diagonals.

    This is an O(n^3) reporting helper, separate from the O(n) triangulation core.
    """
    n = len(vertices)
    edge_set = set(diagonals)
    for i in range(n):
        edge_set.add(normalized_edge(i, (i + 1) % n))

    triangles: List[Triangle] = []
    for i, j, k in combinations(range(n), 3):
        if (
            normalized_edge(i, j) in edge_set
            and normalized_edge(i, k) in edge_set
            and normalized_edge(j, k) in edge_set
        ):
            if abs(orientation(vertices[i], vertices[j], vertices[k])) > EPS:
                triangles.append((i, j, k))

    return triangles


def triangle_area_abs(vertices: Sequence[Point], tri: Triangle) -> float:
    i, j, k = tri
    return abs(orientation(vertices[i], vertices[j], vertices[k])) * 0.5


def run_demo(vertices: Sequence[Point]) -> None:
    polygon, order, chain_map, diagonals = triangulate_y_monotone_linear(vertices)
    triangles = triangles_from_edges(polygon, diagonals)

    n = len(polygon)
    polygon_area = abs(signed_area2(polygon)) * 0.5
    triangulated_area = sum(triangle_area_abs(polygon, tri) for tri in triangles)

    print("=== Linear-Time Triangulation for Y-Monotone Polygon ===")
    print(f"vertex_count = {n}")
    print(f"triangles_expected = {n - 2}, diagonals_expected = {n - 3}")
    print("\nVertices (CCW, index: (x, y)):")
    for idx, point in enumerate(polygon):
        print(f"  {idx}: {point}")

    print("\nMerged top->bottom processing order:")
    print(" ", order)

    print("\nChain labels (top/bottom/left/right):")
    print(" ", {idx: chain_map[idx] for idx in range(n)})

    print("\nDiagonals (index pairs):")
    print(" ", diagonals)

    print("\nTriangles recovered from edges:")
    for tri in triangles:
        print(f"  {tri}  area={triangle_area_abs(polygon, tri):.6f}")

    print("\nChecks:")
    print(f"  diagonal_count_ok = {len(diagonals) == n - 3}")
    print(f"  triangle_count_ok = {len(triangles) == n - 2}")
    print(f"  area_match_ok = {abs(triangulated_area - polygon_area) < 1e-9}")
    print(f"  polygon_area = {polygon_area:.6f}")
    print(f"  triangulated_area = {triangulated_area:.6f}")


def main() -> None:
    # A fixed simple y-monotone polygon (non-convex), no interactive input.
    sample_polygon: List[Point] = [
        (0.0, 0.0),
        (4.0, 0.2),
        (5.0, 2.0),
        (4.0, 4.0),
        (2.0, 5.0),
        (0.0, 4.0),
        (-1.0, 1.0),
    ]
    run_demo(sample_polygon)


if __name__ == "__main__":
    main()
