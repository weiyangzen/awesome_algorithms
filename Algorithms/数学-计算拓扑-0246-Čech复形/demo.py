"""Cech complex: minimal runnable MVP.

This script builds Cech complexes from fixed 2D point clouds under
ball radius filtration, then computes simplex counts and Betti numbers
over Z/2Z.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


Simplex = Tuple[int, ...]
ComplexByDim = Dict[int, List[Simplex]]


@dataclass
class FiltrationRow:
    radius: float
    n0: int
    n1: int
    n2: int
    beta0: int
    beta1: int
    beta2: int


def ensure_points(points: np.ndarray) -> np.ndarray:
    """Validate and return a 2D finite point array."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"points must be 2D, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("points must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points contain non-finite values")
    return arr


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distance matrix."""
    pts = ensure_points(points)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.linalg.norm(diff, ord=2, axis=2)


def triangle_min_enclosing_radius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return radius of the minimum enclosing circle of three 2D points."""
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    lengths = np.asarray([ab, bc, ca], dtype=np.float64)
    lengths_sq = lengths * lengths
    max_idx = int(np.argmax(lengths_sq))
    max_l2 = float(lengths_sq[max_idx])
    other_l2_sum = float(np.sum(lengths_sq) - max_l2)

    # If triangle is obtuse/right (or almost degenerate), MEC radius is half
    # of the longest edge.
    if max_l2 >= other_l2_sum - 1e-12:
        return 0.5 * float(lengths[max_idx])

    s = float(np.sum(lengths) / 2.0)
    area_sq = s * (s - ab) * (s - bc) * (s - ca)
    if area_sq <= 1e-18:
        return 0.5 * float(np.max(lengths))

    area = float(np.sqrt(area_sq))
    return (ab * bc * ca) / (4.0 * area)


def build_cech_complex_2d(points: np.ndarray, radius: float, max_dim: int) -> ComplexByDim:
    """Build Cech complex in R^2 up to dimension 2.

    For equal-radius closed balls B(x_i, radius), a simplex is included iff
    the intersection of balls over its vertices is non-empty.

    Implemented dimensions:
    - 0-simplex: always include each vertex
    - 1-simplex: ||x_i - x_j|| <= 2 * radius
    - 2-simplex: minimum enclosing circle radius of three points <= radius
    """
    pts = ensure_points(points)
    if radius < 0.0:
        raise ValueError("radius must be >= 0")
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")
    if max_dim > 2:
        raise ValueError("this MVP supports max_dim <= 2")

    n_vertices = pts.shape[0]
    d = pairwise_distance_matrix(pts)
    simplices: ComplexByDim = {0: [(i,) for i in range(n_vertices)]}
    edges: List[Simplex] = []
    edge_set = set()

    if max_dim >= 1:
        threshold = 2.0 * radius + 1e-12
        for i, j in combinations(range(n_vertices), 2):
            if d[i, j] <= threshold:
                edge = (i, j)
                edges.append(edge)
                edge_set.add(edge)
    simplices[1] = edges

    triangles: List[Simplex] = []
    if max_dim >= 2:
        for i, j, k in combinations(range(n_vertices), 3):
            # Quick necessary check: all three edges should exist.
            if (i, j) not in edge_set or (i, k) not in edge_set or (j, k) not in edge_set:
                continue
            mec_r = triangle_min_enclosing_radius(pts[i], pts[j], pts[k])
            if mec_r <= radius + 1e-12:
                triangles.append((i, j, k))
    simplices[2] = triangles

    return simplices


def boundary_matrix_mod2(higher: List[Simplex], lower: List[Simplex]) -> np.ndarray:
    """Build boundary matrix d_k: C_k -> C_{k-1} over Z/2Z."""
    matrix = np.zeros((len(lower), len(higher)), dtype=np.uint8)
    if not higher or not lower:
        return matrix

    lower_index = {simplex: idx for idx, simplex in enumerate(lower)}
    for col, simplex in enumerate(higher):
        for removed in range(len(simplex)):
            face = simplex[:removed] + simplex[removed + 1 :]
            row = lower_index.get(face)
            if row is not None:
                matrix[row, col] ^= 1
    return matrix


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute matrix rank over GF(2) via Gaussian elimination."""
    a = np.asarray(matrix, dtype=np.uint8).copy()
    n_rows, n_cols = a.shape
    rank = 0
    pivot_row = 0

    for col in range(n_cols):
        pivot = None
        for row in range(pivot_row, n_rows):
            if a[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            a[[pivot_row, pivot], :] = a[[pivot, pivot_row], :]

        for row in range(n_rows):
            if row != pivot_row and a[row, col] == 1:
                a[row, :] ^= a[pivot_row, :]

        rank += 1
        pivot_row += 1
        if pivot_row == n_rows:
            break

    return rank


def compute_betti_numbers(simplices: ComplexByDim, max_dim: int) -> Dict[int, int]:
    """Compute Betti numbers beta_k (0 <= k <= max_dim) over Z/2Z."""
    boundary_ranks: Dict[int, int] = {}
    for k in range(1, max_dim + 1):
        d_k = boundary_matrix_mod2(simplices.get(k, []), simplices.get(k - 1, []))
        boundary_ranks[k] = gf2_rank(d_k)

    betti: Dict[int, int] = {}
    for k in range(max_dim + 1):
        n_k = len(simplices.get(k, []))
        rank_dk = boundary_ranks.get(k, 0)
        rank_dk1 = boundary_ranks.get(k + 1, 0)
        betti[k] = int(n_k - rank_dk - rank_dk1)
    return betti


def run_filtration(points: np.ndarray, radii: List[float], max_dim: int = 2) -> List[FiltrationRow]:
    """Evaluate Cech complex statistics for each radius in filtration."""
    rows: List[FiltrationRow] = []
    for radius in radii:
        simplices = build_cech_complex_2d(points, radius=float(radius), max_dim=max_dim)
        betti = compute_betti_numbers(simplices, max_dim=max_dim)
        rows.append(
            FiltrationRow(
                radius=float(radius),
                n0=len(simplices.get(0, [])),
                n1=len(simplices.get(1, [])),
                n2=len(simplices.get(2, [])),
                beta0=betti.get(0, 0),
                beta1=betti.get(1, 0),
                beta2=betti.get(2, 0),
            )
        )
    return rows


def print_rows(title: str, rows: List[FiltrationRow]) -> None:
    """Pretty-print filtration summary."""
    print(f"\n=== {title} ===")
    print("r      | #V   #E   #T   | beta0 beta1 beta2")
    print("-------+----------------+-------------------")
    for r in rows:
        print(
            f"{r.radius:>5.2f}  | {r.n0:>3d} {r.n1:>4d} {r.n2:>4d} |"
            f" {r.beta0:>5d} {r.beta1:>5d} {r.beta2:>5d}"
        )


def make_equilateral_triangle(side: float = 1.0) -> np.ndarray:
    """Three points of an equilateral triangle in R^2."""
    h = np.sqrt(3.0) * side / 2.0
    return np.asarray(
        [
            [0.0, 0.0],
            [side, 0.0],
            [0.5 * side, h],
        ],
        dtype=np.float64,
    )


def make_two_cluster_points() -> np.ndarray:
    """Deterministic two-cluster 2D dataset."""
    cluster_a = np.asarray(
        [
            [-1.00, 0.00],
            [-1.12, 0.10],
            [-0.88, 0.10],
            [-1.10, -0.08],
            [-0.90, -0.06],
            [-1.00, 0.16],
        ],
        dtype=np.float64,
    )
    cluster_b = cluster_a + np.asarray([2.20, 0.00], dtype=np.float64)
    return np.vstack([cluster_a, cluster_b])


def main() -> None:
    max_dim = 2

    tri_points = make_equilateral_triangle(side=1.0)
    tri_radii = [0.40, 0.55, 0.60]
    tri_rows = run_filtration(tri_points, tri_radii, max_dim=max_dim)
    print_rows("Case A: Equilateral triangle", tri_rows)

    cluster_points = make_two_cluster_points()
    cluster_radii = [0.08, 0.18, 1.20]
    cluster_rows = run_filtration(cluster_points, cluster_radii, max_dim=max_dim)
    print_rows("Case B: Two clusters", cluster_rows)

    print("\nNotes:")
    print("1) In Case A, r=0.55 has all edges but no 2-simplex in Cech, so beta1=1.")
    print("2) In Case B, beta0 decreases as radius grows and two clusters merge.")


if __name__ == "__main__":
    main()
