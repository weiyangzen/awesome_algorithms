"""Vietoris-Rips complex: minimal runnable MVP.

This script builds Vietoris-Rips complexes from fixed point clouds,
then computes simplex counts and Betti numbers (over Z/2Z) for several
filtration thresholds.
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
    epsilon: float
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


def build_vietoris_rips_complex(
    distance_matrix: np.ndarray,
    epsilon: float,
    max_dim: int,
) -> ComplexByDim:
    """Build a Vietoris-Rips complex up to max_dim.

    A k-simplex is included iff all pairwise distances among its vertices
    are <= epsilon (clique complex definition).
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square")
    if epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")

    n_vertices = d.shape[0]
    adjacency = d <= (epsilon + 1e-12)
    np.fill_diagonal(adjacency, True)

    simplices: ComplexByDim = {0: [(i,) for i in range(n_vertices)]}
    vertices = range(n_vertices)

    for dim in range(1, max_dim + 1):
        current_dim: List[Simplex] = []
        for simplex in combinations(vertices, dim + 1):
            is_clique = True
            for i, j in combinations(simplex, 2):
                if not adjacency[i, j]:
                    is_clique = False
                    break
            if is_clique:
                current_dim.append(simplex)
        simplices[dim] = current_dim

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


def compute_betti_numbers(simplices: ComplexByDim, max_dim: int) -> Tuple[Dict[int, int], Dict[int, int]]:
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
        beta_k = n_k - rank_dk - rank_dk1
        betti[k] = int(beta_k)

    return betti, boundary_ranks


def run_filtration(points: np.ndarray, eps_values: List[float], max_dim: int) -> List[FiltrationRow]:
    """Evaluate Vietoris-Rips complex statistics for a list of eps values."""
    dmat = pairwise_distance_matrix(points)
    rows: List[FiltrationRow] = []

    for eps in eps_values:
        simplices = build_vietoris_rips_complex(dmat, epsilon=eps, max_dim=max_dim)
        betti, _ = compute_betti_numbers(simplices, max_dim=max_dim)
        rows.append(
            FiltrationRow(
                epsilon=float(eps),
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
    print("eps    | #V   #E   #T   | beta0 beta1 beta2")
    print("-------+----------------+-------------------")
    for r in rows:
        print(
            f"{r.epsilon:>5.2f}  | {r.n0:>3d} {r.n1:>4d} {r.n2:>4d} |"
            f" {r.beta0:>5d} {r.beta1:>5d} {r.beta2:>5d}"
        )


def make_circle_points(n: int = 12, radius: float = 1.0) -> np.ndarray:
    """Deterministic points sampled on a circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float64)


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

    circle_points = make_circle_points(n=12, radius=1.0)
    circle_eps = [0.40, 0.60, 2.05]
    circle_rows = run_filtration(circle_points, circle_eps, max_dim=max_dim)
    print_rows("Case A: Circle points", circle_rows)

    cluster_points = make_two_cluster_points()
    cluster_eps = [0.14, 0.30, 2.30]
    cluster_rows = run_filtration(cluster_points, cluster_eps, max_dim=max_dim)
    print_rows("Case B: Two clusters", cluster_rows)

    print("\nNotes:")
    print("1) Case A typically shows a 1-cycle (beta1=1) at intermediate epsilon.")
    print("2) Case B shows beta0 dropping as epsilon grows and clusters connect.")


if __name__ == "__main__":
    main()
