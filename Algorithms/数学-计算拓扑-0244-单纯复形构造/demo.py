"""Minimal runnable MVP for simplicial complex construction.

This demo builds a 2D Vietoris-Rips simplicial complex from point-cloud data,
then verifies chain-complex consistency with boundary operators.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ModuleNotFoundError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    pdist = None
    squareform = None
    SCIPY_AVAILABLE = False

try:
    from sklearn.datasets import make_circles
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    make_circles = None
    NearestNeighbors = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


Vertex = Tuple[int]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]


@dataclass
class SimplicialComplex:
    vertices: List[Vertex]
    edges: List[Edge]
    triangles: List[Triangle]
    epsilon: float


def generate_point_cloud(n_samples: int = 90, noise: float = 0.06, seed: int = 2026) -> np.ndarray:
    """Generate a deterministic 2D point cloud and standardize it."""
    if SKLEARN_AVAILABLE:
        X, _ = make_circles(n_samples=n_samples, factor=0.45, noise=noise, random_state=seed)
        X = StandardScaler().fit_transform(X)
        return X.astype(float)

    # NumPy fallback if scikit-learn is unavailable.
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = 0.45 * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    X = np.vstack([outer, inner])
    X = X + noise * rng.standard_normal(size=X.shape)
    X = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-12)
    return X.astype(float)


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute full Euclidean distance matrix (SciPy preferred, NumPy fallback)."""
    if SCIPY_AVAILABLE:
        D = squareform(pdist(X, metric="euclidean"))
        np.fill_diagonal(D, 0.0)
        return D

    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, 0.0)
    return D


def choose_epsilon(X: np.ndarray, n_neighbors: int = 6, q: float = 0.78) -> float:
    """Pick a robust Rips threshold from kNN distance quantile."""
    if SKLEARN_AVAILABLE:
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X)
        distances, _ = knn.kneighbors(X)
        kth_dist = distances[:, -1]
        return float(np.quantile(kth_dist, q))

    D = pairwise_distance_matrix(X)
    sorted_dist = np.sort(D, axis=1)
    # sorted_dist[:, 0] is self-distance, so use column n_neighbors.
    kth_col = min(max(n_neighbors, 1), X.shape[0] - 1)
    kth_dist = sorted_dist[:, kth_col]
    return float(np.quantile(kth_dist, q))


def build_vietoris_rips_complex(distance_matrix: np.ndarray, epsilon: float) -> SimplicialComplex:
    """Construct a 2-skeleton Vietoris-Rips complex.

    Rule:
    - Vertex for every sample point
    - Edge (i, j) if d(i, j) <= epsilon
    - Triangle (i, j, k) if all three pairwise edges exist
    """
    n = distance_matrix.shape[0]
    if n == 0:
        raise ValueError("distance_matrix must be non-empty")

    vertices: List[Vertex] = [(i,) for i in range(n)]

    edge_mask = distance_matrix <= epsilon
    np.fill_diagonal(edge_mask, False)

    edges: List[Edge] = []
    for i, j in combinations(range(n), 2):
        if edge_mask[i, j]:
            edges.append((i, j))

    triangles: List[Triangle] = []
    for i, j, k in combinations(range(n), 3):
        if edge_mask[i, j] and edge_mask[i, k] and edge_mask[j, k]:
            triangles.append((i, j, k))

    return SimplicialComplex(vertices=vertices, edges=edges, triangles=triangles, epsilon=epsilon)


def summarize_complex(complex_: SimplicialComplex) -> List[Dict[str, float]]:
    """Create dimension-count summary rows."""
    rows = [
        {"dimension": 0, "simplex_count": len(complex_.vertices)},
        {"dimension": 1, "simplex_count": len(complex_.edges)},
        {"dimension": 2, "simplex_count": len(complex_.triangles)},
    ]
    return rows


def format_summary_table(rows: Sequence[Dict[str, float]]) -> str:
    """Format summary rows, using pandas when available."""
    enriched: List[Dict[str, float]] = []
    for i, row in enumerate(rows):
        prev_count = rows[i - 1]["simplex_count"] if i > 0 else np.nan
        density = np.nan if i == 0 else float(row["simplex_count"] / max(prev_count, 1))
        enriched.append(
            {
                "dimension": int(row["dimension"]),
                "simplex_count": int(row["simplex_count"]),
                "density_vs_previous_dim": density,
            }
        )

    if PANDAS_AVAILABLE:
        return pd.DataFrame(enriched).to_string(index=False)

    header = "dimension  simplex_count  density_vs_previous_dim"
    lines = [header]
    for row in enriched:
        density = row["density_vs_previous_dim"]
        density_text = "nan" if np.isnan(density) else f"{density:.6f}"
        lines.append(
            f"{row['dimension']:9d}  {row['simplex_count']:13d}  {density_text:>22s}"
        )
    return "\n".join(lines)


def euler_characteristic(complex_: SimplicialComplex) -> int:
    """Compute Euler characteristic chi = n0 - n1 + n2."""
    return len(complex_.vertices) - len(complex_.edges) + len(complex_.triangles)


def boundary_matrix_1(vertices: Sequence[Vertex], edges: Sequence[Edge]) -> np.ndarray:
    """Build boundary operator d1: C1 -> C0 as an oriented incidence matrix."""
    n0 = len(vertices)
    n1 = len(edges)
    B1 = np.zeros((n0, n1), dtype=np.int64)
    for col, (u, v) in enumerate(edges):
        B1[u, col] = -1.0
        B1[v, col] = 1.0
    return B1


def boundary_matrix_2(edges: Sequence[Edge], triangles: Sequence[Triangle]) -> np.ndarray:
    """Build boundary operator d2: C2 -> C1 with canonical orientation.

    For oriented simplex (a,b,c):
        d2(a,b,c) = (b,c) - (a,c) + (a,b)
    """
    edge_to_row: Dict[Edge, int] = {e: idx for idx, e in enumerate(edges)}
    n1 = len(edges)
    n2 = len(triangles)
    B2 = np.zeros((n1, n2), dtype=np.int64)

    for col, (a, b, c) in enumerate(triangles):
        oriented_faces = [((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)]
        for face, coeff in oriented_faces:
            row = edge_to_row.get(face)
            if row is None:
                # In a valid Rips complex this should not happen.
                continue
            B2[row, col] = coeff
    return B2


def matrix_rank(M: np.ndarray) -> int:
    """Compute matrix rank using torch if available, otherwise NumPy."""
    if M.size == 0:
        return 0
    M_float = M.astype(np.float64, copy=False)
    if TORCH_AVAILABLE:
        t = torch.from_numpy(M_float)
        return int(torch.linalg.matrix_rank(t).item())
    return int(np.linalg.matrix_rank(M_float))


def betti_numbers_from_boundaries(B1: np.ndarray, B2: np.ndarray, n0: int, n1: int, n2: int) -> Tuple[int, int, int]:
    """Estimate Betti numbers (over R) from boundary matrix ranks.

    beta0 = n0 - rank(d1)
    beta1 = n1 - rank(d1) - rank(d2)
    beta2 = n2 - rank(d2)
    """
    r1 = matrix_rank(B1)
    r2 = matrix_rank(B2)
    beta0 = n0 - r1
    beta1 = n1 - r1 - r2
    beta2 = n2 - r2
    return int(beta0), int(beta1), int(beta2)


def preview_simplices(name: str, simplices: Sequence[Tuple[int, ...]], limit: int = 8) -> str:
    """Format a compact simplex preview string."""
    shown = list(simplices[:limit])
    text = ", ".join(str(s) for s in shown)
    if len(simplices) > limit:
        text += ", ..."
    return f"{name}({len(simplices)}): {text}"


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    X = generate_point_cloud(n_samples=90, noise=0.06, seed=2026)
    epsilon = choose_epsilon(X, n_neighbors=6, q=0.78)
    D = pairwise_distance_matrix(X)

    complex_ = build_vietoris_rips_complex(D, epsilon=epsilon)

    B1 = boundary_matrix_1(complex_.vertices, complex_.edges)
    B2 = boundary_matrix_2(complex_.edges, complex_.triangles)
    if B2.shape[1] > 0:
        composition = B1 @ B2
    else:
        composition = np.zeros((B1.shape[0], 0), dtype=float)

    beta0, beta1, beta2 = betti_numbers_from_boundaries(
        B1,
        B2,
        n0=len(complex_.vertices),
        n1=len(complex_.edges),
        n2=len(complex_.triangles),
    )

    summary_rows = summarize_complex(complex_)

    max_abs_boundary_composition = (
        float(np.max(np.abs(composition))) if composition.size > 0 else 0.0
    )

    print("=== Simplicial Complex Construction MVP (Vietoris-Rips, up to 2-simplex) ===")
    print(f"points={X.shape[0]}, dimension={X.shape[1]}, epsilon={complex_.epsilon:.4f}")
    print(f"backend_for_rank={'torch' if TORCH_AVAILABLE else 'numpy'}")
    print(
        "availability: "
        f"sklearn={SKLEARN_AVAILABLE}, scipy={SCIPY_AVAILABLE}, "
        f"pandas={PANDAS_AVAILABLE}, torch={TORCH_AVAILABLE}"
    )
    print("\n[Simplex Count Summary]")
    print(format_summary_table(summary_rows))

    print("\n[Topological Diagnostics]")
    print(f"Euler characteristic (chi) = {euler_characteristic(complex_)}")
    print(f"Betti estimate over R: beta0={beta0}, beta1={beta1}, beta2={beta2}")
    print(f"max|d1*d2| = {max_abs_boundary_composition:.6g} (should be 0)")

    print("\n[Preview]")
    print(preview_simplices("vertices", complex_.vertices, limit=8))
    print(preview_simplices("edges", complex_.edges, limit=8))
    print(preview_simplices("triangles", complex_.triangles, limit=8))


if __name__ == "__main__":
    main()
