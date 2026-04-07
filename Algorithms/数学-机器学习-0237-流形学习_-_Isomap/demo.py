"""Minimal runnable MVP for Isomap manifold learning.

Implementation policy:
- no interactive input
- deterministic via fixed random seed
- transparent algorithm flow (no direct call to external Isomap black box)
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Sequence, Tuple

import numpy as np


Array = np.ndarray
AdjList = List[List[Tuple[int, float]]]


@dataclass
class IsomapResult:
    embedding: Array
    geodesic_distances: Array
    eigenvalues: Array
    n_neighbors: int


def make_swiss_roll_numpy(
    n_samples: int = 450,
    noise: float = 0.04,
    random_state: int = 2026,
) -> Tuple[Array, Array]:
    """Generate Swiss-roll-like 3D data with NumPy only."""
    rng = np.random.default_rng(random_state)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_samples))
    x = t * np.cos(t)
    y = 21.0 * rng.random(n_samples)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    if noise > 0.0:
        X = X + noise * rng.standard_normal(size=X.shape)
    return X.astype(float), t.astype(float)


def pairwise_euclidean_distances(X: Array) -> Array:
    """Compute full Euclidean distance matrix with vectorized NumPy."""
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, 0.0)
    return D


def build_knn_graph(X: Array, n_neighbors: int) -> AdjList:
    """Build a symmetric weighted kNN graph from Euclidean neighbors."""
    n_samples = X.shape[0]
    if not (1 <= n_neighbors < n_samples):
        raise ValueError("n_neighbors must satisfy 1 <= n_neighbors < n_samples")

    distances = pairwise_euclidean_distances(X)
    indices = np.argsort(distances, axis=1)[:, 1 : n_neighbors + 1]  # skip self

    edge_maps: List[Dict[int, float]] = [dict() for _ in range(n_samples)]
    for i in range(n_samples):
        for j in indices[i]:
            jj = int(j)
            dd = float(distances[i, jj])
            if jj == i:
                continue
            old_ij = edge_maps[i].get(jj)
            if old_ij is None or dd < old_ij:
                edge_maps[i][jj] = dd
            old_ji = edge_maps[jj].get(i)
            if old_ji is None or dd < old_ji:
                edge_maps[jj][i] = dd

    graph: AdjList = []
    for i in range(n_samples):
        nbrs = sorted(edge_maps[i].items(), key=lambda t: t[0])
        graph.append([(j, w) for j, w in nbrs])
    return graph


def is_connected(graph: AdjList) -> bool:
    """Check graph connectivity via DFS."""
    n = len(graph)
    if n == 0:
        return True
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v, _ in graph[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return bool(np.all(visited))


def dijkstra_single_source(graph: AdjList, source: int) -> Array:
    """Shortest paths from one source on non-negative weighted graph."""
    n = len(graph)
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist[u]:
            continue
        for v, w in graph[u]:
            nd = cur_dist + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def all_pairs_geodesic_distances(graph: AdjList) -> Array:
    """Compute all-pairs shortest paths by repeated Dijkstra."""
    n = len(graph)
    out = np.empty((n, n), dtype=float)
    for src in range(n):
        out[src] = dijkstra_single_source(graph, src)
    return out


def classical_mds(distance_matrix: Array, n_components: int) -> Tuple[Array, Array]:
    """Classical MDS from pairwise distances."""
    n = distance_matrix.shape[0]
    if n_components <= 0 or n_components >= n:
        raise ValueError("n_components must satisfy 1 <= n_components < n_samples")

    d2 = distance_matrix ** 2
    row_mean = np.mean(d2, axis=1, keepdims=True)
    col_mean = np.mean(d2, axis=0, keepdims=True)
    total_mean = float(np.mean(d2))
    gram = -0.5 * (d2 - row_mean - col_mean + total_mean)
    gram = 0.5 * (gram + gram.T)

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-12
    if int(np.sum(positive)) < n_components:
        raise ValueError(
            f"Not enough positive eigenvalues for embedding: "
            f"need {n_components}, got {int(np.sum(positive))}"
        )

    lam = eigvals[:n_components]
    vecs = eigvecs[:, :n_components]
    embedding = vecs * np.sqrt(lam)
    return embedding, eigvals


def fit_isomap(
    X: Array,
    n_components: int = 2,
    candidate_neighbors: Sequence[int] = (8, 10, 12, 15, 20),
) -> IsomapResult:
    """Try several k values and return the first connected Isomap result."""
    last_error: Exception | None = None
    for k in candidate_neighbors:
        try:
            graph = build_knn_graph(X, n_neighbors=k)
            if not is_connected(graph):
                raise ValueError(f"kNN graph disconnected for k={k}")
            geodesic = all_pairs_geodesic_distances(graph)
            if not np.all(np.isfinite(geodesic)):
                raise ValueError(f"Found inf geodesic distance for k={k}")
            embedding, eigvals = classical_mds(geodesic, n_components=n_components)
            return IsomapResult(
                embedding=embedding,
                geodesic_distances=geodesic,
                eigenvalues=eigvals,
                n_neighbors=k,
            )
        except ValueError as exc:
            last_error = exc

    msg = "Unable to fit Isomap with provided neighbor candidates."
    if last_error is not None:
        msg += f" Last error: {last_error}"
    raise RuntimeError(msg)


def residual_variance(geodesic_distances: Array, embedding: Array) -> float:
    """Compute residual variance: 1 - corr(d_geo, d_emb)^2."""
    iu = np.triu_indices(geodesic_distances.shape[0], k=1)
    d_geo = geodesic_distances[iu]
    d_emb = pairwise_euclidean_distances(embedding)[iu]
    corr = float(np.corrcoef(d_geo, d_emb)[0, 1])
    return float(1.0 - corr * corr)


def trustworthiness_score(X: Array, Y: Array, n_neighbors: int = 12) -> float:
    """Compute trustworthiness with NumPy implementation."""
    n = X.shape[0]
    if not (1 <= n_neighbors < n // 2):
        raise ValueError("n_neighbors must satisfy 1 <= k < n_samples/2")

    d_x = pairwise_euclidean_distances(X)
    d_y = pairwise_euclidean_distances(Y)

    order_x = np.argsort(d_x, axis=1)
    order_y = np.argsort(d_y, axis=1)
    nbr_x = order_x[:, 1 : n_neighbors + 1]
    nbr_y = order_y[:, 1 : n_neighbors + 1]

    ranks_x = np.empty((n, n), dtype=int)
    for i in range(n):
        ranks_x[i, order_x[i]] = np.arange(n, dtype=int)

    penalty = 0.0
    for i in range(n):
        source_set = set(int(v) for v in nbr_x[i])
        for j in nbr_y[i]:
            jj = int(j)
            if jj not in source_set:
                penalty += float(ranks_x[i, jj] - n_neighbors)

    denom = n * n_neighbors * (2.0 * n - 3.0 * n_neighbors - 1.0)
    return float(1.0 - (2.0 / denom) * penalty)


def preview_lines(embedding: Array, roll_param: Array, n_rows: int = 8) -> List[str]:
    """Generate compact preview lines without extra dependencies."""
    lines = ["sample_id      z1        z2    swiss_roll_t"]
    for i in range(min(n_rows, embedding.shape[0])):
        lines.append(
            f"{i:8d} {embedding[i, 0]:8.4f} {embedding[i, 1]:8.4f} {roll_param[i]:12.4f}"
        )
    return lines


def main() -> None:
    rng_seed = 2026
    np.set_printoptions(precision=4, suppress=True)

    X, t = make_swiss_roll_numpy(n_samples=450, noise=0.04, random_state=rng_seed)

    result = fit_isomap(X, n_components=2, candidate_neighbors=(8, 10, 12, 15, 20))
    rv = residual_variance(result.geodesic_distances, result.embedding)
    trust = trustworthiness_score(X, result.embedding, n_neighbors=12)

    print("=== Isomap MVP (manual pipeline) ===")
    print(f"samples={X.shape[0]}, ambient_dim={X.shape[1]}, embed_dim=2")
    print(f"chosen_k={result.n_neighbors}")
    print(f"trustworthiness@12={trust:.4f}")
    print(f"residual_variance={rv:.4f} (lower is better)")
    print("top-5 kernel eigenvalues:", np.round(result.eigenvalues[:5], 6))
    print("\nEmbedding preview:")
    for line in preview_lines(result.embedding, t):
        print(line)

    # Lightweight quality guards for automated validation.
    assert trust > 0.88, f"Unexpected low trustworthiness: {trust:.4f}"
    assert rv < 0.50, f"Unexpected high residual variance: {rv:.4f}"
    assert result.eigenvalues[0] > 0 and result.eigenvalues[1] > 0, "Top eigenvalues must be positive"
    print("\nAssertions passed: Isomap embedding looks valid.")


if __name__ == "__main__":
    main()
