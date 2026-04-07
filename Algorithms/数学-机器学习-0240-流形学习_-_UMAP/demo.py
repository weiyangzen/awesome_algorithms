"""Educational UMAP-style manifold learning MVP.

This script avoids black-box UMAP wrappers and implements the key flow:
1) kNN graph + local scale,
2) fuzzy graph union,
3) low-dimensional optimization with attraction/repulsion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy import sparse as sp_sparse

    HAS_SCIPY = True
except ModuleNotFoundError:
    sp_sparse = None
    HAS_SCIPY = False

try:
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.manifold import trustworthiness as sk_trustworthiness
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ModuleNotFoundError:
    load_digits = None
    SkPCA = None
    sk_trustworthiness = None
    NearestNeighbors = None
    StandardScaler = None
    HAS_SKLEARN = False

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    HAS_TORCH = False


@dataclass
class GraphData:
    """Container for fuzzy graph edge list."""

    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    nnz: int


def standardize_data(X: np.ndarray) -> np.ndarray:
    """Standardize each feature column."""
    if HAS_SKLEARN:
        return StandardScaler().fit_transform(X)
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma


def pca_2d(X: np.ndarray, seed: int) -> np.ndarray:
    """2D PCA with sklearn when available; otherwise SVD fallback."""
    if HAS_SKLEARN:
        return SkPCA(n_components=2, random_state=seed).fit_transform(X).astype(np.float32)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    emb = np.einsum("nd,cd->nc", Xc, vt[:2], optimize=True)
    return emb.astype(np.float32)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Dense Euclidean distance matrix."""
    Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    diff = Xn[:, None, :] - Xn[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2)


def knn_query(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and indices of kNN, including self at column 0."""
    if HAS_SKLEARN:
        model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        model.fit(X)
        distances, indices = model.kneighbors(X, return_distance=True)
        return distances, indices

    D = pairwise_distances(X)
    k = n_neighbors + 1
    idx0 = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    d0 = np.take_along_axis(D, idx0, axis=1)
    order = np.argsort(d0, axis=1)
    indices = np.take_along_axis(idx0, order, axis=1)
    distances = np.take_along_axis(D, indices, axis=1)
    return distances, indices


def trustworthiness_metric(X: np.ndarray, Y: np.ndarray, n_neighbors: int) -> float:
    """Trustworthiness score; sklearn if available, else NumPy fallback."""
    if HAS_SKLEARN:
        return float(sk_trustworthiness(X, Y, n_neighbors=n_neighbors))

    n = X.shape[0]
    k = n_neighbors
    if not (1 <= k < n // 2):
        raise ValueError("n_neighbors must satisfy 1 <= k < n/2 for trustworthiness.")

    DX = pairwise_distances(X)
    DY = pairwise_distances(Y)

    rank = np.empty((n, n), dtype=np.int32)
    order_x = np.argsort(DX, axis=1)
    for i in range(n):
        rank[i, order_x[i]] = np.arange(n, dtype=np.int32)

    order_y = np.argsort(DY, axis=1)[:, 1 : k + 1]
    order_x_k = np.argsort(DX, axis=1)[:, 1 : k + 1]

    penalty = 0.0
    for i in range(n):
        high_set = set(order_x_k[i].tolist())
        for j in order_y[i]:
            if j not in high_set:
                penalty += float(rank[i, j] - k)

    normalizer = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - normalizer * penalty


def _binary_search_sigma(distances: np.ndarray, rho: float, target: float, max_iter: int = 32) -> float:
    """Solve local smooth radius sigma_i with binary search."""
    lo, hi = 1e-3, 64.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        probs = np.exp(-np.maximum(0.0, distances - rho) / mid)
        if float(np.sum(probs)) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_fuzzy_graph(X: np.ndarray, n_neighbors: int, seed: int = 42) -> GraphData:
    """Build UMAP-style fuzzy simplicial set graph."""
    n = X.shape[0]
    distances, indices = knn_query(X, n_neighbors=n_neighbors)

    # Remove self-neighbor.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    target = np.log2(n_neighbors)
    rhos = np.zeros(n, dtype=np.float64)
    sigmas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        d_i = distances[i]
        nonzero = d_i[d_i > 0.0]
        rhos[i] = float(np.min(nonzero)) if nonzero.size > 0 else 0.0
        sigmas[i] = _binary_search_sigma(d_i, rhos[i], target)

    probs = np.exp(-np.maximum(0.0, distances - rhos[:, None]) / sigmas[:, None])
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.reshape(-1)
    vals = probs.reshape(-1)

    if HAS_SCIPY:
        directed = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        union = directed + directed.T - directed.multiply(directed.T)
        union.setdiag(0.0)
        union.eliminate_zeros()
        coo = union.tocoo()
        mask = coo.row != coo.col
        row_u = coo.row[mask].astype(np.int64)
        col_u = coo.col[mask].astype(np.int64)
        w_u = coo.data[mask].astype(np.float32)
        nnz = int(union.nnz)
    else:
        directed = np.zeros((n, n), dtype=np.float32)
        directed[rows, cols] = vals.astype(np.float32)
        union = directed + directed.T - directed * directed.T
        np.fill_diagonal(union, 0.0)
        row_u, col_u = np.nonzero(union)
        w_u = union[row_u, col_u].astype(np.float32)
        nnz = int(np.count_nonzero(union))

    rng = np.random.default_rng(seed)
    order = np.arange(row_u.shape[0])
    rng.shuffle(order)
    return GraphData(rows=row_u[order], cols=col_u[order], weights=w_u[order], nnz=nnz)


def optimize_embedding(
    graph: GraphData,
    init_embedding: np.ndarray,
    epochs: int = 220,
    neg_samples: int = 4,
    lr: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """Train low-dimensional points with attraction + repulsion."""
    if HAS_TORCH:
        return _optimize_embedding_torch(
            graph=graph,
            init_embedding=init_embedding,
            epochs=epochs,
            neg_samples=neg_samples,
            lr=lr,
            seed=seed,
        )
    print("PyTorch not found; falling back to NumPy optimizer.")
    return _optimize_embedding_numpy(
        graph=graph,
        init_embedding=init_embedding,
        epochs=epochs,
        neg_samples=neg_samples,
        lr=lr,
        seed=seed,
    )


def _optimize_embedding_torch(
    graph: GraphData,
    init_embedding: np.ndarray,
    epochs: int,
    neg_samples: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """PyTorch optimizer path."""
    torch.manual_seed(seed)
    Y = torch.tensor(init_embedding, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([Y], lr=lr)

    src = torch.tensor(graph.rows, dtype=torch.long)
    dst = torch.tensor(graph.cols, dtype=torch.long)
    w = torch.tensor(graph.weights, dtype=torch.float32)
    n = init_embedding.shape[0]
    eps = 1e-6

    for epoch in range(epochs):
        optimizer.zero_grad()

        pos_diff = Y[src] - Y[dst]
        pos_dist2 = torch.sum(pos_diff * pos_diff, dim=1)
        q_pos = 1.0 / (1.0 + pos_dist2)
        loss_pos = -(w * torch.log(q_pos + eps)).mean()

        src_neg = src.repeat_interleave(neg_samples)
        dst_neg = torch.randint(0, n, (src_neg.shape[0],), dtype=torch.long)
        neg_diff = Y[src_neg] - Y[dst_neg]
        neg_dist2 = torch.sum(neg_diff * neg_diff, dim=1)
        q_neg = 1.0 / (1.0 + neg_dist2)
        loss_neg = -torch.log(1.0 - q_neg + eps).mean()

        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Y -= Y.mean(dim=0, keepdim=True)

        if epoch in {0, 49, 99, 149, 199, epochs - 1}:
            print(
                f"[epoch {epoch + 1:03d}/{epochs}] "
                f"loss={loss.item():.4f} pos={loss_pos.item():.4f} neg={loss_neg.item():.4f}"
            )

    return Y.detach().cpu().numpy()


def _optimize_embedding_numpy(
    graph: GraphData,
    init_embedding: np.ndarray,
    epochs: int,
    neg_samples: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """NumPy fallback optimizer when PyTorch is unavailable."""
    rng = np.random.default_rng(seed)
    Y = init_embedding.astype(np.float64).copy()
    src = graph.rows
    dst = graph.cols
    w = graph.weights.astype(np.float64)
    n = Y.shape[0]
    m_edges = max(1, src.shape[0])
    eps = 1e-6

    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        grad = np.zeros_like(Y)

        pos_diff = Y[src] - Y[dst]
        pos_dist2 = np.sum(pos_diff * pos_diff, axis=1) + eps
        q_pos = 1.0 / (1.0 + pos_dist2)
        loss_pos = float(np.mean(-w * np.log(q_pos + eps)))

        coeff_pos = (2.0 * w * q_pos) / float(m_edges)
        grad_pos = coeff_pos[:, None] * pos_diff
        np.add.at(grad, src, grad_pos)
        np.add.at(grad, dst, -grad_pos)

        src_neg = np.repeat(src, neg_samples)
        dst_neg = rng.integers(0, n, size=src_neg.shape[0], endpoint=False)
        neg_diff = Y[src_neg] - Y[dst_neg]
        neg_dist2 = np.sum(neg_diff * neg_diff, axis=1) + eps
        q_neg = 1.0 / (1.0 + neg_dist2)
        loss_neg = float(np.mean(-np.log(1.0 - q_neg + eps)))

        denom = np.maximum(neg_dist2 * (1.0 + neg_dist2), 1e-4)
        coeff_neg = -2.0 / denom
        coeff_neg /= float(max(1, src_neg.shape[0]))
        coeff_neg = np.clip(coeff_neg, -10.0, 10.0)
        grad_neg = coeff_neg[:, None] * neg_diff
        np.add.at(grad, src_neg, grad_neg)
        np.add.at(grad, dst_neg, -grad_neg)

        t = epoch + 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        Y -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        Y -= np.mean(Y, axis=0, keepdims=True)

        if epoch in {0, 49, 99, 149, 199, epochs - 1}:
            print(
                f"[epoch {epoch + 1:03d}/{epochs}] "
                f"loss={loss_pos + loss_neg:.4f} pos={loss_pos:.4f} neg={loss_neg:.4f}"
            )

    return Y.astype(np.float32)


def run_umap_mvp(X: np.ndarray, n_neighbors: int = 15, seed: int = 42) -> tuple[np.ndarray, GraphData]:
    """Pipeline: fuzzy graph + PCA init + optimization."""
    graph = build_fuzzy_graph(X, n_neighbors=n_neighbors, seed=seed)
    init = pca_2d(X, seed=seed)
    emb = optimize_embedding(graph, init_embedding=init, seed=seed)
    return emb, graph


def load_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load digits when sklearn exists, else generate synthetic manifold data."""
    rng = np.random.default_rng(seed)
    if HAS_SKLEARN:
        digits = load_digits()
        X = digits.data[:1200].astype(np.float64)
        y = digits.target[:1200].astype(int)
        return X, y

    n = 700
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    x = t * np.cos(t)
    y0 = 21.0 * rng.random(n)
    z = t * np.sin(t)
    noise = 0.15 * rng.normal(size=(n, 7))
    X = np.column_stack([x, y0, z, noise]).astype(np.float64)
    y = np.floor((t - t.min()) / (t.max() - t.min() + 1e-12) * 10).astype(int)
    return X, y


def main() -> None:
    seed = 42
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)

    print(
        "Backends:"
        f" sklearn={'yes' if HAS_SKLEARN else 'no'},"
        f" scipy={'yes' if HAS_SCIPY else 'no'},"
        f" torch={'yes' if HAS_TORCH else 'no'}"
    )

    X, y = load_dataset(seed=seed)
    X = standardize_data(X)

    print("Building + optimizing UMAP-style embedding...")
    t0 = time.perf_counter()
    emb_umap, graph = run_umap_mvp(X, n_neighbors=15, seed=seed)
    t1 = time.perf_counter()

    print("Running PCA baseline...")
    emb_pca = pca_2d(X, seed=seed)

    tw_umap = trustworthiness_metric(X, emb_umap, n_neighbors=15)
    tw_pca = trustworthiness_metric(X, emb_pca, n_neighbors=15)

    summary = pd.DataFrame(
        [
            {
                "method": "UMAP_MVP",
                "trustworthiness@15": round(float(tw_umap), 4),
                "runtime_sec": round(float(t1 - t0), 3),
                "graph_edges": int(graph.nnz),
            },
            {
                "method": "PCA_2D",
                "trustworthiness@15": round(float(tw_pca), 4),
                "runtime_sec": 0.0,
                "graph_edges": 0,
            },
        ]
    )
    print("\n=== Metric Summary ===")
    print(summary.to_string(index=False))

    preview = pd.DataFrame({"x": emb_umap[:, 0], "y": emb_umap[:, 1], "label": y.astype(int)}).head(10)
    print("\n=== Embedding Preview (first 10 points) ===")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
