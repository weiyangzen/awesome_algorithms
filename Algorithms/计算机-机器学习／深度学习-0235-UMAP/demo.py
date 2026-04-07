"""Minimal runnable UMAP MVP (non-black-box implementation).

This script implements a compact UMAP-style pipeline:
1) kNN graph
2) fuzzy simplicial set
3) spectral initialization
4) cross-entropy-like optimization with negative sampling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SimpleUMAPConfig:
    n_neighbors: int = 15
    n_components: int = 2
    min_dist: float = 0.1
    spread: float = 1.0
    n_epochs: int = 120
    learning_rate: float = 0.6
    negative_sample_rate: int = 3
    gamma: float = 1.0
    random_state: int = 20260407


def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit low-dimensional distance curve parameters a, b.

    UMAP uses: q(d) = 1 / (1 + a * d^(2b))
    """

    def curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * np.power(x, 2.0 * b))

    xv = np.linspace(0.0, spread * 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    params, _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0), bounds=(0.0, np.inf), maxfev=20000)
    return float(params[0]), float(params[1])


def smooth_knn_dist(
    distances: np.ndarray,
    k: int,
    n_iter: int = 64,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point smooth radius (sigma) and local offset (rho)."""
    n_samples = distances.shape[0]
    target = math.log2(k) * bandwidth
    sigmas = np.zeros(n_samples, dtype=np.float64)
    rhos = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        ith_distances = distances[i]
        non_zero = ith_distances[ith_distances > 0.0]

        if non_zero.size > 0:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                base = non_zero[min(index - 1, non_zero.size - 1)]
                if interpolation > 1e-6 and index < non_zero.size:
                    base = base + interpolation * (non_zero[index] - base)
                rhos[i] = base
            else:
                rhos[i] = interpolation * non_zero[0]

        lo = 0.0
        hi = np.inf
        mid = 1.0

        for _ in range(n_iter):
            psum = 0.0
            for d_ij in ith_distances:
                if d_ij - rhos[i] <= 0.0:
                    psum += 1.0
                else:
                    psum += math.exp(-((d_ij - rhos[i]) / mid))

            if abs(psum - target) <= 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if np.isinf(hi):
                    mid *= 2.0
                else:
                    mid = (lo + hi) / 2.0

        mean_distance = np.mean(ith_distances)
        if rhos[i] > 0.0:
            sigmas[i] = max(mid, 1e-3 * mean_distance)
        else:
            sigmas[i] = max(mid, 1e-3)

    return sigmas, rhos


def compute_fuzzy_simplicial_set(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
    n_samples: int,
) -> sp.csr_matrix:
    """Build symmetric fuzzy simplicial set graph."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(n_samples):
        for j, d_ij in zip(knn_indices[i], knn_distances[i]):
            if j < 0 or i == j:
                continue
            if d_ij - rhos[i] <= 0.0 or sigmas[i] <= 0.0:
                value = 1.0
            else:
                value = math.exp(-((d_ij - rhos[i]) / sigmas[i]))
            rows.append(i)
            cols.append(int(j))
            vals.append(value)

    graph = sp.coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)).tocsr()
    transpose = graph.transpose().tocsr()
    product = graph.multiply(transpose)
    fuzzy_graph = graph + transpose - product
    fuzzy_graph.eliminate_zeros()
    return fuzzy_graph.tocsr()


def spectral_layout(graph: sp.csr_matrix, n_components: int, random_state: int) -> np.ndarray:
    """Initialize embedding with normalized Laplacian eigenvectors."""
    n_samples = graph.shape[0]
    if n_samples <= n_components + 1:
        rng = np.random.default_rng(random_state)
        return rng.normal(scale=1e-3, size=(n_samples, n_components)).astype(np.float32)

    degree = np.asarray(graph.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    d_mat = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(n_samples, format="csr") - d_mat @ graph @ d_mat

    try:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=n_components + 1,
            which="SM",
            tol=1e-3,
            maxiter=3000,
        )
        order = np.argsort(eigenvalues)
        emb = eigenvectors[:, order[1 : n_components + 1]]
    except Exception:
        rng = np.random.default_rng(random_state)
        emb = rng.normal(scale=1e-3, size=(n_samples, n_components))

    emb = emb - emb.mean(axis=0, keepdims=True)
    emb = emb / (emb.std(axis=0, keepdims=True) + 1e-9)
    return emb.astype(np.float32)


def extract_graph_edges(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sparse graph to undirected edge list."""
    upper = sp.triu(graph, k=1).tocoo()
    mask = upper.data > 0
    return (
        upper.row[mask].astype(np.int64),
        upper.col[mask].astype(np.int64),
        upper.data[mask].astype(np.float64),
    )


def optimize_embedding_numpy(
    init_embedding: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    a: float,
    b: float,
    cfg: SimpleUMAPConfig,
) -> np.ndarray:
    """UMAP-style SGD with negative sampling in NumPy."""
    rng = np.random.default_rng(cfg.random_state)
    embedding = init_embedding.astype(np.float64, copy=True)

    probabilities = edge_w / np.sum(edge_w)
    n_edges = edge_w.shape[0]
    samples_per_epoch = min(4096, n_edges)

    for epoch in range(cfg.n_epochs):
        lr = cfg.learning_rate * (1.0 - (epoch / max(cfg.n_epochs - 1, 1)))
        sampled = rng.choice(n_edges, size=samples_per_epoch, replace=True, p=probabilities)

        for idx in sampled:
            i = int(edge_u[idx])
            j = int(edge_v[idx])
            w = float(edge_w[idx])

            diff = embedding[i] - embedding[j]
            dist2 = float(np.dot(diff, diff)) + 1e-6
            grad_coeff = -2.0 * a * b * (dist2 ** (b - 1.0)) / (a * (dist2**b) + 1.0)
            grad = np.clip(grad_coeff * diff, -4.0, 4.0)
            embedding[i] += lr * w * grad
            embedding[j] -= lr * w * grad

            for _ in range(cfg.negative_sample_rate):
                k = int(rng.integers(0, embedding.shape[0]))
                if k == i:
                    continue
                diff_neg = embedding[i] - embedding[k]
                dist2_neg = float(np.dot(diff_neg, diff_neg)) + 1e-6
                grad_coeff_neg = 2.0 * cfg.gamma * b / ((0.001 + dist2_neg) * (a * (dist2_neg**b) + 1.0))
                grad_neg = np.clip(grad_coeff_neg * diff_neg, -4.0, 4.0)
                embedding[i] += lr * w * grad_neg

        embedding -= embedding.mean(axis=0, keepdims=True)

    return embedding.astype(np.float32)


def refine_with_torch(
    embedding: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    a: float,
    b: float,
    cfg: SimpleUMAPConfig,
) -> np.ndarray:
    """Small torch refinement pass to minimize sampled cross-entropy."""
    torch.manual_seed(cfg.random_state)

    y = torch.tensor(embedding, dtype=torch.float32, requires_grad=True)
    u = torch.tensor(edge_u, dtype=torch.long)
    v = torch.tensor(edge_v, dtype=torch.long)
    w = torch.tensor(edge_w, dtype=torch.float32)
    probs = w / w.sum()

    optimizer = torch.optim.Adam([y], lr=0.03)
    n_points = y.shape[0]
    batch_size = int(min(2048, edge_u.shape[0]))

    for _ in range(60):
        batch = torch.multinomial(probs, batch_size, replacement=True)
        ui = u[batch]
        vi = v[batch]
        wi = w[batch]

        pos_diff = y[ui] - y[vi]
        pos_dist2 = torch.sum(pos_diff * pos_diff, dim=1) + 1e-6
        q_pos = 1.0 / (1.0 + a * torch.pow(pos_dist2, b))
        pos_loss = -(wi * torch.log(q_pos + 1e-8)).mean()

        neg = torch.randint(0, n_points, (batch_size,), dtype=torch.long)
        neg_diff = y[ui] - y[neg]
        neg_dist2 = torch.sum(neg_diff * neg_diff, dim=1) + 1e-6
        q_neg = 1.0 / (1.0 + a * torch.pow(neg_dist2, b))
        rep_loss = -(cfg.gamma * torch.log1p(-q_neg.clamp(max=1.0 - 1e-6))).mean()

        loss = pos_loss + rep_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y -= y.mean(dim=0, keepdim=True)

    return y.detach().cpu().numpy().astype(np.float32)


def simple_umap_fit_transform(X: np.ndarray, cfg: SimpleUMAPConfig) -> tuple[np.ndarray, sp.csr_matrix, float, float]:
    """Run compact UMAP pipeline and return low-dimensional embedding."""
    nn = NearestNeighbors(n_neighbors=cfg.n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)

    # Remove the self-neighbor at column 0.
    knn_idx = knn_idx[:, 1:]
    knn_dist = knn_dist[:, 1:]

    sigmas, rhos = smooth_knn_dist(knn_dist, k=cfg.n_neighbors)
    graph = compute_fuzzy_simplicial_set(knn_idx, knn_dist, sigmas, rhos, n_samples=X.shape[0])

    a, b = find_ab_params(spread=cfg.spread, min_dist=cfg.min_dist)
    init = spectral_layout(graph=graph, n_components=cfg.n_components, random_state=cfg.random_state)

    edge_u, edge_v, edge_w = extract_graph_edges(graph)
    emb = optimize_embedding_numpy(init, edge_u, edge_v, edge_w, a, b, cfg)
    emb = refine_with_torch(emb, edge_u, edge_v, edge_w, a, b, cfg)
    return emb, graph, a, b


def neighbor_recall(high_dim: np.ndarray, low_dim: np.ndarray, k: int = 10) -> float:
    """Mean overlap ratio between high-dim and low-dim kNN."""
    nn_hi = NearestNeighbors(n_neighbors=k + 1).fit(high_dim)
    nn_lo = NearestNeighbors(n_neighbors=k + 1).fit(low_dim)
    idx_hi = nn_hi.kneighbors(return_distance=False)[:, 1:]
    idx_lo = nn_lo.kneighbors(return_distance=False)[:, 1:]

    overlap = [len(set(a).intersection(set(b))) / k for a, b in zip(idx_hi, idx_lo)]
    return float(np.mean(overlap))


def main() -> None:
    cfg = SimpleUMAPConfig()

    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    embedding, graph, a, b = simple_umap_fit_transform(X, cfg)

    trust = trustworthiness(X, embedding, n_neighbors=10)
    recall = neighbor_recall(X, embedding, k=10)

    output_df = pd.DataFrame(
        {
            "id": np.arange(X.shape[0], dtype=np.int64),
            "label": y,
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
        }
    )

    out_dir = Path(__file__).resolve().parent
    out_csv = out_dir / "embedding.csv"
    output_df.to_csv(out_csv, index=False)

    print("=== Simple UMAP MVP ===")
    print(f"samples={X.shape[0]}, features={X.shape[1]}, neighbors={cfg.n_neighbors}")
    print(f"fuzzy_edges={graph.nnz}, undirected_edges={extract_graph_edges(graph)[0].shape[0]}")
    print(f"curve_params: a={a:.6f}, b={b:.6f}")
    print(f"trustworthiness@10={trust:.4f}")
    print(f"neighbor_recall@10={recall:.4f}")
    print(f"saved_embedding={out_csv}")
    print("preview:")
    print(output_df.head(8).to_string(index=False))

    # Keep thresholds moderate because this is a compact MVP, not a full production UMAP.
    assert trust > 0.80, f"trustworthiness too low: {trust:.4f}"
    assert recall > 0.25, f"neighbor recall too low: {recall:.4f}"


if __name__ == "__main__":
    main()
