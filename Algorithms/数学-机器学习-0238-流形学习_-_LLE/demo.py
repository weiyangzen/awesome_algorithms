"""Locally Linear Embedding (LLE) minimal runnable MVP.

This script intentionally implements the core LLE algorithm in source code
instead of calling a single black-box API:
1) k-nearest-neighbor graph,
2) local linear reconstruction weights,
3) global spectral embedding from (I - W)^T(I - W).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LLEConfig:
    """Hyperparameters for the manual LLE solver."""

    n_neighbors: int = 12
    n_components: int = 2
    reg: float = 1e-3
    random_state: int = 42


@dataclass
class LLEResult:
    """Container for LLE outputs."""

    embedding: np.ndarray
    neighbor_indices: np.ndarray
    local_weights: np.ndarray
    eigenvalues: np.ndarray
    reconstruction_error: float


def build_dataset(n_samples: int = 1200, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a noisy high-dimensional swiss-roll dataset.

    Returns:
        X: standardized swiss-roll features, shape (n_samples, 3)
        t: original swiss-roll intrinsic coordinate
        labels: coarse bins from t for readable preview only
    """
    X3, t = make_swiss_roll(n_samples=n_samples, noise=0.04, random_state=random_state)
    X = StandardScaler().fit_transform(X3)

    # Use 6 bins only for quick textual inspection in output.
    bin_edges = np.quantile(t, np.linspace(0.0, 1.0, num=7))
    bin_edges[0] -= 1e-6
    labels = np.digitize(t, bins=bin_edges[1:-1], right=False)
    return X.astype(np.float64), t.astype(np.float64), labels.astype(np.int64)


def knn_graph(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return kNN distances/indices excluding self-neighbor."""
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    model.fit(X)
    distances, indices = model.kneighbors(X, return_distance=True)
    return distances[:, 1:], indices[:, 1:]


def solve_local_weights(X: np.ndarray, neighbor_indices: np.ndarray, reg: float) -> np.ndarray:
    """Solve constrained least-squares reconstruction weights for each sample.

    For each i, solve w_i:
        min ||x_i - sum_j w_ij x_{ij}||^2,
        s.t. sum_j w_ij = 1.
    """
    n_samples, n_features = X.shape
    del n_features
    k = neighbor_indices.shape[1]

    weights = np.zeros((n_samples, k), dtype=np.float64)
    ones = np.ones(k, dtype=np.float64)

    for i in range(n_samples):
        nbr_idx = neighbor_indices[i]
        Z = X[nbr_idx] - X[i]  # Shape: (k, d)
        C = Z @ Z.T  # Local Gram matrix (k, k)

        trace = float(np.trace(C))
        ridge = reg * (trace if trace > 1e-12 else 1.0)
        C = C + np.eye(k, dtype=np.float64) * ridge

        try:
            w = np.linalg.solve(C, ones)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(C, ones, rcond=None)

        w_sum = float(np.sum(w))
        if abs(w_sum) < 1e-12:
            w = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            w = w / w_sum
        weights[i] = w

    return weights


def build_weight_matrix(neighbor_indices: np.ndarray, weights: np.ndarray, n_samples: int) -> sparse.csr_matrix:
    """Construct sparse global weight matrix W."""
    k = neighbor_indices.shape[1]
    rows = np.repeat(np.arange(n_samples), k)
    cols = neighbor_indices.reshape(-1)
    vals = weights.reshape(-1)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float64)
    return W


def solve_global_embedding(W: sparse.csr_matrix, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute embedding from the bottom nontrivial eigenvectors of M=(I-W)^T(I-W)."""
    n_samples = W.shape[0]
    I = sparse.identity(n_samples, format="csr", dtype=np.float64)
    M = (I - W).T @ (I - W)
    M = (M + M.T) * 0.5  # Numerical symmetry stabilization.

    n_eigs = n_components + 1
    if n_eigs >= n_samples:
        raise ValueError("n_components is too large for current sample size.")

    try:
        evals, evecs = eigsh(M, k=n_eigs, which="SM", tol=1e-8, maxiter=20_000)
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is not None and exc.eigenvectors is not None and len(exc.eigenvalues) >= n_eigs:
            evals = exc.eigenvalues
            evecs = exc.eigenvectors
        else:
            dense = M.toarray()
            evals, evecs = np.linalg.eigh(dense)
            order = np.argsort(evals)
            evals = evals[order][:n_eigs]
            evecs = evecs[:, order][:, :n_eigs]

    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)

    embedding = evecs[:, 1 : n_components + 1]
    return embedding, evals


def local_reconstruction_error(
    X: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Average local reconstruction MSE in original feature space."""
    recon = np.einsum("nk,nkd->nd", weights, X[neighbor_indices], optimize=True)
    mse = np.mean(np.sum((X - recon) ** 2, axis=1))
    return float(mse)


def neighbor_overlap_score(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int) -> float:
    """Mean kNN overlap ratio between high and low spaces."""
    _, idx_h = knn_graph(X_high, n_neighbors=n_neighbors)
    _, idx_l = knn_graph(X_low, n_neighbors=n_neighbors)

    overlap = 0.0
    for i in range(X_high.shape[0]):
        overlap += len(set(idx_h[i].tolist()) & set(idx_l[i].tolist())) / n_neighbors
    return float(overlap / X_high.shape[0])


def run_manual_lle(X: np.ndarray, cfg: LLEConfig) -> LLEResult:
    """End-to-end manual LLE pipeline."""
    _, neighbor_indices = knn_graph(X, n_neighbors=cfg.n_neighbors)
    weights = solve_local_weights(X, neighbor_indices=neighbor_indices, reg=cfg.reg)
    W = build_weight_matrix(neighbor_indices=neighbor_indices, weights=weights, n_samples=X.shape[0])
    embedding, evals = solve_global_embedding(W=W, n_components=cfg.n_components)
    recon_err = local_reconstruction_error(X=X, neighbor_indices=neighbor_indices, weights=weights)
    return LLEResult(
        embedding=embedding,
        neighbor_indices=neighbor_indices,
        local_weights=weights,
        eigenvalues=evals,
        reconstruction_error=recon_err,
    )


def evaluate_embeddings(
    X_high: np.ndarray,
    emb_manual: np.ndarray,
    emb_sklearn: np.ndarray,
    emb_pca: np.ndarray,
    n_neighbors: int,
) -> pd.DataFrame:
    """Build a compact metrics table for multiple embeddings."""
    rows = []
    for name, emb in (
        ("Manual_LLE", emb_manual),
        ("Sklearn_LLE", emb_sklearn),
        ("PCA_2D", emb_pca),
    ):
        rows.append(
            {
                "method": name,
                "trustworthiness": trustworthiness(X_high, emb, n_neighbors=n_neighbors),
                "neighbor_overlap": neighbor_overlap_score(X_high, emb, n_neighbors=n_neighbors),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()

    cfg = LLEConfig(n_neighbors=10, n_components=2, reg=1e-3, random_state=42)
    X, t, labels = build_dataset(n_samples=1000, random_state=cfg.random_state)

    t0 = time.perf_counter()
    manual = run_manual_lle(X=X, cfg=cfg)
    t1 = time.perf_counter()

    sk_lle = LocallyLinearEmbedding(
        n_neighbors=cfg.n_neighbors,
        n_components=cfg.n_components,
        reg=cfg.reg,
        eigen_solver="arpack",
        method="standard",
        random_state=cfg.random_state,
    )
    emb_sklearn = sk_lle.fit_transform(X)

    emb_pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state).fit_transform(X)

    metrics = evaluate_embeddings(
        X_high=X,
        emb_manual=manual.embedding,
        emb_sklearn=emb_sklearn,
        emb_pca=emb_pca,
        n_neighbors=cfg.n_neighbors,
    )

    elapsed = time.perf_counter() - start

    print("== LLE MVP: Manual vs Baselines ==")
    print(f"Data shape: {X.shape}, intrinsic target bins: {np.unique(labels).size}")
    print(
        "Manual LLE config: "
        f"n_neighbors={cfg.n_neighbors}, n_components={cfg.n_components}, reg={cfg.reg:.1e}"
    )
    print(f"Manual LLE runtime: {t1 - t0:.3f}s")
    print(f"Total runtime: {elapsed:.3f}s")
    print(f"Manual local reconstruction MSE: {manual.reconstruction_error:.6f}")
    print(f"Manual smallest eigenvalues: {np.round(manual.eigenvalues[:5], 8).tolist()}")
    print("\nMetrics:")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    preview = pd.DataFrame(
        {
            "t": t[:10],
            "label": labels[:10],
            "y1_manual": manual.embedding[:10, 0],
            "y2_manual": manual.embedding[:10, 1],
        }
    )
    print("\nManual embedding preview (first 10 rows):")
    print(preview.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    manual_trust = float(metrics.loc[metrics["method"] == "Manual_LLE", "trustworthiness"].iloc[0])
    sklearn_trust = float(metrics.loc[metrics["method"] == "Sklearn_LLE", "trustworthiness"].iloc[0])
    pca_trust = float(metrics.loc[metrics["method"] == "PCA_2D", "trustworthiness"].iloc[0])

    # Lightweight quality gates: enforce finite output and non-trivial manifold quality.
    assert manual.embedding.shape == (X.shape[0], cfg.n_components)
    assert np.isfinite(manual.embedding).all()
    assert np.isfinite(manual.eigenvalues).all()
    assert manual.reconstruction_error < 4.0
    assert manual_trust >= 0.90
    assert manual_trust >= pca_trust + 0.08
    assert manual_trust >= sklearn_trust - 0.08

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
