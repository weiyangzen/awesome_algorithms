"""Educational t-SNE MVP with transparent algorithm flow.

Key properties:
- no interactive input
- deterministic via fixed seed
- no direct black-box call to sklearn.manifold.TSNE
- explicit high-dimensional affinity construction + KL optimization
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import pdist, squareform

    HAS_SCIPY = True
except ModuleNotFoundError:
    pdist = None
    squareform = None
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


Array = np.ndarray


@dataclass
class TSNEResult:
    embedding: Array
    p_matrix: Array
    losses: list[float]
    optimizer: str


def pairwise_squared_distances(X: Array) -> Array:
    """Squared Euclidean distance matrix."""
    Xf = np.asarray(X, dtype=np.float64)
    if HAS_SCIPY:
        return squareform(pdist(Xf, metric="sqeuclidean"))

    diff = Xf[:, None, :] - Xf[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def pca_init(X: Array, n_components: int, seed: int) -> Array:
    """PCA initialization, scaled to small magnitude for stable t-SNE start."""
    if HAS_SKLEARN:
        Y0 = SkPCA(n_components=n_components, random_state=seed).fit_transform(X)
    else:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        Y0 = np.einsum("nd,cd->nc", Xc, vt[:n_components], optimize=True)

    Y0 = Y0.astype(np.float64)
    scale = np.std(Y0[:, 0])
    if scale < 1e-12:
        scale = 1.0
    return (Y0 / scale) * 1e-4


def make_swiss_roll_numpy(
    n_samples: int = 600,
    noise: float = 0.04,
    random_state: int = 2026,
) -> tuple[Array, Array]:
    """Generate a Swiss-roll-like 3D manifold with NumPy only."""
    rng = np.random.default_rng(random_state)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_samples))
    x = t * np.cos(t)
    y = 21.0 * rng.random(n_samples)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    if noise > 0.0:
        X = X + noise * rng.standard_normal(size=X.shape)
    return X.astype(np.float64), t.astype(np.float64)


def standardize_numpy(X: Array) -> Array:
    """Feature standardization with NumPy fallback."""
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma


def _hbeta(dist_row: Array, beta: float) -> tuple[float, Array]:
    """Return entropy H and normalized probs for one row under precision beta."""
    probs = np.exp(-dist_row * beta)
    sum_probs = float(np.sum(probs))
    if sum_probs < 1e-12:
        probs = np.full_like(dist_row, 1.0 / dist_row.size)
        return float(math.log(dist_row.size)), probs

    probs = probs / sum_probs
    entropy = math.log(sum_probs) + beta * float(np.sum(dist_row * np.exp(-dist_row * beta))) / sum_probs
    return entropy, probs


def conditional_probabilities(
    distance_sq: Array,
    perplexity: float,
    tolerance: float = 1e-5,
    max_iter: int = 60,
) -> Array:
    """Compute conditional probabilities P(j|i) with binary search on beta."""
    n = distance_sq.shape[0]
    if perplexity <= 1.0:
        raise ValueError("perplexity must be > 1")
    if n < 3:
        raise ValueError("Need at least 3 samples")

    log_u = math.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dist_i = distance_sq[i, mask]

        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf

        entropy, this_p = _hbeta(dist_i, beta)
        entropy_diff = entropy - log_u

        for _ in range(max_iter):
            if abs(entropy_diff) <= tolerance:
                break
            if entropy_diff > 0.0:
                beta_min = beta
                beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

            entropy, this_p = _hbeta(dist_i, beta)
            entropy_diff = entropy - log_u

        P[i, mask] = this_p

    return P


def joint_probabilities(P_cond: Array) -> Array:
    """Symmetrize conditional probabilities into joint P matrix."""
    n = P_cond.shape[0]
    P = (P_cond + P_cond.T) / (2.0 * n)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, 1e-12)
    P = P / np.sum(P)
    return P


def _tsne_step_numpy(Y: Array, P_used: Array, learning_rate: float, momentum: float, velocity: Array) -> tuple[Array, Array, float]:
    """One optimization step with NumPy backend."""
    eps = 1e-12
    diff = Y[:, None, :] - Y[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.maximum(dist2, 0.0, out=dist2)

    num = 1.0 / (1.0 + dist2)
    np.fill_diagonal(num, 0.0)

    q = num / np.sum(num)
    q = np.maximum(q, eps)

    pq = (P_used - q) * num
    row_sum = np.sum(pq, axis=1, keepdims=True)
    grad = 4.0 * (row_sum * Y - np.einsum("ij,jd->id", pq, Y, optimize=True))

    velocity = momentum * velocity - learning_rate * grad
    Y = Y + velocity
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    kl = float(np.sum(P_used * np.log((P_used + eps) / q)))
    return Y, velocity, kl


def optimize_embedding_numpy(
    Y_init: Array,
    P: Array,
    n_iter: int,
    learning_rate: float,
    early_exaggeration: float,
    exaggeration_iters: int,
    momentum_schedule: tuple[float, float],
) -> tuple[Array, list[float]]:
    """Optimize t-SNE embedding using NumPy updates."""
    Y = np.array(Y_init, dtype=np.float64, copy=True)
    velocity = np.zeros_like(Y)
    losses: list[float] = []

    for it in range(n_iter):
        momentum = momentum_schedule[0] if it < 120 else momentum_schedule[1]
        P_used = P * early_exaggeration if it < exaggeration_iters else P
        Y, velocity, kl = _tsne_step_numpy(
            Y=Y,
            P_used=P_used,
            learning_rate=learning_rate,
            momentum=momentum,
            velocity=velocity,
        )
        np.clip(Y, -250.0, 250.0, out=Y)
        np.clip(velocity, -50.0, 50.0, out=velocity)
        if it in {0, 49, 99, 149, 199, n_iter - 1}:
            losses.append(kl)
            print(f"[numpy iter {it + 1:03d}/{n_iter}] KL={kl:.6f}")

    return Y, losses


def optimize_embedding_torch(
    Y_init: Array,
    P: Array,
    n_iter: int,
    learning_rate: float,
    early_exaggeration: float,
    exaggeration_iters: int,
    momentum_schedule: tuple[float, float],
) -> tuple[Array, list[float]]:
    """Optimize t-SNE embedding using PyTorch tensor math (manual gradients)."""
    Y = torch.tensor(Y_init, dtype=torch.float64)
    velocity = torch.zeros_like(Y)
    P_t = torch.tensor(P, dtype=torch.float64)
    losses: list[float] = []

    for it in range(n_iter):
        momentum = momentum_schedule[0] if it < 120 else momentum_schedule[1]
        P_used = P_t * early_exaggeration if it < exaggeration_iters else P_t

        sum_y = torch.sum(Y * Y, dim=1, keepdim=True)
        dist2 = sum_y - 2.0 * (Y @ Y.T) + sum_y.T
        dist2 = torch.clamp(dist2, min=0.0)

        num = 1.0 / (1.0 + dist2)
        num.fill_diagonal_(0.0)

        q = num / torch.sum(num)
        q = torch.clamp(q, min=1e-12)

        pq = (P_used - q) * num
        row_sum = torch.sum(pq, dim=1, keepdim=True)
        grad = 4.0 * (row_sum * Y - pq @ Y)

        velocity = momentum * velocity - learning_rate * grad
        Y = Y + velocity
        Y = Y - torch.mean(Y, dim=0, keepdim=True)
        Y = torch.clamp(Y, min=-250.0, max=250.0)
        velocity = torch.clamp(velocity, min=-50.0, max=50.0)

        if it in {0, 49, 99, 149, 199, n_iter - 1}:
            kl = float(torch.sum(P_used * torch.log((P_used + 1e-12) / q)).item())
            losses.append(kl)
            print(f"[torch iter {it + 1:03d}/{n_iter}] KL={kl:.6f}")

    return Y.detach().cpu().numpy(), losses


def trustworthiness_metric(X: Array, Y: Array, n_neighbors: int = 12) -> float:
    """Compute trustworthiness; sklearn path first, NumPy fallback second."""
    if HAS_SKLEARN:
        return float(sk_trustworthiness(X, Y, n_neighbors=n_neighbors))

    n = X.shape[0]
    k = n_neighbors
    if not (1 <= k < n // 2):
        raise ValueError("n_neighbors must satisfy 1 <= k < n/2")

    d_x = pairwise_squared_distances(X)
    d_y = pairwise_squared_distances(Y)
    order_x = np.argsort(d_x, axis=1)
    order_y = np.argsort(d_y, axis=1)

    ranks = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        ranks[i, order_x[i]] = np.arange(n, dtype=np.int32)

    penalty = 0.0
    for i in range(n):
        nn_x = set(order_x[i, 1 : k + 1].tolist())
        for j in order_y[i, 1 : k + 1]:
            if int(j) not in nn_x:
                penalty += float(ranks[i, j] - k)

    normalizer = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return float(1.0 - normalizer * penalty)


def knn_preservation(X: Array, Y: Array, k: int = 10) -> float:
    """Average Jaccard overlap of kNN sets between high/low space."""
    n = X.shape[0]
    if HAS_SKLEARN:
        nx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        ny = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    else:
        dx = pairwise_squared_distances(X)
        dy = pairwise_squared_distances(Y)
        nx = np.argsort(dx, axis=1)[:, 1 : k + 1]
        ny = np.argsort(dy, axis=1)[:, 1 : k + 1]

    score = 0.0
    for i in range(n):
        sx = set(int(v) for v in nx[i])
        sy = set(int(v) for v in ny[i])
        inter = len(sx.intersection(sy))
        union = len(sx.union(sy))
        score += inter / union
    return float(score / n)


def run_tsne_mvp(
    X: Array,
    perplexity: float = 30.0,
    n_components: int = 2,
    n_iter: int = 320,
    learning_rate: float = 70.0,
    early_exaggeration: float = 4.0,
    exaggeration_iters: int = 90,
    random_state: int = 2026,
) -> TSNEResult:
    """End-to-end t-SNE pipeline with explicit internals."""
    np.random.seed(random_state)

    d2 = pairwise_squared_distances(X)
    p_cond = conditional_probabilities(d2, perplexity=perplexity)
    p_joint = joint_probabilities(p_cond)

    y0 = pca_init(X, n_components=n_components, seed=random_state)
    momentum_schedule = (0.5, 0.8)

    if HAS_TORCH:
        emb, losses = optimize_embedding_torch(
            Y_init=y0,
            P=p_joint,
            n_iter=n_iter,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            exaggeration_iters=exaggeration_iters,
            momentum_schedule=momentum_schedule,
        )
        optimizer = "torch"
    else:
        emb, losses = optimize_embedding_numpy(
            Y_init=y0,
            P=p_joint,
            n_iter=n_iter,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            exaggeration_iters=exaggeration_iters,
            momentum_schedule=momentum_schedule,
        )
        optimizer = "numpy"

    return TSNEResult(embedding=emb, p_matrix=p_joint, losses=losses, optimizer=optimizer)


def load_demo_data(n_samples: int = 600) -> tuple[Array, Array]:
    """Load and standardize a compact digits subset for fast deterministic demo."""
    if not HAS_SKLEARN:
        rng = np.random.default_rng(2026)
        X3, t = make_swiss_roll_numpy(n_samples=n_samples, noise=0.05, random_state=2026)
        proj = rng.normal(size=(3, 32))
        nonlinear = np.concatenate([np.sin(X3), np.cos(X3)], axis=1)
        X = np.einsum("nd,df->nf", X3, proj, optimize=True)
        X = X + 0.6 * np.einsum("nd,df->nf", nonlinear, rng.normal(size=(6, 32)), optimize=True)
        X = standardize_numpy(X)
        bins = np.quantile(t, [0.2, 0.4, 0.6, 0.8])
        y = np.digitize(t, bins=bins)
        return X.astype(np.float64), y.astype(np.int64)

    digits = load_digits()
    X = digits.data[:n_samples].astype(np.float64)
    y = digits.target[:n_samples].astype(np.int64)
    X = StandardScaler().fit_transform(X)
    return X, y


def preview_table(Y: Array, y: Array, n_rows: int = 12) -> pd.DataFrame:
    """Construct a readable preview table."""
    rows = min(n_rows, Y.shape[0])
    df = pd.DataFrame(
        {
            "sample_id": np.arange(rows, dtype=int),
            "label": y[:rows],
            "z1": np.round(Y[:rows, 0], 5),
            "z2": np.round(Y[:rows, 1], 5),
        }
    )
    return df


def main() -> None:
    start = time.perf_counter()
    np.set_printoptions(precision=5, suppress=True)

    X, y = load_demo_data(n_samples=600)
    baseline_pca = pca_init(X, n_components=2, seed=2026)

    result = run_tsne_mvp(
        X,
        perplexity=30.0,
        n_components=2,
        n_iter=320,
        learning_rate=70.0,
        early_exaggeration=4.0,
        exaggeration_iters=90,
        random_state=2026,
    )

    trust_tsne = trustworthiness_metric(X, result.embedding, n_neighbors=12)
    trust_pca = trustworthiness_metric(X, baseline_pca, n_neighbors=12)

    knn_jacc_tsne = knn_preservation(X, result.embedding, k=10)
    knn_jacc_pca = knn_preservation(X, baseline_pca, k=10)

    metrics = pd.DataFrame(
        [
            {
                "method": "TSNE_MVP",
                "optimizer": result.optimizer,
                "trustworthiness@12": round(trust_tsne, 6),
                "knn_jaccard@10": round(knn_jacc_tsne, 6),
                "checkpoints": len(result.losses),
                "last_KL": round(result.losses[-1], 6),
            },
            {
                "method": "PCA_2D",
                "optimizer": "closed_form",
                "trustworthiness@12": round(trust_pca, 6),
                "knn_jaccard@10": round(knn_jacc_pca, 6),
                "checkpoints": 0,
                "last_KL": np.nan,
            },
        ]
    )

    print("=== t-SNE MVP (explicit pipeline, non-black-box) ===")
    print(f"samples={X.shape[0]}, features={X.shape[1]}, embed_dim=2")
    print("KL checkpoints:", [round(v, 6) for v in result.losses])
    print("\nMetrics:")
    print(metrics.to_string(index=False))

    print("\nEmbedding preview:")
    print(preview_table(result.embedding, y, n_rows=12).to_string(index=False))

    elapsed = time.perf_counter() - start
    print(f"\nElapsed: {elapsed:.2f}s")

    # Lightweight quality guards for automated validation.
    assert np.all(np.isfinite(result.embedding)), "Embedding contains non-finite values"
    assert result.losses[-1] < result.losses[0], "KL did not decrease from the first checkpoint"
    if HAS_SKLEARN:
        assert trust_tsne > 0.85, f"Unexpected low trustworthiness: {trust_tsne:.4f}"
        assert trust_tsne > trust_pca, "t-SNE trustworthiness should beat PCA on this dataset"
    else:
        assert trust_tsne > 0.70, f"Unexpected low trustworthiness (fallback mode): {trust_tsne:.4f}"
    print("Assertions passed: t-SNE embedding looks valid.")


if __name__ == "__main__":
    main()
