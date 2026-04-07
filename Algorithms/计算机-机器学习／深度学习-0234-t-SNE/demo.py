"""t-SNE MVP: exact implementation from scratch + sklearn cross-check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler


@dataclass
class IterationRecord:
    iteration: int
    kl: float
    grad_norm: float
    momentum: float


@dataclass
class TSNEConfig:
    perplexity: float = 30.0
    learning_rate: float = 80.0
    max_iter: int = 750
    early_exaggeration: float = 12.0
    early_exaggeration_iters: int = 250
    initial_momentum: float = 0.5
    final_momentum: float = 0.8
    momentum_switch_iter: int = 250
    min_grad_norm: float = 1e-7
    random_state: int = 42


@dataclass
class TSNEResult:
    embedding: np.ndarray
    history: list[IterationRecord]
    final_kl: float
    affinities: np.ndarray


def validate_config(cfg: TSNEConfig, n_samples: int) -> None:
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 samples")
    if cfg.perplexity <= 1.0:
        raise ValueError("perplexity must be > 1")
    if cfg.perplexity >= n_samples - 1:
        raise ValueError("perplexity must be < n_samples - 1")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if cfg.max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if cfg.early_exaggeration <= 0:
        raise ValueError("early_exaggeration must be > 0")
    if cfg.early_exaggeration_iters < 0:
        raise ValueError("early_exaggeration_iters must be >= 0")
    if cfg.early_exaggeration_iters > cfg.max_iter:
        raise ValueError("early_exaggeration_iters must be <= max_iter")
    if cfg.min_grad_norm <= 0:
        raise ValueError("min_grad_norm must be > 0")


def _hbeta(dist_row_sq: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    weights = np.exp(-dist_row_sq * beta)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 1e-300:
        weights = np.full_like(dist_row_sq, 1.0 / max(len(dist_row_sq), 1))
        entropy = np.log(len(dist_row_sq) + 1e-12)
        return float(entropy), weights

    probs = weights / weights_sum
    entropy = np.log(weights_sum) + beta * float(np.sum(dist_row_sq * weights) / weights_sum)
    return float(entropy), probs


def _binary_search_conditional_probs(
    dist_row_sq: np.ndarray,
    target_log_perplexity: float,
    max_steps: int = 60,
    tol: float = 1e-5,
) -> np.ndarray:
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf

    probs = np.full_like(dist_row_sq, 1.0 / max(len(dist_row_sq), 1), dtype=np.float64)

    for _ in range(max_steps):
        entropy, probs = _hbeta(dist_row_sq, beta)
        entropy_diff = entropy - target_log_perplexity

        if abs(entropy_diff) < tol:
            break

        if entropy_diff > 0:
            beta_min = beta
            if np.isinf(beta_max):
                beta *= 2.0
            else:
                beta = 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            if np.isinf(beta_min):
                beta *= 0.5
            else:
                beta = 0.5 * (beta + beta_min)

    probs = np.maximum(probs, 1e-300)
    probs /= np.sum(probs)
    return probs


def compute_joint_probabilities(X: np.ndarray, perplexity: float) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    n_samples = X.shape[0]
    if perplexity >= n_samples - 1:
        raise ValueError("perplexity must be smaller than n_samples - 1")

    dist_sq = squareform(pdist(X, metric="sqeuclidean"))
    target_log_perp = np.log(perplexity)

    conditional = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        mask = np.arange(n_samples) != i
        row_probs = _binary_search_conditional_probs(dist_sq[i, mask], target_log_perp)
        conditional[i, mask] = row_probs

    joint = (conditional + conditional.T) / (2.0 * n_samples)
    np.fill_diagonal(joint, 0.0)
    joint /= np.sum(joint)
    return joint


def _pairwise_squared_distances(Y: np.ndarray) -> np.ndarray:
    y_norm = np.sum(Y * Y, axis=1, keepdims=True)
    dist_sq = y_norm + y_norm.T - 2.0 * (Y @ Y.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    np.fill_diagonal(dist_sq, 0.0)
    return dist_sq


def kl_and_gradient(P: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    dist_sq = _pairwise_squared_distances(Y)
    num = 1.0 / (1.0 + dist_sq)
    np.fill_diagonal(num, 0.0)

    q_denom = float(np.sum(num))
    if q_denom <= 1e-300:
        raise ValueError("Low-dimensional affinity denominator underflow")

    Q = num / q_denom
    np.fill_diagonal(Q, 0.0)

    eps = 1e-12
    kl = float(np.sum(P * np.log((P + eps) / (Q + eps))))

    b = (P - Q) * num
    row_sum = np.sum(b, axis=1)
    grad = 4.0 * (row_sum[:, None] * Y - b @ Y)

    return kl, grad, Q


def pca_init(X: np.ndarray, random_state: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    Y = pca.fit_transform(X)
    Y = Y / max(np.std(Y), 1e-12)
    return 1e-4 * Y


def tsne_exact(X: np.ndarray, cfg: TSNEConfig) -> TSNEResult:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN/Inf")

    n_samples = X.shape[0]
    validate_config(cfg, n_samples)

    P = compute_joint_probabilities(X, perplexity=cfg.perplexity)

    Y = pca_init(X, random_state=cfg.random_state)
    updates = np.zeros_like(Y)
    gains = np.ones_like(Y)

    history: list[IterationRecord] = []

    for it in range(1, cfg.max_iter + 1):
        momentum = cfg.initial_momentum if it <= cfg.momentum_switch_iter else cfg.final_momentum
        exaggeration = cfg.early_exaggeration if it <= cfg.early_exaggeration_iters else 1.0

        kl_ex, grad, _ = kl_and_gradient(P * exaggeration, Y)

        sign_changed = np.sign(grad) != np.sign(updates)
        gains = np.where(sign_changed, gains + 0.2, gains * 0.8)
        gains = np.clip(gains, 0.01, 10.0)

        adjusted_grad = gains * grad
        updates = momentum * updates - cfg.learning_rate * adjusted_grad
        Y += updates
        Y -= np.mean(Y, axis=0, keepdims=True)

        kl_true, grad_true, _ = kl_and_gradient(P, Y)
        grad_norm = float(np.linalg.norm(grad_true))
        history.append(
            IterationRecord(
                iteration=it,
                kl=kl_true,
                grad_norm=grad_norm,
                momentum=momentum,
            )
        )

        if grad_norm < cfg.min_grad_norm:
            break

        # Keep an explicit reference to the exaggerated objective for debug parity.
        _ = kl_ex

    final_kl = history[-1].kl
    return TSNEResult(
        embedding=Y,
        history=history,
        final_kl=final_kl,
        affinities=P,
    )


def history_snapshot(history: list[IterationRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "iter": [h.iteration for h in history],
            "kl": [h.kl for h in history],
            "grad_norm": [h.grad_norm for h in history],
            "momentum": [h.momentum for h in history],
        }
    )
    if len(frame) <= 10:
        return frame

    idx = sorted(
        {
            0,
            1,
            2,
            9,
            len(frame) // 2,
            len(frame) // 2 + 1,
            len(frame) - 3,
            len(frame) - 2,
            len(frame) - 1,
        }
    )
    return frame.iloc[idx].reset_index(drop=True)


def knn_overlap(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 10) -> float:
    if emb_a.shape[0] != emb_b.shape[0]:
        raise ValueError("emb_a and emb_b must have the same number of samples")
    if k <= 0 or k >= emb_a.shape[0]:
        raise ValueError("k must be in [1, n_samples-1]")

    def knn_indices(emb: np.ndarray) -> np.ndarray:
        dist_sq = _pairwise_squared_distances(emb)
        order = np.argsort(dist_sq, axis=1)
        return order[:, 1 : k + 1]

    nn_a = knn_indices(emb_a)
    nn_b = knn_indices(emb_b)

    overlaps = []
    for i in range(emb_a.shape[0]):
        overlaps.append(len(set(nn_a[i]) & set(nn_b[i])) / k)
    return float(np.mean(overlaps))


def run_quality_checks(
    custom_result: TSNEResult,
    tw_custom: float,
    tw_sklearn: float,
    overlap_10nn: float,
) -> None:
    if len(custom_result.history) < 50:
        raise AssertionError("History too short; optimization likely failed early")

    kl_values = np.array([h.kl for h in custom_result.history], dtype=np.float64)
    if not np.all(np.isfinite(kl_values)):
        raise AssertionError("Found non-finite KL values")
    if kl_values[-1] >= kl_values[0]:
        raise AssertionError("KL did not improve over training")

    if tw_custom < 0.90:
        raise AssertionError(f"Custom trustworthiness too low: {tw_custom:.4f}")
    if tw_sklearn < 0.90:
        raise AssertionError(f"Sklearn trustworthiness too low: {tw_sklearn:.4f}")
    if abs(tw_custom - tw_sklearn) > 0.08:
        raise AssertionError(
            f"Trustworthiness gap too large: {abs(tw_custom - tw_sklearn):.4f}"
        )
    if overlap_10nn < 0.30:
        raise AssertionError(f"10-NN overlap too low: {overlap_10nn:.4f}")


def main() -> None:
    rng = np.random.default_rng(2026)

    digits = load_digits()
    X_full = digits.data.astype(np.float64)
    y_full = digits.target.astype(np.int64)

    sample_size = 420
    subset_idx = rng.choice(X_full.shape[0], size=sample_size, replace=False)
    X = X_full[subset_idx]
    y = y_full[subset_idx]

    X = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=30, random_state=0).fit_transform(X)

    cfg = TSNEConfig(
        perplexity=30.0,
        learning_rate=80.0,
        max_iter=750,
        early_exaggeration=12.0,
        early_exaggeration_iters=250,
        initial_momentum=0.5,
        final_momentum=0.8,
        momentum_switch_iter=250,
        min_grad_norm=1e-7,
        random_state=42,
    )

    custom = tsne_exact(X_pca, cfg)

    sklearn_tsne = TSNE(
        n_components=2,
        perplexity=cfg.perplexity,
        early_exaggeration=cfg.early_exaggeration,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        init="pca",
        method="exact",
        random_state=cfg.random_state,
        verbose=0,
    )
    emb_sklearn = sklearn_tsne.fit_transform(X_pca)

    tw_custom = trustworthiness(X_pca, custom.embedding, n_neighbors=10)
    tw_sklearn = trustworthiness(X_pca, emb_sklearn, n_neighbors=10)

    overlap_10nn = knn_overlap(custom.embedding, emb_sklearn, k=10)

    label_frame = pd.DataFrame(
        {
            "label": y,
            "custom_x": custom.embedding[:, 0],
            "custom_y": custom.embedding[:, 1],
            "sk_x": emb_sklearn[:, 0],
            "sk_y": emb_sklearn[:, 1],
        }
    )
    centroid_table = (
        label_frame.groupby("label")[["custom_x", "custom_y", "sk_x", "sk_y"]]
        .mean()
        .reset_index()
        .sort_values("label")
    )

    summary = pd.DataFrame(
        [
            {
                "model": "custom_exact",
                "final_kl": custom.final_kl,
                "trustworthiness@10": tw_custom,
            },
            {
                "model": "sklearn_exact",
                "final_kl": float("nan"),
                "trustworthiness@10": tw_sklearn,
            },
        ]
    )

    print("=== t-SNE (Exact) MVP ===")
    print(
        f"samples={X_pca.shape[0]}, pca_features={X_pca.shape[1]}, classes={len(np.unique(y))}"
    )
    print(
        f"perplexity={cfg.perplexity:.1f}, learning_rate={cfg.learning_rate:.1f}, "
        f"max_iter={cfg.max_iter}, early_exaggeration_iters={cfg.early_exaggeration_iters}"
    )

    print("\n[Metrics]")
    print(
        summary.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}" if np.isfinite(v) else "nan",
        )
    )
    print(f"\n10-NN overlap(custom vs sklearn) = {overlap_10nn:.4f}")

    print("\n[Custom Training Snapshot]")
    print(
        history_snapshot(custom.history).to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print("\n[Class Centroids in 2D Embedding Space]")
    print(
        centroid_table.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )

    run_quality_checks(
        custom_result=custom,
        tw_custom=tw_custom,
        tw_sklearn=tw_sklearn,
        overlap_10nn=overlap_10nn,
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
