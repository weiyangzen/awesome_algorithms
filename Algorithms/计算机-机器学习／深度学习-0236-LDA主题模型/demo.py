"""LDA topic modeling MVP (collapsed Gibbs from scratch + sklearn VB baseline).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation


@dataclass(frozen=True)
class LDAConfig:
    """Configuration for synthetic corpus and training."""

    n_topics: int = 3
    alpha: float = 0.25
    beta: float = 0.06
    n_docs: int = 90
    doc_len_min: int = 45
    doc_len_max: int = 72
    gibbs_iters: int = 260
    random_state: int = 236
    top_n_words: int = 8
    vb_max_iter: int = 80


@dataclass
class EvalResult:
    """Evaluation summary against known synthetic ground truth."""

    model: str
    perplexity_token: float
    theta_l1: float
    phi_l1: float


def build_synthetic_corpus(
    cfg: LDAConfig,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray, list[str]]:
    """Generate corpus from known theta/phi so recovery quality can be measured."""
    topic_names = ["tech", "sports", "food"]
    topic_words = {
        "tech": [
            "chip",
            "gpu",
            "model",
            "python",
            "server",
            "cloud",
            "tensor",
            "vector",
            "kernel",
            "query",
        ],
        "sports": [
            "team",
            "coach",
            "match",
            "goal",
            "league",
            "stadium",
            "score",
            "defense",
            "offense",
            "training",
        ],
        "food": [
            "recipe",
            "kitchen",
            "flavor",
            "spicy",
            "sweet",
            "noodle",
            "soup",
            "dish",
            "chef",
            "sauce",
        ],
    }

    if cfg.n_topics != len(topic_names):
        raise ValueError(f"This MVP expects n_topics={len(topic_names)}, got {cfg.n_topics}")

    vocab: list[str] = []
    for topic in topic_names:
        vocab.extend(topic_words[topic])
    vocab_size = len(vocab)

    word2id = {w: i for i, w in enumerate(vocab)}
    phi_true = np.full((cfg.n_topics, vocab_size), 0.0025, dtype=np.float64)

    for k, topic in enumerate(topic_names):
        for w in topic_words[topic]:
            phi_true[k, word2id[w]] = 1.0
        phi_true[k] /= phi_true[k].sum()

    alpha_vec = np.full(cfg.n_topics, cfg.alpha, dtype=np.float64)
    theta_true = rng.dirichlet(alpha=alpha_vec, size=cfg.n_docs)

    docs: list[np.ndarray] = []
    for d in range(cfg.n_docs):
        length = int(rng.integers(cfg.doc_len_min, cfg.doc_len_max + 1))
        z = rng.choice(cfg.n_topics, size=length, p=theta_true[d])
        tokens = np.empty(length, dtype=np.int64)
        for i, topic in enumerate(z):
            tokens[i] = int(rng.choice(vocab_size, p=phi_true[topic]))
        docs.append(tokens)

    return docs, vocab, theta_true, phi_true, topic_names


def docs_to_count_matrix(docs: Sequence[np.ndarray], vocab_size: int) -> np.ndarray:
    """Convert tokenized docs into a document-term count matrix."""
    x = np.zeros((len(docs), vocab_size), dtype=np.int64)
    for d, doc in enumerate(docs):
        x[d] = np.bincount(doc, minlength=vocab_size)
    return x


def initialize_gibbs_state(
    docs: Sequence[np.ndarray],
    n_topics: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Random topic assignments and count tables."""
    n_docs = len(docs)
    ndk = np.zeros((n_docs, n_topics), dtype=np.int64)
    nkw = np.zeros((n_topics, vocab_size), dtype=np.int64)
    nk = np.zeros(n_topics, dtype=np.int64)
    z_dn: list[np.ndarray] = []

    for d, doc in enumerate(docs):
        z = rng.integers(0, n_topics, size=len(doc), dtype=np.int64)
        z_dn.append(z)
        for i, word in enumerate(doc):
            topic = int(z[i])
            ndk[d, topic] += 1
            nkw[topic, int(word)] += 1
            nk[topic] += 1

    return z_dn, ndk, nkw, nk


def sample_categorical(prob: np.ndarray, rng: np.random.Generator) -> int:
    """Numerically safe categorical sampler."""
    total = float(np.sum(prob))
    if (not np.isfinite(total)) or total <= 0.0:
        prob = np.full(prob.shape[0], 1.0 / prob.shape[0], dtype=np.float64)
    else:
        prob = prob / total

    cdf = np.cumsum(prob)
    u = float(rng.random())
    idx = int(np.searchsorted(cdf, u, side="right"))
    return min(idx, prob.shape[0] - 1)


def gibbs_sample(
    docs: Sequence[np.ndarray],
    z_dn: list[np.ndarray],
    ndk: np.ndarray,
    nkw: np.ndarray,
    nk: np.ndarray,
    cfg: LDAConfig,
    rng: np.random.Generator,
) -> None:
    """Collapsed Gibbs sampling loop (in-place state updates)."""
    vocab_size = nkw.shape[1]

    for _ in range(cfg.gibbs_iters):
        for d, doc in enumerate(docs):
            for i, token in enumerate(doc):
                w = int(token)
                old_topic = int(z_dn[d][i])

                ndk[d, old_topic] -= 1
                nkw[old_topic, w] -= 1
                nk[old_topic] -= 1

                left = ndk[d].astype(np.float64) + cfg.alpha
                right = (nkw[:, w].astype(np.float64) + cfg.beta) / (
                    nk.astype(np.float64) + cfg.beta * vocab_size
                )
                prob = left * right

                new_topic = sample_categorical(prob, rng)

                z_dn[d][i] = new_topic
                ndk[d, new_topic] += 1
                nkw[new_topic, w] += 1
                nk[new_topic] += 1


def estimate_theta_phi(
    ndk: np.ndarray,
    nkw: np.ndarray,
    nk: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover theta and phi from final counts with Dirichlet smoothing."""
    n_docs, n_topics = ndk.shape
    _, vocab_size = nkw.shape

    theta = (ndk.astype(np.float64) + alpha) / (
        ndk.sum(axis=1, keepdims=True).astype(np.float64) + n_topics * alpha
    )
    phi = (nkw.astype(np.float64) + beta) / (nk.astype(np.float64)[:, None] + vocab_size * beta)

    if theta.shape != (n_docs, n_topics):
        raise RuntimeError("theta shape mismatch")
    return theta, phi


def token_perplexity(docs: Sequence[np.ndarray], theta: np.ndarray, phi: np.ndarray) -> float:
    """Compute token-level perplexity from explicit theta and phi."""
    total_log_prob = 0.0
    total_tokens = 0
    for d, doc in enumerate(docs):
        for token in doc:
            prob = float(np.dot(theta[d], phi[:, int(token)]))
            prob = max(prob, 1e-12)
            total_log_prob += np.log(prob)
            total_tokens += 1
    return float(np.exp(-total_log_prob / max(total_tokens, 1)))


def align_topics(est_phi: np.ndarray, ref_phi: np.ndarray) -> np.ndarray:
    """Return permutation index so est topics align to reference topic order."""
    similarity = est_phi @ ref_phi.T
    row_ind, col_ind = linear_sum_assignment(-similarity)
    order = row_ind[np.argsort(col_ind)]
    return order.astype(np.int64)


def evaluate_against_truth(
    model_name: str,
    theta_est: np.ndarray,
    phi_est: np.ndarray,
    theta_true: np.ndarray,
    phi_true: np.ndarray,
    docs: Sequence[np.ndarray],
) -> tuple[EvalResult, np.ndarray, np.ndarray]:
    """Align estimated topics to truth and compute simple recovery errors."""
    order = align_topics(phi_est, phi_true)
    phi_aligned = phi_est[order]
    theta_aligned = theta_est[:, order]

    theta_l1 = float(np.mean(np.abs(theta_aligned - theta_true)))
    phi_l1 = float(np.mean(np.abs(phi_aligned - phi_true)))
    perp = token_perplexity(docs, theta_aligned, phi_aligned)

    return (
        EvalResult(model=model_name, perplexity_token=perp, theta_l1=theta_l1, phi_l1=phi_l1),
        theta_aligned,
        phi_aligned,
    )


def top_words(phi: np.ndarray, vocab: Sequence[str], top_n: int) -> list[list[str]]:
    """Top-N words per topic."""
    out: list[list[str]] = []
    for k in range(phi.shape[0]):
        idx = np.argsort(phi[k])[::-1][:top_n]
        out.append([vocab[int(i)] for i in idx])
    return out


def main() -> None:
    cfg = LDAConfig()
    rng = np.random.default_rng(cfg.random_state)

    docs, vocab, theta_true, phi_true, topic_names = build_synthetic_corpus(cfg, rng)
    vocab_size = len(vocab)
    x_counts = docs_to_count_matrix(docs, vocab_size)

    z_dn, ndk, nkw, nk = initialize_gibbs_state(docs, cfg.n_topics, vocab_size, rng)
    gibbs_sample(docs, z_dn, ndk, nkw, nk, cfg, rng)
    theta_gibbs, phi_gibbs = estimate_theta_phi(ndk, nkw, nk, cfg.alpha, cfg.beta)

    vb_model = LatentDirichletAllocation(
        n_components=cfg.n_topics,
        doc_topic_prior=cfg.alpha,
        topic_word_prior=cfg.beta,
        learning_method="batch",
        max_iter=cfg.vb_max_iter,
        random_state=cfg.random_state,
    )
    vb_model.fit(x_counts)
    theta_vb = vb_model.transform(x_counts)
    phi_vb = vb_model.components_ / vb_model.components_.sum(axis=1, keepdims=True)

    eval_gibbs, theta_gibbs_aligned, phi_gibbs_aligned = evaluate_against_truth(
        "scratch-gibbs",
        theta_gibbs,
        phi_gibbs,
        theta_true,
        phi_true,
        docs,
    )
    eval_vb, theta_vb_aligned, phi_vb_aligned = evaluate_against_truth(
        "sklearn-vb",
        theta_vb,
        phi_vb,
        theta_true,
        phi_true,
        docs,
    )

    summary = pd.DataFrame(
        [
            {
                "model": eval_gibbs.model,
                "token_perplexity": eval_gibbs.perplexity_token,
                "theta_mean_l1": eval_gibbs.theta_l1,
                "phi_mean_l1": eval_gibbs.phi_l1,
            },
            {
                "model": eval_vb.model,
                "token_perplexity": eval_vb.perplexity_token,
                "theta_mean_l1": eval_vb.theta_l1,
                "phi_mean_l1": eval_vb.phi_l1,
            },
        ]
    )

    top_words_gibbs = top_words(phi_gibbs_aligned, vocab, cfg.top_n_words)
    top_words_vb = top_words(phi_vb_aligned, vocab, cfg.top_n_words)

    preview = pd.DataFrame(
        {
            "doc_id": np.arange(min(6, cfg.n_docs)),
            "true_theta": [np.round(theta_true[i], 3).tolist() for i in range(min(6, cfg.n_docs))],
            "gibbs_theta": [
                np.round(theta_gibbs_aligned[i], 3).tolist() for i in range(min(6, cfg.n_docs))
            ],
            "vb_theta": [np.round(theta_vb_aligned[i], 3).tolist() for i in range(min(6, cfg.n_docs))],
        }
    )

    print("LDA Topic Model MVP (CS-0107)")
    print("=" * 78)
    print(
        "config:",
        {
            "n_topics": cfg.n_topics,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "n_docs": cfg.n_docs,
            "gibbs_iters": cfg.gibbs_iters,
            "vb_max_iter": cfg.vb_max_iter,
            "seed": cfg.random_state,
        },
    )
    print(f"vocab_size={vocab_size}, total_tokens={int(sum(len(doc) for doc in docs))}")
    print(f"sklearn_bound_perplexity={float(vb_model.perplexity(x_counts)):.6f}")
    print()
    print("[Recovery Summary]")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("[Top Words by Topic - scratch-gibbs]")
    for k, words in enumerate(top_words_gibbs):
        print(f"  Topic {k} ({topic_names[k]}): {', '.join(words)}")
    print()

    print("[Top Words by Topic - sklearn-vb]")
    for k, words in enumerate(top_words_vb):
        print(f"  Topic {k} ({topic_names[k]}): {', '.join(words)}")
    print()

    print("[Doc Topic Preview]")
    print(preview.to_string(index=False))

    if not np.allclose(theta_gibbs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("gibbs theta rows do not sum to 1")
    if not np.allclose(phi_gibbs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("gibbs phi rows do not sum to 1")
    if eval_gibbs.theta_l1 > 0.20:
        raise RuntimeError(f"gibbs theta recovery too weak: {eval_gibbs.theta_l1:.4f}")
    if eval_gibbs.phi_l1 > 0.12:
        raise RuntimeError(f"gibbs phi recovery too weak: {eval_gibbs.phi_l1:.4f}")
    if eval_gibbs.perplexity_token > eval_vb.perplexity_token * 1.35:
        raise RuntimeError("gibbs token perplexity is unexpectedly much worse than VB baseline")

    print("=" * 78)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
