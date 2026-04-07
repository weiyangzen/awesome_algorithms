"""Minimal runnable MVP for Latent Dirichlet Allocation (LDA).

This demo implements collapsed Gibbs sampling from scratch with NumPy.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LDAConfig:
    n_topics: int = 3
    alpha: float = 0.3
    beta: float = 0.08
    n_iters: int = 250
    seed: int = 7
    n_docs: int = 24
    doc_len_min: int = 45
    doc_len_max: int = 70
    top_n_words: int = 8


def build_synthetic_corpus(
    cfg: LDAConfig,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[str], np.ndarray, np.ndarray, List[str]]:
    """Generate documents from known topic-word distributions."""
    topic_names = ["tech", "sports", "food"]
    topic_words = {
        "tech": ["chip", "gpu", "model", "neural", "data", "python", "server", "cloud"],
        "sports": ["team", "match", "goal", "coach", "player", "league", "stadium", "score"],
        "food": ["recipe", "kitchen", "flavor", "spicy", "sweet", "soup", "noodle", "dish"],
    }

    if cfg.n_topics != len(topic_names):
        raise ValueError(
            f"This MVP synthetic generator expects n_topics={len(topic_names)}, got {cfg.n_topics}."
        )

    vocab: List[str] = []
    for t_name in topic_names:
        vocab.extend(topic_words[t_name])
    vocab_size = len(vocab)

    word2id: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
    phi_true = np.full((cfg.n_topics, vocab_size), 0.01, dtype=np.float64)

    for k, t_name in enumerate(topic_names):
        for w in topic_words[t_name]:
            phi_true[k, word2id[w]] = 1.0
        phi_true[k] /= phi_true[k].sum()

    theta_true = rng.dirichlet(alpha=np.full(cfg.n_topics, cfg.alpha), size=cfg.n_docs)

    docs: List[np.ndarray] = []
    for d in range(cfg.n_docs):
        length = int(rng.integers(cfg.doc_len_min, cfg.doc_len_max + 1))
        z = rng.choice(cfg.n_topics, size=length, p=theta_true[d])
        tokens = np.empty(length, dtype=np.int64)
        for i, topic in enumerate(z):
            tokens[i] = int(rng.choice(vocab_size, p=phi_true[topic]))
        docs.append(tokens)

    return docs, vocab, theta_true, phi_true, topic_names


def initialize_state(
    docs: Sequence[np.ndarray],
    n_topics: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Randomly assign topics to tokens and build count matrices."""
    n_docs = len(docs)
    ndk = np.zeros((n_docs, n_topics), dtype=np.int64)
    nkw = np.zeros((n_topics, vocab_size), dtype=np.int64)
    nk = np.zeros(n_topics, dtype=np.int64)
    z_dn: List[np.ndarray] = []

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
    """Sample a category index from a probability vector."""
    total = float(prob.sum())
    if not np.isfinite(total) or total <= 0.0:
        prob = np.full_like(prob, 1.0 / len(prob), dtype=np.float64)
    else:
        prob = prob / total

    cdf = np.cumsum(prob)
    u = float(rng.random())
    idx = int(np.searchsorted(cdf, u, side="right"))
    return min(idx, len(prob) - 1)


def gibbs_sample(
    docs: Sequence[np.ndarray],
    z_dn: List[np.ndarray],
    ndk: np.ndarray,
    nkw: np.ndarray,
    nk: np.ndarray,
    cfg: LDAConfig,
    rng: np.random.Generator,
) -> None:
    """Run collapsed Gibbs sampling in-place."""
    vocab_size = nkw.shape[1]

    for _ in range(cfg.n_iters):
        for d, doc in enumerate(docs):
            for i, word in enumerate(doc):
                old_topic = int(z_dn[d][i])
                w = int(word)

                ndk[d, old_topic] -= 1
                nkw[old_topic, w] -= 1
                nk[old_topic] -= 1

                left = ndk[d].astype(np.float64) + cfg.alpha
                right = (nkw[:, w].astype(np.float64) + cfg.beta) / (
                    nk.astype(np.float64) + cfg.beta * vocab_size
                )
                p = left * right

                new_topic = sample_categorical(p, rng)

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover theta and phi from count matrices with Dirichlet smoothing."""
    n_docs, n_topics = ndk.shape
    _, vocab_size = nkw.shape

    theta = (ndk.astype(np.float64) + alpha) / (
        ndk.sum(axis=1, keepdims=True).astype(np.float64) + n_topics * alpha
    )
    phi = (nkw.astype(np.float64) + beta) / (
        nk.astype(np.float64)[:, None] + vocab_size * beta
    )

    if theta.shape[0] != n_docs:
        raise RuntimeError("theta shape mismatch")
    return theta, phi


def compute_perplexity(docs: Sequence[np.ndarray], theta: np.ndarray, phi: np.ndarray) -> float:
    """Compute perplexity on training corpus."""
    total_log_prob = 0.0
    total_tokens = 0

    for d, doc in enumerate(docs):
        for word in doc:
            prob = float(np.dot(theta[d], phi[:, int(word)]))
            prob = max(prob, 1e-12)
            total_log_prob += np.log(prob)
            total_tokens += 1

    return float(np.exp(-total_log_prob / max(total_tokens, 1)))


def top_words_per_topic(phi: np.ndarray, vocab: Sequence[str], top_n: int) -> List[List[str]]:
    """Return top-n words for each topic."""
    result: List[List[str]] = []
    for k in range(phi.shape[0]):
        top_idx = np.argsort(phi[k])[::-1][:top_n]
        result.append([vocab[int(i)] for i in top_idx])
    return result


def align_topics_to_reference(phi_est: np.ndarray, phi_ref: np.ndarray) -> np.ndarray:
    """Find a permutation so inferred topics align with reference topics for display."""
    k_topics = phi_est.shape[0]
    best_perm = tuple(range(k_topics))
    best_score = -np.inf

    for perm in permutations(range(k_topics)):
        score = 0.0
        for k in range(k_topics):
            score += float(np.dot(phi_est[perm[k]], phi_ref[k]))
        if score > best_score:
            best_score = score
            best_perm = perm

    return np.asarray(best_perm, dtype=np.int64)


def main() -> None:
    cfg = LDAConfig()
    rng = np.random.default_rng(cfg.seed)

    docs, vocab, theta_true, phi_true, topic_names = build_synthetic_corpus(cfg, rng)
    z_dn, ndk, nkw, nk = initialize_state(docs, cfg.n_topics, len(vocab), rng)
    gibbs_sample(docs, z_dn, ndk, nkw, nk, cfg, rng)
    theta, phi = estimate_theta_phi(ndk, nkw, nk, cfg.alpha, cfg.beta)
    perm = align_topics_to_reference(phi, phi_true)
    theta = theta[:, perm]
    phi = phi[perm]
    perplexity = compute_perplexity(docs, theta, phi)
    topic_words = top_words_per_topic(phi, vocab, cfg.top_n_words)

    if not np.allclose(theta.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("theta rows do not sum to 1")
    if not np.allclose(phi.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("phi rows do not sum to 1")
    if (not np.isfinite(perplexity)) or perplexity <= 0.0:
        raise RuntimeError("invalid perplexity")

    print("=== LDA Collapsed Gibbs MVP ===")
    print(
        "Config:",
        {
            "n_topics": cfg.n_topics,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "n_iters": cfg.n_iters,
            "seed": cfg.seed,
        },
    )
    print(f"Vocabulary size: {len(vocab)}, Documents: {len(docs)}")

    print("\nTopic-word summaries:")
    for k, words in enumerate(topic_words):
        print(f"  Topic {k} (reference cluster: {topic_names[k]}): {', '.join(words)}")

    print("\nDocument-topic mixtures (first 5 docs):")
    for d in range(min(5, len(docs))):
        mix = ", ".join(f"T{k}:{theta[d, k]:.3f}" for k in range(cfg.n_topics))
        true_mix = ", ".join(f"T{k}:{theta_true[d, k]:.3f}" for k in range(cfg.n_topics))
        print(f"  Doc {d:02d} inferred -> [{mix}] | true -> [{true_mix}]")

    print(f"\nTraining perplexity: {perplexity:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
