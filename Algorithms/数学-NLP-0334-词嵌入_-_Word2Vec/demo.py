"""Minimal runnable MVP for Word2Vec (MATH-0334).

Implementation scope:
- Skip-gram with Negative Sampling (SGNS)
- Pure NumPy training loop (no black-box word2vec call)
- Deterministic toy corpus and built-in quality checks
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingReport:
    vocab_size: int
    pair_count: int
    epochs: int
    final_avg_loss: float


class SkipGramWord2Vec:
    """Compact SGNS trainer for educational and auditable use."""

    def __init__(
        self,
        embedding_dim: int = 24,
        window_size: int = 2,
        negatives: int = 6,
        learning_rate: float = 0.035,
        epochs: int = 24,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if negatives <= 0:
            raise ValueError("negatives must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if min_count <= 0:
            raise ValueError("min_count must be positive")

        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negatives = negatives
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.seed = seed

        self.word_to_id_: dict[str, int] | None = None
        self.id_to_word_: list[str] | None = None
        self.embeddings_: np.ndarray | None = None

        self._w_in: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._neg_probs: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)

    def fit(self, corpus: list[list[str]]) -> TrainingReport:
        if not corpus:
            raise ValueError("corpus must not be empty")
        if any(len(sent) < 2 for sent in corpus):
            raise ValueError("every sentence in corpus must contain at least 2 tokens")

        word_to_id, id_to_word, token_counts = self._build_vocab(corpus)
        encoded_sentences = [[word_to_id[w] for w in sent if w in word_to_id] for sent in corpus]
        pairs = self._generate_skipgram_pairs(encoded_sentences)
        if not pairs:
            raise RuntimeError("no training pairs were generated")

        vocab_size = len(id_to_word)
        scale = 0.5 / max(1, self.embedding_dim)
        self._w_in = self._rng.uniform(-scale, scale, size=(vocab_size, self.embedding_dim))
        self._w_out = np.zeros((vocab_size, self.embedding_dim), dtype=float)

        freqs = np.array([token_counts[word] for word in id_to_word], dtype=float)
        probs = np.power(freqs, 0.75)
        self._neg_probs = probs / probs.sum()

        last_avg_loss = float("nan")
        for _ in range(self.epochs):
            order = self._rng.permutation(len(pairs))
            total_loss = 0.0
            for idx in order:
                center_id, context_id = pairs[idx]
                neg_ids = self._sample_negatives(context_id)
                total_loss += self._sgd_step(center_id, context_id, neg_ids)
            last_avg_loss = total_loss / len(pairs)

        combined = self._w_in + self._w_out
        self.embeddings_ = self._normalize_rows(combined)
        self.word_to_id_ = word_to_id
        self.id_to_word_ = id_to_word

        return TrainingReport(
            vocab_size=vocab_size,
            pair_count=len(pairs),
            epochs=self.epochs,
            final_avg_loss=float(last_avg_loss),
        )

    def most_similar(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vec = self.vector(word)
        sims = self._all_similarities(vec)
        word_id = self._word_id(word)
        sims[word_id] = -1.0

        idx = np.argsort(-sims)[:top_k]
        return [(self._id_to_word(i), float(sims[i])) for i in idx]

    def analogy(
        self,
        positive1: str,
        negative: str,
        positive2: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        target = self.vector(positive1) - self.vector(negative) + self.vector(positive2)
        target = target / (np.linalg.norm(target) + 1e-12)
        sims = self._all_similarities(target)
        for token in (positive1, negative, positive2):
            sims[self._word_id(token)] = -1.0

        idx = np.argsort(-sims)[:top_k]
        return [(self._id_to_word(i), float(sims[i])) for i in idx]

    def similarity(self, word_a: str, word_b: str) -> float:
        va = self.vector(word_a)
        vb = self.vector(word_b)
        return float(np.dot(va, vb))

    def vector(self, word: str) -> np.ndarray:
        self._check_fitted()
        word_id = self._word_id(word)
        assert self.embeddings_ is not None
        return self.embeddings_[word_id]

    def _build_vocab(self, corpus: list[list[str]]) -> tuple[dict[str, int], list[str], Counter[str]]:
        counts: Counter[str] = Counter()
        for sent in corpus:
            counts.update(sent)

        kept = [word for word, c in counts.items() if c >= self.min_count]
        if len(kept) < 2:
            raise ValueError("vocabulary too small after min_count filtering")

        kept.sort(key=lambda w: (-counts[w], w))
        word_to_id = {word: i for i, word in enumerate(kept)}
        return word_to_id, kept, counts

    def _generate_skipgram_pairs(
        self, encoded_sentences: list[list[int]]
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        w = self.window_size
        for sent in encoded_sentences:
            n = len(sent)
            for i, center in enumerate(sent):
                left = max(0, i - w)
                right = min(n, i + w + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    pairs.append((center, sent[j]))
        return pairs

    def _sample_negatives(self, positive_context_id: int) -> np.ndarray:
        assert self._neg_probs is not None
        vocab_size = self._neg_probs.size
        neg_ids = self._rng.choice(vocab_size, size=self.negatives, p=self._neg_probs)

        # Avoid using positive context as a negative label.
        mask = neg_ids == positive_context_id
        while np.any(mask):
            neg_ids[mask] = self._rng.choice(vocab_size, size=np.sum(mask), p=self._neg_probs)
            mask = neg_ids == positive_context_id
        return neg_ids

    def _sgd_step(self, center_id: int, context_id: int, neg_ids: np.ndarray) -> float:
        assert self._w_in is not None
        assert self._w_out is not None

        v = self._w_in[center_id].copy()
        u_pos = self._w_out[context_id].copy()
        u_neg = self._w_out[neg_ids].copy()

        pos_score = float(np.dot(u_pos, v))
        neg_scores = u_neg @ v

        pos_sig = self._sigmoid_scalar(pos_score)
        neg_sig = self._sigmoid(neg_scores)

        eps = 1e-12
        loss = -np.log(pos_sig + eps) - np.sum(np.log(1.0 - neg_sig + eps))

        grad_v = (pos_sig - 1.0) * u_pos + np.sum(neg_sig[:, None] * u_neg, axis=0)
        grad_u_pos = (pos_sig - 1.0) * v
        grad_u_neg = neg_sig[:, None] * v[None, :]

        lr = self.learning_rate
        self._w_in[center_id] -= lr * grad_v
        self._w_out[context_id] -= lr * grad_u_pos
        np.add.at(self._w_out, neg_ids, -lr * grad_u_neg)

        return float(loss)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x_clip))

    @staticmethod
    def _sigmoid_scalar(x: float) -> float:
        x_clip = min(20.0, max(-20.0, x))
        return float(1.0 / (1.0 + np.exp(-x_clip)))

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-12)

    def _all_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.embeddings_ is not None
        return self.embeddings_ @ query_vec

    def _check_fitted(self) -> None:
        if self.embeddings_ is None or self.word_to_id_ is None or self.id_to_word_ is None:
            raise RuntimeError("model is not fitted")

    def _word_id(self, word: str) -> int:
        assert self.word_to_id_ is not None
        if word not in self.word_to_id_:
            raise KeyError(f"word not in vocabulary: {word}")
        return self.word_to_id_[word]

    def _id_to_word(self, word_id: int) -> str:
        assert self.id_to_word_ is not None
        return self.id_to_word_[word_id]


def build_toy_corpus(seed: int = 2026) -> list[list[str]]:
    """Create a deterministic tiny corpus with multiple semantic groups."""

    base_sentences = [
        ["king", "queen", "royal", "palace", "crown"],
        ["king", "man", "royal", "male", "leader"],
        ["queen", "woman", "royal", "female", "leader"],
        ["prince", "boy", "royal", "male", "palace"],
        ["princess", "girl", "royal", "female", "palace"],
        ["man", "father", "male", "family", "home"],
        ["woman", "mother", "female", "family", "home"],
        ["dog", "puppy", "pet", "animal", "home"],
        ["cat", "kitten", "pet", "animal", "home"],
        ["dog", "cat", "pet", "animal", "playful"],
        ["apple", "banana", "orange", "fruit", "sweet"],
        ["banana", "orange", "fruit", "fresh", "market"],
        ["car", "bus", "train", "vehicle", "road"],
        ["car", "truck", "vehicle", "road", "engine"],
        ["bus", "train", "vehicle", "travel", "city"],
        ["king", "queen", "man", "woman", "royal"],
        ["dog", "cat", "pet", "home", "family"],
        ["car", "bus", "train", "city", "travel"],
    ]

    rng = np.random.default_rng(seed)
    corpus: list[list[str]] = []
    repeats = 30
    for sentence in base_sentences:
        for _ in range(repeats):
            shuffled = sentence.copy()
            rng.shuffle(shuffled)
            corpus.append(shuffled)
    return corpus


def run_quality_checks(model: SkipGramWord2Vec, report: TrainingReport) -> None:
    if not np.isfinite(report.final_avg_loss):
        raise AssertionError("final loss is not finite")
    if report.vocab_size < 20:
        raise AssertionError("vocab size unexpectedly small")
    if report.pair_count < 1000:
        raise AssertionError("pair count unexpectedly small")

    sim_king_queen = model.similarity("king", "queen")
    sim_king_banana = model.similarity("king", "banana")
    sim_dog_cat = model.similarity("dog", "cat")
    sim_dog_train = model.similarity("dog", "train")

    if sim_king_queen <= sim_king_banana + 0.15:
        raise AssertionError("royalty semantic relation is too weak")
    if sim_dog_cat <= sim_dog_train + 0.15:
        raise AssertionError("animal semantic relation is too weak")

    top_words = [w for w, _ in model.analogy("king", "man", "woman", top_k=6)]
    if "queen" not in top_words:
        raise AssertionError("analogy king - man + woman -> queen not recovered")


def main() -> None:
    corpus = build_toy_corpus()

    model = SkipGramWord2Vec(
        embedding_dim=24,
        window_size=2,
        negatives=6,
        learning_rate=0.035,
        epochs=24,
        min_count=1,
        seed=42,
    )
    report = model.fit(corpus)

    print("Word2Vec (Skip-gram + Negative Sampling) MVP")
    print(f"Vocabulary size: {report.vocab_size}")
    print(f"Training pairs: {report.pair_count}")
    print(f"Epochs: {report.epochs}")
    print(f"Final average loss: {report.final_avg_loss:.4f}")
    print()

    probes = ["king", "queen", "dog", "apple", "car"]
    for word in probes:
        neighbors = model.most_similar(word, top_k=5)
        rendered = ", ".join(f"{w}:{s:.3f}" for w, s in neighbors)
        print(f"Nearest to '{word}': {rendered}")

    print()
    analogy_result = model.analogy("king", "man", "woman", top_k=5)
    rendered_analogy = ", ".join(f"{w}:{s:.3f}" for w, s in analogy_result)
    print("Analogy: king - man + woman")
    print(f"Top candidates: {rendered_analogy}")

    run_quality_checks(model, report)
    print("All checks passed.")


if __name__ == "__main__":
    main()
