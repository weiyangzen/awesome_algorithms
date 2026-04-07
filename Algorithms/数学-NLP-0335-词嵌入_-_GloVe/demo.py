"""Minimal runnable MVP for GloVe (MATH-0335).

Implementation scope:
- Build global word-word co-occurrence counts with a context window
- Optimize the weighted least-squares GloVe objective using AdaGrad
- Use pure NumPy (no black-box glove package)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class TrainingReport:
    vocab_size: int
    cooc_pairs: int
    epochs: int
    final_avg_loss: float


class GloveEmbeddings:
    """Compact and auditable GloVe trainer."""

    def __init__(
        self,
        embedding_dim: int = 32,
        window_size: int = 2,
        x_max: float = 10.0,
        alpha: float = 0.75,
        learning_rate: float = 0.05,
        epochs: int = 70,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if x_max <= 0:
            raise ValueError("x_max must be positive")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if min_count <= 0:
            raise ValueError("min_count must be positive")

        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.seed = seed

        self.word_to_id_: dict[str, int] | None = None
        self.id_to_word_: list[str] | None = None
        self.embeddings_: np.ndarray | None = None

        self._w_main: np.ndarray | None = None
        self._w_context: np.ndarray | None = None
        self._b_main: np.ndarray | None = None
        self._b_context: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)

    def fit(self, corpus: list[list[str]]) -> TrainingReport:
        if not corpus:
            raise ValueError("corpus must not be empty")
        if any(len(sent) < 2 for sent in corpus):
            raise ValueError("every sentence in corpus must contain at least 2 tokens")

        word_to_id, id_to_word = self._build_vocab(corpus)
        encoded = [[word_to_id[w] for w in sent if w in word_to_id] for sent in corpus]

        i_idx, j_idx, x_ij = self._build_cooccurrence_pairs(encoded)
        if x_ij.size == 0:
            raise RuntimeError("co-occurrence matrix is empty")

        vocab_size = len(id_to_word)
        scale = 0.5 / max(1, self.embedding_dim)
        self._w_main = self._rng.uniform(-scale, scale, size=(vocab_size, self.embedding_dim))
        self._w_context = self._rng.uniform(-scale, scale, size=(vocab_size, self.embedding_dim))
        self._b_main = np.zeros(vocab_size, dtype=float)
        self._b_context = np.zeros(vocab_size, dtype=float)

        g_w_main = np.ones_like(self._w_main)
        g_w_context = np.ones_like(self._w_context)
        g_b_main = np.ones_like(self._b_main)
        g_b_context = np.ones_like(self._b_context)

        eps = 1e-8
        last_avg_loss = float("nan")

        for _ in range(self.epochs):
            order = self._rng.permutation(x_ij.size)
            total_loss = 0.0
            for p in order:
                i = int(i_idx[p])
                j = int(j_idx[p])
                x = float(x_ij[p])

                weight = self._weight_fn(x)
                log_x = math.log(x)

                assert self._w_main is not None
                assert self._w_context is not None
                assert self._b_main is not None
                assert self._b_context is not None

                pred = (
                    float(np.dot(self._w_main[i], self._w_context[j]))
                    + float(self._b_main[i])
                    + float(self._b_context[j])
                )
                diff = pred - log_x
                weighted_diff = weight * diff
                total_loss += weighted_diff * diff

                grad_common = 2.0 * weighted_diff
                grad_wi = grad_common * self._w_context[j]
                grad_wj = grad_common * self._w_main[i]
                grad_bi = grad_common
                grad_bj = grad_common

                g_w_main[i] += grad_wi * grad_wi
                g_w_context[j] += grad_wj * grad_wj
                g_b_main[i] += grad_bi * grad_bi
                g_b_context[j] += grad_bj * grad_bj

                self._w_main[i] -= self.learning_rate * grad_wi / np.sqrt(g_w_main[i] + eps)
                self._w_context[j] -= self.learning_rate * grad_wj / np.sqrt(g_w_context[j] + eps)
                self._b_main[i] -= self.learning_rate * grad_bi / math.sqrt(g_b_main[i] + eps)
                self._b_context[j] -= self.learning_rate * grad_bj / math.sqrt(g_b_context[j] + eps)

            last_avg_loss = total_loss / x_ij.size

        combined = self._w_main + self._w_context
        self.embeddings_ = self._normalize_rows(combined)
        self.word_to_id_ = word_to_id
        self.id_to_word_ = id_to_word

        return TrainingReport(
            vocab_size=vocab_size,
            cooc_pairs=int(x_ij.size),
            epochs=self.epochs,
            final_avg_loss=float(last_avg_loss),
        )

    def most_similar(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = self.vector(word)
        sims = self._all_similarities(query)
        sims[self._word_id(word)] = -1.0
        idx = np.argsort(-sims)[:top_k]
        return [(self._id_to_word(i), float(sims[i])) for i in idx]

    def analogy(self, positive1: str, negative: str, positive2: str, top_k: int = 5) -> list[tuple[str, float]]:
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
        assert self.embeddings_ is not None
        return self.embeddings_[self._word_id(word)]

    def _build_vocab(self, corpus: list[list[str]]) -> tuple[dict[str, int], list[str]]:
        counts: Counter[str] = Counter()
        for sent in corpus:
            counts.update(sent)

        kept = [word for word, c in counts.items() if c >= self.min_count]
        if len(kept) < 2:
            raise ValueError("vocabulary too small after min_count filtering")

        kept.sort(key=lambda w: (-counts[w], w))
        word_to_id = {word: idx for idx, word in enumerate(kept)}
        return word_to_id, kept

    def _build_cooccurrence_pairs(
        self,
        encoded_sentences: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cooc: defaultdict[tuple[int, int], float] = defaultdict(float)
        window = self.window_size

        for sent in encoded_sentences:
            n = len(sent)
            for i, wi in enumerate(sent):
                left = max(0, i - window)
                right = min(n, i + window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    wj = sent[j]
                    distance = abs(i - j)
                    cooc[(wi, wj)] += 1.0 / distance

        if not cooc:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=float),
            )

        items = sorted(cooc.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        i_idx = np.array([pair[0] for pair, _ in items], dtype=np.int32)
        j_idx = np.array([pair[1] for pair, _ in items], dtype=np.int32)
        x_ij = np.array([value for _, value in items], dtype=float)
        return i_idx, j_idx, x_ij

    def _weight_fn(self, x: float) -> float:
        if x < self.x_max:
            return float((x / self.x_max) ** self.alpha)
        return 1.0

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    def _all_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.embeddings_ is not None
        return self.embeddings_ @ query_vec

    def _word_id(self, word: str) -> int:
        assert self.word_to_id_ is not None
        if word not in self.word_to_id_:
            raise KeyError(f"word not in vocabulary: {word}")
        return self.word_to_id_[word]

    def _id_to_word(self, idx: int) -> str:
        assert self.id_to_word_ is not None
        return self.id_to_word_[idx]

    def _check_fitted(self) -> None:
        if self.embeddings_ is None or self.word_to_id_ is None or self.id_to_word_ is None:
            raise RuntimeError("model is not fitted")


def build_toy_corpus(seed: int = 2026) -> list[list[str]]:
    """Create a deterministic corpus with several semantic clusters."""

    base_sentences = [
        ["king", "queen", "royal", "palace", "crown"],
        ["king", "man", "male", "leader", "royal"],
        ["queen", "woman", "female", "leader", "royal"],
        ["prince", "boy", "male", "royal", "palace"],
        ["princess", "girl", "female", "royal", "palace"],
        ["dog", "cat", "pet", "animal", "home"],
        ["dog", "puppy", "pet", "animal", "playful"],
        ["cat", "kitten", "pet", "animal", "playful"],
        ["apple", "banana", "orange", "fruit", "sweet"],
        ["banana", "orange", "fruit", "fresh", "market"],
        ["apple", "fruit", "fresh", "market", "sweet"],
        ["car", "bus", "train", "vehicle", "travel"],
        ["car", "truck", "vehicle", "road", "engine"],
        ["bus", "train", "vehicle", "city", "travel"],
        ["paris", "france", "city", "europe", "capital"],
        ["berlin", "germany", "city", "europe", "capital"],
        ["tokyo", "japan", "city", "asia", "capital"],
        ["doctor", "nurse", "hospital", "health", "care"],
        ["teacher", "student", "school", "learn", "class"],
        ["engineer", "code", "computer", "system", "build"],
    ]

    corpus = base_sentences * 8
    rng = np.random.default_rng(seed)
    rng.shuffle(corpus)
    return corpus


def run_quality_checks(model: GloveEmbeddings, report: TrainingReport) -> None:
    if not np.isfinite(report.final_avg_loss):
        raise AssertionError("loss is not finite")
    if report.vocab_size < 25:
        raise AssertionError("vocabulary unexpectedly small")
    if report.cooc_pairs < 120:
        raise AssertionError("too few co-occurrence pairs")

    sim_king_queen = model.similarity("king", "queen")
    sim_king_banana = model.similarity("king", "banana")
    if sim_king_queen <= sim_king_banana:
        raise AssertionError("expected king-queen to be closer than king-banana")

    sim_dog_cat = model.similarity("dog", "cat")
    sim_dog_train = model.similarity("dog", "train")
    if sim_dog_cat <= sim_dog_train:
        raise AssertionError("expected dog-cat to be closer than dog-train")

    analogy_results = model.analogy("king", "man", "woman", top_k=5)
    candidates = {word for word, _ in analogy_results}
    if "queen" not in candidates:
        raise AssertionError("analogy check failed: queen missing in top candidates")


def main() -> None:
    corpus = build_toy_corpus(seed=2026)
    model = GloveEmbeddings(
        embedding_dim=32,
        window_size=2,
        x_max=10.0,
        alpha=0.75,
        learning_rate=0.05,
        epochs=70,
        min_count=1,
        seed=42,
    )

    report = model.fit(corpus)

    print("=== GloVe MVP Report ===")
    print(f"Vocabulary size: {report.vocab_size}")
    print(f"Co-occurrence pairs: {report.cooc_pairs}")
    print(f"Epochs: {report.epochs}")
    print(f"Final average loss: {report.final_avg_loss:.6f}")

    for token in ("king", "queen", "dog", "apple", "city", "paris"):
        neighbors = model.most_similar(token, top_k=5)
        print(f"Nearest to '{token}': {neighbors}")

    analogy = model.analogy("king", "man", "woman", top_k=5)
    print(f"Analogy: king - man + woman -> {analogy}")

    run_quality_checks(model, report)
    print("All checks passed.")


if __name__ == "__main__":
    main()
