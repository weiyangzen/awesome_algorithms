"""Vector Space Model (TF-IDF + cosine) minimal runnable MVP for CS-0319.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


class VectorSpaceModel:
    """A small, explicit VSM implementation with sparse TF-IDF vectors."""

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
        self.documents: list[Document] = []
        self.doc_term_counts: sparse.csr_matrix | None = None
        self.doc_tfidf: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None

    def fit(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("documents must be non-empty")

        ids = [d.doc_id for d in documents]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate doc_id found: {ids}")

        texts = [d.text.strip() for d in documents]
        if any(not t for t in texts):
            raise ValueError("document text must not be blank")

        self.documents = documents
        counts = self.vectorizer.fit_transform(texts).tocsr()
        self.doc_term_counts = counts

        self.idf = self._compute_idf(counts)
        self.doc_tfidf = self._tfidf_from_counts(counts, self.idf)

    def _compute_idf(self, counts: sparse.csr_matrix) -> np.ndarray:
        n_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel().astype(float)
        # Smoothed IDF: avoids division-by-zero and keeps finite weights.
        idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        return idf

    def _tfidf_from_counts(self, counts: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
        row_sum = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        inv_row_sum = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)

        tf = counts.multiply(inv_row_sum[:, None])
        tfidf = tf.multiply(idf)

        l2 = np.sqrt(np.asarray(tfidf.power(2).sum(axis=1)).ravel())
        inv_l2 = np.divide(1.0, l2, out=np.zeros_like(l2), where=l2 > 0)
        normalized = tfidf.multiply(inv_l2[:, None]).tocsr()
        return normalized

    def encode_query(self, query: str) -> sparse.csr_matrix:
        if self.idf is None:
            raise RuntimeError("call fit() before encode_query()")
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")

        q_counts = self.vectorizer.transform([query]).tocsr()
        if q_counts.nnz == 0:
            # All query terms are out-of-vocabulary; return explicit zero vector.
            return sparse.csr_matrix((1, len(self.idf)), dtype=float)

        return self._tfidf_from_counts(q_counts, self.idf)

    def _score_with_torch(self, query_vec: sparse.csr_matrix) -> np.ndarray:
        if self.doc_tfidf is None:
            raise RuntimeError("index not built")

        doc_dense = torch.from_numpy(self.doc_tfidf.toarray()).to(dtype=torch.float32)
        query_dense = torch.from_numpy(query_vec.toarray().ravel()).to(dtype=torch.float32)
        scores = torch.mv(doc_dense, query_dense)
        return scores.cpu().numpy()

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        if self.doc_tfidf is None:
            raise RuntimeError("call fit() before search()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        q_vec = self.encode_query(query)
        sparse_scores = (self.doc_tfidf @ q_vec.T).toarray().ravel()
        torch_scores = self._score_with_torch(q_vec)

        if not np.allclose(sparse_scores, torch_scores, atol=1e-6):
            raise RuntimeError("score mismatch between SciPy and PyTorch paths")

        order = np.argsort(-sparse_scores)
        take = order[: min(top_k, len(order))]

        rows: list[dict[str, object]] = []
        for rank, idx in enumerate(take, start=1):
            doc = self.documents[int(idx)]
            rows.append(
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "score": float(sparse_scores[idx]),
                }
            )

        return pd.DataFrame(rows)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    hit = sum(1 for doc_id in topk if doc_id in relevant_ids)
    return float(hit / k)


def build_demo_corpus() -> list[Document]:
    """Whitespace-tokenized corpus for deterministic behavior."""
    return [
        Document("D1", "Lamport逻辑时钟", "lamport logical clock happens_before event ordering distributed system"),
        Document("D2", "向量时钟", "vector clock causal consistency concurrent events partial order"),
        Document("D3", "分布式快照", "chandy lamport snapshot consistent global state channel messages"),
        Document("D4", "CRDT简介", "crdt commutative replicated data type eventual consistency merge"),
        Document("D5", "Raft共识", "raft consensus leader election log replication commit"),
        Document("D6", "TF IDF检索", "tf idf vector space model text retrieval cosine similarity"),
        Document("D7", "BM25检索", "bm25 sparse retrieval term frequency inverse document frequency"),
        Document("D8", "并行矩阵乘法", "parallel matrix multiplication block partition gpu cpu speedup"),
    ]


def print_index_stats(vsm: VectorSpaceModel) -> None:
    if vsm.doc_tfidf is None:
        raise RuntimeError("index not built")

    n_docs, n_terms = vsm.doc_tfidf.shape
    nnz = int(vsm.doc_tfidf.nnz)
    density = nnz / float(n_docs * n_terms)
    print("=== Index Stats ===")
    print(f"documents={n_docs}, vocabulary_size={n_terms}, nnz={nnz}, density={density:.4f}")


def run_demo() -> None:
    corpus = build_demo_corpus()
    vsm = VectorSpaceModel()
    vsm.fit(corpus)

    print_index_stats(vsm)

    queries: list[tuple[str, set[str]]] = [
        ("vector clock concurrent events", {"D2"}),
        ("snapshot global state channel", {"D3"}),
        ("tf idf cosine retrieval", {"D6", "D7"}),
        ("log replication leader", {"D5"}),
    ]

    p_at_3_values: list[float] = []

    for idx, (query, relevant) in enumerate(queries, start=1):
        result = vsm.search(query, top_k=3)
        retrieved = result["doc_id"].tolist()
        p3 = precision_at_k(retrieved, relevant, 3)
        p_at_3_values.append(p3)

        print(f"\n=== Query {idx} ===")
        print(f"query: {query}")
        print(f"relevant: {sorted(relevant)}")
        print(result.to_string(index=False, formatters={"score": lambda x: f"{x:.4f}"}))
        print(f"Precision@3: {p3:.4f}")

    mean_p3 = float(np.mean(np.array(p_at_3_values, dtype=float)))
    print("\n=== Aggregate ===")
    print(f"Mean Precision@3: {mean_p3:.4f}")
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
