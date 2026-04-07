"""Minimal runnable MVP for TF-IDF (CS-0320).

Implementation scope:
- Build TF and DF statistics with a map-reduce style parallel map stage
- Compute smoothed IDF and L2-normalized TF-IDF with SciPy sparse matrices
- Validate numerical consistency against scikit-learn (not used as algorithm black box)
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class TfidfReport:
    n_documents: int
    vocab_size: int
    nnz: int
    num_partitions: int
    max_abs_diff_vs_sklearn: float


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _map_partition(
    partition: list[tuple[int, list[str]]],
) -> tuple[list[tuple[int, Counter[str]]], Counter[str]]:
    """Map phase on one partition: term count per document + local document frequency."""

    local_doc_counts: list[tuple[int, Counter[str]]] = []
    local_df: Counter[str] = Counter()

    for doc_id, tokens in partition:
        counts = Counter(tokens)
        local_doc_counts.append((doc_id, counts))
        local_df.update(counts.keys())

    return local_doc_counts, local_df


class MapReduceTfidf:
    """Small, auditable TF-IDF implementation."""

    def __init__(self, num_partitions: int = 3, num_workers: int = 2) -> None:
        if num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        self.num_partitions = num_partitions
        self.num_workers = num_workers

        self.vocabulary_: list[str] | None = None
        self.term_to_id_: dict[str, int] | None = None
        self.idf_: np.ndarray | None = None
        self.matrix_: sparse.csr_matrix | None = None

    def fit_transform(self, tokenized_docs: list[list[str]]) -> sparse.csr_matrix:
        if not tokenized_docs:
            raise ValueError("tokenized_docs must not be empty")
        if any(len(tokens) == 0 for tokens in tokenized_docs):
            raise ValueError("every document must contain at least one token")

        doc_term_counts, doc_freq = self._build_statistics_map_reduce(tokenized_docs)
        if not doc_freq:
            raise RuntimeError("empty vocabulary after statistics aggregation")

        vocabulary = sorted(doc_freq.keys())
        term_to_id = {term: idx for idx, term in enumerate(vocabulary)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for doc_id, counts in enumerate(doc_term_counts):
            for term, count in counts.items():
                rows.append(doc_id)
                cols.append(term_to_id[term])
                data.append(float(count))

        tf = sparse.csr_matrix(
            (np.array(data, dtype=np.float64), (rows, cols)),
            shape=(len(tokenized_docs), len(vocabulary)),
            dtype=np.float64,
        )

        df_vector = np.array([doc_freq[term] for term in vocabulary], dtype=np.float64)
        idf = np.log((1.0 + len(tokenized_docs)) / (1.0 + df_vector)) + 1.0

        tfidf = tf.multiply(idf)
        tfidf = self._l2_normalize_rows(tfidf.tocsr())

        self.vocabulary_ = vocabulary
        self.term_to_id_ = term_to_id
        self.idf_ = idf
        self.matrix_ = tfidf
        return tfidf

    def _build_statistics_map_reduce(
        self,
        tokenized_docs: list[list[str]],
    ) -> tuple[list[Counter[str]], Counter[str]]:
        indexed_docs = list(enumerate(tokenized_docs))
        shard_indices = np.array_split(np.arange(len(indexed_docs)), self.num_partitions)
        partitions: list[list[tuple[int, list[str]]]] = []

        for shard in shard_indices:
            if shard.size == 0:
                continue
            partitions.append([indexed_docs[int(i)] for i in shard])

        if not partitions:
            raise RuntimeError("no partitions built for map-reduce")

        if self.num_workers == 1 or len(partitions) == 1:
            mapped_results = [_map_partition(partition) for partition in partitions]
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                mapped_results = list(executor.map(_map_partition, partitions))

        doc_term_counts: list[Counter[str]] = [Counter() for _ in tokenized_docs]
        global_df: Counter[str] = Counter()

        for local_docs, local_df in mapped_results:
            for doc_id, counts in local_docs:
                doc_term_counts[doc_id] = counts
            global_df.update(local_df)

        return doc_term_counts, global_df

    @staticmethod
    def _l2_normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        inv = sparse.diags(1.0 / safe_norms)
        return (inv @ matrix).tocsr()


def build_demo_corpus() -> list[dict[str, str]]:
    """Deterministic corpus with distributed-system flavored documents."""

    return [
        {
            "title": "MapReduce Primer",
            "text": (
                "distributed map reduce cluster worker shard partition "
                "fault tolerance scheduler"
            ),
        },
        {
            "title": "Spark Pipeline",
            "text": (
                "distributed spark dag executor partition cache shuffle "
                "cluster scheduler"
            ),
        },
        {
            "title": "Consensus Notes",
            "text": (
                "distributed consensus raft paxos quorum leader election "
                "log replication"
            ),
        },
        {
            "title": "Storage Replication",
            "text": (
                "distributed storage replication erasure coding availability "
                "consistency quorum"
            ),
        },
        {
            "title": "Graph Processing",
            "text": (
                "parallel graph processing distributed vertex edge partition "
                "message passing"
            ),
        },
        {
            "title": "TFIDF Search",
            "text": (
                "document retrieval tfidf ranking sparse vector idf term "
                "frequency cosine"
            ),
        },
        {
            "title": "Image Training",
            "text": (
                "image convolution gpu tensor batch gradient optimizer "
                "learning rate"
            ),
        },
        {
            "title": "Sharded Database",
            "text": (
                "distributed database sharding transaction replica read write "
                "consistency"
            ),
        },
    ]


def top_terms_table(
    matrix: sparse.csr_matrix,
    vocabulary: list[str],
    titles: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []

    for doc_id in range(matrix.shape[0]):
        row = matrix.getrow(doc_id)
        if row.nnz == 0:
            terms_text = ""
        else:
            cols = row.indices
            vals = row.data
            order = np.argsort(vals)[::-1][:top_k]
            terms_text = ", ".join(
                f"{vocabulary[int(cols[i])]}:{vals[i]:.3f}" for i in order
            )

        rows.append(
            {
                "doc_id": doc_id,
                "title": titles[doc_id],
                "top_terms": terms_text,
            }
        )

    return pd.DataFrame(rows)


def cosine_similarity_rows(
    matrix: sparse.csr_matrix,
    row_a: int,
    row_b: int,
) -> float:
    value = matrix.getrow(row_a).multiply(matrix.getrow(row_b)).sum()
    return float(value)


def max_abs_diff(a: sparse.csr_matrix, b: sparse.csr_matrix) -> float:
    delta = (a - b).tocoo()
    if delta.nnz == 0:
        return 0.0
    return float(np.max(np.abs(delta.data)))


def validate_against_sklearn(
    tokenized_docs: list[list[str]],
    vocabulary: list[str],
    reference: sparse.csr_matrix,
) -> float:
    joined_docs = [" ".join(tokens) for tokens in tokenized_docs]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        vocabulary=vocabulary,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
    )
    sklearn_matrix = vectorizer.fit_transform(joined_docs).tocsr()
    return max_abs_diff(reference, sklearn_matrix)


def run_quality_checks(
    model: MapReduceTfidf,
    matrix: sparse.csr_matrix,
    tokenized_docs: list[list[str]],
    diff_vs_sklearn: float,
) -> None:
    if model.vocabulary_ is None or model.term_to_id_ is None or model.idf_ is None:
        raise RuntimeError("model must be fitted before quality checks")

    if matrix.shape[0] != len(tokenized_docs):
        raise AssertionError("document count mismatch")
    if matrix.shape[1] != len(model.vocabulary_):
        raise AssertionError("vocabulary size mismatch")
    if not np.all(np.isfinite(matrix.data)):
        raise AssertionError("non-finite values in tf-idf matrix")

    term_to_id = model.term_to_id_
    idf = model.idf_

    if "consensus" not in term_to_id or "distributed" not in term_to_id:
        raise AssertionError("expected control terms missing from vocabulary")

    if not idf[term_to_id["consensus"]] > idf[term_to_id["distributed"]]:
        raise AssertionError("idf should be higher for rarer term: consensus")

    sim_mapreduce_spark = cosine_similarity_rows(matrix, 0, 1)
    sim_mapreduce_image = cosine_similarity_rows(matrix, 0, 6)
    if not sim_mapreduce_spark > sim_mapreduce_image:
        raise AssertionError("semantic similarity check failed")

    if diff_vs_sklearn > 1e-12:
        raise AssertionError(f"implementation diverges from sklearn baseline: {diff_vs_sklearn:.3e}")


def main() -> None:
    corpus = build_demo_corpus()
    titles = [item["title"] for item in corpus]
    tokenized_docs = [tokenize(item["text"]) for item in corpus]

    model = MapReduceTfidf(num_partitions=3, num_workers=2)
    tfidf_matrix = model.fit_transform(tokenized_docs)

    assert model.vocabulary_ is not None

    diff_vs_sklearn = validate_against_sklearn(
        tokenized_docs=tokenized_docs,
        vocabulary=model.vocabulary_,
        reference=tfidf_matrix,
    )

    report = TfidfReport(
        n_documents=len(tokenized_docs),
        vocab_size=len(model.vocabulary_),
        nnz=int(tfidf_matrix.nnz),
        num_partitions=model.num_partitions,
        max_abs_diff_vs_sklearn=diff_vs_sklearn,
    )

    print("TF-IDF Map-Reduce MVP Report")
    print(f"- Documents: {report.n_documents}")
    print(f"- Vocabulary size: {report.vocab_size}")
    print(f"- Non-zero entries: {report.nnz}")
    print(f"- Partitions: {report.num_partitions}")
    print(f"- Max abs diff vs sklearn: {report.max_abs_diff_vs_sklearn:.3e}")

    top_df = top_terms_table(tfidf_matrix, model.vocabulary_, titles, top_k=5)
    print("\nTop TF-IDF terms per document:")
    print(top_df.to_string(index=False))

    run_quality_checks(
        model=model,
        matrix=tfidf_matrix,
        tokenized_docs=tokenized_docs,
        diff_vs_sklearn=diff_vs_sklearn,
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
