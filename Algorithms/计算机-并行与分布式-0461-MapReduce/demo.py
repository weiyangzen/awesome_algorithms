"""Minimal runnable MVP for MapReduce (CS-0301).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import re

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-zA-Z]+")


@dataclass
class MapperStats:
    """Statistics of one mapper task."""

    mapper_id: int
    doc_ids: list[int]
    raw_emit_count: int
    combined_emit_count: int


@dataclass
class ReducerStats:
    """Statistics of one reducer task."""

    reducer_id: int
    input_key_count: int
    input_value_count: int
    output_key_count: int


@dataclass
class MapReduceResult:
    """Container for deterministic MVP outputs and traces."""

    final_counts: dict[str, int]
    mapper_stats: list[MapperStats]
    reducer_stats: list[ReducerStats]
    shuffle_buckets: list[dict[str, list[int]]]
    raw_emit_total: int
    combined_emit_total: int


def tokenize(text: str) -> list[str]:
    """Lowercase + alphabetic tokenization for stable word-count examples."""
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def stable_partition(key: str, num_reducers: int) -> int:
    """Deterministic partition id from key (independent of Python hash randomization)."""
    if num_reducers <= 0:
        raise ValueError("num_reducers must be >= 1")
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % num_reducers


def split_document_shards(documents: list[str], num_mappers: int) -> list[list[int]]:
    """Split document indices into mapper shards using numpy array split."""
    if num_mappers <= 0:
        raise ValueError("num_mappers must be >= 1")
    if not documents:
        raise ValueError("documents must not be empty")

    indices = np.arange(len(documents), dtype=np.int64)
    shards = np.array_split(indices, num_mappers)
    return [shard.tolist() for shard in shards]


def map_with_combiner(
    mapper_id: int,
    shard_doc_ids: list[int],
    documents: list[str],
) -> tuple[dict[str, int], MapperStats]:
    """Map phase with local combiner: (word, 1) -> local (word, partial_count)."""
    local_counter: Counter[str] = Counter()
    raw_emit_count = 0

    for doc_id in shard_doc_ids:
        tokens = tokenize(documents[doc_id])
        raw_emit_count += len(tokens)
        local_counter.update(tokens)

    combined_counts = dict(sorted((word, int(cnt)) for word, cnt in local_counter.items()))
    stats = MapperStats(
        mapper_id=mapper_id,
        doc_ids=shard_doc_ids,
        raw_emit_count=raw_emit_count,
        combined_emit_count=len(combined_counts),
    )
    return combined_counts, stats


def shuffle_sort(
    mapper_outputs: list[dict[str, int]],
    num_reducers: int,
) -> list[dict[str, list[int]]]:
    """Shuffle + sort: group mapper partial counts by reducer and key."""
    buckets: list[defaultdict[str, list[int]]] = [defaultdict(list) for _ in range(num_reducers)]

    for partial_counts in mapper_outputs:
        for key, partial_count in partial_counts.items():
            reducer_id = stable_partition(key, num_reducers)
            buckets[reducer_id][key].append(int(partial_count))

    sorted_buckets: list[dict[str, list[int]]] = []
    for bucket in buckets:
        ordered_bucket = {key: bucket[key] for key in sorted(bucket.keys())}
        sorted_buckets.append(ordered_bucket)
    return sorted_buckets


def reduce_bucket(reducer_id: int, bucket: dict[str, list[int]]) -> tuple[dict[str, int], ReducerStats]:
    """Reduce phase: aggregate partial counts for each key."""
    reduced_counts: dict[str, int] = {}

    for key, partial_values in bucket.items():
        reduced_counts[key] = int(np.sum(np.asarray(partial_values, dtype=np.int64)))

    stats = ReducerStats(
        reducer_id=reducer_id,
        input_key_count=len(bucket),
        input_value_count=sum(len(values) for values in bucket.values()),
        output_key_count=len(reduced_counts),
    )
    return reduced_counts, stats


def direct_word_count(documents: list[str]) -> dict[str, int]:
    """Sequential baseline for correctness validation."""
    counter: Counter[str] = Counter()
    for text in documents:
        counter.update(tokenize(text))
    return dict(sorted((word, int(cnt)) for word, cnt in counter.items()))


def run_mapreduce(
    documents: list[str],
    num_mappers: int = 3,
    num_reducers: int = 2,
) -> MapReduceResult:
    """Run a deterministic MapReduce word-count job end-to-end."""
    if num_mappers <= 0:
        raise ValueError("num_mappers must be >= 1")
    if num_reducers <= 0:
        raise ValueError("num_reducers must be >= 1")
    if not documents:
        raise ValueError("documents must not be empty")

    shards = split_document_shards(documents, num_mappers)

    mapper_outputs: list[dict[str, int]] = []
    mapper_stats: list[MapperStats] = []
    for mapper_id, shard_doc_ids in enumerate(shards):
        partial_counts, stats = map_with_combiner(mapper_id, shard_doc_ids, documents)
        mapper_outputs.append(partial_counts)
        mapper_stats.append(stats)

    raw_emit_total = sum(s.raw_emit_count for s in mapper_stats)
    combined_emit_total = sum(s.combined_emit_count for s in mapper_stats)

    shuffle_buckets = shuffle_sort(mapper_outputs, num_reducers)

    reducer_outputs: list[dict[str, int]] = []
    reducer_stats: list[ReducerStats] = []
    for reducer_id, bucket in enumerate(shuffle_buckets):
        reduced_counts, stats = reduce_bucket(reducer_id, bucket)
        reducer_outputs.append(reduced_counts)
        reducer_stats.append(stats)

    merged: dict[str, int] = {}
    for reduced_counts in reducer_outputs:
        for key, value in reduced_counts.items():
            if key in merged:
                raise AssertionError(f"Key {key!r} appears in multiple reducers")
            merged[key] = int(value)
    final_counts = dict(sorted(merged.items()))

    # Correctness checks: partition invariants + baseline equality.
    for reducer_id, bucket in enumerate(shuffle_buckets):
        for key in bucket.keys():
            if stable_partition(key, num_reducers) != reducer_id:
                raise AssertionError(f"Partition mismatch for key={key}, reducer={reducer_id}")

    baseline = direct_word_count(documents)
    if final_counts != baseline:
        raise AssertionError("MapReduce result mismatches sequential baseline")

    if combined_emit_total > raw_emit_total:
        raise AssertionError("Combiner should not increase intermediate pair count")

    return MapReduceResult(
        final_counts=final_counts,
        mapper_stats=mapper_stats,
        reducer_stats=reducer_stats,
        shuffle_buckets=shuffle_buckets,
        raw_emit_total=raw_emit_total,
        combined_emit_total=combined_emit_total,
    )


def mapper_stats_frame(stats: list[MapperStats]) -> pd.DataFrame:
    """Pretty table for mapper-level diagnostics."""
    rows = [
        {
            "mapper": s.mapper_id,
            "doc_ids": s.doc_ids,
            "raw_emit": s.raw_emit_count,
            "after_combiner": s.combined_emit_count,
        }
        for s in stats
    ]
    return pd.DataFrame(rows)


def reducer_stats_frame(stats: list[ReducerStats]) -> pd.DataFrame:
    """Pretty table for reducer-level diagnostics."""
    rows = [
        {
            "reducer": s.reducer_id,
            "input_keys": s.input_key_count,
            "partial_values": s.input_value_count,
            "output_keys": s.output_key_count,
        }
        for s in stats
    ]
    return pd.DataFrame(rows)


def result_frame(final_counts: dict[str, int]) -> pd.DataFrame:
    """Sorted frequency table."""
    df = pd.DataFrame(
        [{"word": word, "count": count} for word, count in final_counts.items()]
    )
    return df.sort_values(by=["count", "word"], ascending=[False, True], ignore_index=True)


def main() -> None:
    print("=== MapReduce MVP: Word Count (CS-0301) ===")

    documents = [
        "MapReduce splits large data tasks into mapper and reducer stages.",
        "Each mapper emits key value pairs from local data blocks.",
        "Shuffle groups records with the same key before reduce.",
        "A reducer aggregates counts and writes deterministic output.",
        "MapReduce tolerates worker failure by re running tasks.",
        "Combiner reduces network traffic for repetitive key emissions.",
        "This demo implements MapReduce without distributed runtime.",
        "The algorithm remains useful for log analytics and indexing.",
        "MapReduce style thinking still appears in modern data systems.",
        "Reliable systems monitor mapper latency and reducer skew.",
    ]
    num_mappers = 3
    num_reducers = 2

    result = run_mapreduce(documents, num_mappers=num_mappers, num_reducers=num_reducers)
    word_count_df = result_frame(result.final_counts)

    reduction_ratio = 0.0
    if result.raw_emit_total > 0:
        reduction_ratio = 1.0 - (result.combined_emit_total / result.raw_emit_total)

    print(f"Documents: {len(documents)} | mappers={num_mappers} | reducers={num_reducers}")
    print(
        f"Intermediate pairs: raw={result.raw_emit_total}, "
        f"after_combiner={result.combined_emit_total}, "
        f"reduction={reduction_ratio:.2%}"
    )

    print("\nMapper stats:")
    print(mapper_stats_frame(result.mapper_stats).to_string(index=False))

    print("\nReducer stats:")
    print(reducer_stats_frame(result.reducer_stats).to_string(index=False))

    print("\nTop 12 words:")
    print(word_count_df.head(12).to_string(index=False))

    print("\nAll checks passed for CS-0301 (MapReduce).")


if __name__ == "__main__":
    main()
