"""Hash Index minimal runnable MVP.

This demo shows how a database-style hash index can be implemented
with explicit buckets, collision handling, and rehashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class IndexEntry:
    key: Any
    row_ids: List[int]


@dataclass
class LookupResult:
    row_ids: List[int]
    probes: int


class HashIndex:
    """A small hash index using separate chaining."""

    def __init__(self, bucket_count: int = 4, load_factor_threshold: float = 0.75) -> None:
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if not (0 < load_factor_threshold < 1.0):
            raise ValueError("load_factor_threshold must be in (0, 1)")

        self.bucket_count = bucket_count
        self.load_factor_threshold = load_factor_threshold
        self.buckets: List[List[IndexEntry]] = [[] for _ in range(bucket_count)]

        # Number of distinct keys stored in index entries.
        self.entry_count = 0

        # Statistics for observability.
        self.rehash_count = 0
        self.collision_inserts = 0

    def _bucket_id(self, key: Any) -> int:
        return hash(key) % self.bucket_count

    def _load_factor(self) -> float:
        return self.entry_count / self.bucket_count

    def _rehash(self, new_bucket_count: int) -> None:
        old_buckets = self.buckets
        self.bucket_count = new_bucket_count
        self.buckets = [[] for _ in range(new_bucket_count)]

        for bucket in old_buckets:
            for entry in bucket:
                new_id = self._bucket_id(entry.key)
                self.buckets[new_id].append(IndexEntry(entry.key, list(entry.row_ids)))

        self.rehash_count += 1

    def insert(self, key: Any, row_id: int) -> None:
        if key is None:
            raise ValueError("index key cannot be None")

        b_id = self._bucket_id(key)
        bucket = self.buckets[b_id]

        for entry in bucket:
            if entry.key == key:
                entry.row_ids.append(row_id)
                return

        if bucket:
            self.collision_inserts += 1

        bucket.append(IndexEntry(key=key, row_ids=[row_id]))
        self.entry_count += 1

        if self._load_factor() > self.load_factor_threshold:
            self._rehash(self.bucket_count * 2)

    def lookup(self, key: Any) -> LookupResult:
        if key is None:
            raise ValueError("lookup key cannot be None")

        b_id = self._bucket_id(key)
        bucket = self.buckets[b_id]
        probes = 0

        for entry in bucket:
            probes += 1
            if entry.key == key:
                return LookupResult(row_ids=list(entry.row_ids), probes=probes)

        return LookupResult(row_ids=[], probes=probes)

    def delete(self, key: Any, row_id: int) -> bool:
        if key is None:
            raise ValueError("delete key cannot be None")

        b_id = self._bucket_id(key)
        bucket = self.buckets[b_id]

        for idx, entry in enumerate(bucket):
            if entry.key != key:
                continue

            if row_id not in entry.row_ids:
                return False

            entry.row_ids.remove(row_id)
            if not entry.row_ids:
                del bucket[idx]
                self.entry_count -= 1
            return True

        return False

    def build(self, records: Sequence[Dict[str, Any]], key_column: str) -> None:
        for row_id, row in enumerate(records):
            self.insert(row.get(key_column), row_id)

    def bucket_histogram(self) -> List[int]:
        return [len(bucket) for bucket in self.buckets]


def linear_scan(records: Sequence[Dict[str, Any]], key_column: str, key: Any) -> List[int]:
    return [i for i, row in enumerate(records) if row.get(key_column) == key]


def fetch_rows(records: Sequence[Dict[str, Any]], row_ids: Sequence[int]) -> List[Dict[str, Any]]:
    return [records[i] for i in row_ids]


def main() -> None:
    orders: List[Dict[str, Any]] = [
        {"order_id": 1, "customer_id": 101, "amount": 120.0},
        {"order_id": 2, "customer_id": 102, "amount": 80.0},
        {"order_id": 3, "customer_id": 101, "amount": 15.5},
        {"order_id": 4, "customer_id": 103, "amount": 240.0},
        {"order_id": 5, "customer_id": 104, "amount": 55.0},
        {"order_id": 6, "customer_id": 101, "amount": 33.0},
        {"order_id": 7, "customer_id": 105, "amount": 9.9},
    ]

    index = HashIndex(bucket_count=4, load_factor_threshold=0.75)
    index.build(orders, key_column="customer_id")

    # Equality query: customer_id=101 should return 3 rows.
    key = 101
    lookup_101 = index.lookup(key)
    scan_101 = linear_scan(orders, "customer_id", key)
    assert sorted(lookup_101.row_ids) == sorted(scan_101)

    rows_101 = fetch_rows(orders, lookup_101.row_ids)
    assert len(rows_101) == 3

    # Query for absent key.
    missing = 999
    lookup_missing = index.lookup(missing)
    scan_missing = linear_scan(orders, "customer_id", missing)
    assert lookup_missing.row_ids == scan_missing == []

    # Insert a new order and update index incrementally.
    new_row = {"order_id": 8, "customer_id": 102, "amount": 61.0}
    orders.append(new_row)
    new_row_id = len(orders) - 1
    index.insert(new_row["customer_id"], new_row_id)

    lookup_102 = index.lookup(102)
    scan_102 = linear_scan(orders, "customer_id", 102)
    assert sorted(lookup_102.row_ids) == sorted(scan_102)
    assert len(lookup_102.row_ids) == 2

    # Delete one indexed row and verify consistency.
    deleted = index.delete(101, 2)
    assert deleted is True

    lookup_101_after_delete = index.lookup(101)
    expected_after_delete = [rid for rid in linear_scan(orders, "customer_id", 101) if rid != 2]
    assert sorted(lookup_101_after_delete.row_ids) == sorted(expected_after_delete)

    print("== Hash Index MVP ==")
    print(f"bucket_count={index.bucket_count}")
    print(f"entry_count={index.entry_count}")
    print(f"load_factor={index.entry_count / index.bucket_count:.3f}")
    print(f"collision_inserts={index.collision_inserts}")
    print(f"rehash_count={index.rehash_count}")
    print(f"bucket_histogram={index.bucket_histogram()}")

    print("\nQuery: customer_id=101")
    print(f"row_ids={lookup_101_after_delete.row_ids}, probes={lookup_101_after_delete.probes}")
    print(f"rows={fetch_rows(orders, lookup_101_after_delete.row_ids)}")

    print("\nQuery: customer_id=102")
    print(f"row_ids={lookup_102.row_ids}, probes={lookup_102.probes}")
    print(f"rows={fetch_rows(orders, lookup_102.row_ids)}")

    print("\nQuery: customer_id=999")
    print(f"row_ids={lookup_missing.row_ids}, probes={lookup_missing.probes}")

    print("\nAll assertions passed. Hash Index MVP is working.")


if __name__ == "__main__":
    main()
