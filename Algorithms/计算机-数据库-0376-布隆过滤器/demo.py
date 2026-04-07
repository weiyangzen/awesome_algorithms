"""Bloom Filter minimal runnable MVP.

This demo implements a transparent Bloom Filter with:
- explicit bit-array updates
- deterministic double hashing
- empirical vs. theoretical false-positive evaluation
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class QueryRecord:
    key: str
    in_ground_truth: bool
    bloom_result: bool
    is_false_positive: bool


class BloomFilter:
    """A small Bloom Filter with numpy bit array and double hashing."""

    def __init__(self, bit_size: int, hash_count: int) -> None:
        if bit_size <= 0:
            raise ValueError("bit_size must be positive")
        if hash_count <= 0:
            raise ValueError("hash_count must be positive")

        self.bit_size = int(bit_size)
        self.hash_count = int(hash_count)
        self.bits = np.zeros(self.bit_size, dtype=np.uint8)

        self.insert_count = 0

    def _hash_pair(self, key: str) -> tuple[int, int]:
        key_bytes = key.encode("utf-8")

        h1 = int.from_bytes(
            hashlib.blake2b(key_bytes, digest_size=8, person=b"BFH1v1").digest(),
            byteorder="big",
            signed=False,
        )
        h2 = int.from_bytes(
            hashlib.blake2b(key_bytes, digest_size=8, person=b"BFH2v1").digest(),
            byteorder="big",
            signed=False,
        )

        # Avoid zero stride in double hashing.
        h2 = (h2 % self.bit_size) or 1
        return h1, h2

    def _positions(self, key: str) -> List[int]:
        h1, h2 = self._hash_pair(key)
        return [int((h1 + i * h2 + i * i) % self.bit_size) for i in range(self.hash_count)]

    def add(self, key: str) -> None:
        for pos in self._positions(key):
            self.bits[pos] = 1
        self.insert_count += 1

    def might_contain(self, key: str) -> bool:
        positions = self._positions(key)
        return bool(np.all(self.bits[positions] == 1))

    def bit_density(self) -> float:
        return float(self.bits.mean())


def choose_parameters(expected_items: int, target_fp_rate: float) -> tuple[int, int]:
    """Compute (m, k) from standard Bloom Filter formulas."""
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if not (0.0 < target_fp_rate < 1.0):
        raise ValueError("target_fp_rate must be in (0, 1)")

    ln2 = math.log(2.0)
    m = math.ceil(-(expected_items * math.log(target_fp_rate)) / (ln2 * ln2))
    k = max(1, round((m / expected_items) * ln2))
    return int(m), int(k)


def theoretical_fp_rate(item_count: int, bit_size: int, hash_count: int) -> float:
    """p ~= (1 - exp(-k*n/m))^k"""
    n = float(item_count)
    m = float(bit_size)
    k = float(hash_count)
    return float((1.0 - math.exp(-k * n / m)) ** k)


def evaluate_queries(
    bloom: BloomFilter,
    inserted: Sequence[str],
    negatives: Sequence[str],
) -> List[QueryRecord]:
    inserted_set = set(inserted)

    records: List[QueryRecord] = []
    for key in list(inserted) + list(negatives):
        in_truth = key in inserted_set
        pred = bloom.might_contain(key)
        records.append(
            QueryRecord(
                key=key,
                in_ground_truth=in_truth,
                bloom_result=pred,
                is_false_positive=(not in_truth and pred),
            )
        )
    return records


def main() -> None:
    print("=== Bloom Filter MVP Demo ===")

    expected_items = 250
    target_fp = 0.02
    bit_size, hash_count = choose_parameters(expected_items=expected_items, target_fp_rate=target_fp)

    bloom = BloomFilter(bit_size=bit_size, hash_count=hash_count)

    inserted = [f"user:{i:04d}" for i in range(expected_items)]
    negatives = [f"probe:{i:04d}" for i in range(3000)]

    for key in inserted:
        bloom.add(key)

    # Correctness check 1: Bloom Filter must have no false negatives.
    inserted_results = np.array([bloom.might_contain(k) for k in inserted], dtype=bool)
    if not bool(np.all(inserted_results)):
        raise AssertionError("Bloom Filter produced a false negative")

    # Correctness check 2: empirical false-positive rate should be near theory.
    negative_results = np.array([bloom.might_contain(k) for k in negatives], dtype=bool)
    fp_count = int(negative_results.sum())
    observed_fp = fp_count / len(negatives)
    expected_fp = theoretical_fp_rate(
        item_count=len(inserted),
        bit_size=bloom.bit_size,
        hash_count=bloom.hash_count,
    )

    tolerance = max(0.015, expected_fp * 0.60)
    if observed_fp > expected_fp + tolerance:
        raise AssertionError(
            f"observed FP rate too high: observed={observed_fp:.4f}, expected={expected_fp:.4f}, "
            f"tolerance={tolerance:.4f}"
        )

    # Correctness check 3: bit density should be close to theory: 1-exp(-k*n/m).
    observed_density = bloom.bit_density()
    expected_density = 1.0 - math.exp(-(bloom.hash_count * len(inserted)) / bloom.bit_size)
    if abs(observed_density - expected_density) > 0.08:
        raise AssertionError(
            f"unexpected bit density: observed={observed_density:.4f}, expected={expected_density:.4f}"
        )

    query_records = evaluate_queries(bloom=bloom, inserted=inserted[:12], negatives=negatives[:12])
    table = pd.DataFrame(
        {
            "key": [r.key for r in query_records],
            "in_ground_truth": [r.in_ground_truth for r in query_records],
            "bloom_result": [r.bloom_result for r in query_records],
            "is_false_positive": [r.is_false_positive for r in query_records],
        }
    )

    print(f"bit_size(m)={bloom.bit_size}, hash_count(k)={bloom.hash_count}")
    print(f"inserted_items(n)={len(inserted)}")
    print(f"target_fp_rate={target_fp:.4f}")
    print(f"theoretical_fp_rate={expected_fp:.4f}")
    print(f"observed_fp_rate={observed_fp:.4f} ({fp_count}/{len(negatives)})")
    print(f"expected_bit_density={expected_density:.4f}")
    print(f"observed_bit_density={observed_density:.4f}")

    print("\nSample query table:")
    with pd.option_context("display.width", 140):
        print(table.to_string(index=False))

    print("\nAll checks passed. Bloom Filter MVP is working.")


if __name__ == "__main__":
    main()
