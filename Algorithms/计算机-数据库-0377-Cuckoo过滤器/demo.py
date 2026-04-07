"""Cuckoo Filter minimal runnable MVP.

This script implements a compact, source-level Cuckoo Filter:
- supports insert / contains / delete
- measures load factor and false positive rate
- runs end-to-end without interactive input
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np


def _u64_hash(data: bytes) -> int:
    """Stable 64-bit hash from bytes."""
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _index_hash(key: str, num_buckets: int) -> int:
    return _u64_hash(b"i|" + key.encode("utf-8")) & (num_buckets - 1)


def _fingerprint_hash(key: str, fingerprint_bits: int) -> int:
    mask = (1 << fingerprint_bits) - 1
    fp = _u64_hash(b"f|" + key.encode("utf-8")) & mask
    return fp if fp != 0 else 1


def _alt_index(index: int, fingerprint: int, num_buckets: int) -> int:
    fp_bytes = fingerprint.to_bytes(8, byteorder="little", signed=False)
    mixed = _u64_hash(b"a|" + fp_bytes)
    return (index ^ mixed) & (num_buckets - 1)


@dataclass(frozen=True)
class CuckooConfig:
    capacity: int = 8_000
    bucket_size: int = 4
    fingerprint_bits: int = 12
    max_kicks: int = 500


class CuckooFilter:
    def __init__(self, config: CuckooConfig, rng: np.random.Generator | None = None) -> None:
        if config.capacity <= 0:
            raise ValueError("capacity must be positive")
        if config.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if config.fingerprint_bits < 2:
            raise ValueError("fingerprint_bits must be >= 2")
        if config.max_kicks <= 0:
            raise ValueError("max_kicks must be positive")

        self.capacity = config.capacity
        self.bucket_size = config.bucket_size
        self.fingerprint_bits = config.fingerprint_bits
        self.max_kicks = config.max_kicks

        raw_buckets = math.ceil(self.capacity / self.bucket_size)
        self.num_buckets = _next_power_of_two(raw_buckets)
        self.buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]
        self.size = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def _try_insert_fp(self, bucket_idx: int, fp: int) -> bool:
        bucket = self.buckets[bucket_idx]
        if len(bucket) < self.bucket_size:
            bucket.append(fp)
            return True
        return False

    def insert(self, key: str) -> bool:
        fp = _fingerprint_hash(key, self.fingerprint_bits)
        i1 = _index_hash(key, self.num_buckets)
        i2 = _alt_index(i1, fp, self.num_buckets)

        if self._try_insert_fp(i1, fp) or self._try_insert_fp(i2, fp):
            self.size += 1
            return True

        idx = i1 if self.rng.random() < 0.5 else i2
        cur_fp = fp
        for _ in range(self.max_kicks):
            if self._try_insert_fp(idx, cur_fp):
                self.size += 1
                return True

            slot = int(self.rng.integers(0, self.bucket_size))
            self.buckets[idx][slot], cur_fp = cur_fp, self.buckets[idx][slot]
            idx = _alt_index(idx, cur_fp, self.num_buckets)

        return False

    def contains(self, key: str) -> bool:
        fp = _fingerprint_hash(key, self.fingerprint_bits)
        i1 = _index_hash(key, self.num_buckets)
        i2 = _alt_index(i1, fp, self.num_buckets)
        return (fp in self.buckets[i1]) or (fp in self.buckets[i2])

    def delete(self, key: str) -> bool:
        fp = _fingerprint_hash(key, self.fingerprint_bits)
        i1 = _index_hash(key, self.num_buckets)
        i2 = _alt_index(i1, fp, self.num_buckets)
        for idx in (i1, i2):
            bucket = self.buckets[idx]
            for j, val in enumerate(bucket):
                if val == fp:
                    bucket.pop(j)
                    self.size -= 1
                    return True
        return False

    def load_factor(self) -> float:
        return self.size / (self.num_buckets * self.bucket_size)


def _to_rate(bits: Iterable[bool]) -> float:
    arr = np.fromiter((1 if x else 0 for x in bits), dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def main() -> None:
    rng = np.random.default_rng(20260407)
    config = CuckooConfig(capacity=8_000, bucket_size=4, fingerprint_bits=12, max_kicks=500)
    filt = CuckooFilter(config=config, rng=rng)

    # 1) insertion workload
    insert_keys = [f"user-{i}" for i in range(7_000)]
    insert_ok = 0
    for key in insert_keys:
        if filt.insert(key):
            insert_ok += 1
        else:
            break
    inserted = insert_keys[:insert_ok]

    # 2) false negative check on inserted keys
    false_negative_rate = _to_rate(not filt.contains(k) for k in inserted)

    # 3) delete a sample and verify deletion
    delete_count = min(1_000, len(inserted))
    delete_idx = rng.choice(len(inserted), size=delete_count, replace=False)
    delete_keys = [inserted[int(i)] for i in delete_idx]
    deleted_ok = sum(1 for k in delete_keys if filt.delete(k))
    deleted_still_present_rate = _to_rate(filt.contains(k) for k in delete_keys)

    # 4) false positive probe on disjoint keys
    probe_keys = [f"probe-{i}" for i in range(20_000)]
    false_positive_rate = _to_rate(filt.contains(k) for k in probe_keys)

    # textbook-style approximation: eps ~= 1 - (1 - 2^-f)^(2*b)
    f = config.fingerprint_bits
    b = config.bucket_size
    expected_fp = 1.0 - (1.0 - 2.0 ** (-f)) ** (2.0 * b)

    print("=== Cuckoo Filter MVP ===")
    print(f"capacity_target={config.capacity}")
    print(f"num_buckets={filt.num_buckets}, bucket_size={config.bucket_size}")
    print(f"fingerprint_bits={config.fingerprint_bits}, max_kicks={config.max_kicks}")
    print(f"inserted={insert_ok}, load_factor={filt.load_factor():.4f}")
    print(f"false_negative_rate={false_negative_rate:.6f}")
    print(
        f"deleted_ok={deleted_ok}/{delete_count}, "
        f"deleted_still_present_rate={deleted_still_present_rate:.6f}"
    )
    print(
        f"false_positive_rate={false_positive_rate:.6f}, "
        f"expected_approx={expected_fp:.6f}"
    )


if __name__ == "__main__":
    main()
