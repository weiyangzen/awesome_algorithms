"""MemTable minimal runnable MVP.

This script demonstrates a transparent MemTable implementation with:
- versioned PUT/DEL records
- snapshot reads
- sorted range scans
- immutable-run export for flush preparation
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

OpType = Literal["PUT", "DEL"]


@dataclass(frozen=True)
class VersionedRecord:
    seq: int
    op: OpType
    value: Optional[str]


@dataclass(frozen=True)
class OperationRecord:
    seq: int
    op: OpType
    key: str
    value: Optional[str]


class MemTable:
    """A compact educational MemTable with version chains per key."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)

        self._versions: dict[str, list[VersionedRecord]] = {}
        self._sorted_keys: list[str] = []
        self._seq = 0
        self._entry_count = 0
        self._approx_bytes = 0

    @property
    def current_seq(self) -> int:
        return self._seq

    @property
    def entry_count(self) -> int:
        return self._entry_count

    @property
    def key_count(self) -> int:
        return len(self._sorted_keys)

    @property
    def approx_bytes(self) -> int:
        return self._approx_bytes

    @staticmethod
    def _estimate_record_bytes(key: str, value: Optional[str]) -> int:
        # A rough accounting for demo purposes (metadata + payload).
        return 24 + len(key.encode("utf-8")) + (0 if value is None else len(value.encode("utf-8")))

    def _append_record(self, key: str, op: OpType, value: Optional[str]) -> int:
        self._seq += 1
        if key not in self._versions:
            bisect.insort(self._sorted_keys, key)
            self._versions[key] = []

        self._versions[key].append(VersionedRecord(seq=self._seq, op=op, value=value))
        self._entry_count += 1
        self._approx_bytes += self._estimate_record_bytes(key, value)
        return self._seq

    def put(self, key: str, value: str) -> int:
        return self._append_record(key=key, op="PUT", value=value)

    def delete(self, key: str) -> int:
        return self._append_record(key=key, op="DEL", value=None)

    def _latest_record_leq(self, key: str, snapshot_seq: Optional[int]) -> Optional[VersionedRecord]:
        if key not in self._versions:
            return None

        if snapshot_seq is None:
            snapshot_seq = self._seq
        if snapshot_seq <= 0:
            return None

        versions = self._versions[key]
        for rec in reversed(versions):
            if rec.seq <= snapshot_seq:
                return rec
        return None

    def get(self, key: str, snapshot_seq: Optional[int] = None) -> Optional[str]:
        rec = self._latest_record_leq(key=key, snapshot_seq=snapshot_seq)
        if rec is None or rec.op == "DEL":
            return None
        return rec.value

    def range_scan(
        self,
        start_key: str,
        end_key: str,
        snapshot_seq: Optional[int] = None,
    ) -> list[tuple[str, str]]:
        if start_key > end_key:
            raise ValueError("start_key must be <= end_key")

        if snapshot_seq is None:
            snapshot_seq = self._seq

        left = bisect.bisect_left(self._sorted_keys, start_key)
        right = bisect.bisect_right(self._sorted_keys, end_key)

        visible: list[tuple[str, str]] = []
        for key in self._sorted_keys[left:right]:
            rec = self._latest_record_leq(key=key, snapshot_seq=snapshot_seq)
            if rec is not None and rec.op == "PUT" and rec.value is not None:
                visible.append((key, rec.value))
        return visible

    def freeze_to_immutable_run(
        self,
        snapshot_seq: Optional[int] = None,
    ) -> list[tuple[str, OpType, Optional[str], int]]:
        """Return one latest record per key (including tombstones) in sorted-key order."""
        if snapshot_seq is None:
            snapshot_seq = self._seq

        run: list[tuple[str, OpType, Optional[str], int]] = []
        for key in self._sorted_keys:
            rec = self._latest_record_leq(key=key, snapshot_seq=snapshot_seq)
            if rec is not None:
                run.append((key, rec.op, rec.value, rec.seq))
        return run

    def should_flush(self) -> bool:
        return self._entry_count >= self.max_entries or self._approx_bytes >= self.max_bytes

    def stats(self) -> dict[str, float | int]:
        version_counts = np.array([len(v) for v in self._versions.values()], dtype=np.int64)
        avg_versions = float(version_counts.mean()) if version_counts.size else 0.0
        max_versions = int(version_counts.max()) if version_counts.size else 0
        return {
            "current_seq": self._seq,
            "entry_count": self._entry_count,
            "key_count": self.key_count,
            "approx_bytes": self._approx_bytes,
            "avg_versions_per_key": avg_versions,
            "max_versions_per_key": max_versions,
        }


def reference_get(history: Sequence[OperationRecord], key: str, snapshot_seq: int) -> Optional[str]:
    if snapshot_seq <= 0:
        return None
    for rec in reversed(history):
        if rec.seq <= snapshot_seq and rec.key == key:
            return rec.value if rec.op == "PUT" else None
    return None


def reference_range(
    history: Sequence[OperationRecord],
    start_key: str,
    end_key: str,
    snapshot_seq: int,
) -> list[tuple[str, str]]:
    state: dict[str, str] = {}
    for rec in history:
        if rec.seq > snapshot_seq:
            break
        if rec.op == "PUT" and rec.value is not None:
            state[rec.key] = rec.value
        else:
            state.pop(rec.key, None)
    return [(k, v) for k, v in sorted(state.items()) if start_key <= k <= end_key]


def apply_workload_and_validate(
    memtable: MemTable,
    workload: Sequence[tuple[OpType, str, Optional[str]]],
) -> list[OperationRecord]:
    history: list[OperationRecord] = []
    observed_keys: set[str] = set()

    for op, key, value in workload:
        if op == "PUT":
            if value is None:
                raise ValueError("PUT must have non-None value")
            seq = memtable.put(key=key, value=value)
        else:
            seq = memtable.delete(key=key)

        history.append(OperationRecord(seq=seq, op=op, key=key, value=value))
        observed_keys.add(key)

        # Validate current snapshot against a reference model.
        for probe_key in sorted(observed_keys):
            expected = reference_get(history=history, key=probe_key, snapshot_seq=seq)
            got = memtable.get(key=probe_key, snapshot_seq=seq)
            if got != expected:
                raise AssertionError(
                    f"mismatch at seq={seq}, key={probe_key}: expected={expected}, got={got}"
                )

        # Validate previous snapshot (if exists) for the touched key.
        if seq > 1:
            expected_prev = reference_get(history=history, key=key, snapshot_seq=seq - 1)
            got_prev = memtable.get(key=key, snapshot_seq=seq - 1)
            if got_prev != expected_prev:
                raise AssertionError(
                    f"snapshot mismatch at seq={seq - 1}, key={key}: "
                    f"expected={expected_prev}, got={got_prev}"
                )

    return history


def build_workload() -> list[tuple[OpType, str, Optional[str]]]:
    return [
        ("PUT", "user:001", "alice"),
        ("PUT", "user:002", "bob"),
        ("PUT", "user:003", "carol"),
        ("PUT", "user:002", "bobby"),
        ("DEL", "user:003", None),
        ("PUT", "user:004", "dave"),
        ("PUT", "user:010", "zoe"),
        ("DEL", "user:999", None),
        ("PUT", "user:003", "carol_v2"),
    ]


def main() -> None:
    print("=== MemTable MVP Demo ===")

    memtable = MemTable(max_entries=9, max_bytes=4096)
    workload = build_workload()
    history = apply_workload_and_validate(memtable=memtable, workload=workload)

    latest_seq = memtable.current_seq
    delete_seq = next(rec.seq for rec in history if rec.op == "DEL" and rec.key == "user:003")

    # Tombstone and snapshot checks.
    if memtable.get("user:003", snapshot_seq=delete_seq) is not None:
        raise AssertionError("user:003 should be deleted at delete snapshot")
    if memtable.get("user:003", snapshot_seq=delete_seq - 1) != "carol":
        raise AssertionError("user:003 should be visible as carol before deletion")
    if memtable.get("user:003", snapshot_seq=latest_seq) != "carol_v2":
        raise AssertionError("user:003 should be resurrected as carol_v2 at latest snapshot")

    visible_range = memtable.range_scan("user:000", "user:999", snapshot_seq=latest_seq)
    expected_range = reference_range(history, "user:000", "user:999", snapshot_seq=latest_seq)
    if visible_range != expected_range:
        raise AssertionError("range_scan result mismatches reference model")

    immutable_run = memtable.freeze_to_immutable_run(snapshot_seq=latest_seq)
    immutable_keys = [row[0] for row in immutable_run]
    if immutable_keys != sorted(immutable_keys):
        raise AssertionError("immutable run keys must be globally sorted")

    run_visible = [(k, v) for k, op, v, _ in immutable_run if op == "PUT" and v is not None]
    if run_visible != visible_range:
        raise AssertionError("visible rows in immutable run should match latest range_scan")

    if not memtable.should_flush():
        raise AssertionError("expected flush trigger was not reached")

    stats = memtable.stats()

    operations_df = pd.DataFrame(
        {
            "seq": [r.seq for r in history],
            "op": [r.op for r in history],
            "key": [r.key for r in history],
            "value": [r.value for r in history],
        }
    )
    immutable_df = pd.DataFrame(immutable_run, columns=["key", "op", "value", "seq"])
    visible_df = pd.DataFrame(visible_range, columns=["key", "value"])

    print(
        "stats:",
        (
            f"current_seq={stats['current_seq']}, entry_count={stats['entry_count']}, "
            f"key_count={stats['key_count']}, approx_bytes={stats['approx_bytes']}, "
            f"avg_versions_per_key={stats['avg_versions_per_key']:.2f}, "
            f"max_versions_per_key={stats['max_versions_per_key']}"
        ),
    )
    print(f"should_flush={memtable.should_flush()} (max_entries=9, max_bytes=4096)")

    print("\nOperation log:")
    with pd.option_context("display.width", 140):
        print(operations_df.to_string(index=False))

    print("\nImmutable run (flush candidate):")
    with pd.option_context("display.width", 140):
        print(immutable_df.to_string(index=False))

    print("\nVisible range [user:000, user:999]:")
    with pd.option_context("display.width", 140):
        print(visible_df.to_string(index=False))

    print("\nAll checks passed. MemTable MVP is working.")


if __name__ == "__main__":
    main()
