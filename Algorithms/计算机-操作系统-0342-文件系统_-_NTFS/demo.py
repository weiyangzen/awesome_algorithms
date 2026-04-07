"""Simplified NTFS MVP: MFT + bitmap + runlist + journal recovery."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class JournalEntry:
    lsn: int
    txid: int
    op: str
    payload: dict[str, Any]


@dataclass
class MFTRecord:
    file_ref: int
    name: str
    size: int
    resident: bool
    resident_data: bytes
    runlist_delta: str


class MiniNTFS:
    """A teaching-oriented NTFS skeleton with deterministic in-memory recovery."""

    def __init__(
        self,
        total_clusters: int = 48,
        cluster_size: int = 16,
        resident_threshold: int = 28,
    ) -> None:
        self.total_clusters = total_clusters
        self.cluster_size = cluster_size
        self.resident_threshold = resident_threshold
        self.reserved_clusters = {0, 1, 10, 11, 25}

        self.journal: list[JournalEntry] = []
        self._next_lsn = 1
        self._next_txid = 1

        self._reset_state()

    def _reset_state(self) -> None:
        self.bitmap = np.zeros(self.total_clusters, dtype=bool)
        for cid in self.reserved_clusters:
            self.bitmap[cid] = True
        self.clusters: dict[int, bytes] = {}
        self.records: dict[str, MFTRecord] = {}
        self._next_file_ref = 24  # NTFS user files are after system records conceptually.

    def _log(self, txid: int, op: str, payload: dict[str, Any]) -> None:
        self.journal.append(JournalEntry(self._next_lsn, txid, op, payload))
        self._next_lsn += 1

    def _begin_tx(self) -> int:
        txid = self._next_txid
        self._next_txid += 1
        self._log(txid, "BEGIN", {})
        return txid

    def _commit_tx(self, txid: int) -> None:
        self._log(txid, "COMMIT", {})

    @staticmethod
    def encode_runlist_delta(runs: list[tuple[int, int]]) -> str:
        """
        Encode runs as: length:delta|length:delta...
        delta is relative to previous run start LCN, matching NTFS delta idea.
        """
        prev_lcn = 0
        tokens: list[str] = []
        for start, length in runs:
            delta = start - prev_lcn
            tokens.append(f"{length}:{delta}")
            prev_lcn = start
        return "|".join(tokens)

    @staticmethod
    def decode_runlist_delta(encoded: str) -> list[tuple[int, int]]:
        if not encoded:
            return []
        runs: list[tuple[int, int]] = []
        prev_lcn = 0
        for token in encoded.split("|"):
            length_s, delta_s = token.split(":")
            length = int(length_s)
            delta = int(delta_s)
            start = prev_lcn + delta
            runs.append((start, length))
            prev_lcn = start
        return runs

    def _allocate_runs_first_fit(self, needed_clusters: int) -> list[tuple[int, int]]:
        free_count = int((~self.bitmap).sum())
        if free_count < needed_clusters:
            raise RuntimeError(
                f"Not enough space: need={needed_clusters}, free={free_count}"
            )

        runs: list[tuple[int, int]] = []
        i = 0
        while needed_clusters > 0:
            while i < self.total_clusters and self.bitmap[i]:
                i += 1
            if i >= self.total_clusters:
                raise RuntimeError("Unexpected allocation failure during bitmap scan.")

            start = i
            length = 0
            while i < self.total_clusters and (not self.bitmap[i]) and needed_clusters > 0:
                self.bitmap[i] = True
                length += 1
                needed_clusters -= 1
                i += 1
            runs.append((start, length))
        return runs

    def _apply_create_file(self, name: str, data: bytes) -> None:
        if name in self.records:
            raise RuntimeError(f"File already exists in current state: {name}")

        size = len(data)
        if size <= self.resident_threshold:
            record = MFTRecord(
                file_ref=self._next_file_ref,
                name=name,
                size=size,
                resident=True,
                resident_data=data,
                runlist_delta="",
            )
        else:
            needed_clusters = ceil(size / self.cluster_size)
            runs = self._allocate_runs_first_fit(needed_clusters)
            runlist_delta = self.encode_runlist_delta(runs)
            decoded_runs = self.decode_runlist_delta(runlist_delta)
            assert decoded_runs == runs, "Runlist delta encode/decode mismatch."

            cursor = 0
            for start, length in decoded_runs:
                for cid in range(start, start + length):
                    chunk = data[cursor : cursor + self.cluster_size]
                    cursor += self.cluster_size
                    self.clusters[cid] = chunk.ljust(self.cluster_size, b"\x00")

            record = MFTRecord(
                file_ref=self._next_file_ref,
                name=name,
                size=size,
                resident=False,
                resident_data=b"",
                runlist_delta=runlist_delta,
            )

        self.records[name] = record
        self._next_file_ref += 1

    def create_file(self, name: str, data: bytes, commit: bool = True) -> int:
        txid = self._begin_tx()
        self._log(txid, "CREATE_FILE", {"name": name, "data": data})
        if commit:
            self._commit_tx(txid)
            self._apply_create_file(name, data)
        return txid

    def read_file(self, name: str) -> bytes | None:
        record = self.records.get(name)
        if record is None:
            return None
        if record.resident:
            return record.resident_data

        chunks: list[bytes] = []
        for start, length in self.decode_runlist_delta(record.runlist_delta):
            for cid in range(start, start + length):
                chunks.append(self.clusters[cid])
        return b"".join(chunks)[: record.size]

    def recover_from_journal(self) -> None:
        """Crash recovery: rebuild state from committed transactions only."""
        self._reset_state()
        active_ops: dict[int, list[JournalEntry]] = {}

        for entry in self.journal:
            if entry.op == "BEGIN":
                active_ops[entry.txid] = []
            elif entry.op == "COMMIT":
                for op_entry in active_ops.get(entry.txid, []):
                    if op_entry.op == "CREATE_FILE":
                        payload = op_entry.payload
                        self._apply_create_file(payload["name"], payload["data"])
            else:
                active_ops.setdefault(entry.txid, []).append(entry)

    def mft_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for rec in sorted(self.records.values(), key=lambda r: r.file_ref):
            runs = self.decode_runlist_delta(rec.runlist_delta)
            rows.append(
                {
                    "file_ref": rec.file_ref,
                    "name": rec.name,
                    "size": rec.size,
                    "resident": rec.resident,
                    "run_count": len(runs),
                    "clusters_used": int(sum(length for _, length in runs)),
                    "runlist_delta": rec.runlist_delta or "<resident>",
                }
            )
        return pd.DataFrame(rows)

    def stats(self) -> dict[str, float]:
        used_clusters = int(self.bitmap.sum())
        free_clusters = int(self.total_clusters - used_clusters)
        non_resident_records = [r for r in self.records.values() if not r.resident]
        extent_counts = np.array(
            [len(self.decode_runlist_delta(r.runlist_delta)) for r in non_resident_records],
            dtype=int,
        )
        avg_extents = float(extent_counts.mean()) if extent_counts.size > 0 else 0.0
        max_extents = int(extent_counts.max()) if extent_counts.size > 0 else 0
        return {
            "file_count": float(len(self.records)),
            "used_clusters": float(used_clusters),
            "free_clusters": float(free_clusters),
            "avg_extents_non_resident": avg_extents,
            "max_extents_non_resident": float(max_extents),
        }


def main() -> None:
    fs = MiniNTFS(total_clusters=48, cluster_size=16, resident_threshold=28)

    tiny = "tiny-ntfs".encode("utf-8")
    manual = ("NTFS runlist + journal recovery demo. ".encode("utf-8")) * 8
    temp = b"this file should disappear after crash recovery"

    fs.create_file("tiny.txt", tiny, commit=True)
    fs.create_file("manual.bin", manual, commit=True)
    fs.create_file("temp.log", temp, commit=False)  # crash happens before COMMIT

    # Simulate crash and boot-time recovery from journal.
    fs.recover_from_journal()

    tiny_after = fs.read_file("tiny.txt")
    manual_after = fs.read_file("manual.bin")
    temp_after = fs.read_file("temp.log")

    assert tiny_after == tiny
    assert manual_after == manual
    assert temp_after is None, "Uncommitted transaction must not survive recovery."

    mft_df = fs.mft_dataframe()
    stats = fs.stats()

    print("Recovered files:", sorted(fs.records.keys()))
    print("\nMFT Snapshot:")
    print(mft_df.to_string(index=False))
    print("\nStats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    # Sanity checks for allocator + runlist decoder path.
    for rec in fs.records.values():
        if rec.resident:
            assert rec.runlist_delta == ""
        else:
            runs = fs.decode_runlist_delta(rec.runlist_delta)
            assert len(runs) >= 1
            needed = ceil(rec.size / fs.cluster_size)
            assert sum(length for _, length in runs) == needed

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
