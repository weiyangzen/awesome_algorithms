"""Simplified ext2/ext3/ext4 MVP: allocation policy + journaling recovery."""

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
class Inode:
    inode_id: int
    name: str
    size: int
    block_ptrs: list[int]
    extents: list[tuple[int, int]]


class MiniExtFS:
    """A teaching-oriented ext-family filesystem model.

    - ext2: block pointer mapping, no journal replay.
    - ext3: block pointer mapping + journal replay (ordered-like simplification).
    - ext4: extent mapping + journal replay.
    """

    def __init__(
        self,
        fs_type: str,
        total_blocks: int = 160,
        block_size: int = 64,
    ) -> None:
        if fs_type not in {"ext2", "ext3", "ext4"}:
            raise ValueError("fs_type must be one of ext2/ext3/ext4")
        if total_blocks <= 32:
            raise ValueError("total_blocks must be > 32")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        self.fs_type = fs_type
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.journal_enabled = fs_type in {"ext3", "ext4"}
        self.extent_enabled = fs_type == "ext4"

        # Simulate metadata + pre-existing sparse occupation to create fragmentation.
        self.reserved_blocks = {
            0,
            1,
            2,
            3,
            18,
            19,
            36,
            37,
            54,
            55,
            72,
            73,
            90,
            91,
            108,
            109,
        }

        self.journal: list[JournalEntry] = []
        self._next_lsn = 1
        self._next_txid = 1
        self._next_inode = 11

        self._reset_state()

    def _reset_state(self) -> None:
        self.bitmap = np.zeros(self.total_blocks, dtype=bool)
        for bid in self.reserved_blocks:
            self.bitmap[bid] = True
        self.blocks: dict[int, bytes] = {}
        self.inodes: dict[str, Inode] = {}

    def _log(self, txid: int, op: str, payload: dict[str, Any]) -> None:
        self.journal.append(JournalEntry(lsn=self._next_lsn, txid=txid, op=op, payload=payload))
        self._next_lsn += 1

    def _begin_tx(self) -> int:
        txid = self._next_txid
        self._next_txid += 1
        self._log(txid, "BEGIN", {})
        return txid

    def _commit_tx(self, txid: int) -> None:
        self._log(txid, "COMMIT", {})

    def _free_runs(self) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        i = 0
        while i < self.total_blocks:
            while i < self.total_blocks and self.bitmap[i]:
                i += 1
            if i >= self.total_blocks:
                break
            start = i
            length = 0
            while i < self.total_blocks and (not self.bitmap[i]):
                length += 1
                i += 1
            runs.append((start, length))
        return runs

    def _allocate_block_map(self, needed: int) -> tuple[list[int], list[tuple[int, int]]]:
        free_count = int((~self.bitmap).sum())
        if free_count < needed:
            raise RuntimeError(f"Insufficient space: need={needed}, free={free_count}")

        block_ptrs: list[int] = []
        for bid in range(self.total_blocks):
            if not self.bitmap[bid]:
                self.bitmap[bid] = True
                block_ptrs.append(bid)
                if len(block_ptrs) == needed:
                    break

        if len(block_ptrs) != needed:
            raise RuntimeError("Block allocation failed unexpectedly.")
        return block_ptrs, []

    def _allocate_extents(self, needed: int) -> tuple[list[int], list[tuple[int, int]]]:
        free_count = int((~self.bitmap).sum())
        if free_count < needed:
            raise RuntimeError(f"Insufficient space: need={needed}, free={free_count}")

        extents: list[tuple[int, int]] = []
        block_order: list[int] = []

        # Largest-first among free runs, mimicking ext4's preference for long contiguous chunks.
        runs = sorted(self._free_runs(), key=lambda x: x[1], reverse=True)
        remaining = needed
        for start, run_len in runs:
            if remaining == 0:
                break
            take = min(run_len, remaining)
            if take <= 0:
                continue
            extents.append((start, take))
            block_order.extend(range(start, start + take))
            remaining -= take

        if remaining != 0:
            raise RuntimeError("Extent allocation failed unexpectedly.")

        for bid in block_order:
            if self.bitmap[bid]:
                raise RuntimeError("Allocator produced overlapping allocation.")
            self.bitmap[bid] = True

        return block_order, extents

    def _apply_create_file(self, name: str, data: bytes) -> None:
        if name in self.inodes:
            raise RuntimeError(f"File already exists: {name}")

        block_count = ceil(len(data) / self.block_size)
        if self.extent_enabled:
            block_ptrs, extents = self._allocate_extents(block_count)
        else:
            block_ptrs, extents = self._allocate_block_map(block_count)

        cursor = 0
        for bid in block_ptrs:
            chunk = data[cursor : cursor + self.block_size]
            cursor += self.block_size
            self.blocks[bid] = chunk.ljust(self.block_size, b"\x00")

        inode = Inode(
            inode_id=self._next_inode,
            name=name,
            size=len(data),
            block_ptrs=block_ptrs,
            extents=extents,
        )
        self._next_inode += 1
        self.inodes[name] = inode

    def create_file(self, name: str, data: bytes, commit: bool = True) -> int:
        txid = self._begin_tx()
        self._log(txid, "CREATE", {"name": name, "data": data})

        # Apply to current memory state first (dirty cache). Durability depends on COMMIT + replay.
        self._apply_create_file(name, data)

        if self.journal_enabled and commit:
            self._commit_tx(txid)
        return txid

    def read_file(self, name: str) -> bytes | None:
        inode = self.inodes.get(name)
        if inode is None:
            return None
        payload = b"".join(self.blocks[bid] for bid in inode.block_ptrs)
        return payload[: inode.size]

    def recover_from_journal(self) -> None:
        """Replay committed operations only for ext3/ext4. ext2 has no replay stage."""
        if not self.journal_enabled:
            return

        self._reset_state()
        active_ops: dict[int, list[JournalEntry]] = {}

        for entry in self.journal:
            if entry.op == "BEGIN":
                active_ops[entry.txid] = []
            elif entry.op == "COMMIT":
                for op_entry in active_ops.get(entry.txid, []):
                    if op_entry.op == "CREATE":
                        payload = op_entry.payload
                        self._apply_create_file(payload["name"], payload["data"])
            else:
                active_ops.setdefault(entry.txid, []).append(entry)

    @staticmethod
    def _runs_from_blocks(blocks: list[int]) -> list[tuple[int, int]]:
        if not blocks:
            return []
        ordered = sorted(blocks)
        runs: list[tuple[int, int]] = []
        start = ordered[0]
        prev = ordered[0]
        for bid in ordered[1:]:
            if bid == prev + 1:
                prev = bid
                continue
            runs.append((start, prev - start + 1))
            start = bid
            prev = bid
        runs.append((start, prev - start + 1))
        return runs

    def files_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for inode in sorted(self.inodes.values(), key=lambda x: x.inode_id):
            runs = inode.extents if self.extent_enabled else self._runs_from_blocks(inode.block_ptrs)
            rows.append(
                {
                    "inode": inode.inode_id,
                    "name": inode.name,
                    "size": inode.size,
                    "data_blocks": len(inode.block_ptrs),
                    "mapping_entries": len(inode.extents) if self.extent_enabled else len(inode.block_ptrs),
                    "run_count": len(runs),
                    "max_run_len": int(max((ln for _, ln in runs), default=0)),
                }
            )
        return pd.DataFrame(rows)

    def stats(self) -> dict[str, float]:
        used_total = int(self.bitmap.sum())
        used_data = used_total - len(self.reserved_blocks)
        free = int(self.total_blocks - used_total)

        mapping_entries = 0
        run_lens: list[int] = []
        for inode in self.inodes.values():
            mapping_entries += len(inode.extents) if self.extent_enabled else len(inode.block_ptrs)
            runs = inode.extents if self.extent_enabled else self._runs_from_blocks(inode.block_ptrs)
            run_lens.extend([ln for _, ln in runs])

        avg_run = float(np.mean(run_lens)) if run_lens else 0.0
        map_per_block = float(mapping_entries / used_data) if used_data else 0.0

        return {
            "files": float(len(self.inodes)),
            "data_blocks": float(used_data),
            "mapping_entries": float(mapping_entries),
            "avg_contiguous_run": avg_run,
            "mapping_entries_per_data_block": map_per_block,
            "free_blocks": float(free),
        }


def _make_payload(rng: np.random.Generator, n_bytes: int) -> bytes:
    return rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()


def run_single(fs_type: str) -> tuple[MiniExtFS, pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(20260407)
    fs = MiniExtFS(fs_type=fs_type, total_blocks=160, block_size=64)

    committed_files = {
        "alpha.bin": _make_payload(rng, 860),
        "beta.bin": _make_payload(rng, 1240),
    }
    volatile_file = ("volatile.tmp", _make_payload(rng, 780))

    for name, payload in committed_files.items():
        fs.create_file(name, payload, commit=True)

    fs.create_file(volatile_file[0], volatile_file[1], commit=False)
    volatile_before = fs.read_file(volatile_file[0]) is not None

    fs.recover_from_journal()
    volatile_after = fs.read_file(volatile_file[0]) is not None

    for name, payload in committed_files.items():
        got = fs.read_file(name)
        if got != payload:
            raise RuntimeError(f"Data mismatch after recovery in {fs_type}: {name}")

    file_df = fs.files_dataframe()
    stats = fs.stats()
    stats["volatile_visible_before_recovery"] = float(volatile_before)
    stats["volatile_visible_after_recovery"] = float(volatile_after)
    return fs, file_df, stats


def main() -> None:
    print("ext Family MVP Demo (ext2/ext3/ext4)\n")

    all_stats: list[dict[str, Any]] = []

    for fs_type in ("ext2", "ext3", "ext4"):
        fs, file_df, stats = run_single(fs_type)
        stats_row = {"fs_type": fs_type, **stats}
        all_stats.append(stats_row)

        print(f"=== {fs_type} file table ===")
        print(file_df.to_string(index=False))
        print()

    summary = pd.DataFrame(all_stats)
    numeric_cols = [
        "files",
        "data_blocks",
        "mapping_entries",
        "avg_contiguous_run",
        "mapping_entries_per_data_block",
        "free_blocks",
        "volatile_visible_before_recovery",
        "volatile_visible_after_recovery",
    ]
    summary[numeric_cols] = summary[numeric_cols].astype(float)

    print("=== Summary ===")
    print(summary.to_string(index=False))
    print()

    # Deterministic behavioral checks.
    ext2_after = bool(summary.loc[summary["fs_type"] == "ext2", "volatile_visible_after_recovery"].iloc[0])
    ext3_after = bool(summary.loc[summary["fs_type"] == "ext3", "volatile_visible_after_recovery"].iloc[0])
    ext4_after = bool(summary.loc[summary["fs_type"] == "ext4", "volatile_visible_after_recovery"].iloc[0])
    if not ext2_after:
        raise RuntimeError("Expected ext2 volatile file to remain visible in this no-journal model.")
    if ext3_after or ext4_after:
        raise RuntimeError("Expected ext3/ext4 replay to discard uncommitted volatile file.")

    ext2_mpb = float(
        summary.loc[summary["fs_type"] == "ext2", "mapping_entries_per_data_block"].iloc[0]
    )
    ext4_mpb = float(
        summary.loc[summary["fs_type"] == "ext4", "mapping_entries_per_data_block"].iloc[0]
    )
    if not (ext4_mpb < ext2_mpb):
        raise RuntimeError("Expected ext4 mapping overhead per block to be lower than ext2.")

    print("Checks passed:")
    print("- ext2 has no journal replay in this MVP, volatile file remains visible.")
    print("- ext3/ext4 replay committed transactions only, volatile file is removed.")
    print("- ext4 extent mapping uses fewer metadata entries per data block than ext2.")


if __name__ == "__main__":
    main()
