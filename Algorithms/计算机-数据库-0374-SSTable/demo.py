"""Minimal runnable MVP for SSTable.

This script builds a tiny LSM-style KV store with:
- MemTable (in-memory mutable map)
- SSTable flush (sorted immutable on-disk files)
- Sparse index lookup
- Tombstone delete
- Simple compaction

Run:
    uv run python demo.py
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

TOMBSTONE = object()
NOT_FOUND = object()
ValueType = Union[str, object]


@dataclass(frozen=True)
class TablePaths:
    """File paths for one SSTable."""

    data: Path
    index: Path


class SSTableWriter:
    """Write immutable sorted key-value records to disk."""

    def __init__(self, table_id: int, table_dir: Path, sparse_step: int = 2) -> None:
        self.table_id = table_id
        self.table_dir = table_dir
        self.sparse_step = max(1, sparse_step)

    def _paths(self) -> TablePaths:
        stem = f"table_{self.table_id:06d}"
        return TablePaths(
            data=self.table_dir / f"{stem}.data",
            index=self.table_dir / f"{stem}.index",
        )

    def write(self, items: List[Tuple[str, ValueType]]) -> TablePaths:
        """Persist sorted items and a sparse index."""
        paths = self._paths()
        sorted_items = sorted(items, key=lambda x: x[0])
        index_entries: List[Dict[str, int | str]] = []

        with paths.data.open("wb") as f:
            for i, (key, value) in enumerate(sorted_items):
                offset = f.tell()
                is_deleted = value is TOMBSTONE
                record = {
                    "k": key,
                    "v": None if is_deleted else value,
                    "d": is_deleted,
                }
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                f.write(line)
                if i % self.sparse_step == 0:
                    index_entries.append({"k": key, "o": offset})

        with paths.index.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "table_id": self.table_id,
                    "sparse_step": self.sparse_step,
                    "entries": index_entries,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return paths


class SSTableReader:
    """Read one immutable SSTable with sparse-index guided scan."""

    def __init__(self, data_path: Path, index_path: Path) -> None:
        self.data_path = data_path
        self.index_path = index_path
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._entries = payload["entries"]
        self._keys = [e["k"] for e in self._entries]
        self._offsets = [e["o"] for e in self._entries]

    def _start_offset(self, key: str) -> int:
        if not self._keys:
            return 0
        pos = bisect_right(self._keys, key) - 1
        return self._offsets[pos] if pos >= 0 else 0

    def get(self, key: str) -> ValueType:
        """Point lookup inside this single table."""
        start = self._start_offset(key)
        with self.data_path.open("rb") as f:
            f.seek(start)
            while True:
                line = f.readline()
                if not line:
                    return NOT_FOUND
                rec = json.loads(line)
                rec_key = rec["k"]
                if rec_key == key:
                    return TOMBSTONE if rec["d"] else rec["v"]
                if rec_key > key:
                    return NOT_FOUND

    def iter_records(self) -> Iterator[Tuple[str, ValueType]]:
        """Sequentially read all records from this table."""
        with self.data_path.open("rb") as f:
            for line in f:
                rec = json.loads(line)
                value = TOMBSTONE if rec["d"] else rec["v"]
                yield rec["k"], value


class LSMTreeMini:
    """Tiny LSM-like KV store for educational SSTable demo."""

    def __init__(self, table_dir: Path, memtable_limit: int = 4, sparse_step: int = 2) -> None:
        self.table_dir = table_dir
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.memtable_limit = max(1, memtable_limit)
        self.sparse_step = max(1, sparse_step)
        self.memtable: Dict[str, ValueType] = {}
        self.table_ids: List[int] = self._discover_table_ids()
        self.next_table_id = (max(self.table_ids) + 1) if self.table_ids else 1

    def _discover_table_ids(self) -> List[int]:
        ids: List[int] = []
        for p in self.table_dir.glob("table_*.data"):
            ids.append(int(p.stem.split("_")[1]))
        # Newest table first.
        return sorted(ids, reverse=True)

    def _paths_for_id(self, table_id: int) -> TablePaths:
        stem = f"table_{table_id:06d}"
        return TablePaths(
            data=self.table_dir / f"{stem}.data",
            index=self.table_dir / f"{stem}.index",
        )

    def _new_table_id(self) -> int:
        table_id = self.next_table_id
        self.next_table_id += 1
        return table_id

    def _maybe_flush(self) -> None:
        if len(self.memtable) >= self.memtable_limit:
            self.flush()

    def put(self, key: str, value: str) -> None:
        self.memtable[key] = value
        self._maybe_flush()

    def delete(self, key: str) -> None:
        self.memtable[key] = TOMBSTONE
        self._maybe_flush()

    def flush(self) -> None:
        if not self.memtable:
            return
        table_id = self._new_table_id()
        writer = SSTableWriter(table_id, self.table_dir, self.sparse_step)
        writer.write(list(self.memtable.items()))
        self.memtable.clear()
        # Newest first.
        self.table_ids.insert(0, table_id)

    def get(self, key: str) -> str | None:
        # Check MemTable first.
        if key in self.memtable:
            v = self.memtable[key]
            return None if v is TOMBSTONE else str(v)

        # Check immutable tables from newest to oldest.
        for table_id in self.table_ids:
            paths = self._paths_for_id(table_id)
            reader = SSTableReader(paths.data, paths.index)
            result = reader.get(key)
            if result is NOT_FOUND:
                continue
            if result is TOMBSTONE:
                return None
            return str(result)
        return None

    def compact(self) -> None:
        """Merge all SSTables into one newest table and drop tombstones."""
        self.flush()
        if len(self.table_ids) <= 1:
            return

        merged: Dict[str, ValueType] = {}
        # Iterate newest -> oldest. First seen key wins.
        for table_id in self.table_ids:
            paths = self._paths_for_id(table_id)
            reader = SSTableReader(paths.data, paths.index)
            for key, value in reader.iter_records():
                if key not in merged:
                    merged[key] = value

        compacted_items = [(k, v) for k, v in sorted(merged.items()) if v is not TOMBSTONE]

        # Remove old tables.
        old_ids = list(self.table_ids)
        for table_id in old_ids:
            paths = self._paths_for_id(table_id)
            if paths.data.exists():
                paths.data.unlink()
            if paths.index.exists():
                paths.index.unlink()

        self.table_ids.clear()

        # Write merged table if there is live data.
        if compacted_items:
            table_id = self._new_table_id()
            writer = SSTableWriter(table_id, self.table_dir, self.sparse_step)
            writer.write(compacted_items)
            self.table_ids = [table_id]


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "_demo_data"
    base_dir.mkdir(parents=True, exist_ok=True)
    for p in base_dir.glob("table_*.*"):
        p.unlink()

    db = LSMTreeMini(base_dir, memtable_limit=3, sparse_step=2)

    # Round 1: first flush triggered automatically at size 3.
    db.put("apple", "red")
    db.put("banana", "yellow")
    db.put("cherry", "dark-red")

    # Round 2: updates and delete in newer table.
    db.put("banana", "green")
    db.delete("cherry")
    db.put("date", "brown")

    # Persist remaining MemTable if any.
    db.flush()

    assert db.get("apple") == "red"
    assert db.get("banana") == "green"
    assert db.get("cherry") is None
    assert db.get("date") == "brown"
    assert db.get("unknown") is None

    before_compact = len(db.table_ids)
    db.compact()
    after_compact = len(db.table_ids)

    assert db.get("banana") == "green"
    assert db.get("cherry") is None

    print("SSTable MVP demo completed.")
    print(f"Data dir: {base_dir}")
    print(f"SSTable count before compaction: {before_compact}")
    print(f"SSTable count after compaction:  {after_compact}")
    print(
        "Sample lookups:",
        {
            "apple": db.get("apple"),
            "banana": db.get("banana"),
            "cherry": db.get("cherry"),
            "date": db.get("date"),
        },
    )


if __name__ == "__main__":
    main()
