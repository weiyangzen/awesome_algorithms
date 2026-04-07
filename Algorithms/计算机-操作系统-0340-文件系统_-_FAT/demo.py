"""Simplified FAT MVP: cluster-chain allocation, read/delete, and fsck checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

FREE = -1
EOF = -2
RESERVED = -3


@dataclass
class DirectoryEntry:
    name: str
    size: int
    start_cluster: int


class MiniFAT:
    """Teaching-oriented FAT filesystem model.

    The model keeps only core FAT ideas:
    - FAT table maps cluster -> next cluster / EOF / FREE.
    - Directory entry stores file name + size + start cluster.
    - File read traverses cluster chain sequentially.
    """

    def __init__(
        self,
        total_clusters: int = 96,
        cluster_size: int = 32,
        reserved_clusters: set[int] | None = None,
        bad_clusters: set[int] | None = None,
    ) -> None:
        if total_clusters <= 16:
            raise ValueError("total_clusters must be > 16")
        if cluster_size <= 0:
            raise ValueError("cluster_size must be positive")

        self.total_clusters = total_clusters
        self.cluster_size = cluster_size
        self.reserved_clusters = set(reserved_clusters or {0, 1})
        self.bad_clusters = set(bad_clusters or {11, 12, 26, 27, 43, 60})

        overlap = self.reserved_clusters & self.bad_clusters
        if overlap:
            raise ValueError(f"reserved and bad clusters overlap: {sorted(overlap)}")

        self.fat = np.full(self.total_clusters, FREE, dtype=np.int32)
        for cid in self.reserved_clusters | self.bad_clusters:
            self._assert_cluster_range(cid)
            self.fat[cid] = RESERVED

        self.directory: dict[str, DirectoryEntry] = {}
        self.cluster_data: dict[int, bytes] = {}

    def _assert_cluster_range(self, cid: int) -> None:
        if cid < 0 or cid >= self.total_clusters:
            raise RuntimeError(f"Cluster out of range: {cid}")

    def _free_cluster_ids(self) -> list[int]:
        return [cid for cid in range(self.total_clusters) if int(self.fat[cid]) == FREE]

    def _allocate_chain_first_fit(self, needed_clusters: int) -> list[int]:
        free_ids = self._free_cluster_ids()
        if len(free_ids) < needed_clusters:
            raise RuntimeError(
                f"Insufficient clusters: needed={needed_clusters}, free={len(free_ids)}"
            )

        chain = free_ids[:needed_clusters]
        for i, cid in enumerate(chain):
            nxt = chain[i + 1] if i + 1 < len(chain) else EOF
            self.fat[cid] = nxt
        return chain

    def _chain_from_start(self, start_cluster: int) -> list[int]:
        if start_cluster < 0:
            return []

        chain: list[int] = []
        visited: set[int] = set()
        current = start_cluster
        while True:
            self._assert_cluster_range(current)
            if current in visited:
                raise RuntimeError(f"Loop detected in FAT chain at cluster={current}")
            visited.add(current)
            chain.append(current)

            nxt = int(self.fat[current])
            if nxt == EOF:
                break
            if nxt == FREE:
                raise RuntimeError(f"Broken chain: next cluster is FREE at cluster={current}")
            if nxt == RESERVED:
                raise RuntimeError(f"Broken chain: next cluster is RESERVED at cluster={current}")
            current = nxt
            if len(chain) > self.total_clusters:
                raise RuntimeError("Chain length exceeded cluster count (corruption suspected)")

        return chain

    def _run_count(self, chain: list[int]) -> int:
        if not chain:
            return 0
        runs = 1
        for a, b in zip(chain, chain[1:]):
            if b != a + 1:
                runs += 1
        return runs

    def create_file(self, name: str, data: bytes) -> DirectoryEntry:
        if name in self.directory:
            raise RuntimeError(f"File already exists: {name}")
        if len(data) == 0:
            raise RuntimeError("Zero-length file is intentionally unsupported in this MVP")

        needed = ceil(len(data) / self.cluster_size)
        chain = self._allocate_chain_first_fit(needed)

        cursor = 0
        for cid in chain:
            chunk = data[cursor : cursor + self.cluster_size]
            cursor += self.cluster_size
            self.cluster_data[cid] = chunk.ljust(self.cluster_size, b"\x00")

        entry = DirectoryEntry(name=name, size=len(data), start_cluster=chain[0])
        self.directory[name] = entry
        return entry

    def read_file(self, name: str) -> bytes | None:
        entry = self.directory.get(name)
        if entry is None:
            return None

        chain = self._chain_from_start(entry.start_cluster)
        payload = b"".join(self.cluster_data[cid] for cid in chain)
        return payload[: entry.size]

    def delete_file(self, name: str) -> bool:
        entry = self.directory.pop(name, None)
        if entry is None:
            return False

        chain = self._chain_from_start(entry.start_cluster)
        for cid in chain:
            self.fat[cid] = FREE
            self.cluster_data.pop(cid, None)
        return True

    def files_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for entry in sorted(self.directory.values(), key=lambda x: x.name):
            chain = self._chain_from_start(entry.start_cluster)
            run_count = self._run_count(chain)
            max_run = 0
            current_run = 1
            for a, b in zip(chain, chain[1:]):
                if b == a + 1:
                    current_run += 1
                else:
                    max_run = max(max_run, current_run)
                    current_run = 1
            max_run = max(max_run, current_run) if chain else 0

            rows.append(
                {
                    "name": entry.name,
                    "size": entry.size,
                    "start_cluster": entry.start_cluster,
                    "cluster_count": len(chain),
                    "run_count": run_count,
                    "max_run_len": max_run,
                    "chain_preview": "->".join(map(str, chain[:8])) + ("..." if len(chain) > 8 else ""),
                }
            )

        return pd.DataFrame(rows)

    def fsck(self) -> dict[str, float]:
        cross_link_count = 0
        broken_files = 0
        owned_by: dict[int, str] = {}

        for entry in self.directory.values():
            try:
                chain = self._chain_from_start(entry.start_cluster)
            except RuntimeError:
                broken_files += 1
                continue

            needed = ceil(entry.size / self.cluster_size)
            if len(chain) < needed:
                broken_files += 1

            for cid in chain:
                prev_owner = owned_by.get(cid)
                if prev_owner is None:
                    owned_by[cid] = entry.name
                elif prev_owner != entry.name:
                    cross_link_count += 1

        return {
            "broken_files": float(broken_files),
            "cross_link_count": float(cross_link_count),
        }

    def stats(self) -> dict[str, float]:
        file_chains = [self._chain_from_start(e.start_cluster) for e in self.directory.values()]
        run_counts = np.array([self._run_count(chain) for chain in file_chains], dtype=np.int32)
        chain_lens = np.array([len(chain) for chain in file_chains], dtype=np.int32)

        used_clusters = int(np.sum(self.fat != FREE))
        free_clusters = int(np.sum(self.fat == FREE))
        data_clusters = int(chain_lens.sum()) if chain_lens.size else 0

        frag_flags = np.array([1 if r > 1 else 0 for r in run_counts], dtype=np.int32)
        avg_chain = float(chain_lens.mean()) if chain_lens.size else 0.0
        avg_runs = float(run_counts.mean()) if run_counts.size else 0.0
        fragmented_ratio = float(frag_flags.mean()) if frag_flags.size else 0.0
        avg_run_len = float(data_clusters / run_counts.sum()) if run_counts.sum() > 0 else 0.0

        return {
            "file_count": float(len(self.directory)),
            "used_clusters_total": float(used_clusters),
            "free_clusters": float(free_clusters),
            "data_clusters": float(data_clusters),
            "avg_chain_len": avg_chain,
            "avg_runs_per_file": avg_runs,
            "avg_run_len": avg_run_len,
            "fragmented_file_ratio": fragmented_ratio,
        }


def _make_payload(rng: np.random.Generator, n_bytes: int) -> bytes:
    return rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()


def main() -> None:
    rng = np.random.default_rng(20260407)

    fs = MiniFAT(
        total_clusters=96,
        cluster_size=32,
        reserved_clusters={0, 1},
        bad_clusters={11, 12, 26, 27, 43, 60},
    )

    kernel = _make_payload(rng, 950)
    video = _make_payload(rng, 640)
    notes = _make_payload(rng, 210)
    dataset = _make_payload(rng, 820)

    fs.create_file("kernel.sys", kernel)
    fs.create_file("video.raw", video)
    fs.create_file("notes.txt", notes)

    deleted = fs.delete_file("video.raw")
    assert deleted is True

    fs.create_file("dataset.bin", dataset)

    assert fs.read_file("kernel.sys") == kernel
    assert fs.read_file("notes.txt") == notes
    assert fs.read_file("dataset.bin") == dataset
    assert fs.read_file("video.raw") is None

    fsck_report = fs.fsck()
    assert fsck_report["broken_files"] == 0.0
    assert fsck_report["cross_link_count"] == 0.0

    df = fs.files_dataframe()
    stats = fs.stats()

    dataset_row = df[df["name"] == "dataset.bin"]
    if dataset_row.empty:
        raise RuntimeError("dataset.bin not found in output table")
    dataset_runs = int(dataset_row["run_count"].iloc[0])
    assert dataset_runs >= 2, "dataset.bin should be fragmented in this scenario"

    print("FAT Directory Snapshot:")
    print(df.to_string(index=False))
    print("\nFAT Stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    print("\nFSCK Report:")
    for key, value in fsck_report.items():
        print(f"  - {key}: {value}")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
