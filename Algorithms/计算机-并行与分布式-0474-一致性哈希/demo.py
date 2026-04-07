"""Minimal runnable MVP for Consistent Hashing (CS-0313).

Run:
    uv run python demo.py
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections import Counter

import numpy as np
import pandas as pd


class ConsistentHashRing:
    """A deterministic consistent hash ring with virtual nodes."""

    def __init__(self, replicas_per_node: int = 128) -> None:
        if replicas_per_node <= 0:
            raise ValueError("replicas_per_node must be > 0")
        self.replicas_per_node = replicas_per_node
        self.positions: list[int] = []
        self.owners: list[str] = []
        self.node_positions: dict[str, list[int]] = {}

    @staticmethod
    def _hash_to_int(text: str) -> int:
        # md5 is used only for deterministic hashing, not for cryptographic security.
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _insert_point(self, position: int, node: str) -> None:
        idx = bisect_left(self.positions, position)
        self.positions.insert(idx, position)
        self.owners.insert(idx, node)

    def add_node(self, node: str) -> None:
        if node in self.node_positions:
            raise ValueError(f"node already exists: {node}")
        placements: list[int] = []
        for replica_id in range(self.replicas_per_node):
            # In the rare event of a collision, append an extra salt and retry.
            salt = 0
            while True:
                token = f"{node}#{replica_id}#{salt}"
                position = self._hash_to_int(token)
                idx = bisect_left(self.positions, position)
                if idx == len(self.positions) or self.positions[idx] != position:
                    break
                salt += 1
            self._insert_point(position, node)
            placements.append(position)
        self.node_positions[node] = placements

    def remove_node(self, node: str) -> None:
        if node not in self.node_positions:
            raise ValueError(f"node does not exist: {node}")
        for position in self.node_positions[node]:
            idx = bisect_left(self.positions, position)
            if idx >= len(self.positions) or self.positions[idx] != position:
                raise RuntimeError("ring corruption detected while removing node")
            self.positions.pop(idx)
            self.owners.pop(idx)
        del self.node_positions[node]

    def get_node(self, key: str) -> str:
        if not self.positions:
            raise ValueError("ring is empty")
        position = self._hash_to_int(key)
        idx = bisect_right(self.positions, position)
        if idx == len(self.positions):
            idx = 0
        return self.owners[idx]

    def assign_keys(self, keys: list[str]) -> dict[str, str]:
        return {key: self.get_node(key) for key in keys}


def distribution_df(mapping: dict[str, str], all_nodes: list[str]) -> pd.DataFrame:
    counts = Counter(mapping.values())
    rows = []
    total = len(mapping)
    for node in sorted(all_nodes):
        c = counts.get(node, 0)
        rows.append({"node": node, "keys": c, "ratio": c / total})
    df = pd.DataFrame(rows).sort_values("node").reset_index(drop=True)
    return df


def moved_ratio(before: dict[str, str], after: dict[str, str], keys: list[str]) -> float:
    moved = sum(1 for key in keys if before[key] != after[key])
    return moved / len(keys)


def print_distribution(title: str, df: pd.DataFrame) -> None:
    print(title)
    print(df.to_string(index=False, formatters={"ratio": "{:.3%}".format}))
    print("")


def main() -> None:
    print("=== Consistent Hashing MVP ===")

    keys = [f"user-{i:05d}" for i in range(6000)]
    base_nodes = ["cache-a", "cache-b", "cache-c"]

    ring = ConsistentHashRing(replicas_per_node=128)
    for node in base_nodes:
        ring.add_node(node)

    mapping_base = ring.assign_keys(keys)
    df_base = distribution_df(mapping_base, sorted(base_nodes))

    ring.add_node("cache-d")
    mapping_after_add = ring.assign_keys(keys)
    df_after_add = distribution_df(mapping_after_add, ["cache-a", "cache-b", "cache-c", "cache-d"])
    ratio_add = moved_ratio(mapping_base, mapping_after_add, keys)

    moved_keys_add = {k for k in keys if mapping_base[k] != mapping_after_add[k]}
    new_node_keys = {k for k in keys if mapping_after_add[k] == "cache-d"}
    if moved_keys_add != new_node_keys:
        raise AssertionError("after adding one node, moved keys should equal keys now owned by the new node")

    ring.remove_node("cache-b")
    mapping_after_remove = ring.assign_keys(keys)
    df_after_remove = distribution_df(mapping_after_remove, ["cache-a", "cache-c", "cache-d"])
    ratio_remove = moved_ratio(mapping_after_add, mapping_after_remove, keys)

    moved_keys_remove = {k for k in keys if mapping_after_add[k] != mapping_after_remove[k]}
    removed_node_keys = {k for k in keys if mapping_after_add[k] == "cache-b"}
    if moved_keys_remove != removed_node_keys:
        raise AssertionError("after removing one node, moved keys should equal keys previously owned by removed node")

    if any(node == "cache-b" for node in mapping_after_remove.values()):
        raise AssertionError("removed node still receives keys")

    if not (0.15 <= ratio_add <= 0.35):
        raise AssertionError(f"unexpected add-node moved ratio: {ratio_add:.4f}")

    if not (0.15 <= ratio_remove <= 0.35):
        raise AssertionError(f"unexpected remove-node moved ratio: {ratio_remove:.4f}")

    # Check load balance quality (coefficient of variation).
    cv_base = float(np.std(df_base["keys"].to_numpy()) / np.mean(df_base["keys"].to_numpy()))
    cv_after_add = float(
        np.std(df_after_add["keys"].to_numpy()) / np.mean(df_after_add["keys"].to_numpy())
    )
    cv_after_remove = float(
        np.std(df_after_remove["keys"].to_numpy()) / np.mean(df_after_remove["keys"].to_numpy())
    )

    print(f"Key count: {len(keys)}")
    print(f"Virtual nodes per physical node: {ring.replicas_per_node}\n")

    print_distribution("Distribution with 3 nodes:", df_base)
    print_distribution("Distribution after adding cache-d:", df_after_add)
    print_distribution("Distribution after removing cache-b:", df_after_remove)

    print(f"Moved ratio (3 -> 4 nodes): {ratio_add:.3%}")
    print(f"Moved ratio (4 -> 3 nodes by removing cache-b): {ratio_remove:.3%}")
    print(f"Load balance CV (3 nodes): {cv_base:.4f}")
    print(f"Load balance CV (4 nodes): {cv_after_add:.4f}")
    print(f"Load balance CV (remove one node): {cv_after_remove:.4f}")
    print("\nAll checks passed for CS-0313 (一致性哈希).")


if __name__ == "__main__":
    main()
