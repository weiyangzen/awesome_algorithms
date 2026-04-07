"""Minimal runnable MVP for B+ tree index."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable in minimal envs
    np = None


@dataclass
class LeafNode:
    """Leaf node of a B+ tree: stores key -> list[value] and next-leaf pointer."""

    keys: list[int] = field(default_factory=list)
    values: list[list[Any]] = field(default_factory=list)
    next: Optional["LeafNode"] = None
    is_leaf: bool = field(default=True, init=False)


@dataclass
class InternalNode:
    """Internal node of a B+ tree: stores separator keys and child pointers."""

    keys: list[int] = field(default_factory=list)
    children: list[LeafNode | "InternalNode"] = field(default_factory=list)
    is_leaf: bool = field(default=False, init=False)


Node = LeafNode | InternalNode


class BPlusTree:
    """A small educational B+ tree supporting insert, point lookup and range scan."""

    def __init__(self, max_keys: int = 4) -> None:
        if max_keys < 3:
            raise ValueError("max_keys must be >= 3")
        self.max_keys = max_keys
        self.root: Node = LeafNode()

    def search(self, key: int) -> Optional[list[Any]]:
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return list(leaf.values[i])
        return None

    def insert(self, key: int, value: Any) -> None:
        split_info = self._insert_recursive(self.root, key, value)
        if split_info is None:
            return

        promoted_key, right_child = split_info
        new_root = InternalNode(
            keys=[promoted_key],
            children=[self.root, right_child],
        )
        self.root = new_root

    def range_query(self, low: int, high: int) -> list[tuple[int, Any]]:
        if low > high:
            raise ValueError("low must be <= high")

        output: list[tuple[int, Any]] = []
        leaf = self._find_leaf(low)
        while leaf is not None:
            for key, values in zip(leaf.keys, leaf.values):
                if key < low:
                    continue
                if key > high:
                    return output
                for value in values:
                    output.append((key, value))
            leaf = leaf.next
        return output

    def all_items(self) -> list[tuple[int, Any]]:
        items: list[tuple[int, Any]] = []
        leaf = self._leftmost_leaf()
        while leaf is not None:
            for key, values in zip(leaf.keys, leaf.values):
                for value in values:
                    items.append((key, value))
            leaf = leaf.next
        return items

    def validate_structure(self) -> dict[str, int]:
        leaf_depths: list[int] = []
        leaf_count = 0

        def dfs(node: Node, depth: int) -> None:
            nonlocal leaf_count
            if node.is_leaf:
                leaf_count += 1
                leaf_depths.append(depth)
                if node.keys != sorted(node.keys):
                    raise AssertionError("Leaf keys must be sorted")
                if len(node.keys) != len(node.values):
                    raise AssertionError("Leaf keys/values length mismatch")
                return

            internal = node
            if len(internal.children) != len(internal.keys) + 1:
                raise AssertionError("Internal node must satisfy children = keys + 1")
            if internal.keys != sorted(internal.keys):
                raise AssertionError("Internal keys must be sorted")
            for child in internal.children:
                dfs(child, depth + 1)

        dfs(self.root, 0)
        if not leaf_depths:
            raise AssertionError("Tree must have at least one leaf")
        if len(set(leaf_depths)) != 1:
            raise AssertionError("All leaves must be at same depth")

        linked_items: list[tuple[int, Any]] = []
        leaf = self._leftmost_leaf()
        while leaf is not None:
            if leaf.next is not None and leaf.keys and leaf.next.keys:
                if leaf.keys[-1] > leaf.next.keys[0]:
                    raise AssertionError("Leaf linked list key order is invalid")
            for key, values in zip(leaf.keys, leaf.values):
                for value in values:
                    linked_items.append((key, value))
            leaf = leaf.next

        keys_only = [key for key, _ in linked_items]
        if keys_only != sorted(keys_only):
            raise AssertionError("Global key order must be non-decreasing")

        return {
            "height": leaf_depths[0] + 1,
            "leaf_nodes": leaf_count,
            "total_items": len(linked_items),
        }

    def _insert_recursive(self, node: Node, key: int, value: Any) -> Optional[tuple[int, Node]]:
        if node.is_leaf:
            return self._insert_into_leaf(node, key, value)

        internal = node
        child_idx = bisect_right(internal.keys, key)
        split_info = self._insert_recursive(internal.children[child_idx], key, value)
        if split_info is None:
            return None

        promoted_key, right_child = split_info
        insert_at = bisect_right(internal.keys, promoted_key)
        internal.keys.insert(insert_at, promoted_key)
        internal.children.insert(insert_at + 1, right_child)

        if len(internal.keys) <= self.max_keys:
            return None
        return self._split_internal(internal)

    def _insert_into_leaf(self, leaf: LeafNode, key: int, value: Any) -> Optional[tuple[int, Node]]:
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            leaf.values[i].append(value)
        else:
            leaf.keys.insert(i, key)
            leaf.values.insert(i, [value])

        if len(leaf.keys) <= self.max_keys:
            return None
        return self._split_leaf(leaf)

    def _split_leaf(self, leaf: LeafNode) -> tuple[int, Node]:
        mid = len(leaf.keys) // 2
        right_leaf = LeafNode(
            keys=leaf.keys[mid:],
            values=leaf.values[mid:],
            next=leaf.next,
        )
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.next = right_leaf

        promoted_key = right_leaf.keys[0]
        return promoted_key, right_leaf

    def _split_internal(self, node: InternalNode) -> tuple[int, Node]:
        mid = len(node.keys) // 2
        promoted_key = node.keys[mid]

        right_node = InternalNode(
            keys=node.keys[mid + 1 :],
            children=node.children[mid + 1 :],
        )
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]

        return promoted_key, right_node

    def _find_leaf(self, key: int) -> LeafNode:
        node = self.root
        while not node.is_leaf:
            internal = node
            child_idx = bisect_right(internal.keys, key)
            node = internal.children[child_idx]
        return node

    def _leftmost_leaf(self) -> LeafNode:
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        return node


def build_demo_records(n_rows: int = 48, key_upper: int = 20) -> list[tuple[int, str]]:
    if np is None:
        import random

        rng = random.Random(2026)
        keys = [rng.randrange(0, key_upper) for _ in range(n_rows)]
    else:
        rng = np.random.default_rng(seed=2026)
        keys = rng.integers(low=0, high=key_upper, size=n_rows).tolist()

    return [(int(k), f"row_{i:03d}") for i, k in enumerate(keys)]


def build_reference(records: list[tuple[int, str]]) -> dict[int, list[str]]:
    reference: dict[int, list[str]] = defaultdict(list)
    for key, value in records:
        reference[key].append(value)
    return dict(reference)


def assert_tree_matches_reference(tree: BPlusTree, reference: dict[int, list[str]], key_upper: int) -> None:
    for key in range(key_upper):
        expected = reference.get(key)
        got = tree.search(key)
        if expected is None:
            if got is not None:
                raise AssertionError(f"Key {key} should be missing")
        else:
            if got != expected:
                raise AssertionError(f"Key {key} mismatch: expected={expected}, got={got}")

    test_ranges = [(0, 5), (3, 9), (8, 14), (0, key_upper - 1)]
    for low, high in test_ranges:
        got = tree.range_query(low, high)
        expected: list[tuple[int, str]] = []
        for key in sorted(reference):
            if low <= key <= high:
                for value in reference[key]:
                    expected.append((key, value))
        if got != expected:
            raise AssertionError(
                f"Range [{low}, {high}] mismatch: expected={expected}, got={got}"
            )


def main() -> None:
    records = build_demo_records(n_rows=48, key_upper=20)
    reference = build_reference(records)

    tree = BPlusTree(max_keys=4)
    for key, value in records:
        tree.insert(key, value)

    assert_tree_matches_reference(tree, reference, key_upper=20)
    stats = tree.validate_structure()

    print("B+ tree MVP demo finished.")
    print(f"Inserted rows: {len(records)}")
    print(f"Unique keys: {len(reference)}")
    print(f"Tree stats: {stats}")
    print("Sample range [4, 10]:")
    print(tree.range_query(4, 10))


if __name__ == "__main__":
    main()
