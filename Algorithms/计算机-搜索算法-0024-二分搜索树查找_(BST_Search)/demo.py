"""BST Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Node:
    key: int
    left: "Node | None" = None
    right: "Node | None" = None


class BST:
    """Minimal BST supporting insert and iterative search."""

    def __init__(self) -> None:
        self.root: Node | None = None

    def insert(self, key: int) -> bool:
        """Insert key if absent. Return True if inserted, False if duplicate."""
        if self.root is None:
            self.root = Node(key)
            return True

        current = self.root
        while True:
            if key == current.key:
                return False
            if key < current.key:
                if current.left is None:
                    current.left = Node(key)
                    return True
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(key)
                    return True
                current = current.right

    def search(self, target: int) -> Node | None:
        """Return node with target key, or None if absent."""
        current = self.root
        while current is not None:
            if target == current.key:
                return current
            if target < current.key:
                current = current.left
            else:
                current = current.right
        return None

    def inorder(self) -> list[int]:
        """Return in-order traversal for structure validation."""
        result: list[int] = []

        def _dfs(node: Node | None) -> None:
            if node is None:
                return
            _dfs(node.left)
            result.append(node.key)
            _dfs(node.right)

        _dfs(self.root)
        return result

    def search_with_trace(self, target: int) -> tuple[Node | None, list[int]]:
        """Search target and also return visited keys path."""
        trace: list[int] = []
        current = self.root
        while current is not None:
            trace.append(current.key)
            if target == current.key:
                return current, trace
            if target < current.key:
                current = current.left
            else:
                current = current.right
        return None, trace


def build_bst(values: Iterable[int]) -> BST:
    tree = BST()
    for v in values:
        tree.insert(v)
    return tree


def main() -> None:
    # Canonical BST example values.
    values = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    tree = build_bst(values)
    value_set = set(values)

    print("BST Search MVP demo")
    print("-" * 32)
    print(f"Inserted values (order): {values}")
    inorder_values = tree.inorder()
    print(f"In-order traversal:       {inorder_values}")

    # In-order traversal must be sorted and contain unique keys.
    if inorder_values != sorted(value_set):
        raise AssertionError("In-order traversal check failed: BST property broken")

    queries = [6, 13, 2, 8, 15, 1, 14]
    for idx, q in enumerate(queries, start=1):
        node, trace = tree.search_with_trace(q)
        found = node is not None
        expected = q in value_set
        if found != expected:
            raise AssertionError(
                f"Case {idx}: mismatch for target={q}, found={found}, expected={expected}, trace={trace}"
            )
        print(
            f"Case {idx:02d}: target={q:>2}, found={found}, "
            f"trace={' -> '.join(map(str, trace)) if trace else 'EMPTY'}"
        )

    # Duplicate insertion policy check: duplicates should not be inserted.
    duplicate_key = 6
    inserted = tree.insert(duplicate_key)
    if inserted:
        raise AssertionError("Duplicate insertion policy failed: duplicate key was inserted")
    if tree.inorder() != inorder_values:
        raise AssertionError("Tree changed after duplicate insertion attempt")
    print("Duplicate insertion check: passed (duplicate key ignored)")

    # Empty tree search check.
    empty_tree = BST()
    if empty_tree.search(42) is not None:
        raise AssertionError("Empty-tree search failed: expected None")
    print("Empty tree search check: passed")

    print("All checks passed.")


if __name__ == "__main__":
    main()
