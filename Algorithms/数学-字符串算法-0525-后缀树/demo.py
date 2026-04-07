"""Minimal runnable MVP: suffix tree via compressed suffix trie insertion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Edge:
    """Directed edge labeled by text[start:end)."""

    start: int
    end: int
    child: int


@dataclass
class Node:
    """Suffix tree node."""

    children: Dict[str, Edge] = field(default_factory=dict)
    suffix_index: Optional[int] = None


class SuffixTree:
    """Compressed suffix tree built by inserting every suffix (O(n^2) MVP)."""

    def __init__(self, text: str) -> None:
        self.original_text = text
        self.terminal = self._pick_terminal(text)
        self.full_text = text + self.terminal
        self.original_size = len(text)

        self.nodes: List[Node] = [Node()]  # root at index 0
        self.root = 0

        for start in range(len(self.full_text)):
            self._insert_suffix(start)

    @staticmethod
    def _pick_terminal(text: str) -> str:
        for candidate in ("$", "#", "@", "%", "&", "|", "~", "\u0000", "\u0001", "\uE000"):
            if candidate not in text:
                return candidate
        for code in range(0xE001, 0xF8FF):
            ch = chr(code)
            if ch not in text:
                return ch
        raise ValueError("Unable to find a unique terminal character")

    def _new_node(self, suffix_index: Optional[int] = None) -> int:
        self.nodes.append(Node(suffix_index=suffix_index))
        return len(self.nodes) - 1

    def _edge_label(self, edge: Edge) -> str:
        return self.full_text[edge.start : edge.end]

    def _insert_suffix(self, suffix_start: int) -> None:
        node_index = self.root
        pos = suffix_start

        while pos < len(self.full_text):
            current_char = self.full_text[pos]
            edge = self.nodes[node_index].children.get(current_char)

            if edge is None:
                leaf = self._new_node(suffix_index=suffix_start)
                self.nodes[node_index].children[current_char] = Edge(
                    start=pos,
                    end=len(self.full_text),
                    child=leaf,
                )
                return

            label_start = edge.start
            label_end = edge.end
            i = 0
            while (
                label_start + i < label_end
                and pos + i < len(self.full_text)
                and self.full_text[label_start + i] == self.full_text[pos + i]
            ):
                i += 1

            edge_len = label_end - label_start
            if i == edge_len:
                node_index = edge.child
                pos += i
                continue

            # Split existing edge at mismatch position.
            split = self._new_node(suffix_index=None)
            old_child = edge.child

            self.nodes[node_index].children[current_char] = Edge(
                start=label_start,
                end=label_start + i,
                child=split,
            )

            old_char = self.full_text[label_start + i]
            self.nodes[split].children[old_char] = Edge(
                start=label_start + i,
                end=label_end,
                child=old_child,
            )

            new_leaf = self._new_node(suffix_index=suffix_start)
            new_char = self.full_text[pos + i]
            self.nodes[split].children[new_char] = Edge(
                start=pos + i,
                end=len(self.full_text),
                child=new_leaf,
            )
            return

        # In this MVP each suffix includes a unique terminal, so this branch is
        # usually not reached. Keep it for completeness.
        if self.nodes[node_index].suffix_index is None:
            self.nodes[node_index].suffix_index = suffix_start

    def contains(self, pattern: str) -> bool:
        if pattern == "":
            return True
        if self.terminal in pattern:
            return False

        node_index = self.root
        pos = 0

        while pos < len(pattern):
            edge = self.nodes[node_index].children.get(pattern[pos])
            if edge is None:
                return False

            j = edge.start
            while j < edge.end and pos < len(pattern):
                if self.full_text[j] != pattern[pos]:
                    return False
                j += 1
                pos += 1

            if pos == len(pattern):
                return True

            node_index = edge.child

        return True

    def suffix_array(self) -> List[int]:
        result: List[int] = []

        def dfs(node_index: int) -> None:
            node = self.nodes[node_index]
            if not node.children:
                if node.suffix_index is not None and 0 <= node.suffix_index < self.original_size:
                    result.append(node.suffix_index)
                return

            for ch in sorted(node.children):
                dfs(node.children[ch].child)

        dfs(self.root)
        return result

    def leaf_count(self) -> int:
        return sum(1 for node in self.nodes if not node.children)


@dataclass
class CaseReport:
    text: str
    node_count: int
    leaf_count: int
    suffix_array: List[int]
    pattern_results: List[Tuple[str, bool]]


def naive_suffix_array(text: str) -> List[int]:
    return sorted(range(len(text)), key=lambda i: text[i:])


def run_case(text: str, patterns: List[str]) -> CaseReport:
    tree = SuffixTree(text)

    expected_sa = naive_suffix_array(text)
    tree_sa = tree.suffix_array()
    assert tree_sa == expected_sa, f"Suffix array mismatch: {tree_sa} vs {expected_sa}"

    expected_leaf_count = len(text) + 1
    actual_leaf_count = tree.leaf_count()
    assert actual_leaf_count == expected_leaf_count, (
        f"Leaf count mismatch: {actual_leaf_count} vs {expected_leaf_count}"
    )

    for i in range(len(text)):
        suffix = text[i:]
        assert tree.contains(suffix), f"Missing suffix in tree: {suffix!r}"

    pattern_results: List[Tuple[str, bool]] = []
    for pattern in patterns:
        fast = tree.contains(pattern)
        slow = pattern in text
        assert fast == slow, f"Pattern mismatch for {pattern!r}: fast={fast}, slow={slow}"
        pattern_results.append((pattern, fast))

    return CaseReport(
        text=text,
        node_count=len(tree.nodes),
        leaf_count=actual_leaf_count,
        suffix_array=tree_sa,
        pattern_results=pattern_results,
    )


def main() -> None:
    cases = {
        "banana": ["ana", "nana", "ban", "apple", "", "a$"],
        "mississippi": ["issi", "ssip", "sip", "ppi", "miss", "xyz"],
        "xabxac": ["abx", "xac", "bxac", "xacx", ""],
        "aaaaa": ["a", "aa", "aaa", "aaaaaa", "b"],
        "": ["", "a"],
    }

    print("Suffix tree (naive compressed insertion) MVP demo")
    print("=" * 66)

    for text, patterns in cases.items():
        report = run_case(text, patterns)
        print(f"Text: {report.text!r}")
        print(f"  Nodes: {report.node_count}")
        print(f"  Leaves: {report.leaf_count} (expected {len(text) + 1})")
        print(f"  Suffix Array: {report.suffix_array}")
        checks = ", ".join(f"{p!r}:{ok}" for p, ok in report.pattern_results)
        print(f"  Pattern Checks: {checks}")
        print("-" * 66)

    print("All deterministic checks passed.")


if __name__ == "__main__":
    main()
