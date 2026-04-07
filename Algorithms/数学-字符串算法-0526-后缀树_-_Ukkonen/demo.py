"""Minimal runnable MVP: Suffix Tree (Ukkonen) with deterministic validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class End:
    """Mutable end pointer shared by all current leaf edges."""

    value: int


@dataclass
class Node:
    """Suffix tree node represented via outgoing edges by first character."""

    start: int
    end: Union[int, End]
    suffix_link: int
    children: Dict[str, int] = field(default_factory=dict)
    suffix_index: int = -1


class SuffixTree:
    """Ukkonen online construction for a single text with a unique terminal."""

    def __init__(self, text: str) -> None:
        self.original_text = text
        self.terminal = self._pick_terminal(text)
        self.text = text + self.terminal
        self.size = len(self.text)
        self.original_size = len(text)

        self.nodes: List[Node] = []
        self.root = self._new_node(start=-1, end=-1)
        self.nodes[self.root].suffix_link = self.root

        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining_suffix_count = 0
        self.last_new_node = -1

        self.leaf_end = End(-1)

        for pos in range(self.size):
            self._extend(pos)

        self._set_suffix_index_dfs(self.root, 0)

    @staticmethod
    def _pick_terminal(text: str) -> str:
        # Single-char terminal not present in the input text.
        for candidate in ("$", "#", "@", "%", "&", "|", "~", "\u0000", "\u0001", "\uE000"):
            if candidate not in text:
                return candidate
        for code in range(0xE001, 0xF8FF):
            ch = chr(code)
            if ch not in text:
                return ch
        raise ValueError("Unable to find a unique terminal character")

    def _new_node(self, start: int, end: Union[int, End]) -> int:
        node = Node(start=start, end=end, suffix_link=self.root if self.nodes else 0)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _edge_end(self, node_index: int) -> int:
        end = self.nodes[node_index].end
        return end.value if isinstance(end, End) else end

    def _edge_length(self, node_index: int) -> int:
        node = self.nodes[node_index]
        if node.start == -1:
            return 0
        return self._edge_end(node_index) - node.start + 1

    def _walk_down(self, next_node: int) -> bool:
        edge_len = self._edge_length(next_node)
        if self.active_length >= edge_len:
            self.active_edge += edge_len
            self.active_length -= edge_len
            self.active_node = next_node
            return True
        return False

    def _extend(self, pos: int) -> None:
        self.leaf_end.value = pos
        self.remaining_suffix_count += 1
        self.last_new_node = -1

        while self.remaining_suffix_count > 0:
            if self.active_length == 0:
                self.active_edge = pos

            active_char = self.text[self.active_edge]
            active_children = self.nodes[self.active_node].children

            if active_char not in active_children:
                leaf = self._new_node(start=pos, end=self.leaf_end)
                active_children[active_char] = leaf

                if self.last_new_node != -1:
                    self.nodes[self.last_new_node].suffix_link = self.active_node
                    self.last_new_node = -1
            else:
                next_node = active_children[active_char]

                if self._walk_down(next_node):
                    continue

                edge_start = self.nodes[next_node].start
                if self.text[edge_start + self.active_length] == self.text[pos]:
                    if self.last_new_node != -1 and self.active_node != self.root:
                        self.nodes[self.last_new_node].suffix_link = self.active_node
                        self.last_new_node = -1
                    self.active_length += 1
                    break

                split_end = edge_start + self.active_length - 1
                split = self._new_node(start=edge_start, end=split_end)
                self.nodes[self.active_node].children[active_char] = split

                leaf = self._new_node(start=pos, end=self.leaf_end)
                self.nodes[split].children[self.text[pos]] = leaf

                self.nodes[next_node].start += self.active_length
                next_char = self.text[self.nodes[next_node].start]
                self.nodes[split].children[next_char] = next_node

                if self.last_new_node != -1:
                    self.nodes[self.last_new_node].suffix_link = split
                self.last_new_node = split

            self.remaining_suffix_count -= 1

            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining_suffix_count + 1
            elif self.active_node != self.root:
                self.active_node = self.nodes[self.active_node].suffix_link

    def _set_suffix_index_dfs(self, node_index: int, label_height: int) -> None:
        node = self.nodes[node_index]
        if not node.children:
            node.suffix_index = self.size - label_height
            return

        for child in node.children.values():
            self._set_suffix_index_dfs(child, label_height + self._edge_length(child))

    def contains(self, pattern: str) -> bool:
        if pattern == "":
            return True
        if self.terminal in pattern:
            return False

        node_index = self.root
        i = 0

        while i < len(pattern):
            child = self.nodes[node_index].children.get(pattern[i])
            if child is None:
                return False

            start = self.nodes[child].start
            end = self._edge_end(child)
            j = start
            while j <= end and i < len(pattern):
                if self.text[j] != pattern[i]:
                    return False
                j += 1
                i += 1

            if i == len(pattern):
                return True

            node_index = child

        return True

    def suffix_array(self) -> List[int]:
        result: List[int] = []

        def dfs(node_index: int) -> None:
            node = self.nodes[node_index]
            if not node.children:
                if 0 <= node.suffix_index < self.original_size:
                    result.append(node.suffix_index)
                return

            for ch in sorted(node.children):
                dfs(node.children[ch])

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
    pattern_results: List[tuple[str, bool]]


def naive_suffix_array(text: str) -> List[int]:
    return sorted(range(len(text)), key=lambda i: text[i:])


def run_case(text: str, patterns: List[str]) -> CaseReport:
    tree = SuffixTree(text)

    expected_sa = naive_suffix_array(text)
    tree_sa = tree.suffix_array()
    assert tree_sa == expected_sa, f"Suffix array mismatch: {tree_sa} vs {expected_sa}"

    expected_leaf_count = len(text) + 1
    assert tree.leaf_count() == expected_leaf_count, (
        f"Leaf count mismatch: {tree.leaf_count()} vs {expected_leaf_count}"
    )

    for i in range(len(text)):
        suffix = text[i:]
        assert tree.contains(suffix), f"Missing suffix in tree: {suffix!r}"

    pattern_results: List[tuple[str, bool]] = []
    for pattern in patterns:
        fast = tree.contains(pattern)
        slow = pattern in text
        assert fast == slow, f"Pattern mismatch for {pattern!r}: fast={fast}, slow={slow}"
        pattern_results.append((pattern, fast))

    return CaseReport(
        text=text,
        node_count=len(tree.nodes),
        leaf_count=tree.leaf_count(),
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

    print("Ukkonen suffix tree MVP demo")
    print("=" * 60)

    for text, patterns in cases.items():
        report = run_case(text, patterns)
        print(f"Text: {report.text!r}")
        print(f"  Nodes: {report.node_count}")
        print(f"  Leaves: {report.leaf_count} (expected {len(text) + 1})")
        print(f"  Suffix Array: {report.suffix_array}")
        checks = ", ".join(f"{p!r}:{ok}" for p, ok in report.pattern_results)
        print(f"  Pattern Checks: {checks}")
        print("-" * 60)

    print("All deterministic checks passed.")


if __name__ == "__main__":
    main()
