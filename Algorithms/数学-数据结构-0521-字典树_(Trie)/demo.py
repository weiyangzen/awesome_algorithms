"""Minimal runnable MVP for Trie (prefix tree).

This script is intentionally self-contained and uses only Python stdlib.
Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A single node in the Trie.

    Attributes:
        children: outgoing edges keyed by one character.
        pass_count: number of words passing through this node.
        end_count: number of words ending at this node.
    """

    children: dict[str, "TrieNode"] = field(default_factory=dict)
    pass_count: int = 0
    end_count: int = 0


class Trie:
    """Trie supporting insertion, exact query, prefix query, counting and erase."""

    def __init__(self) -> None:
        self.root = TrieNode()

    @staticmethod
    def _validate_text(text: str, *, field_name: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"{field_name} must be str, got {type(text).__name__}")

    def insert(self, word: str) -> None:
        self._validate_text(word, field_name="word")

        node = self.root
        node.pass_count += 1
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.pass_count += 1
        node.end_count += 1

    def _find_node(self, text: str) -> TrieNode | None:
        node = self.root
        for ch in text:
            nxt = node.children.get(ch)
            if nxt is None:
                return None
            node = nxt
        return node

    def search(self, word: str) -> bool:
        self._validate_text(word, field_name="word")
        node = self._find_node(word)
        return bool(node and node.end_count > 0)

    def starts_with(self, prefix: str) -> bool:
        self._validate_text(prefix, field_name="prefix")
        return self._find_node(prefix) is not None

    def count_words_equal_to(self, word: str) -> int:
        self._validate_text(word, field_name="word")
        node = self._find_node(word)
        return 0 if node is None else node.end_count

    def count_words_starting_with(self, prefix: str) -> int:
        self._validate_text(prefix, field_name="prefix")
        node = self._find_node(prefix)
        return 0 if node is None else node.pass_count

    def erase(self, word: str) -> bool:
        """Erase one occurrence of word. Return False if word does not exist."""
        self._validate_text(word, field_name="word")

        # Trace path first so we can safely decrement counters on the way back.
        node = self.root
        path: list[tuple[TrieNode, str]] = []
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                return False
            path.append((node, ch))
            node = nxt

        if node.end_count == 0:
            return False

        node.end_count -= 1
        self.root.pass_count -= 1

        # Decrease pass_count along the path from root to leaf node.
        current = self.root
        for ch in word:
            child = current.children[ch]
            child.pass_count -= 1
            current = child

        # Prune dead branches from the end for compactness.
        for parent, ch in reversed(path):
            child = parent.children[ch]
            if child.pass_count == 0 and child.end_count == 0 and not child.children:
                del parent.children[ch]
            else:
                break

        return True

    def list_words_with_prefix(self, prefix: str, limit: int = 20) -> list[str]:
        self._validate_text(prefix, field_name="prefix")
        if limit <= 0:
            return []

        start = self._find_node(prefix)
        if start is None:
            return []

        result: list[str] = []

        def dfs(node: TrieNode, path: str) -> None:
            if len(result) >= limit:
                return
            if node.end_count > 0:
                result.extend([path] * node.end_count)
            for ch in sorted(node.children.keys()):
                if len(result) >= limit:
                    return
                dfs(node.children[ch], path + ch)

        dfs(start, prefix)
        return result[:limit]


def run_demo() -> None:
    trie = Trie()

    words = [
        "cat",
        "car",
        "cart",
        "dog",
        "dove",
        "do",
        "中文",
        "中秋",
        "中关村",
        "car",  # duplicate
    ]
    for w in words:
        trie.insert(w)

    print("=== Trie Demo ===")
    print(f"Inserted {len(words)} words (with duplicates).")

    checks = [
        ("search('car')", trie.search("car"), True),
        ("search('cars')", trie.search("cars"), False),
        ("starts_with('ca')", trie.starts_with("ca"), True),
        ("starts_with('cap')", trie.starts_with("cap"), False),
        ("count_words_equal_to('car')", trie.count_words_equal_to("car"), 2),
        (
            "count_words_starting_with('do')",
            trie.count_words_starting_with("do"),
            3,
        ),
        (
            "count_words_starting_with('中')",
            trie.count_words_starting_with("中"),
            3,
        ),
    ]

    for label, got, expected in checks:
        status = "OK" if got == expected else "FAIL"
        print(f"{label:<36} -> {got!r:<8} expected={expected!r:<8} [{status}]")
        assert got == expected, f"{label}: got {got!r}, expected {expected!r}"

    print("\nWords with prefix 'ca':", trie.list_words_with_prefix("ca"))
    print("Words with prefix '中':", trie.list_words_with_prefix("中"))

    erased_first = trie.erase("car")
    erased_second = trie.erase("car")
    erased_third = trie.erase("car")

    print("\nErase 'car' three times:")
    print("1st erase:", erased_first)
    print("2nd erase:", erased_second)
    print("3rd erase:", erased_third)
    print("count_words_equal_to('car') after erase:", trie.count_words_equal_to("car"))

    assert erased_first is True
    assert erased_second is True
    assert erased_third is False
    assert trie.count_words_equal_to("car") == 0
    assert trie.search("cart") is True

    print("\nAll assertions passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
