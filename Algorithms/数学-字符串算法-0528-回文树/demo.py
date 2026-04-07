"""回文树（Eertree）最小可运行示例。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """回文树节点：对应一个唯一回文子串。"""

    length: int
    suffix_link: int
    next: dict[str, int] = field(default_factory=dict)
    occ: int = 0
    first_pos: int = -1


class Eertree:
    """回文树（Palindromic Tree / Eertree）。"""

    def __init__(self) -> None:
        # 0: 长度 -1 的奇根；1: 长度 0 的偶根
        self.nodes: list[Node] = [Node(length=-1, suffix_link=0), Node(length=0, suffix_link=0)]
        self.last: int = 1
        self.text: str = ""

    def _find_longest_suffix(self, pos: int, ch: str) -> int:
        """找到可被 ch 向两侧扩展的最长后缀回文节点。"""
        cur = self.last
        while True:
            cur_len = self.nodes[cur].length
            left = pos - cur_len - 1
            if left >= 0 and self.text[left] == ch:
                return cur
            cur = self.nodes[cur].suffix_link

    def add_char(self, ch: str) -> bool:
        """向回文树追加一个字符。返回值表示是否创建了新节点。"""
        pos = len(self.text)
        self.text += ch
        cur = self._find_longest_suffix(pos, ch)

        # 该回文已存在，直接复用节点
        if ch in self.nodes[cur].next:
            self.last = self.nodes[cur].next[ch]
            self.nodes[self.last].occ += 1
            return False

        # 创建新回文节点
        new_len = self.nodes[cur].length + 2
        new_idx = len(self.nodes)
        self.nodes.append(Node(length=new_len, suffix_link=0, occ=1, first_pos=pos))
        self.nodes[cur].next[ch] = new_idx

        # 单字符回文的后缀链接指向空串根
        if new_len == 1:
            self.nodes[new_idx].suffix_link = 1
            self.last = new_idx
            return True

        # 查找新节点的 suffix link
        link_candidate = self.nodes[cur].suffix_link
        while True:
            cand_len = self.nodes[link_candidate].length
            left = pos - cand_len - 1
            if left >= 0 and self.text[left] == ch:
                self.nodes[new_idx].suffix_link = self.nodes[link_candidate].next[ch]
                break
            link_candidate = self.nodes[link_candidate].suffix_link

        self.last = new_idx
        return True

    def build(self, text: str) -> None:
        for ch in text:
            self.add_char(ch)
        self.propagate_occurrences()

    def propagate_occurrences(self) -> None:
        """将更长回文的出现次数沿 suffix link 回传到更短回文。"""
        order = sorted(range(2, len(self.nodes)), key=lambda i: self.nodes[i].length, reverse=True)
        for idx in order:
            link = self.nodes[idx].suffix_link
            self.nodes[link].occ += self.nodes[idx].occ

    def node_to_string(self, idx: int) -> str:
        node = self.nodes[idx]
        if node.length <= 0:
            return ""
        end = node.first_pos
        start = end - node.length + 1
        return self.text[start : end + 1]

    def distinct_count(self) -> int:
        return len(self.nodes) - 2

    def longest_palindrome(self) -> str:
        if len(self.nodes) <= 2:
            return ""
        best_idx = max(range(2, len(self.nodes)), key=lambda i: self.nodes[i].length)
        return self.node_to_string(best_idx)

    def palindrome_occurrences(self) -> dict[str, int]:
        return {self.node_to_string(i): self.nodes[i].occ for i in range(2, len(self.nodes))}


def brute_force_occurrences(text: str) -> dict[str, int]:
    """仅用于小样本验证正确性的暴力计数。"""
    result: dict[str, int] = {}
    n = len(text)
    for i in range(n):
        for j in range(i, n):
            s = text[i : j + 1]
            if s == s[::-1]:
                result[s] = result.get(s, 0) + 1
    return result


def validate_with_bruteforce(text: str) -> tuple[Eertree, dict[str, int], dict[str, int]]:
    tree = Eertree()
    tree.build(text)
    tree_map = tree.palindrome_occurrences()
    brute_map = brute_force_occurrences(text)
    if tree_map != brute_map:
        raise AssertionError(f"结果不一致: tree={tree_map}, brute={brute_map}")
    return tree, tree_map, brute_map


def main() -> None:
    samples = [
        "abacabaeae",
        "aaaaa",
        "abacdfgdcaba",
        "levelupmadamimadam",
    ]

    for text in samples:
        tree, tree_map, _ = validate_with_bruteforce(text)
        print("=" * 72)
        print(f"text: {text}")
        print(f"distinct palindromes: {tree.distinct_count()}")
        print(f"longest palindrome: {tree.longest_palindrome()!r}")
        ranked = sorted(tree_map.items(), key=lambda kv: (-len(kv[0]), kv[0]))
        print("top palindromes by length (up to 10):")
        for p, cnt in ranked[:10]:
            print(f"  {p!r:<20} len={len(p):<2} occ={cnt}")
    print("=" * 72)
    print("All samples passed Eertree vs brute-force validation.")


if __name__ == "__main__":
    main()
