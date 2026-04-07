"""A minimal runnable MVP of Aho-Corasick automaton (AC自动机)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import random
from typing import Deque, Dict, List, Tuple


@dataclass
class Node:
    """Single automaton node."""

    nxt: Dict[str, int] = field(default_factory=dict)
    fail: int = 0
    out: List[int] = field(default_factory=list)  # pattern ids ending here


Match = Tuple[str, int, int]  # (pattern, start, end)


class AhoCorasick:
    """A simple, explicit implementation of AC automaton."""

    def __init__(self, patterns: List[str]) -> None:
        if not patterns:
            raise ValueError("patterns must not be empty")
        if any(p == "" for p in patterns):
            raise ValueError("empty pattern is not allowed in this MVP")
        self.patterns = patterns
        self.nodes: List[Node] = [Node()]  # root at index 0
        self._build_trie()
        self._build_fail_links()

    def _build_trie(self) -> None:
        for pid, pat in enumerate(self.patterns):
            cur = 0
            for ch in pat:
                if ch not in self.nodes[cur].nxt:
                    self.nodes[cur].nxt[ch] = len(self.nodes)
                    self.nodes.append(Node())
                cur = self.nodes[cur].nxt[ch]
            self.nodes[cur].out.append(pid)

    def _build_fail_links(self) -> None:
        q: Deque[int] = deque()

        # Root children: fail = 0
        for child in self.nodes[0].nxt.values():
            self.nodes[child].fail = 0
            q.append(child)

        # BFS over trie levels
        while q:
            u = q.popleft()
            for ch, v in self.nodes[u].nxt.items():
                q.append(v)
                f = self.nodes[u].fail
                while f != 0 and ch not in self.nodes[f].nxt:
                    f = self.nodes[f].fail
                if ch in self.nodes[f].nxt:
                    self.nodes[v].fail = self.nodes[f].nxt[ch]
                else:
                    self.nodes[v].fail = 0

                # Inherit outputs from fail state to catch suffix patterns
                self.nodes[v].out.extend(self.nodes[self.nodes[v].fail].out)

    def search(self, text: str) -> List[Match]:
        matches: List[Match] = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self.nodes[state].nxt:
                state = self.nodes[state].fail
            if ch in self.nodes[state].nxt:
                state = self.nodes[state].nxt[ch]
            else:
                state = 0

            if self.nodes[state].out:
                for pid in self.nodes[state].out:
                    pat = self.patterns[pid]
                    start = i - len(pat) + 1
                    matches.append((pat, start, i))
        return matches


def naive_search(patterns: List[str], text: str) -> List[Match]:
    """Reference implementation for verification."""
    out: List[Match] = []
    n = len(text)
    for pat in patterns:
        m = len(pat)
        for i in range(0, n - m + 1):
            if text[i : i + m] == pat:
                out.append((pat, i, i + m - 1))
    return out


def count_by_pattern(matches: List[Match]) -> Counter:
    return Counter(pat for pat, _, _ in matches)


def run_fixed_demo() -> None:
    patterns = ["he", "she", "his", "hers", "is"]
    text = "ahishersheis"

    ac = AhoCorasick(patterns)
    ac_matches = ac.search(text)
    naive_matches = naive_search(patterns, text)

    if sorted(ac_matches) != sorted(naive_matches):
        raise AssertionError("fixed demo mismatch between AC and naive outputs")

    print("=== AC Automaton Fixed Demo ===")
    print(f"patterns: {patterns}")
    print(f"text:     {text}")
    print("matches (pattern, start, end):")
    for item in sorted(ac_matches, key=lambda x: (x[1], x[2], x[0])):
        print(f"  {item}")
    print("count by pattern:")
    for pat in patterns:
        print(f"  {pat:>4}: {count_by_pattern(ac_matches)[pat]}")
    print("fixed demo check: PASS")


def run_random_regression(rounds: int = 80, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    alphabet = "abcd"
    for r in range(rounds):
        text_len = rng.randint(20, 80)
        k = rng.randint(3, 10)
        text = "".join(rng.choice(alphabet) for _ in range(text_len))
        patterns = []
        for _ in range(k):
            plen = rng.randint(1, 6)
            patterns.append("".join(rng.choice(alphabet) for _ in range(plen)))

        ac = AhoCorasick(patterns)
        ac_matches = sorted(ac.search(text))
        naive_matches = sorted(naive_search(patterns, text))
        if ac_matches != naive_matches:
            raise AssertionError(
                f"random regression failed at round={r}\n"
                f"patterns={patterns}\ntext={text}\n"
                f"ac={ac_matches}\nnaive={naive_matches}"
            )
    print(f"random regression: PASS ({rounds} rounds, seed={seed})")


def main() -> None:
    run_fixed_demo()
    run_random_regression()


if __name__ == "__main__":
    main()
