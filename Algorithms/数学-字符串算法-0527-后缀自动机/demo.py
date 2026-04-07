"""Suffix Automaton (SAM) MVP with self-checking demos.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class State:
    """A SAM state."""

    length: int = 0
    link: int = -1
    next: dict[str, int] = field(default_factory=dict)
    occ: int = 0


class SuffixAutomaton:
    """Minimal suffix automaton implementation for one base string."""

    def __init__(self) -> None:
        self.states: list[State] = [State(length=0, link=-1)]
        self.last: int = 0
        self._occ_finalized: bool = False

    def extend(self, ch: str) -> None:
        if len(ch) != 1:
            raise ValueError("extend expects a single character")

        self._occ_finalized = False

        cur = len(self.states)
        self.states.append(State(length=self.states[self.last].length + 1, occ=1))

        p = self.last
        while p != -1 and ch not in self.states[p].next:
            self.states[p].next[ch] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[ch]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = len(self.states)
                self.states.append(
                    State(
                        length=self.states[p].length + 1,
                        link=self.states[q].link,
                        next=self.states[q].next.copy(),
                        occ=0,
                    )
                )

                while p != -1 and self.states[p].next.get(ch) == q:
                    self.states[p].next[ch] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self.last = cur

    def build(self, text: str) -> None:
        for ch in text:
            self.extend(ch)

    def finalize_occurrences(self) -> None:
        if self._occ_finalized:
            return

        order = sorted(range(1, len(self.states)), key=lambda i: self.states[i].length, reverse=True)
        for v in order:
            parent = self.states[v].link
            if parent >= 0:
                self.states[parent].occ += self.states[v].occ

        self._occ_finalized = True

    def contains(self, pattern: str) -> bool:
        state = 0
        for ch in pattern:
            nxt = self.states[state].next.get(ch)
            if nxt is None:
                return False
            state = nxt
        return True

    def _state_of_pattern(self, pattern: str) -> int:
        state = 0
        for ch in pattern:
            nxt = self.states[state].next.get(ch)
            if nxt is None:
                return -1
            state = nxt
        return state

    def count_occurrences(self, pattern: str) -> int:
        if not pattern:
            return 0
        self.finalize_occurrences()
        state = self._state_of_pattern(pattern)
        if state == -1:
            return 0
        return self.states[state].occ

    def count_distinct_substrings(self) -> int:
        total = 0
        for i in range(1, len(self.states)):
            link = self.states[i].link
            total += self.states[i].length - self.states[link].length
        return total

    def edge_count(self) -> int:
        return sum(len(st.next) for st in self.states)

    def longest_common_substring(self, other: str) -> tuple[int, str]:
        state = 0
        length = 0
        best_len = 0
        best_end = -1

        for idx, ch in enumerate(other):
            while state != 0 and ch not in self.states[state].next:
                state = self.states[state].link
                length = self.states[state].length

            nxt = self.states[state].next.get(ch)
            if nxt is not None:
                state = nxt
                length += 1
            else:
                state = 0
                length = 0

            if length > best_len:
                best_len = length
                best_end = idx

        if best_len == 0:
            return 0, ""
        return best_len, other[best_end - best_len + 1 : best_end + 1]


def brute_distinct_substrings(s: str) -> int:
    seen: set[str] = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            seen.add(s[i:j])
    return len(seen)


def brute_occurrences(s: str, pattern: str) -> int:
    if not pattern:
        return 0
    count = 0
    m = len(pattern)
    for i in range(len(s) - m + 1):
        if s[i : i + m] == pattern:
            count += 1
    return count


def brute_lcs(a: str, b: str) -> tuple[int, str]:
    if not a or not b:
        return 0, ""

    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end = -1

    for i, ca in enumerate(a, start=1):
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_end = i - 1
        prev = cur

    if best_len == 0:
        return 0, ""
    return best_len, a[best_end - best_len + 1 : best_end + 1]


def run_case(text: str, patterns: list[str], other: str) -> None:
    print("=" * 72)
    print(f"text: {text!r}")

    sam = SuffixAutomaton()
    sam.build(text)

    distinct = sam.count_distinct_substrings()
    distinct_brute = brute_distinct_substrings(text)
    print(f"states={len(sam.states)}, edges={sam.edge_count()}")
    print(f"distinct_substrings: sam={distinct}, brute={distinct_brute}")
    assert distinct == distinct_brute, "Distinct substring count mismatch"

    for pattern in patterns:
        exists = sam.contains(pattern)
        exists_brute = pattern in text
        occ = sam.count_occurrences(pattern)
        occ_brute = brute_occurrences(text, pattern)
        print(
            f"pattern={pattern!r:<8} contains={exists} (brute={exists_brute}) "
            f"occurrences={occ} (brute={occ_brute})"
        )
        assert exists == exists_brute, f"contains mismatch for pattern={pattern!r}"
        assert occ == occ_brute, f"occurrence mismatch for pattern={pattern!r}"

    lcs_len, lcs_value = sam.longest_common_substring(other)
    lcs_len_brute, lcs_value_brute = brute_lcs(text, other)
    print(
        f"LCS with {other!r}: len={lcs_len}, value={lcs_value!r} "
        f"(brute_len={lcs_len_brute}, brute_value={lcs_value_brute!r})"
    )
    assert lcs_len == lcs_len_brute, "LCS length mismatch"


def main() -> None:
    cases = [
        {
            "text": "ababa",
            "patterns": ["a", "aba", "ba", "bab", "c", ""],
            "other": "bababca",
        },
        {
            "text": "banana",
            "patterns": ["ana", "na", "nana", "banana", "bananas", ""],
            "other": "ananas",
        },
    ]

    for case in cases:
        run_case(case["text"], case["patterns"], case["other"])

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
