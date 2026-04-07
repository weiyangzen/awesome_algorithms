"""Word Break MVP with DP trace and cross-check solvers.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Iterable, Sequence

import numpy as np


@dataclass
class WordBreakResult:
    can_break: bool
    segments: list[str]
    cut_positions: list[int]


def normalize_inputs(text: str, word_dict: Sequence[str]) -> tuple[str, set[str], np.ndarray]:
    """Validate and normalize inputs.

    Returns:
        text: original input string
        word_set: deduplicated non-empty words
        lengths: sorted unique word lengths (np.int64)
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be str, got {type(text)!r}.")

    if not isinstance(word_dict, Sequence):
        raise ValueError("word_dict must be a sequence of strings.")

    clean_words: set[str] = set()
    for i, word in enumerate(word_dict):
        if not isinstance(word, str):
            raise ValueError(f"word_dict[{i}] must be str, got {type(word)!r}.")
        if word == "":
            raise ValueError("word_dict cannot contain empty string.")
        clean_words.add(word)

    lengths = np.array(sorted({len(w) for w in clean_words}), dtype=np.int64)
    return text, clean_words, lengths


def word_break_dp_trace(text: str, word_dict: Sequence[str]) -> WordBreakResult:
    """DP solver with one-solution reconstruction."""
    s, word_set, lengths = normalize_inputs(text, word_dict)
    n = len(s)

    if n == 0:
        return WordBreakResult(can_break=True, segments=[], cut_positions=[0])
    if not word_set:
        return WordBreakResult(can_break=False, segments=[], cut_positions=[])

    dp = np.zeros(n + 1, dtype=bool)
    dp[0] = True
    parent = np.full(n + 1, -1, dtype=np.int64)
    picked_word: list[str] = [""] * (n + 1)
    parent[0] = 0

    for i in range(1, n + 1):
        for wlen in lengths:
            j = i - int(wlen)
            if j < 0:
                break
            if not dp[j]:
                continue
            chunk = s[j:i]
            if chunk in word_set:
                dp[i] = True
                parent[i] = j
                picked_word[i] = chunk
                break

    if not dp[n]:
        return WordBreakResult(can_break=False, segments=[], cut_positions=[])

    rev_segments: list[str] = []
    rev_positions: list[int] = [n]
    cur = n
    while cur != 0:
        p = int(parent[cur])
        if p < 0 or p >= cur:
            raise AssertionError("Invalid parent chain while reconstructing segmentation.")
        rev_segments.append(picked_word[cur])
        rev_positions.append(p)
        cur = p

    segments = list(reversed(rev_segments))
    cut_positions = list(reversed(rev_positions))
    return WordBreakResult(can_break=True, segments=segments, cut_positions=cut_positions)


def word_break_bfs(text: str, word_dict: Sequence[str]) -> bool:
    """BFS reachability solver on index graph."""
    s, word_set, lengths = normalize_inputs(text, word_dict)
    n = len(s)

    if n == 0:
        return True
    if not word_set:
        return False

    visited = np.zeros(n + 1, dtype=bool)
    queue: list[int] = [0]
    visited[0] = True
    head = 0

    while head < len(queue):
        start = queue[head]
        head += 1

        for wlen in lengths:
            end = start + int(wlen)
            if end > n:
                break
            if visited[end]:
                continue
            if s[start:end] not in word_set:
                continue
            if end == n:
                return True
            visited[end] = True
            queue.append(end)

    return False


def word_break_dfs_memo(text: str, word_dict: Sequence[str]) -> WordBreakResult:
    """Memoized DFS exact solver for cross-check."""
    s, word_set, lengths = normalize_inputs(text, word_dict)
    n = len(s)

    if n == 0:
        return WordBreakResult(can_break=True, segments=[], cut_positions=[0])
    if not word_set:
        return WordBreakResult(can_break=False, segments=[], cut_positions=[])

    @lru_cache(maxsize=None)
    def dfs(start: int) -> tuple[bool, tuple[str, ...]]:
        if start == n:
            return True, ()

        for wlen in lengths:
            end = start + int(wlen)
            if end > n:
                break
            chunk = s[start:end]
            if chunk not in word_set:
                continue
            ok, suffix = dfs(end)
            if ok:
                return True, (chunk,) + suffix
        return False, ()

    ok, segmentation = dfs(0)
    if not ok:
        return WordBreakResult(can_break=False, segments=[], cut_positions=[])

    segments = list(segmentation)
    cut_positions = [0]
    cursor = 0
    for seg in segments:
        cursor += len(seg)
        cut_positions.append(cursor)
    return WordBreakResult(can_break=True, segments=segments, cut_positions=cut_positions)


def is_valid_segmentation(
    text: str,
    word_dict: Iterable[str],
    segments: Sequence[str],
    can_break: bool,
) -> bool:
    word_set = set(word_dict)

    if not can_break:
        return len(segments) == 0

    if text == "":
        return len(segments) == 0

    if len(segments) == 0:
        return False
    if any(seg == "" for seg in segments):
        return False
    if any(seg not in word_set for seg in segments):
        return False

    return "".join(segments) == text


def run_case(name: str, text: str, word_dict: Sequence[str]) -> None:
    dp_result = word_break_dp_trace(text, word_dict)
    bfs_result = word_break_bfs(text, word_dict)
    dfs_result = word_break_dfs_memo(text, word_dict) if len(text) <= 24 else None

    valid = is_valid_segmentation(text, word_dict, dp_result.segments, dp_result.can_break)

    print(f"=== {name} ===")
    print(f"text         = {text!r}")
    print(f"word_dict    = {list(word_dict)}")
    print(
        "dp_trace     -> "
        f"can_break={dp_result.can_break}, segments={dp_result.segments}, cuts={dp_result.cut_positions}"
    )
    print(f"bfs          -> can_break={bfs_result}")

    if dfs_result is not None:
        print(
            "dfs_memo     -> "
            f"can_break={dfs_result.can_break}, segments={dfs_result.segments}, cuts={dfs_result.cut_positions}"
        )

    print(
        "checks       -> "
        f"segmentation_valid={valid}, "
        f"bfs_match={bfs_result == dp_result.can_break}, "
        f"dfs_match={None if dfs_result is None else dfs_result.can_break == dp_result.can_break}"
    )
    print()

    if not valid:
        raise AssertionError(f"Invalid segmentation in case '{name}'.")
    if bfs_result != dp_result.can_break:
        raise AssertionError(f"BFS mismatch in case '{name}'.")
    if dfs_result is not None and dfs_result.can_break != dp_result.can_break:
        raise AssertionError(f"DFS mismatch in case '{name}'.")


def randomized_regression(seed: int = 2026, rounds: int = 300) -> None:
    """Randomized consistency test among DP, BFS, and DFS."""
    rng = Random(seed)
    alphabet = "abcd"

    for _ in range(rounds):
        n = rng.randint(0, 14)
        text = "".join(rng.choice(alphabet) for _ in range(n))

        candidate_words: set[str] = set()
        for _ in range(rng.randint(1, 12)):
            if n > 0 and rng.random() < 0.6:
                i = rng.randint(0, n - 1)
                j = rng.randint(i + 1, min(n, i + 4))
                candidate_words.add(text[i:j])
            else:
                k = rng.randint(1, 4)
                candidate_words.add("".join(rng.choice(alphabet) for _ in range(k)))

        word_dict = sorted(candidate_words)

        dp_result = word_break_dp_trace(text, word_dict)
        bfs_result = word_break_bfs(text, word_dict)
        dfs_result = word_break_dfs_memo(text, word_dict)

        assert is_valid_segmentation(text, word_dict, dp_result.segments, dp_result.can_break)
        assert bfs_result == dp_result.can_break
        assert dfs_result.can_break == dp_result.can_break

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, n_range=[0,14], alphabet='abcd'"
    )


def main() -> None:
    cases = [
        ("Case 1: basic true", "leetcode", ["leet", "code"]),
        ("Case 2: basic true reuse", "applepenapple", ["apple", "pen"]),
        ("Case 3: classic false", "catsandog", ["cats", "dog", "sand", "and", "cat"]),
        ("Case 4: empty string", "", ["a", "abc"]),
        ("Case 5: empty dict false", "a", []),
        ("Case 6: overlapping true", "cars", ["car", "ca", "rs"]),
        ("Case 7: many options", "pineapplepenapple", ["apple", "pen", "applepen", "pine", "pineapple"]),
        ("Case 8: repeated words", "aaaaaaa", ["aaaa", "aaa"]),
    ]

    for name, text, word_dict in cases:
        run_case(name, text, word_dict)

    randomized_regression()


if __name__ == "__main__":
    main()
