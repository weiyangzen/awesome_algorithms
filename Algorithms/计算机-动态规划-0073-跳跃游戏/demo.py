"""Jump Game MVP with DP trace, greedy check, and brute-force cross-check.

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
class JumpGameResult:
    reachable: bool
    path_indices: list[int]
    jumps: list[int]


def to_1d_nonnegative_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Validate and normalize input into a 1D non-negative integer array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Input must be a 1D sequence, got shape={arr.shape}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values (nan or inf).")

    rounded = np.rint(arr)
    if arr.size > 0 and not np.allclose(arr, rounded):
        raise ValueError("Jump lengths must be integers.")

    ints = rounded.astype(np.int64)
    if ints.size > 0 and np.any(ints < 0):
        raise ValueError("Jump lengths must be non-negative.")
    return ints


def jump_game_dp_trace(values: Sequence[int] | np.ndarray) -> JumpGameResult:
    """O(n^2) DP reachability with parent reconstruction."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)

    if n == 0:
        return JumpGameResult(reachable=False, path_indices=[], jumps=[])
    if n == 1:
        return JumpGameResult(reachable=True, path_indices=[0], jumps=[])

    reachable = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=np.int64)

    reachable[0] = True
    parent[0] = 0

    for i in range(n):
        if not reachable[i]:
            continue

        furthest = min(n - 1, i + int(nums[i]))
        for j in range(i + 1, furthest + 1):
            if not reachable[j]:
                reachable[j] = True
                parent[j] = i

        if reachable[n - 1]:
            break

    if not reachable[n - 1]:
        return JumpGameResult(reachable=False, path_indices=[], jumps=[])

    path_rev: list[int] = []
    cur = n - 1
    while cur != 0:
        path_rev.append(cur)
        p = int(parent[cur])
        if p < 0 or p == cur:
            raise AssertionError("Invalid parent chain while reconstructing path.")
        cur = p
    path_rev.append(0)

    path = list(reversed(path_rev))
    jumps = [path[i + 1] - path[i] for i in range(len(path) - 1)]
    return JumpGameResult(reachable=True, path_indices=path, jumps=jumps)


def jump_game_greedy(values: Sequence[int] | np.ndarray) -> bool:
    """O(n) greedy reachability check."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)

    if n == 0:
        return False

    farthest = 0
    for i, step in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + int(step))
        if farthest >= n - 1:
            return True
    return farthest >= n - 1


def jump_game_bruteforce(values: Sequence[int] | np.ndarray) -> JumpGameResult:
    """Exact solver for small n using DFS + memoization."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)

    if n == 0:
        return JumpGameResult(reachable=False, path_indices=[], jumps=[])

    @lru_cache(maxsize=None)
    def dfs(i: int) -> tuple[bool, tuple[int, ...]]:
        if i >= n - 1:
            return True, (n - 1,)

        furthest = min(n - 1, i + int(nums[i]))
        for nxt in range(i + 1, furthest + 1):
            ok, suffix = dfs(nxt)
            if ok:
                if suffix and suffix[0] == nxt:
                    return True, (i,) + suffix
                return True, (i, nxt) + suffix

        return False, ()

    ok, path_tuple = dfs(0)
    path = list(path_tuple) if ok else []

    if ok and (not path or path[-1] != n - 1):
        path.append(n - 1)

    jumps = [path[i + 1] - path[i] for i in range(len(path) - 1)] if ok else []
    return JumpGameResult(reachable=ok, path_indices=path, jumps=jumps)


def is_valid_path(
    values: Sequence[int] | np.ndarray,
    path_indices: Iterable[int],
    reachable: bool,
) -> bool:
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)
    path = list(path_indices)

    if not reachable:
        return path == []

    if n == 0:
        return False

    if not path:
        return False
    if path[0] != 0 or path[-1] != n - 1:
        return False

    if any(path[i] >= path[i + 1] for i in range(len(path) - 1)):
        return False

    for i in range(len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]
        if nxt - cur > int(nums[cur]):
            return False

    return True


def run_case(name: str, values: Sequence[int]) -> None:
    dp_result = jump_game_dp_trace(values)
    greedy_result = jump_game_greedy(values)
    brute_result = jump_game_bruteforce(values) if len(values) <= 20 else None

    valid_path = is_valid_path(values, dp_result.path_indices, dp_result.reachable)
    jumps_match = dp_result.jumps == [
        dp_result.path_indices[i + 1] - dp_result.path_indices[i]
        for i in range(len(dp_result.path_indices) - 1)
    ]

    print(f"=== {name} ===")
    print(f"nums         = {list(values)}")
    print(
        "dp_trace     -> "
        f"reachable={dp_result.reachable}, path={dp_result.path_indices}, jumps={dp_result.jumps}"
    )
    print(f"greedy       -> reachable={greedy_result}")

    if brute_result is not None:
        print(
            "bruteforce   -> "
            f"reachable={brute_result.reachable}, "
            f"path={brute_result.path_indices}, jumps={brute_result.jumps}"
        )

    print(
        "checks       -> "
        f"valid_path={valid_path}, jumps_match={jumps_match}, "
        f"greedy_match={greedy_result == dp_result.reachable}, "
        f"bruteforce_match={None if brute_result is None else brute_result.reachable == dp_result.reachable}"
    )
    print()

    if not valid_path:
        raise AssertionError(f"Invalid reconstructed path in case '{name}'.")
    if not jumps_match:
        raise AssertionError(f"Jump list mismatch in case '{name}'.")
    if greedy_result != dp_result.reachable:
        raise AssertionError(f"Greedy mismatch in case '{name}'.")
    if brute_result is not None and brute_result.reachable != dp_result.reachable:
        raise AssertionError(f"Bruteforce mismatch in case '{name}'.")


def randomized_regression(seed: int = 2026, rounds: int = 300) -> None:
    """Randomized consistency test among DP, greedy, and brute-force methods."""
    rng = Random(seed)

    for _ in range(rounds):
        n = rng.randint(0, 14)
        values = [rng.randint(0, 8) for _ in range(n)]

        dp_result = jump_game_dp_trace(values)
        greedy_result = jump_game_greedy(values)
        brute_result = jump_game_bruteforce(values)

        assert is_valid_path(values, dp_result.path_indices, dp_result.reachable)
        assert greedy_result == dp_result.reachable
        assert brute_result.reachable == dp_result.reachable

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, n_range=[0,14], step_range=[0,8]"
    )


def main() -> None:
    cases = {
        "Case 1: reachable classic": [2, 3, 1, 1, 4],
        "Case 2: unreachable trap": [3, 2, 1, 0, 4],
        "Case 3: single zero": [0],
        "Case 4: short fail": [1, 0, 0],
        "Case 5: direct jump": [2, 0, 0],
        "Case 6: empty": [],
        "Case 7: large early jump": [2, 5, 0, 0],
        "Case 8: unit steps": [1, 1, 1, 1, 1],
    }

    for name, values in cases.items():
        run_case(name, values)

    randomized_regression()


if __name__ == "__main__":
    main()
