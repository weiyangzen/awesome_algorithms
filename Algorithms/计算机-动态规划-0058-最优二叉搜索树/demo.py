"""最优二叉搜索树（Optimal BST）MVP：区间 DP + 记忆化基线 + 小规模暴力校验。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class OBSTNode:
    key_index: int
    left: "OBSTNode | None"
    right: "OBSTNode | None"


@dataclass
class OptimalBSTResult:
    n_keys: int
    p: list[float]
    q: list[float]
    min_expected_cost: float
    root_table: np.ndarray
    expected_cost_table: np.ndarray
    weight_table: np.ndarray
    tree: OBSTNode | None


def to_probability_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} contains negative probabilities")
    return arr


def validate_probabilities(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_arr = to_probability_vector(p, name="p")
    q_arr = to_probability_vector(q, name="q")

    n = int(p_arr.size)
    if q_arr.size != n + 1:
        raise ValueError(
            f"q length must be n+1 where n=len(p), got len(p)={n}, len(q)={q_arr.size}"
        )

    total = float(p_arr.sum() + q_arr.sum())
    if total <= 0.0:
        raise ValueError("total probability mass must be positive")
    if not np.isclose(total, 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError(
            f"p and q must sum to 1.0, got total={total:.12f}"
        )

    return p_arr, q_arr


def build_prefix_sums(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_prefix = np.concatenate(([0.0], np.cumsum(p)))
    q_prefix = np.concatenate(([0.0], np.cumsum(q)))
    return p_prefix, q_prefix


def interval_weight(
    i: int,
    j: int,
    p_prefix: np.ndarray,
    q_prefix: np.ndarray,
) -> float:
    # 区间 [i, j) 的权重：sum(p[i:j]) + sum(q[i:j+1])
    p_part = float(p_prefix[j] - p_prefix[i])
    q_part = float(q_prefix[j + 1] - q_prefix[i])
    return p_part + q_part


def build_tree(root: np.ndarray, i: int, j: int) -> OBSTNode | None:
    if i == j:
        return None

    r = int(root[i, j])
    if r < i or r >= j:
        raise RuntimeError(f"invalid root entry at ({i}, {j}): {r}")

    return OBSTNode(
        key_index=r,
        left=build_tree(root, i, r),
        right=build_tree(root, r + 1, j),
    )


def tree_to_expression(node: OBSTNode | None) -> str:
    if node is None:
        return "∅"
    return (
        f"k{node.key_index + 1}("
        f"{tree_to_expression(node.left)},"
        f"{tree_to_expression(node.right)})"
    )


def evaluate_expected_cost_from_tree(
    node: OBSTNode | None,
    p: np.ndarray,
    q: np.ndarray,
    i: int,
    j: int,
    depth: int,
) -> float:
    if i == j:
        return float(q[i] * depth)
    if node is None:
        raise RuntimeError("non-empty interval has no root node")

    r = int(node.key_index)
    if r < i or r >= j:
        raise RuntimeError(f"tree node key index out of interval: key={r}, interval=[{i},{j})")

    here = float(p[r] * depth)
    left = evaluate_expected_cost_from_tree(node.left, p, q, i, r, depth + 1)
    right = evaluate_expected_cost_from_tree(node.right, p, q, r + 1, j, depth + 1)
    return here + left + right


def optimal_bst_dp(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> OptimalBSTResult:
    """自底向上区间 DP，返回最优期望代价与根表。"""
    p_arr, q_arr = validate_probabilities(p, q)
    n = int(p_arr.size)

    e = np.full((n + 1, n + 1), np.inf, dtype=float)
    w = np.zeros((n + 1, n + 1), dtype=float)
    root = np.full((n + 1, n + 1), -1, dtype=int)

    for i in range(n + 1):
        e[i, i] = float(q_arr[i])
        w[i, i] = float(q_arr[i])

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            w[i, j] = w[i, j - 1] + float(p_arr[j - 1]) + float(q_arr[j])

            best = np.inf
            best_r = -1
            for r in range(i, j):
                candidate = e[i, r] + e[r + 1, j] + w[i, j]
                if candidate < best:
                    best = candidate
                    best_r = r

            e[i, j] = best
            root[i, j] = best_r

    tree = build_tree(root, 0, n)

    return OptimalBSTResult(
        n_keys=n,
        p=[float(x) for x in p_arr.tolist()],
        q=[float(x) for x in q_arr.tolist()],
        min_expected_cost=float(e[0, n]),
        root_table=root,
        expected_cost_table=e,
        weight_table=w,
        tree=tree,
    )


def optimal_bst_top_down_cost(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> float:
    """记忆化递归基线：只返回最优期望代价。"""
    p_arr, q_arr = validate_probabilities(p, q)
    n = int(p_arr.size)
    p_prefix, q_prefix = build_prefix_sums(p_arr, q_arr)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> float:
        if i == j:
            return float(q_arr[i])

        w_ij = interval_weight(i, j, p_prefix, q_prefix)
        best = np.inf
        for r in range(i, j):
            candidate = solve(i, r) + solve(r + 1, j) + w_ij
            if candidate < best:
                best = candidate
        return float(best)

    return float(solve(0, n))


def optimal_bst_bruteforce_cost(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> float:
    """无缓存暴力递归，仅用于小规模交叉校验。"""
    p_arr, q_arr = validate_probabilities(p, q)
    n = int(p_arr.size)
    p_prefix, q_prefix = build_prefix_sums(p_arr, q_arr)

    def solve(i: int, j: int) -> float:
        if i == j:
            return float(q_arr[i])

        w_ij = interval_weight(i, j, p_prefix, q_prefix)
        best = np.inf
        for r in range(i, j):
            candidate = solve(i, r) + solve(r + 1, j) + w_ij
            if candidate < best:
                best = candidate
        return float(best)

    return float(solve(0, n))


def left_biased_policy_cost(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> float:
    """非最优基线：每个区间都选最左键作为根。"""
    p_arr, q_arr = validate_probabilities(p, q)
    n = int(p_arr.size)
    p_prefix, q_prefix = build_prefix_sums(p_arr, q_arr)

    def solve(i: int, j: int) -> float:
        if i == j:
            return float(q_arr[i])
        r = i
        w_ij = interval_weight(i, j, p_prefix, q_prefix)
        return solve(i, r) + solve(r + 1, j) + w_ij

    return float(solve(0, n))


def normalize_random_probabilities(raw: np.ndarray) -> np.ndarray:
    shifted = np.asarray(raw, dtype=float) + 1e-6
    total = float(shifted.sum())
    return shifted / total


def run_case(
    name: str,
    p: Sequence[float],
    q: Sequence[float],
    expected_cost: float | None = None,
    bruteforce_limit: int = 6,
) -> None:
    result = optimal_bst_dp(p, q)

    p_arr = np.asarray(result.p, dtype=float)
    q_arr = np.asarray(result.q, dtype=float)

    top_down = optimal_bst_top_down_cost(p_arr, q_arr)
    brute_force = None
    if result.n_keys <= bruteforce_limit:
        brute_force = optimal_bst_bruteforce_cost(p_arr, q_arr)

    rebuilt = evaluate_expected_cost_from_tree(
        result.tree,
        p_arr,
        q_arr,
        i=0,
        j=result.n_keys,
        depth=1,
    )

    left_cost = left_biased_policy_cost(p_arr, q_arr)

    print(f"=== {name} ===")
    print(f"p={np.round(p_arr, 6).tolist()}")
    print(f"q={np.round(q_arr, 6).tolist()}")
    print(
        "optimal => "
        f"cost={result.min_expected_cost:.12f}, "
        f"tree={tree_to_expression(result.tree)}"
    )
    print(
        "cross-check => "
        f"top_down={top_down:.12f}, "
        f"brute_force={('N/A' if brute_force is None else f'{brute_force:.12f}')}, "
        f"rebuilt={rebuilt:.12f}, "
        f"left_biased={left_cost:.12f}"
    )
    print()

    if not np.isclose(result.min_expected_cost, top_down, atol=1e-12, rtol=1e-12):
        raise AssertionError("bottom-up DP and top-down baseline mismatch")

    if brute_force is not None and not np.isclose(
        result.min_expected_cost,
        brute_force,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError("bottom-up DP and brute-force baseline mismatch")

    if not np.isclose(result.min_expected_cost, rebuilt, atol=1e-12, rtol=1e-12):
        raise AssertionError("tree reconstruction expected cost mismatch")

    if expected_cost is not None and not np.isclose(
        result.min_expected_cost,
        expected_cost,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            f"unexpected optimal cost: got {result.min_expected_cost}, expected {expected_cost}"
        )


def randomized_cross_check(
    trials: int = 200,
    max_keys: int = 7,
    bruteforce_limit: int = 5,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        n = int(rng.integers(1, max_keys + 1))

        raw = rng.random(size=(2 * n + 1,))
        probs = normalize_random_probabilities(raw)
        p = probs[:n]
        q = probs[n:]

        result = optimal_bst_dp(p, q)
        top_down = optimal_bst_top_down_cost(p, q)
        if not np.isclose(result.min_expected_cost, top_down, atol=1e-12, rtol=1e-12):
            raise AssertionError("random check failed: DP vs top-down mismatch")

        if n <= bruteforce_limit:
            brute_force = optimal_bst_bruteforce_cost(p, q)
            if not np.isclose(
                result.min_expected_cost,
                brute_force,
                atol=1e-12,
                rtol=1e-12,
            ):
                raise AssertionError("random check failed: DP vs brute-force mismatch")

        rebuilt = evaluate_expected_cost_from_tree(
            result.tree,
            np.asarray(result.p, dtype=float),
            np.asarray(result.q, dtype=float),
            i=0,
            j=n,
            depth=1,
        )
        if not np.isclose(result.min_expected_cost, rebuilt, atol=1e-12, rtol=1e-12):
            raise AssertionError("random check failed: tree reconstruction mismatch")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_keys={max_keys}, seed={seed})."
    )


def main() -> None:
    # CLRS 经典样例：最优期望代价为 2.75
    run_case(
        name="Case 1: CLRS classic",
        p=[0.15, 0.10, 0.05, 0.10, 0.20],
        q=[0.05, 0.10, 0.05, 0.05, 0.05, 0.10],
        expected_cost=2.75,
        bruteforce_limit=0,
    )

    run_case(
        name="Case 2: skewed successful probabilities",
        p=[0.30, 0.20, 0.10],
        q=[0.05, 0.10, 0.15, 0.10],
        expected_cost=None,
        bruteforce_limit=7,
    )

    run_case(
        name="Case 3: near-uniform distribution",
        p=[0.12, 0.12, 0.12, 0.12],
        q=[0.104, 0.104, 0.104, 0.104, 0.104],
        expected_cost=None,
        bruteforce_limit=6,
    )

    randomized_cross_check(
        trials=200,
        max_keys=7,
        bruteforce_limit=5,
        seed=2026,
    )


if __name__ == "__main__":
    main()
