"""Gale-Shapley 稳定婚姻问题的最小可运行 MVP。

实现内容：
1) 手写提案方（proposer）主导的 Gale-Shapley 算法。
2) 对输出匹配执行稳定性验证（阻塞对检测）。
3) 在小规模下穷举全部稳定匹配，验证“提案方最优”性质。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

IntArray = np.ndarray
BlockingPair = Tuple[int, int, int, int]


@dataclass(frozen=True)
class StableMarriageResult:
    """稳定婚姻算法输出。"""

    proposer_to_receiver: IntArray
    receiver_to_proposer: IntArray
    proposal_count: int


def validate_preferences(name: str, prefs: Sequence[Sequence[int]]) -> IntArray:
    """检查偏好矩阵是 n*n 且每行是 [0, n-1] 的一个排列。"""
    arr = np.asarray(prefs, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got ndim={arr.ndim}")
    n_rows, n_cols = arr.shape
    if n_rows != n_cols:
        raise ValueError(f"{name} must be square, got shape={arr.shape}")

    n = n_rows
    expected = np.arange(n, dtype=np.int64)
    for i, row in enumerate(arr):
        if np.any((row < 0) | (row >= n)):
            raise ValueError(f"{name}[{i}] contains out-of-range index")
        if not np.array_equal(np.sort(row), expected):
            raise ValueError(f"{name}[{i}] is not a permutation of 0..{n-1}")
    return arr


def build_rank_matrix(prefs: IntArray) -> IntArray:
    """从偏好序构造 rank[agent, partner]，数值越小表示越偏好。"""
    n = prefs.shape[0]
    rank = np.empty((n, n), dtype=np.int64)
    for agent in range(n):
        for pos, partner in enumerate(prefs[agent]):
            rank[agent, partner] = pos
    return rank


def gale_shapley(proposer_prefs: IntArray, receiver_prefs: IntArray) -> StableMarriageResult:
    """提案方主导 Gale-Shapley。"""
    proposer_prefs = validate_preferences("proposer_prefs", proposer_prefs)
    receiver_prefs = validate_preferences("receiver_prefs", receiver_prefs)
    if proposer_prefs.shape != receiver_prefs.shape:
        raise ValueError("proposer_prefs and receiver_prefs must have same shape")

    n = proposer_prefs.shape[0]
    receiver_rank = build_rank_matrix(receiver_prefs)

    next_choice_idx = np.zeros(n, dtype=np.int64)
    proposer_match = np.full(n, -1, dtype=np.int64)
    receiver_match = np.full(n, -1, dtype=np.int64)

    free_proposers = deque(range(n))
    proposal_count = 0

    while free_proposers:
        p = free_proposers.popleft()
        if proposer_match[p] != -1:
            continue

        if next_choice_idx[p] >= n:
            raise RuntimeError(f"Proposer {p} exhausted all options before matching")

        r = int(proposer_prefs[p, next_choice_idx[p]])
        next_choice_idx[p] += 1
        proposal_count += 1

        current = int(receiver_match[r])
        if current == -1:
            receiver_match[r] = p
            proposer_match[p] = r
            continue

        if receiver_rank[r, p] < receiver_rank[r, current]:
            receiver_match[r] = p
            proposer_match[p] = r
            proposer_match[current] = -1
            free_proposers.append(current)
        else:
            free_proposers.append(p)

    return StableMarriageResult(
        proposer_to_receiver=proposer_match,
        receiver_to_proposer=receiver_match,
        proposal_count=proposal_count,
    )


def is_perfect_matching(result: StableMarriageResult) -> bool:
    """检查是否形成双射匹配。"""
    p2r = result.proposer_to_receiver
    r2p = result.receiver_to_proposer
    n = len(p2r)

    if len(r2p) != n:
        return False
    if np.any((p2r < 0) | (p2r >= n)):
        return False
    if np.any((r2p < 0) | (r2p >= n)):
        return False
    if len(set(p2r.tolist())) != n:
        return False
    if len(set(r2p.tolist())) != n:
        return False

    for p, r in enumerate(p2r):
        if int(r2p[r]) != p:
            return False
    return True


def find_blocking_pairs(
    proposer_prefs: IntArray,
    receiver_prefs: IntArray,
    proposer_to_receiver: IntArray,
) -> List[BlockingPair]:
    """返回阻塞对列表。

    每个元素为 `(p, r, p_current_of_r, r_current_of_p)`。
    """
    proposer_prefs = validate_preferences("proposer_prefs", proposer_prefs)
    receiver_prefs = validate_preferences("receiver_prefs", receiver_prefs)
    n = proposer_prefs.shape[0]

    match = np.asarray(proposer_to_receiver, dtype=np.int64)
    if match.shape != (n,):
        raise ValueError(f"proposer_to_receiver must be shape ({n},)")

    if len(set(match.tolist())) != n or np.any((match < 0) | (match >= n)):
        raise ValueError("proposer_to_receiver must be a bijection over receivers")

    receiver_to_proposer = np.empty(n, dtype=np.int64)
    for p, r in enumerate(match):
        receiver_to_proposer[r] = p

    proposer_rank = build_rank_matrix(proposer_prefs)
    receiver_rank = build_rank_matrix(receiver_prefs)

    blocking_pairs: List[BlockingPair] = []
    for p in range(n):
        current_r = int(match[p])
        current_rank = int(proposer_rank[p, current_r])

        better_receivers = proposer_prefs[p, :current_rank]
        for r in better_receivers:
            r = int(r)
            current_p_of_r = int(receiver_to_proposer[r])
            if receiver_rank[r, p] < receiver_rank[r, current_p_of_r]:
                blocking_pairs.append((p, r, current_p_of_r, current_r))
    return blocking_pairs


def enumerate_stable_matchings(
    proposer_prefs: IntArray,
    receiver_prefs: IntArray,
    max_n: int = 8,
) -> List[IntArray]:
    """穷举全部稳定匹配（仅用于小规模校验）。"""
    proposer_prefs = validate_preferences("proposer_prefs", proposer_prefs)
    receiver_prefs = validate_preferences("receiver_prefs", receiver_prefs)
    n = proposer_prefs.shape[0]

    if n > max_n:
        raise ValueError(f"n={n} is too large for brute force enumeration")

    stable: List[IntArray] = []
    for perm in permutations(range(n)):
        candidate = np.asarray(perm, dtype=np.int64)
        if not find_blocking_pairs(proposer_prefs, receiver_prefs, candidate):
            stable.append(candidate)
    return stable


def evaluate_proposer_optimality(
    proposer_prefs: IntArray,
    stable_matchings: Sequence[IntArray],
    gs_matching: IntArray,
) -> Tuple[bool, IntArray, IntArray]:
    """检查 GS 结果是否对每位提案方都达到稳定匹配集合内最优。"""
    proposer_prefs = validate_preferences("proposer_prefs", proposer_prefs)
    n = proposer_prefs.shape[0]

    if len(stable_matchings) == 0:
        raise ValueError("stable_matchings must be non-empty")

    rank = build_rank_matrix(proposer_prefs)
    gs_rank = rank[np.arange(n), gs_matching]

    best_rank = np.full(n, n, dtype=np.int64)
    for match in stable_matchings:
        m = np.asarray(match, dtype=np.int64)
        if m.shape != (n,):
            raise ValueError("Each stable matching must have shape (n,)")
        current = rank[np.arange(n), m]
        best_rank = np.minimum(best_rank, current)

    return bool(np.array_equal(gs_rank, best_rank)), gs_rank, best_rank


def random_preferences(rng: np.random.Generator, n: int) -> IntArray:
    """生成 n*n 的随机严格偏好。"""
    base = np.arange(n, dtype=np.int64)
    out = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        out[i] = rng.permutation(base)
    return out


def matching_to_dataframe(
    proposer_names: Sequence[str],
    receiver_names: Sequence[str],
    proposer_prefs: IntArray,
    receiver_prefs: IntArray,
    result: StableMarriageResult,
) -> pd.DataFrame:
    """把匹配结果转成便于阅读的表格。"""
    proposer_rank = build_rank_matrix(proposer_prefs)
    receiver_rank = build_rank_matrix(receiver_prefs)

    rows: List[Dict[str, object]] = []
    for p, p_name in enumerate(proposer_names):
        r = int(result.proposer_to_receiver[p])
        rows.append(
            {
                "proposer": p_name,
                "receiver": receiver_names[r],
                "p_rank(0-best)": int(proposer_rank[p, r]),
                "r_rank(0-best)": int(receiver_rank[r, p]),
            }
        )
    return pd.DataFrame(rows)


def run_case(
    title: str,
    proposer_prefs: IntArray,
    receiver_prefs: IntArray,
    proposer_names: Sequence[str],
    receiver_names: Sequence[str],
) -> None:
    """执行单个案例并输出校验结果。"""
    print(f"\n=== {title} ===")

    result = gale_shapley(proposer_prefs, receiver_prefs)
    if not is_perfect_matching(result):
        raise RuntimeError("Result is not a perfect matching")

    blocking = find_blocking_pairs(
        proposer_prefs=proposer_prefs,
        receiver_prefs=receiver_prefs,
        proposer_to_receiver=result.proposer_to_receiver,
    )
    if blocking:
        raise RuntimeError(f"Unexpected blocking pairs: {blocking}")

    n = len(proposer_names)
    stable_matchings = enumerate_stable_matchings(proposer_prefs, receiver_prefs, max_n=8)
    proposer_optimal, gs_rank, best_rank = evaluate_proposer_optimality(
        proposer_prefs=proposer_prefs,
        stable_matchings=stable_matchings,
        gs_matching=result.proposer_to_receiver,
    )

    table = matching_to_dataframe(
        proposer_names=proposer_names,
        receiver_names=receiver_names,
        proposer_prefs=proposer_prefs,
        receiver_prefs=receiver_prefs,
        result=result,
    )

    print(f"participants per side: {n}")
    print(f"total proposals: {result.proposal_count}")
    print(f"stable matchings found by brute force: {len(stable_matchings)}")
    print(f"blocking pairs in GS output: {len(blocking)}")
    print(f"proposer-optimal verified: {proposer_optimal}")
    print(table.to_string(index=False))
    print("gs_rank_per_proposer:", gs_rank.tolist())
    print("best_rank_among_stable:", best_rank.tolist())


def main() -> None:
    """固定样例 + 随机样例，均可复现且无需交互输入。"""
    # Case 1: 经典 4x4 教学样例
    proposer_prefs_case1 = np.array(
        [
            [0, 1, 2, 3],
            [2, 0, 1, 3],
            [1, 2, 3, 0],
            [0, 2, 1, 3],
        ],
        dtype=np.int64,
    )
    receiver_prefs_case1 = np.array(
        [
            [1, 0, 2, 3],
            [2, 1, 0, 3],
            [0, 1, 2, 3],
            [0, 1, 3, 2],
        ],
        dtype=np.int64,
    )

    # Case 2: 随机但可复现
    rng = np.random.default_rng(20260407)
    n2 = 6
    proposer_prefs_case2 = random_preferences(rng, n2)
    receiver_prefs_case2 = random_preferences(rng, n2)

    run_case(
        title="Case 1 - Hand-crafted 4x4",
        proposer_prefs=proposer_prefs_case1,
        receiver_prefs=receiver_prefs_case1,
        proposer_names=[f"M{i}" for i in range(4)],
        receiver_names=[f"W{i}" for i in range(4)],
    )

    run_case(
        title="Case 2 - Reproducible random 6x6",
        proposer_prefs=proposer_prefs_case2,
        receiver_prefs=receiver_prefs_case2,
        proposer_names=[f"P{i}" for i in range(n2)],
        receiver_names=[f"R{i}" for i in range(n2)],
    )


if __name__ == "__main__":
    main()
