"""树形DP - 树的最大独立集 MVP。

实现内容：
- 主算法：树形 DP（每个节点 two-state: 选/不选）+ 解重建
- 基线算法：记忆化递归
- 校验算法：小规模暴力枚举

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class TreeMISResult:
    n: int
    best_weight: int
    selected_nodes: list[int]
    selected_weight: int
    include: np.ndarray
    exclude: np.ndarray


def to_node_count(n: int) -> int:
    if not isinstance(n, int):
        raise ValueError(f"n must be int, got {type(n)}")
    if n < 0:
        raise ValueError("n must be non-negative")
    return n


def to_weight_array(n: int, weights: Sequence[int] | np.ndarray | None) -> np.ndarray:
    if n == 0:
        if weights is None:
            return np.zeros(0, dtype=np.int64)
        arr0 = np.asarray(weights)
        if arr0.size != 0:
            raise ValueError("weights must be empty when n=0")
        return np.zeros(0, dtype=np.int64)

    if weights is None:
        return np.ones(n, dtype=np.int64)

    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape={arr.shape}")
    if arr.size != n:
        raise ValueError(f"weights length mismatch: expected {n}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("weights contains non-finite values")

    rounded = np.rint(arr)
    if not np.allclose(arr, rounded):
        raise ValueError("weights must be integer-valued in this MVP")

    ints = rounded.astype(np.int64)
    if np.any(ints < 0):
        raise ValueError("weights must be non-negative")
    return ints


def to_edge_list(n: int, edges: Sequence[tuple[int, int]] | np.ndarray) -> list[tuple[int, int]]:
    if n == 0:
        if len(edges) != 0:
            raise ValueError("n=0 requires empty edges")
        return []

    arr = np.asarray(edges, dtype=int)
    if n == 1:
        if arr.size != 0:
            raise ValueError("single-node tree must have 0 edges")
        return []

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edges must be a 2D array-like with shape (m, 2)")
    if arr.shape[0] != n - 1:
        raise ValueError(f"tree must have exactly n-1 edges, expected {n-1}, got {arr.shape[0]}")

    edge_list: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for u_raw, v_raw in arr.tolist():
        u = int(u_raw)
        v = int(v_raw)

        if u == v:
            raise ValueError("self-loop is not allowed in a tree")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge endpoint out of range: ({u}, {v}), n={n}")

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            raise ValueError(f"duplicate edge detected: ({a}, {b})")
        seen.add((a, b))
        edge_list.append((u, v))

    return edge_list


def build_adjacency(n: int, edge_list: Sequence[tuple[int, int]]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    for nbrs in adj:
        nbrs.sort()
    return adj


def assert_connected_tree(n: int, adj: Sequence[Sequence[int]]) -> None:
    if n <= 1:
        return

    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True

    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    if not np.all(visited):
        raise ValueError("input edges do not form a connected tree")


def prepare_tree(
    n: int,
    edges: Sequence[tuple[int, int]] | np.ndarray,
    weights: Sequence[int] | np.ndarray | None,
) -> tuple[list[tuple[int, int]], list[list[int]], np.ndarray]:
    n_checked = to_node_count(n)
    edge_list = to_edge_list(n_checked, edges)
    adj = build_adjacency(n_checked, edge_list)
    assert_connected_tree(n_checked, adj)
    w = to_weight_array(n_checked, weights)
    return edge_list, adj, w


def build_rooted_tree(
    adj: Sequence[Sequence[int]], root: int
) -> tuple[np.ndarray, list[list[int]], list[int]]:
    n = len(adj)
    if n == 0:
        return np.zeros(0, dtype=np.int64), [], []
    if not (0 <= root < n):
        raise ValueError(f"root out of range: root={root}, n={n}")

    parent = np.full(n, -1, dtype=np.int64)
    children = [[] for _ in range(n)]
    order: list[int] = []

    stack = [root]
    parent[root] = root

    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v == parent[u]:
                continue
            if parent[v] != -1:
                continue
            parent[v] = u
            children[u].append(v)
            stack.append(v)

    if len(order) != n:
        raise ValueError("failed to root the tree: graph appears disconnected")

    for nodes in children:
        nodes.sort()

    return parent, children, order


def tree_max_independent_set(
    n: int,
    edges: Sequence[tuple[int, int]] | np.ndarray,
    weights: Sequence[int] | np.ndarray | None = None,
    root: int = 0,
) -> TreeMISResult:
    """树形 DP 主算法，返回最优权重与一组最优独立集节点。"""
    _, adj, w = prepare_tree(n=n, edges=edges, weights=weights)

    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return TreeMISResult(
            n=0,
            best_weight=0,
            selected_nodes=[],
            selected_weight=0,
            include=empty,
            exclude=empty,
        )

    _, children, order = build_rooted_tree(adj, root=root)

    include = np.zeros(n, dtype=np.int64)
    exclude = np.zeros(n, dtype=np.int64)

    for u in reversed(order):
        take = int(w[u])
        skip = 0
        for v in children[u]:
            take += int(exclude[v])
            skip += int(max(include[v], exclude[v]))
        include[u] = take
        exclude[u] = skip

    selected_nodes: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]

    while stack:
        u, parent_taken = stack.pop()

        if parent_taken:
            take_u = False
        else:
            if include[u] > exclude[u]:
                take_u = True
            elif include[u] < exclude[u]:
                take_u = False
            else:
                # 平局时固定选择 take，保证输出可复现。
                take_u = True

        if take_u:
            selected_nodes.append(u)

        for v in reversed(children[u]):
            stack.append((v, take_u))

    selected_nodes.sort()
    selected_weight = int(np.sum(w[selected_nodes])) if selected_nodes else 0
    best_weight = int(max(include[root], exclude[root]))

    return TreeMISResult(
        n=n,
        best_weight=best_weight,
        selected_nodes=selected_nodes,
        selected_weight=selected_weight,
        include=include,
        exclude=exclude,
    )


def tree_mis_top_down(
    n: int,
    edges: Sequence[tuple[int, int]] | np.ndarray,
    weights: Sequence[int] | np.ndarray | None = None,
    root: int = 0,
) -> int:
    """记忆化递归基线，仅返回最优权重。"""
    _, adj, w = prepare_tree(n=n, edges=edges, weights=weights)
    if n == 0:
        return 0

    _, children, _ = build_rooted_tree(adj, root=root)

    @lru_cache(maxsize=None)
    def solve(u: int, parent_taken: bool) -> int:
        if parent_taken:
            return int(sum(solve(v, False) for v in children[u]))

        take_u = int(w[u]) + int(sum(solve(v, True) for v in children[u]))
        skip_u = int(sum(solve(v, False) for v in children[u]))
        return max(take_u, skip_u)

    return int(solve(root, False))


def brute_force_tree_mis(
    n: int,
    edges: Sequence[tuple[int, int]] | np.ndarray,
    weights: Sequence[int] | np.ndarray | None = None,
) -> tuple[int, list[int]]:
    """小规模暴力枚举基线（指数复杂度）。"""
    edge_list, _, w = prepare_tree(n=n, edges=edges, weights=weights)

    if n == 0:
        return 0, []

    best_weight = -1
    best_nodes: list[int] = []

    for mask in range(1 << n):
        valid = True
        for u, v in edge_list:
            if ((mask >> u) & 1) and ((mask >> v) & 1):
                valid = False
                break
        if not valid:
            continue

        nodes = [i for i in range(n) if (mask >> i) & 1]
        total = int(np.sum(w[nodes])) if nodes else 0

        if total > best_weight or (total == best_weight and nodes < best_nodes):
            best_weight = total
            best_nodes = nodes

    return int(best_weight), best_nodes


def is_independent_set(edge_list: Sequence[tuple[int, int]], nodes: Sequence[int]) -> bool:
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        return False
    for u, v in edge_list:
        if u in node_set and v in node_set:
            return False
    return True


def run_case(
    name: str,
    n: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[int] | None = None,
    expected: int | None = None,
    brute_force_limit: int = 20,
) -> None:
    edge_list, _, w = prepare_tree(n=n, edges=edges, weights=weights)

    result = tree_max_independent_set(n=n, edges=edges, weights=w)
    baseline = tree_mis_top_down(n=n, edges=edges, weights=w)

    brute_weight = None
    brute_nodes = None
    if n <= brute_force_limit:
        brute_weight, brute_nodes = brute_force_tree_mis(n=n, edges=edges, weights=w)

    valid_set = is_independent_set(edge_list=edge_list, nodes=result.selected_nodes)
    selected_sum_match = result.selected_weight == result.best_weight
    baseline_match = baseline == result.best_weight
    brute_match = None if brute_weight is None else (brute_weight == result.best_weight)

    print(f"=== {name} ===")
    print(f"n={n}, edges={list(edge_list)}")
    print(f"weights={w.tolist()}")
    print(
        "dp_result      -> "
        f"best_weight={result.best_weight}, selected_nodes={result.selected_nodes}, "
        f"selected_weight={result.selected_weight}"
    )
    print(f"top_down       -> best_weight={baseline}")
    if brute_weight is None:
        print("bruteforce     -> skipped")
    else:
        print(f"bruteforce     -> best_weight={brute_weight}, selected_nodes={brute_nodes}")

    print(
        "checks         -> "
        f"independent={valid_set}, selected_sum_match={selected_sum_match}, "
        f"top_down_match={baseline_match}, brute_match={brute_match}"
    )
    print()

    if not valid_set:
        raise AssertionError("selected_nodes is not an independent set")
    if not selected_sum_match:
        raise AssertionError("selected nodes weight does not match best_weight")
    if not baseline_match:
        raise AssertionError("tree DP and top-down baseline mismatch")
    if brute_weight is not None and brute_weight != result.best_weight:
        raise AssertionError("tree DP and brute-force baseline mismatch")
    if expected is not None and result.best_weight != expected:
        raise AssertionError(
            f"unexpected best_weight in {name}: got {result.best_weight}, expected {expected}"
        )


def generate_random_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n <= 1:
        return []
    edges: list[tuple[int, int]] = []
    for node in range(1, n):
        parent = int(rng.integers(0, node))
        edges.append((parent, node))
    return edges


def randomized_cross_check(
    trials: int = 300,
    max_n: int = 14,
    max_weight: int = 20,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        n = int(rng.integers(0, max_n + 1))
        edges = generate_random_tree_edges(n, rng)
        weights = (
            rng.integers(1, max_weight + 1, size=n).astype(np.int64).tolist()
            if n > 0
            else []
        )

        result = tree_max_independent_set(n=n, edges=edges, weights=weights)
        baseline = tree_mis_top_down(n=n, edges=edges, weights=weights)
        brute_weight, _ = brute_force_tree_mis(n=n, edges=edges, weights=weights)

        if result.best_weight != baseline:
            raise AssertionError("random check failed: DP vs top-down mismatch")
        if result.best_weight != brute_weight:
            raise AssertionError("random check failed: DP vs brute-force mismatch")

    print(
        "Randomized cross-check passed: "
        f"trials={trials}, max_n={max_n}, max_weight={max_weight}, seed={seed}."
    )


def main() -> None:
    # 无权重（默认每点权重=1）
    run_case(
        name="Case 1: chain-unweighted",
        n=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        weights=None,
        expected=3,
    )

    run_case(
        name="Case 2: star-unweighted",
        n=6,
        edges=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        weights=None,
        expected=5,
    )

    run_case(
        name="Case 3: single-node",
        n=1,
        edges=[],
        weights=None,
        expected=1,
    )

    # 加权版本（更通用；无权版本是其特例）
    run_case(
        name="Case 4: weighted-binary-tree",
        n=7,
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        weights=[5, 3, 4, 4, 1, 2, 6],
        expected=18,
    )

    run_case(
        name="Case 5: empty-tree",
        n=0,
        edges=[],
        weights=[],
        expected=0,
    )

    randomized_cross_check(trials=300, max_n=14, max_weight=20, seed=2026)


if __name__ == "__main__":
    main()
