"""树链剖分（Heavy-Light Decomposition）最小可运行示例。

功能：
1) 点更新
2) 路径和 / 路径最大值查询
3) 子树和 / 子树最大值查询

脚本内置测试数据，无需交互输入，直接运行：
    python3 demo.py
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Tuple


class SegmentTree:
    """线段树：支持区间和、区间最大值、单点更新。"""

    def __init__(self, values: Sequence[int]) -> None:
        self.n = len(values)
        size = 1
        while size < self.n:
            size <<= 1
        self.size = size

        self.tree_sum = [0] * (2 * size)
        self.tree_max = [-10**18] * (2 * size)

        for i, value in enumerate(values):
            idx = size + i
            self.tree_sum[idx] = value
            self.tree_max[idx] = value

        for idx in range(size - 1, 0, -1):
            self.tree_sum[idx] = self.tree_sum[idx << 1] + self.tree_sum[idx << 1 | 1]
            self.tree_max[idx] = max(self.tree_max[idx << 1], self.tree_max[idx << 1 | 1])

    def point_update(self, index: int, value: int) -> None:
        idx = self.size + index
        self.tree_sum[idx] = value
        self.tree_max[idx] = value
        idx >>= 1
        while idx:
            self.tree_sum[idx] = self.tree_sum[idx << 1] + self.tree_sum[idx << 1 | 1]
            self.tree_max[idx] = max(self.tree_max[idx << 1], self.tree_max[idx << 1 | 1])
            idx >>= 1

    def range_sum(self, left: int, right: int) -> int:
        """返回半开区间 [left, right) 的和。"""
        left += self.size
        right += self.size
        result = 0
        while left < right:
            if left & 1:
                result += self.tree_sum[left]
                left += 1
            if right & 1:
                right -= 1
                result += self.tree_sum[right]
            left >>= 1
            right >>= 1
        return result

    def range_max(self, left: int, right: int) -> int:
        """返回半开区间 [left, right) 的最大值。"""
        left += self.size
        right += self.size
        result = -10**18
        while left < right:
            if left & 1:
                result = max(result, self.tree_max[left])
                left += 1
            if right & 1:
                right -= 1
                result = max(result, self.tree_max[right])
            left >>= 1
            right >>= 1
        return result


class HeavyLightDecomposition:
    """点权树链剖分实现。"""

    def __init__(self, n: int, adj: List[List[int]], values: Sequence[int], root: int = 0) -> None:
        self.n = n
        self.adj = adj
        self.root = root
        self.values = list(values)

        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [0] * n
        self.heavy = [-1] * n

        self.head = [0] * n
        self.pos = [0] * n
        self.node_at_pos = [0] * n
        self._current_pos = 0

        self._dfs_sizes(root, -1)
        self._dfs_decompose(root, root)

        base = [0] * n
        for node in range(n):
            base[self.pos[node]] = self.values[node]
        self.seg = SegmentTree(base)

    def _dfs_sizes(self, u: int, p: int) -> None:
        self.parent[u] = p
        self.size[u] = 1
        max_subtree_size = 0

        for v in self.adj[u]:
            if v == p:
                continue
            self.depth[v] = self.depth[u] + 1
            self._dfs_sizes(v, u)
            self.size[u] += self.size[v]
            if self.size[v] > max_subtree_size:
                max_subtree_size = self.size[v]
                self.heavy[u] = v

    def _dfs_decompose(self, u: int, head: int) -> None:
        self.head[u] = head
        self.pos[u] = self._current_pos
        self.node_at_pos[self._current_pos] = u
        self._current_pos += 1

        heavy_child = self.heavy[u]
        if heavy_child != -1:
            self._dfs_decompose(heavy_child, head)

        for v in self.adj[u]:
            if v == self.parent[u] or v == heavy_child:
                continue
            self._dfs_decompose(v, v)

    def update_node(self, u: int, new_value: int) -> None:
        self.values[u] = new_value
        self.seg.point_update(self.pos[u], new_value)

    def query_path_sum(self, u: int, v: int) -> int:
        result = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            head_u = self.head[u]
            result += self.seg.range_sum(self.pos[head_u], self.pos[u] + 1)
            u = self.parent[head_u]

        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result += self.seg.range_sum(self.pos[u], self.pos[v] + 1)
        return result

    def query_path_max(self, u: int, v: int) -> int:
        result = -10**18
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            head_u = self.head[u]
            result = max(result, self.seg.range_max(self.pos[head_u], self.pos[u] + 1))
            u = self.parent[head_u]

        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, self.seg.range_max(self.pos[u], self.pos[v] + 1))
        return result

    def query_subtree_sum(self, u: int) -> int:
        left = self.pos[u]
        right = left + self.size[u]
        return self.seg.range_sum(left, right)

    def query_subtree_max(self, u: int) -> int:
        left = self.pos[u]
        right = left + self.size[u]
        return self.seg.range_max(left, right)


def build_adj(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def naive_path_nodes(u: int, v: int, parent: Sequence[int], depth: Sequence[int]) -> List[int]:
    left_nodes: List[int] = []
    right_nodes: List[int] = []

    uu, vv = u, v
    while depth[uu] > depth[vv]:
        left_nodes.append(uu)
        uu = parent[uu]
    while depth[vv] > depth[uu]:
        right_nodes.append(vv)
        vv = parent[vv]

    while uu != vv:
        left_nodes.append(uu)
        right_nodes.append(vv)
        uu = parent[uu]
        vv = parent[vv]

    lca = uu
    return left_nodes + [lca] + list(reversed(right_nodes))


def naive_subtree_nodes(hld: HeavyLightDecomposition, u: int) -> List[int]:
    left = hld.pos[u]
    right = left + hld.size[u]
    return [hld.node_at_pos[idx] for idx in range(left, right)]


def run_demo() -> None:
    n = 10
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
        (5, 7),
        (5, 8),
        (6, 9),
    ]
    values = [5, 3, 8, 6, 1, 7, 4, 2, 9, 10]

    adj = build_adj(n, edges)
    hld = HeavyLightDecomposition(n=n, adj=adj, values=values, root=0)

    path_queries = [(3, 4), (3, 8), (7, 9), (0, 9)]
    subtree_queries = [0, 1, 2, 5]

    print("=== Initial Path Queries ===")
    for u, v in path_queries:
        nodes = naive_path_nodes(u, v, hld.parent, hld.depth)
        naive_sum = sum(values[node] for node in nodes)
        naive_max = max(values[node] for node in nodes)

        hld_sum = hld.query_path_sum(u, v)
        hld_max = hld.query_path_max(u, v)

        print(
            f"path({u},{v}) -> nodes={nodes}, "
            f"hld_sum={hld_sum}, naive_sum={naive_sum}, "
            f"hld_max={hld_max}, naive_max={naive_max}"
        )
        assert hld_sum == naive_sum
        assert hld_max == naive_max

    print("\n=== Initial Subtree Queries ===")
    for u in subtree_queries:
        nodes = naive_subtree_nodes(hld, u)
        naive_sum = sum(values[node] for node in nodes)
        naive_max = max(values[node] for node in nodes)

        hld_sum = hld.query_subtree_sum(u)
        hld_max = hld.query_subtree_max(u)

        print(
            f"subtree({u}) -> nodes={nodes}, "
            f"hld_sum={hld_sum}, naive_sum={naive_sum}, "
            f"hld_max={hld_max}, naive_max={naive_max}"
        )
        assert hld_sum == naive_sum
        assert hld_max == naive_max

    updates = [(7, 11), (3, -2), (0, 12)]
    print("\n=== Point Updates + Recheck ===")
    for node, new_value in updates:
        print(f"update node {node}: {values[node]} -> {new_value}")
        hld.update_node(node, new_value)
        values[node] = new_value

        u, v = 7, 9
        nodes = naive_path_nodes(u, v, hld.parent, hld.depth)
        naive_sum = sum(values[x] for x in nodes)
        naive_max = max(values[x] for x in nodes)
        hld_sum = hld.query_path_sum(u, v)
        hld_max = hld.query_path_max(u, v)
        print(
            f"  check path({u},{v}) -> hld_sum={hld_sum}, naive_sum={naive_sum}, "
            f"hld_max={hld_max}, naive_max={naive_max}"
        )
        assert hld_sum == naive_sum
        assert hld_max == naive_max

    print("\nAll checks passed: HLD implementation is consistent with naive verification.")


def main() -> None:
    sys.setrecursionlimit(1_000_000)
    run_demo()


if __name__ == "__main__":
    main()
