"""一般图最大匹配（Edmonds Blossom）最小可运行示例。

- 不依赖交互输入，直接运行会执行固定用例与随机回归测试。
- 实现目标：无权一般图最大基数匹配（maximum cardinality matching）。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable, List, Sequence, Tuple


Edge = Tuple[int, int]


@dataclass
class MatchingResult:
    """保存匹配结果。"""

    match: List[int]

    @property
    def cardinality(self) -> int:
        return sum(1 for v in self.match if v != -1) // 2

    def pairs(self) -> List[Tuple[int, int]]:
        res: List[Tuple[int, int]] = []
        for u, v in enumerate(self.match):
            if v != -1 and u < v:
                res.append((u, v))
        return res


class EdmondsBlossom:
    """Edmonds 开花算法（Gabow/CP 常见实现思路，O(V^3)）。"""

    def __init__(self, n: int, edges: Iterable[Edge]) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.g: List[List[int]] = [[] for _ in range(n)]
        seen = set()
        for a, b in edges:
            if not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"edge {(a, b)} contains invalid vertex id")
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in seen:
                continue
            seen.add((u, v))
            self.g[u].append(v)
            self.g[v].append(u)

        self.match = [-1] * n
        self.parent = [-1] * n
        self.base = list(range(n))
        self.used = [False] * n
        self.blossom = [False] * n

    def _lca(self, a: int, b: int) -> int:
        """寻找 a 与 b 在交替树上的最低公共祖先（按 base 压缩后）。"""
        used_path = [False] * self.n
        while True:
            a = self.base[a]
            used_path[a] = True
            if self.match[a] == -1:
                break
            a = self.parent[self.match[a]]
        while True:
            b = self.base[b]
            if used_path[b]:
                return b
            b = self.parent[self.match[b]]

    def _mark_path(self, v: int, b: int, child: int) -> None:
        """把从 v 到基点 b 的路径标记为 blossom，并修正 parent。"""
        while self.base[v] != b:
            bv = self.base[v]
            bm = self.base[self.match[v]]
            self.blossom[bv] = True
            self.blossom[bm] = True
            self.parent[v] = child
            child = self.match[v]
            v = self.parent[self.match[v]]

    def _find_augmenting_path(self, root: int) -> int:
        """BFS 搜索从 root 出发的增广路，找不到则返回 -1。"""
        self.used = [False] * self.n
        self.parent = [-1] * self.n
        self.base = list(range(self.n))

        queue: List[int] = [root]
        self.used[root] = True

        q_head = 0
        while q_head < len(queue):
            v = queue[q_head]
            q_head += 1

            for u in self.g[v]:
                if self.base[v] == self.base[u] or self.match[v] == u:
                    continue

                # 发现奇环（blossom）并进行收缩
                if u == root or (self.match[u] != -1 and self.parent[self.match[u]] != -1):
                    cur_base = self._lca(v, u)
                    self.blossom = [False] * self.n
                    self._mark_path(v, cur_base, u)
                    self._mark_path(u, cur_base, v)

                    for x in range(self.n):
                        if self.blossom[self.base[x]]:
                            self.base[x] = cur_base
                            if not self.used[x]:
                                self.used[x] = True
                                queue.append(x)

                elif self.parent[u] == -1:
                    self.parent[u] = v
                    if self.match[u] == -1:
                        return u
                    nxt = self.match[u]
                    self.used[nxt] = True
                    queue.append(nxt)

        return -1

    def solve(self) -> MatchingResult:
        """求一般图最大基数匹配。"""
        for v in range(self.n):
            if self.match[v] != -1:
                continue
            end = self._find_augmenting_path(v)
            if end == -1:
                continue

            # 沿增广路翻转匹配边与非匹配边
            while end != -1:
                pv = self.parent[end]
                nv = self.match[pv] if pv != -1 else -1
                self.match[end] = pv
                if pv != -1:
                    self.match[pv] = end
                end = nv

        return MatchingResult(match=self.match.copy())


def brute_force_maximum_matching_size(n: int, edges: Sequence[Edge]) -> int:
    """小规模图暴力求最大匹配规模，用于回归校验。"""
    adj = [[False] * n for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
        adj[a][b] = True
        adj[b][a] = True

    memo = {}

    def dfs(mask: int) -> int:
        if mask in memo:
            return memo[mask]

        i = 0
        while i < n and (mask >> i) & 1:
            i += 1
        if i == n:
            memo[mask] = 0
            return 0

        # 选项1：i 不参与匹配
        best = dfs(mask | (1 << i))

        # 选项2：i 与某个可配顶点 j 匹配
        for j in range(i + 1, n):
            if ((mask >> j) & 1) == 0 and adj[i][j]:
                best = max(best, 1 + dfs(mask | (1 << i) | (1 << j)))

        memo[mask] = best
        return best

    return dfs(0)


def run_fixed_demo() -> None:
    print("=== 固定样例：包含奇环（blossom）的一般图 ===")
    n = 7
    edges: List[Edge] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),  # 0-1-2-3-4-0 构成 5 环（奇环）
        (0, 5),
        (2, 6),  # 从奇环引出两个外部点
        (1, 6),
    ]

    solver = EdmondsBlossom(n, edges)
    result = solver.solve()
    pairs = result.pairs()

    print(f"顶点数: {n}")
    print(f"边数: {len(edges)}")
    print(f"匹配边: {pairs}")
    print(f"匹配规模: {result.cardinality}")

    # 小图用暴力法确认最优性
    brute = brute_force_maximum_matching_size(n, edges)
    print(f"暴力最优规模: {brute}")
    assert result.cardinality == brute, "固定样例未达到最优匹配规模"


def run_random_regression_tests(num_tests: int = 40, seed: int = 7) -> None:
    print("\n=== 随机小图回归测试（对比暴力最优）===")
    random.seed(seed)

    passed = 0
    for t in range(1, num_tests + 1):
        n = random.randint(2, 9)
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)

        # 随机采样边，控制图密度在 [0.2, 0.7]
        target_m = random.randint(max(1, int(0.2 * len(all_edges))), max(1, int(0.7 * len(all_edges))))
        edges = all_edges[:target_m]

        solver = EdmondsBlossom(n, edges)
        got = solver.solve().cardinality
        expected = brute_force_maximum_matching_size(n, edges)
        if got != expected:
            raise AssertionError(
                f"测试 {t} 失败: n={n}, m={len(edges)}, got={got}, expected={expected}"
            )
        passed += 1

    print(f"通过 {passed}/{num_tests} 组随机测试。")


def main() -> None:
    run_fixed_demo()
    run_random_regression_tests(num_tests=60, seed=2026)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
