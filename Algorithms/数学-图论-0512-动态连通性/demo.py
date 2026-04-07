"""动态连通性 MVP: 线段树分治时间区间 + 可回滚并查集.

该 demo 支持离线处理如下操作序列（无交互输入）:
- ("add", u, v): 添加无向边
- ("remove", u, v): 删除无向边
- ("query", u, v): 询问当前时刻 u 与 v 是否连通
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence, Tuple

Op = Tuple[str, int, int]
Edge = Tuple[int, int]


def norm_edge(u: int, v: int) -> Edge:
    """规范化无向边表示."""
    return (u, v) if u <= v else (v, u)


class RollbackDSU:
    """可回滚并查集（不做路径压缩，按 size 合并）."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.parent = list(range(n))
        self.size = [1] * n
        self.history: List[Tuple[int, int, int]] = []

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.history.append((rb, ra, self.size[ra]))
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def snapshot(self) -> int:
        return len(self.history)

    def rollback(self, checkpoint: int) -> None:
        while len(self.history) > checkpoint:
            rb, ra, old_size_ra = self.history.pop()
            self.parent[rb] = rb
            self.size[ra] = old_size_ra


@dataclass
class OfflineDynamicConnectivity:
    """离线动态连通性求解器.

    时间轴上的每条边生命周期会被分配到线段树节点，
    DFS 线段树时用 DSU 回滚机制维护“当前区间内有效边集”的连通性状态。
    """

    n: int
    operations: Sequence[Op]

    def _validate_vertex(self, x: int) -> None:
        if not (0 <= x < self.n):
            raise ValueError(f"vertex {x} out of range [0, {self.n - 1}]")

    def _build_edge_intervals(self) -> List[Tuple[int, int, Edge]]:
        """把 add/remove 序列转成边活跃区间 [l, r)."""
        active_count: Dict[Edge, int] = {}
        start_time: Dict[Edge, int] = {}
        intervals: List[Tuple[int, int, Edge]] = []

        for t, (typ, u, v) in enumerate(self.operations):
            self._validate_vertex(u)
            self._validate_vertex(v)
            if typ == "query":
                continue

            e = norm_edge(u, v)
            c = active_count.get(e, 0)

            if typ == "add":
                if c == 0:
                    start_time[e] = t
                active_count[e] = c + 1
            elif typ == "remove":
                if c == 0:
                    raise ValueError(f"remove non-existent edge {e} at t={t}")
                c -= 1
                if c == 0:
                    intervals.append((start_time[e], t, e))
                    del start_time[e]
                    del active_count[e]
                else:
                    active_count[e] = c
            else:
                raise ValueError(f"unknown op type: {typ}")

        q = len(self.operations)
        for e, l in start_time.items():
            intervals.append((l, q, e))
        return intervals

    def solve(self) -> List[bool]:
        q = len(self.operations)
        if q == 0:
            return []

        intervals = self._build_edge_intervals()

        size = 1
        while size < q:
            size <<= 1
        tree: List[List[Edge]] = [[] for _ in range(size << 1)]

        def add_interval(l: int, r: int, e: Edge) -> None:
            l += size
            r += size
            while l < r:
                if l & 1:
                    tree[l].append(e)
                    l += 1
                if r & 1:
                    r -= 1
                    tree[r].append(e)
                l >>= 1
                r >>= 1

        for l, r, e in intervals:
            if l < r:
                add_interval(l, r, e)

        dsu = RollbackDSU(self.n)
        answers_by_time: Dict[int, bool] = {}

        def dfs(idx: int, left: int, right: int) -> None:
            checkpoint = dsu.snapshot()
            for u, v in tree[idx]:
                if u != v:
                    dsu.union(u, v)

            if right - left == 1:
                if left < q:
                    typ, u, v = self.operations[left]
                    if typ == "query":
                        answers_by_time[left] = dsu.connected(u, v)
            else:
                mid = (left + right) // 2
                dfs(idx * 2, left, mid)
                dfs(idx * 2 + 1, mid, right)

            dsu.rollback(checkpoint)

        dfs(1, 0, size)

        # 按 query 出现顺序返回答案
        out: List[bool] = []
        for t, (typ, _, _) in enumerate(self.operations):
            if typ == "query":
                out.append(answers_by_time[t])
        return out


def naive_online_solver(n: int, operations: Sequence[Op]) -> List[bool]:
    """用于对拍的朴素在线求解器（每次 query 现建图 BFS）."""
    active_count: Dict[Edge, int] = {}
    answers: List[bool] = []

    def connected(u: int, v: int) -> bool:
        if u == v:
            return True
        g = [[] for _ in range(n)]
        for (a, b), c in active_count.items():
            if c > 0 and a != b:
                g[a].append(b)
                g[b].append(a)
        dq = deque([u])
        vis = [False] * n
        vis[u] = True
        while dq:
            x = dq.popleft()
            for y in g[x]:
                if not vis[y]:
                    if y == v:
                        return True
                    vis[y] = True
                    dq.append(y)
        return False

    for t, (typ, u, v) in enumerate(operations):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"vertex out of range at t={t}")
        e = norm_edge(u, v)

        if typ == "add":
            active_count[e] = active_count.get(e, 0) + 1
        elif typ == "remove":
            c = active_count.get(e, 0)
            if c == 0:
                raise ValueError(f"remove non-existent edge {e} at t={t}")
            if c == 1:
                del active_count[e]
            else:
                active_count[e] = c - 1
        elif typ == "query":
            answers.append(connected(u, v))
        else:
            raise ValueError(f"unknown op type: {typ}")

    return answers


def format_ops(operations: Iterable[Op]) -> str:
    lines = []
    for i, (typ, u, v) in enumerate(operations):
        lines.append(f"{i:02d}. ({typ:6s}, {u}, {v})")
    return "\n".join(lines)


def build_random_case(
    n: int,
    steps: int,
    rng: random.Random,
) -> List[Op]:
    """生成合法随机序列（remove 仅针对当前活跃边）."""
    ops: List[Op] = []
    active_count: Dict[Edge, int] = {}

    for _ in range(steps):
        p = rng.random()
        can_remove = bool(active_count)

        if p < 0.45:
            u, v = rng.randrange(n), rng.randrange(n)
            ops.append(("query", u, v))
        elif p < 0.75 or not can_remove:
            u, v = rng.randrange(n), rng.randrange(n)
            e = norm_edge(u, v)
            active_count[e] = active_count.get(e, 0) + 1
            ops.append(("add", u, v))
        else:
            e = rng.choice(list(active_count.keys()))
            u, v = e
            c = active_count[e]
            if c == 1:
                del active_count[e]
            else:
                active_count[e] = c - 1
            ops.append(("remove", u, v))

    # 追加几次 query，便于观察尾部状态
    for _ in range(min(5, n)):
        u, v = rng.randrange(n), rng.randrange(n)
        ops.append(("query", u, v))
    return ops


def run_fixed_demo() -> None:
    n = 6
    operations: List[Op] = [
        ("add", 0, 1),
        ("add", 1, 2),
        ("query", 0, 2),
        ("query", 0, 3),
        ("add", 2, 3),
        ("query", 0, 3),
        ("remove", 1, 2),
        ("query", 0, 3),
        ("add", 4, 5),
        ("query", 0, 5),
        ("add", 3, 4),
        ("query", 0, 5),
        ("remove", 2, 3),
        ("query", 0, 5),
    ]

    offline_ans = OfflineDynamicConnectivity(n, operations).solve()
    naive_ans = naive_online_solver(n, operations)
    if offline_ans != naive_ans:
        raise AssertionError("fixed demo mismatch between offline and naive solver")

    print("=== Fixed Demo ===")
    print("Operations:")
    print(format_ops(operations))
    print("\nQuery answers (in order):")
    print(offline_ans)
    print()


def run_random_crosscheck() -> None:
    print("=== Random Cross Check ===")
    rng = random.Random(20260407)
    trials = 20
    for tid in range(1, trials + 1):
        n = rng.randint(5, 10)
        steps = rng.randint(20, 45)
        ops = build_random_case(n, steps, rng)
        offline_ans = OfflineDynamicConnectivity(n, ops).solve()
        naive_ans = naive_online_solver(n, ops)
        if offline_ans != naive_ans:
            print("Mismatch detected!")
            print(f"trial={tid}, n={n}, steps={steps}")
            print(format_ops(ops))
            print("offline:", offline_ans)
            print("naive  :", naive_ans)
            raise AssertionError("random cross check failed")
    print(f"All {trials} random trials passed.\n")


def main() -> None:
    run_fixed_demo()
    run_random_crosscheck()
    print("Dynamic connectivity MVP finished successfully.")


if __name__ == "__main__":
    main()
