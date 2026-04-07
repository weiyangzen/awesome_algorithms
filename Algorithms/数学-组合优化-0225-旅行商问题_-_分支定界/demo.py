"""旅行商问题(TSP) - 分支定界(Branch and Bound) 最小可运行示例.

运行方式:
    python3 demo.py
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchStats:
    expanded_nodes: int = 0
    pruned_nodes: int = 0


class TSPBranchAndBound:
    """对称 TSP 的分支定界求解器.

    使用思路:
    - 状态: 当前路径 + 已访问集合 + 当前累计代价
    - 下界: first_min / second_min 组合下界
    - 剪枝: 当前累计代价 + 下界 >= 已知最优上界 时剪枝
    """

    def __init__(self, distance_matrix: np.ndarray, start: int = 0) -> None:
        dist = np.asarray(distance_matrix, dtype=float)
        if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
            raise ValueError("distance_matrix must be a square matrix")
        if not (0 <= start < dist.shape[0]):
            raise ValueError("start city index out of range")

        self.dist = dist
        self.n = dist.shape[0]
        self.start = start
        self.stats = SearchStats()

        self.best_cost = math.inf
        self.best_path: list[int] = []

        self.first_min = np.zeros(self.n, dtype=float)
        self.second_min = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            a, b = self._two_smallest_finite(self.dist[i, :])
            self.first_min[i] = a
            self.second_min[i] = b

    @staticmethod
    def _two_smallest_finite(row: np.ndarray) -> tuple[float, float]:
        vals = np.sort(row[np.isfinite(row)])
        if vals.size < 2:
            raise ValueError("each city must have at least 2 finite edges")
        return float(vals[0]), float(vals[1])

    def solve(self) -> tuple[float, list[int], SearchStats]:
        if self.n == 1:
            return 0.0, [self.start, self.start], self.stats

        initial_bound = 0.5 * float(np.sum(self.first_min + self.second_min))
        path = [-1] * (self.n + 1)
        visited = [False] * self.n

        path[0] = self.start
        visited[self.start] = True

        self._dfs(
            curr_bound=initial_bound,
            curr_weight=0.0,
            level=1,
            path=path,
            visited=visited,
        )
        return self.best_cost, self.best_path, self.stats

    def _dfs(
        self,
        curr_bound: float,
        curr_weight: float,
        level: int,
        path: list[int],
        visited: list[bool],
    ) -> None:
        if level == self.n:
            last = path[level - 1]
            back = self.dist[last, self.start]
            if np.isfinite(back):
                total = curr_weight + float(back)
                if total < self.best_cost:
                    self.best_cost = total
                    self.best_path = path[:level] + [self.start]
            return

        prev = path[level - 1]
        for nxt in range(self.n):
            if visited[nxt]:
                continue
            edge = self.dist[prev, nxt]
            if not np.isfinite(edge):
                continue

            next_weight = curr_weight + float(edge)
            if level == 1:
                next_bound = curr_bound - (self.first_min[prev] + self.first_min[nxt]) / 2.0
            else:
                next_bound = curr_bound - (self.second_min[prev] + self.first_min[nxt]) / 2.0

            estimate = next_weight + next_bound
            if estimate < self.best_cost:
                self.stats.expanded_nodes += 1
                path[level] = nxt
                visited[nxt] = True
                self._dfs(
                    curr_bound=next_bound,
                    curr_weight=next_weight,
                    level=level + 1,
                    path=path,
                    visited=visited,
                )
                visited[nxt] = False
                path[level] = -1
            else:
                self.stats.pruned_nodes += 1


def build_euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """根据二维坐标构建欧式距离矩阵，对角线置为 inf."""
    points = np.asarray(points, dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist


def brute_force_tsp(dist: np.ndarray, start: int = 0) -> tuple[float, list[int]]:
    """仅用于小规模校验 correctness."""
    n = dist.shape[0]
    others = [i for i in range(n) if i != start]
    best = math.inf
    best_path: list[int] = []
    for perm in itertools.permutations(others):
        path = [start, *perm, start]
        cost = 0.0
        feasible = True
        for i in range(len(path) - 1):
            w = dist[path[i], path[i + 1]]
            if not np.isfinite(w):
                feasible = False
                break
            cost += float(w)
        if feasible and cost < best:
            best = cost
            best_path = path
    return best, best_path


def run_case(title: str, points: np.ndarray, verify_with_bruteforce: bool) -> None:
    dist = build_euclidean_distance_matrix(points)
    solver = TSPBranchAndBound(dist, start=0)

    t0 = time.perf_counter()
    best_cost, best_path, stats = solver.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n== {title} ==")
    print(f"城市数: {len(points)}")
    print(f"B&B 最优路长: {best_cost:.6f}")
    print("B&B 路径:", " -> ".join(map(str, best_path)))
    print(
        f"搜索统计: expanded={stats.expanded_nodes}, "
        f"pruned={stats.pruned_nodes}, time_ms={elapsed_ms:.2f}"
    )

    if verify_with_bruteforce:
        bf_cost, bf_path = brute_force_tsp(dist, start=0)
        ok = abs(best_cost - bf_cost) <= 1e-9
        print(f"Bruteforce 最优路长: {bf_cost:.6f}")
        print("Bruteforce 路径:", " -> ".join(map(str, bf_path)))
        print(f"最优性校验: {'PASS' if ok else 'FAIL'}")


def main() -> None:
    points_small = np.array(
        [
            [0.0, 0.0],
            [1.0, 5.0],
            [4.0, 4.0],
            [6.5, 1.0],
            [2.0, 2.5],
            [5.0, 6.0],
            [7.0, 4.0],
            [3.0, 7.0],
        ],
        dtype=float,
    )
    run_case("Case A: 8 城市(含暴力校验)", points_small, verify_with_bruteforce=True)

    rng = np.random.default_rng(seed=42)
    points_random = rng.uniform(low=0.0, high=10.0, size=(9, 2))
    run_case("Case B: 9 城市随机点", points_random, verify_with_bruteforce=False)


if __name__ == "__main__":
    main()
