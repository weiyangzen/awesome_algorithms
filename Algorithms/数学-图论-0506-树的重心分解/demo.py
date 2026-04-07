"""Tree centroid decomposition MVP.

This demo provides:
1) building the centroid decomposition tree,
2) dynamic "activate node" updates,
3) nearest activated-node distance query,
4) cross-check against a naive BFS oracle.
"""

from __future__ import annotations

from collections import deque
import random
from typing import List, Optional, Tuple


class CentroidDecomposition:
    """Centroid decomposition + nearest activated node queries on a static tree."""

    def __init__(self, n: int, edges: List[Tuple[int, int]]) -> None:
        self.n = n
        self._validate_tree_input(n, edges)

        self.graph: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)

        self.parent: List[int] = [-1] * n
        self.level: List[int] = [-1] * n
        self.removed: List[bool] = [False] * n
        self.sub_size: List[int] = [0] * n

        # dist_to_centroids[u] stores pairs (centroid, distance(u, centroid))
        # along the centroid-ancestor chain of u.
        self.dist_to_centroids: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        self._decompose(entry=0, parent=-1, depth=0)

        self.inf = 10**18
        self.best: List[int] = [self.inf] * n

    @staticmethod
    def _validate_tree_input(n: int, edges: List[Tuple[int, int]]) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if len(edges) != n - 1:
            raise ValueError("A tree with n nodes must have exactly n-1 edges")

        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u}, {v}) out of node range [0, {n - 1}]")
            if u == v:
                raise ValueError("self loop is not allowed in a tree")

        # Connectivity check.
        g: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        seen = [False] * n
        q: deque[int] = deque([0])
        seen[0] = True
        while q:
            u = q.popleft()
            for nxt in g[u]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        if not all(seen):
            raise ValueError("input graph is not connected, so it is not a tree")

    def _compute_subtree_sizes(self, u: int, p: int) -> int:
        self.sub_size[u] = 1
        for v in self.graph[u]:
            if v == p or self.removed[v]:
                continue
            self.sub_size[u] += self._compute_subtree_sizes(v, u)
        return self.sub_size[u]

    def _find_centroid(self, u: int, p: int, total: int) -> int:
        for v in self.graph[u]:
            if v == p or self.removed[v]:
                continue
            if self.sub_size[v] > total // 2:
                return self._find_centroid(v, u, total)
        return u

    def _collect_distances(self, u: int, p: int, dist: int, centroid: int) -> None:
        self.dist_to_centroids[u].append((centroid, dist))
        for v in self.graph[u]:
            if v == p or self.removed[v]:
                continue
            self._collect_distances(v, u, dist + 1, centroid)

    def _decompose(self, entry: int, parent: int, depth: int) -> int:
        total = self._compute_subtree_sizes(entry, -1)
        c = self._find_centroid(entry, -1, total)

        self.parent[c] = parent
        self.level[c] = depth

        # Record distance to this centroid for all nodes in current component.
        self._collect_distances(c, -1, 0, c)
        self.removed[c] = True

        for v in self.graph[c]:
            if self.removed[v]:
                continue
            self._decompose(v, c, depth + 1)
        return c

    def activate(self, node: int) -> None:
        self._check_node(node)
        for centroid, dist in self.dist_to_centroids[node]:
            if dist < self.best[centroid]:
                self.best[centroid] = dist

    def query_nearest(self, node: int) -> Optional[int]:
        self._check_node(node)
        ans = self.inf
        for centroid, dist in self.dist_to_centroids[node]:
            cand = self.best[centroid] + dist
            if cand < ans:
                ans = cand
        return None if ans == self.inf else ans

    def _check_node(self, node: int) -> None:
        if not (0 <= node < self.n):
            raise ValueError(f"node {node} out of range [0, {self.n - 1}]")


class TreeDistanceOracle:
    """Naive checker via all-pairs shortest paths on tree (BFS from each node)."""

    def __init__(self, n: int, edges: List[Tuple[int, int]]) -> None:
        self.n = n
        self.graph: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        self.all_dist = [self._bfs(i) for i in range(n)]
        self.active = set()

    def _bfs(self, src: int) -> List[int]:
        dist = [-1] * self.n
        dist[src] = 0
        q: deque[int] = deque([src])
        while q:
            u = q.popleft()
            for v in self.graph[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def activate(self, node: int) -> None:
        self.active.add(node)

    def query_nearest(self, node: int) -> Optional[int]:
        if not self.active:
            return None
        return min(self.all_dist[node][a] for a in self.active)


def build_random_tree(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for v in range(1, n):
        p = rng.randrange(v)
        edges.append((p, v))
    return edges


def run_fixed_demo() -> None:
    print("=== Fixed Demo: centroid decomposition nearest-active query ===")
    n = 6
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]
    cd = CentroidDecomposition(n, edges)

    print("Edges:", edges)
    print("Centroid parent:", cd.parent)
    print("Centroid level:", cd.level)

    ops = [
        ("query", 5),
        ("activate", 2),
        ("query", 5),
        ("activate", 4),
        ("query", 5),
        ("query", 0),
    ]

    for op, x in ops:
        if op == "activate":
            cd.activate(x)
            print(f"activate({x})")
        else:
            ans = cd.query_nearest(x)
            print(f"query_nearest({x}) -> {ans}")
    print()


def run_random_crosscheck(
    rounds: int = 30,
    n_min: int = 5,
    n_max: int = 40,
    steps: int = 120,
    seed: int = 20260407,
) -> None:
    print("=== Random Cross-check ===")
    rng = random.Random(seed)

    for case_id in range(1, rounds + 1):
        n = rng.randint(n_min, n_max)
        edges = build_random_tree(n, rng)
        cd = CentroidDecomposition(n, edges)
        oracle = TreeDistanceOracle(n, edges)

        for _ in range(steps):
            node = rng.randrange(n)
            if rng.random() < 0.4:
                cd.activate(node)
                oracle.activate(node)
            else:
                got = cd.query_nearest(node)
                expected = oracle.query_nearest(node)
                if got != expected:
                    raise AssertionError(
                        "Mismatch detected: "
                        f"case={case_id}, node={node}, got={got}, expected={expected}"
                    )

    print(
        f"Cross-check passed: rounds={rounds}, "
        f"steps_per_round={steps}, seed={seed}"
    )


def main() -> None:
    run_fixed_demo()
    run_random_crosscheck()


if __name__ == "__main__":
    main()
