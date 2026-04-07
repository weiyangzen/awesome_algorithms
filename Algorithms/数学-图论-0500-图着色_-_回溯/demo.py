"""Graph coloring with backtracking (MVP, non-interactive)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class SearchStats:
    recursive_calls: int = 0
    safety_checks: int = 0
    backtracks: int = 0


class GraphColoringBacktracking:
    """Exact m-coloring solver via DFS backtracking."""

    def __init__(self, num_vertices: int, edges: Sequence[Tuple[int, int]]) -> None:
        if num_vertices <= 0:
            raise ValueError("num_vertices must be positive")

        self.n = num_vertices
        self.adj: List[List[int]] = [[] for _ in range(num_vertices)]

        for u, v in edges:
            if not (0 <= u < num_vertices and 0 <= v < num_vertices):
                raise ValueError(f"edge ({u}, {v}) contains out-of-range vertex")
            if u == v:
                raise ValueError(f"self-loop detected at vertex {u}; graph is not colorable")
            self.adj[u].append(v)
            self.adj[v].append(u)

        for i in range(num_vertices):
            # Deduplicate and keep deterministic traversal order.
            self.adj[i] = sorted(set(self.adj[i]))

        # High-degree-first ordering is a lightweight pruning heuristic.
        self.order = sorted(range(self.n), key=lambda x: len(self.adj[x]), reverse=True)

    def _is_safe(self, vertex: int, color: int, colors: Sequence[int], stats: SearchStats) -> bool:
        for neighbor in self.adj[vertex]:
            stats.safety_checks += 1
            if colors[neighbor] == color:
                return False
        return True

    def _backtrack(self, pos: int, m: int, colors: List[int], stats: SearchStats) -> bool:
        stats.recursive_calls += 1

        if pos == self.n:
            return True

        vertex = self.order[pos]
        for color in range(m):
            if self._is_safe(vertex, color, colors, stats):
                colors[vertex] = color
                if self._backtrack(pos + 1, m, colors, stats):
                    return True
                colors[vertex] = -1
                stats.backtracks += 1

        return False

    def color_with_m(self, m: int) -> Tuple[bool, List[int], SearchStats]:
        if m <= 0:
            raise ValueError("m must be positive")

        colors = [-1] * self.n
        stats = SearchStats()
        feasible = self._backtrack(0, m, colors, stats)
        return feasible, colors, stats


def build_cycle_graph(n: int) -> Tuple[int, List[Tuple[int, int]]]:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges


def build_complete_graph(n: int) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return n, edges


def build_bipartite_graph(left_size: int, right_size: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = left_size + right_size
    edges: List[Tuple[int, int]] = []
    for i in range(left_size):
        for j in range(right_size):
            edges.append((i, left_size + j))
    return n, edges


def validate_coloring(adj: Sequence[Sequence[int]], colors: Sequence[int]) -> bool:
    for u, neighbors in enumerate(adj):
        if colors[u] < 0:
            return False
        for v in neighbors:
            if colors[u] == colors[v]:
                return False
    return True


def run_case(name: str, n: int, edges: Sequence[Tuple[int, int]], test_ms: Sequence[int]) -> None:
    solver = GraphColoringBacktracking(n, edges)
    print(f"=== {name} ===")
    print(f"vertices={n}, edges={len(edges)}, order={solver.order}")

    for m in test_ms:
        feasible, colors, stats = solver.color_with_m(m)
        status = "YES" if feasible else "NO"

        print(
            f"m={m}: feasible={status}, "
            f"recursive_calls={stats.recursive_calls}, "
            f"safety_checks={stats.safety_checks}, "
            f"backtracks={stats.backtracks}"
        )

        if feasible:
            assert validate_coloring(solver.adj, colors), "internal error: invalid coloring produced"
            print(f"  coloring={colors}")
        else:
            print("  coloring=None")

    print()


def main() -> None:
    cases: Dict[str, Tuple[int, List[Tuple[int, int]], List[int]]] = {}

    n1, e1 = build_cycle_graph(5)  # odd cycle: chi=3
    cases["C5"] = (n1, e1, [2, 3])

    n2, e2 = build_complete_graph(4)  # K4: chi=4
    cases["K4"] = (n2, e2, [3, 4])

    n3, e3 = build_bipartite_graph(3, 4)  # complete bipartite graph: chi=2
    cases["Bipartite_3x4"] = (n3, e3, [1, 2])

    for name, (n, edges, ms) in cases.items():
        run_case(name, n, edges, ms)


if __name__ == "__main__":
    main()
