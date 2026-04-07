"""Minimal runnable MVP for MATH-0501: Planarity testing.

This implementation is dependency-free (standard library only).
It provides an exact planarity decision for small/medium graphs by combining:
1) Structure-preserving reductions (remove degree<=1, smooth degree-2 vertices),
2) Necessary Euler bounds,
3) Exact forbidden-minor search for K5 / K3,3 on the reduced graph.

Important scope note:
- The forbidden-minor search is exponential in the reduced vertex count.
- This MVP is intentionally bounded for small instances and educational traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Dict, Iterable, List, Set, Tuple


def _popcount(x: int) -> int:
    """Python-version-compatible popcount."""
    if hasattr(int, "bit_count"):
        return x.bit_count()  # type: ignore[attr-defined]
    return bin(x).count("1")


@dataclass(frozen=True)
class SimpleGraph:
    """Undirected simple graph using bitset adjacency."""

    n: int
    adj: Tuple[int, ...]

    @staticmethod
    def from_edges(n: int, edges: Iterable[Tuple[int, int]]) -> "SimpleGraph":
        """Build a simple graph; self loops are ignored, parallel edges collapsed."""
        if n < 0:
            raise ValueError("n must be non-negative")

        adj_sets: List[Set[int]] = [set() for _ in range(n)]
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u}, {v}) is out of range for n={n}")
            if u == v:
                continue
            adj_sets[u].add(v)
            adj_sets[v].add(u)

        adj_bits: List[int] = [0] * n
        for u in range(n):
            bits = 0
            for v in adj_sets[u]:
                bits |= 1 << v
            adj_bits[u] = bits
        return SimpleGraph(n=n, adj=tuple(adj_bits))

    def degree(self, v: int) -> int:
        return _popcount(self.adj[v])

    def edge_count(self) -> int:
        return sum(_popcount(mask) for mask in self.adj) // 2

    def edges(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for u in range(self.n):
            mask = self.adj[u]
            while mask:
                low = mask & -mask
                v = low.bit_length() - 1
                if u < v:
                    out.append((u, v))
                mask ^= low
        return out

    def induced_subgraph(self, vertices: List[int]) -> "SimpleGraph":
        """Return relabeled induced subgraph on given vertex list."""
        remap: Dict[int, int] = {old: i for i, old in enumerate(vertices)}
        sub_edges: List[Tuple[int, int]] = []
        in_set = set(vertices)
        for u in vertices:
            mask = self.adj[u]
            while mask:
                low = mask & -mask
                v = low.bit_length() - 1
                mask ^= low
                if v in in_set and u < v:
                    sub_edges.append((remap[u], remap[v]))
        return SimpleGraph.from_edges(len(vertices), sub_edges)


@dataclass(frozen=True)
class MinorTarget:
    """Fixed minor target graph."""

    name: str
    k: int
    edges: Tuple[Tuple[int, int], ...]


K5_TARGET = MinorTarget(
    name="K5",
    k=5,
    edges=tuple((i, j) for i in range(5) for j in range(i + 1, 5)),
)

K33_TARGET = MinorTarget(
    name="K3,3",
    k=6,
    edges=tuple((a, b) for a in range(3) for b in range(3, 6)),
)


def connected_components(g: SimpleGraph) -> List[List[int]]:
    """Connected components as vertex lists."""
    seen = [False] * g.n
    comps: List[List[int]] = []

    for s in range(g.n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            mask = g.adj[u]
            while mask:
                low = mask & -mask
                v = low.bit_length() - 1
                mask ^= low
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def is_bipartite(g: SimpleGraph) -> bool:
    """Bipartite check via BFS coloring."""
    color = [-1] * g.n
    for s in range(g.n):
        if color[s] != -1:
            continue
        color[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            mask = g.adj[u]
            while mask:
                low = mask & -mask
                v = low.bit_length() - 1
                mask ^= low
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def reduce_graph_for_planarity(g: SimpleGraph) -> SimpleGraph:
    """Apply planarity-preserving reductions.

    Reductions:
    - Remove degree-0/1 vertices.
    - Smooth degree-2 vertices (replace path u-v-w by edge u-w).

    These operations preserve planarity equivalence.
    """
    n = g.n
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u in range(n):
        mask = g.adj[u]
        while mask:
            low = mask & -mask
            v = low.bit_length() - 1
            mask ^= low
            if u < v:
                adj[u].add(v)
                adj[v].add(u)

    active = set(range(n))

    def deg(v: int) -> int:
        return len(adj[v] & active)

    changed = True
    while changed:
        changed = False

        # 1) peel degree <= 1
        peeled = True
        while peeled:
            peeled = False
            for v in list(active):
                if deg(v) <= 1:
                    for u in list(adj[v] & active):
                        adj[u].discard(v)
                    active.discard(v)
                    peeled = True
                    changed = True

        # 2) smooth one degree-2 vertex at a time
        smoothed_one = True
        while smoothed_one:
            smoothed_one = False
            for v in list(active):
                nbrs = list(adj[v] & active)
                if len(nbrs) != 2:
                    continue
                a, b = nbrs
                adj[a].discard(v)
                adj[b].discard(v)
                active.discard(v)
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
                changed = True
                smoothed_one = True
                break

    kept = sorted(active)
    if not kept:
        return SimpleGraph.from_edges(0, [])

    remap = {old: i for i, old in enumerate(kept)}
    red_edges: List[Tuple[int, int]] = []
    for u in kept:
        for v in adj[u]:
            if v in active and u < v:
                red_edges.append((remap[u], remap[v]))

    return SimpleGraph.from_edges(len(kept), red_edges)


def _is_connected_mask(g: SimpleGraph, mask: int) -> bool:
    """Check connectivity of the vertex-induced subgraph represented by bit mask."""
    if mask == 0:
        return False
    start_bit = mask & -mask
    seen = start_bit
    stack = [start_bit.bit_length() - 1]

    while stack:
        u = stack.pop()
        nbr = g.adj[u] & mask
        new = nbr & ~seen
        while new:
            low = new & -new
            v = low.bit_length() - 1
            new ^= low
            seen |= low
            stack.append(v)
    return seen == mask


def _has_edge_between_masks(g: SimpleGraph, a_mask: int, b_mask: int) -> bool:
    """Whether there exists an edge crossing between two disjoint masks."""
    x = a_mask
    while x:
        low = x & -x
        u = low.bit_length() - 1
        x ^= low
        if g.adj[u] & b_mask:
            return True
    return False


def has_target_minor(
    g: SimpleGraph,
    target: MinorTarget,
    max_states: int = 2_000_000,
) -> bool:
    """Exact minor existence check by branch-set partition backtracking.

    We search for k disjoint non-empty branch sets B_0..B_{k-1} such that:
    - each B_i induces a connected subgraph in G,
    - for every edge (i,j) in target, at least one edge connects B_i and B_j.

    This is exponential and intended for reduced small graphs.
    """
    k = target.k
    n = g.n
    if n < k:
        return False
    if g.edge_count() < len(target.edges):
        return False

    order = sorted(range(n), key=lambda v: g.degree(v), reverse=True)
    edges = target.edges

    state_counter = {"count": 0}

    @lru_cache(maxsize=None)
    def dfs(pos: int, used: int, masks: Tuple[int, ...]) -> bool:
        state_counter["count"] += 1
        if state_counter["count"] > max_states:
            raise RuntimeError(
                f"minor search exceeded max_states={max_states}; "
                "increase max_states or reduce graph size"
            )

        remaining = n - pos
        if remaining < (k - used):
            return False

        if pos == n:
            if used != k:
                return False

            # connectedness of branch sets
            for i in range(k):
                if not _is_connected_mask(g, masks[i]):
                    return False

            # edge requirements of target minor
            for a, b in edges:
                if not _has_edge_between_masks(g, masks[a], masks[b]):
                    return False
            return True

        v = order[pos]
        bit = 1 << v

        # Option 1: keep vertex unused (deleted in minor model).
        if dfs(pos + 1, used, masks):
            return True

        # Option 2: assign to an existing branch set.
        for label in range(used):
            updated = list(masks)
            updated[label] |= bit
            if dfs(pos + 1, used, tuple(updated)):
                return True

        # Option 3: start one new branch set (symmetry broken by contiguous labels).
        if used < k:
            updated = list(masks)
            updated[used] = bit
            if dfs(pos + 1, used + 1, tuple(updated)):
                return True

        return False

    init_masks = tuple(0 for _ in range(k))
    return dfs(0, 0, init_masks)


@dataclass(frozen=True)
class PlanarityResult:
    planar: bool
    reason: str


def is_planar_mvp(
    g: SimpleGraph,
    max_exact_vertices: int = 11,
    max_states: int = 2_000_000,
) -> PlanarityResult:
    """Planarity decision for small/medium graphs.

    Strategy per connected component:
    1) reduce graph (degree peeling + degree-2 smoothing),
    2) use Euler necessary bounds for quick non-planarity,
    3) exact K5/K3,3 minor detection on reduced component.
    """
    if g.n <= 4:
        return PlanarityResult(True, "n <= 4, always planar")

    for comp_idx, comp_vertices in enumerate(connected_components(g), start=1):
        comp = g.induced_subgraph(comp_vertices)
        red = reduce_graph_for_planarity(comp)

        n = red.n
        m = red.edge_count()
        if n <= 4:
            continue

        # Necessary condition for simple planar graphs.
        if m > 3 * n - 6:
            return PlanarityResult(
                False,
                f"component {comp_idx}: violates m <= 3n-6 after reduction ({m} > {3*n-6})",
            )

        # Stronger necessary condition for bipartite planar graphs.
        if is_bipartite(red) and m > 2 * n - 4:
            return PlanarityResult(
                False,
                f"component {comp_idx}: bipartite bound violated ({m} > {2*n-4})",
            )

        if n > max_exact_vertices:
            raise RuntimeError(
                "reduced component too large for exact MVP minor search: "
                f"n={n} > max_exact_vertices={max_exact_vertices}"
            )

        if has_target_minor(red, K5_TARGET, max_states=max_states):
            return PlanarityResult(False, f"component {comp_idx}: contains K5 minor")

        if has_target_minor(red, K33_TARGET, max_states=max_states):
            return PlanarityResult(False, f"component {comp_idx}: contains K3,3 minor")

    return PlanarityResult(True, "no forbidden minor found in reduced components")


def build_demo_graphs() -> List[Tuple[str, SimpleGraph, bool]]:
    """Create deterministic demo cases with expected labels."""
    cases: List[Tuple[str, SimpleGraph, bool]] = []

    # 1) K4 is planar.
    k4_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    cases.append(("K4", SimpleGraph.from_edges(4, k4_edges), True))

    # 2) K5 is non-planar.
    k5_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    cases.append(("K5", SimpleGraph.from_edges(5, k5_edges), False))

    # 3) K3,3 is non-planar.
    k33_edges = [(a, b) for a in range(3) for b in range(3, 6)]
    cases.append(("K3,3", SimpleGraph.from_edges(6, k33_edges), False))

    # 4) Cube graph (8 vertices, planar).
    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    cases.append(("Cube", SimpleGraph.from_edges(8, cube_edges), True))

    # 5) A subdivision of K3,3 (still non-planar).
    # Replace edge (0,3) by 0-6-3.
    k33_sub_edges = [(a, b) for a in range(3) for b in range(3, 6) if not (a == 0 and b == 3)]
    k33_sub_edges.extend([(0, 6), (6, 3)])
    cases.append(("Subdivision(K3,3)", SimpleGraph.from_edges(7, k33_sub_edges), False))

    # 6) Wheel W6 (6-cycle + center), planar.
    w6_edges = []
    for i in range(1, 7):
        w6_edges.append((i, 1 + (i % 6)))
        w6_edges.append((0, i))
    cases.append(("Wheel W6", SimpleGraph.from_edges(7, w6_edges), True))

    return cases


def main() -> None:
    print("Planarity Testing MVP (MATH-0501)")
    print("=" * 72)
    print("Method: reductions + Euler bounds + exact K5/K3,3 minor search")

    cases = build_demo_graphs()
    for name, g, expected in cases:
        t0 = perf_counter()
        result = is_planar_mvp(g, max_exact_vertices=11, max_states=2_000_000)
        dt_ms = (perf_counter() - t0) * 1000.0

        status = "planar" if result.planar else "non-planar"
        print(
            f"[{name:<18}] n={g.n:>2}, m={g.edge_count():>2} -> "
            f"{status:<10} | reason: {result.reason} | {dt_ms:7.2f} ms"
        )

        if result.planar != expected:
            raise RuntimeError(
                f"case '{name}' mismatch: expected {expected}, got {result.planar}"
            )

    print("-" * 72)
    print("All demo cases passed.")


if __name__ == "__main__":
    main()
