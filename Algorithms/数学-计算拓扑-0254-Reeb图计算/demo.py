"""Reeb graph computation MVP (discrete, union-find sweep on a scalar grid).

This script builds a small, explicit Reeb-like graph for a 2D scalar field.
To keep the implementation honest and transparent, we do not rely on black-box
TDA libraries. Instead, we use a lower-star sweep and track component events:
- birth (minimum),
- merge (saddle),
- termination (maximum).

The output is a directed acyclic graph with edges ordered by scalar value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ReebNode:
    """A critical event node in the discrete Reeb graph approximation."""

    node_id: int
    kind: str  # "minimum" | "saddle" | "maximum"
    vertex: int
    row: int
    col: int
    value: float


@dataclass(frozen=True)
class ReebGraphResult:
    """Container for graph output and useful diagnostics."""

    shape: Tuple[int, int]
    nodes: List[ReebNode]
    edges: List[Tuple[int, int]]
    minimum_count: int
    saddle_count: int
    maximum_count: int


class UnionFind:
    """Union-find with component peak tracking."""

    def __init__(self, n: int, values: np.ndarray) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)
        self.peak = np.arange(n, dtype=np.int64)
        self.values = values

    def activate(self, i: int) -> None:
        self.parent[i] = i
        self.rank[i] = 0
        self.peak[i] = i

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != i:
            nxt = self.parent[i]
            self.parent[i] = root
            i = nxt
        return root

    def _better_peak(self, a: int, b: int) -> int:
        va = float(self.values[a])
        vb = float(self.values[b])
        if va > vb:
            return a
        if vb > va:
            return b
        return min(a, b)

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        self.peak[ra] = self._better_peak(int(self.peak[ra]), int(self.peak[rb]))
        return ra


class NodeBuilder:
    """Helper for creating numbered critical nodes."""

    def __init__(self, h: int, w: int, values: np.ndarray) -> None:
        self.h = h
        self.w = w
        self.values = values
        self.nodes: List[ReebNode] = []
        self.next_id = 0

    def add(self, kind: str, vertex: int) -> int:
        r, c = divmod(vertex, self.w)
        node = ReebNode(
            node_id=self.next_id,
            kind=kind,
            vertex=vertex,
            row=r,
            col=c,
            value=float(self.values[vertex]),
        )
        self.nodes.append(node)
        self.next_id += 1
        return node.node_id


def validate_scalar_field(field: np.ndarray) -> np.ndarray:
    """Validate and normalize input scalar field."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("scalar field must be a 2D array")
    if arr.size == 0:
        raise ValueError("scalar field cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError("scalar field contains NaN or inf")
    return arr


def grid_neighbors(v: int, h: int, w: int) -> List[int]:
    """4-neighborhood for flattened index v."""
    r, c = divmod(v, w)
    out: List[int] = []
    if r > 0:
        out.append((r - 1) * w + c)
    if r + 1 < h:
        out.append((r + 1) * w + c)
    if c > 0:
        out.append(r * w + (c - 1))
    if c + 1 < w:
        out.append(r * w + (c + 1))
    return out


def compute_reeb_graph(field: np.ndarray) -> ReebGraphResult:
    """Compute a discrete Reeb graph approximation via lower-star sweep.

    The sweep order is ascending scalar value with deterministic vertex index
    tie-break. This yields a merge-tree style Reeb skeleton for a connected
    2D domain.
    """
    arr = validate_scalar_field(field)
    h, w = arr.shape
    values = arr.reshape(-1)
    n = values.size

    # Deterministic strict ordering even if original values contain ties.
    order = np.lexsort((np.arange(n, dtype=np.int64), values))

    uf = UnionFind(n=n, values=values)
    active = np.zeros(n, dtype=bool)
    builder = NodeBuilder(h=h, w=w, values=values)

    # root -> latest critical node ID on this connected component branch
    component_frontier: Dict[int, int] = {}
    edge_set: set[Tuple[int, int]] = set()

    for v in order:
        v_int = int(v)
        uf.activate(v_int)
        active[v_int] = True

        lower_neighbor_roots: List[int] = []
        for nb in grid_neighbors(v_int, h, w):
            if active[nb]:
                r = uf.find(nb)
                if r not in lower_neighbor_roots:
                    lower_neighbor_roots.append(r)

        if len(lower_neighbor_roots) == 0:
            # Birth of a new component => local minimum event.
            nid = builder.add("minimum", v_int)
            component_frontier[uf.find(v_int)] = nid
            continue

        if len(lower_neighbor_roots) == 1:
            # Regular vertex: component continues without creating a node.
            root0 = lower_neighbor_roots[0]
            prev_node = component_frontier.pop(root0)
            new_root = uf.union(v_int, root0)
            component_frontier[new_root] = prev_node
            continue

        # Merge event => saddle node.
        saddle_id = builder.add("saddle", v_int)
        for r in lower_neighbor_roots:
            src = component_frontier[r]
            edge_set.add((src, saddle_id))

        merged_root = v_int
        for r in lower_neighbor_roots:
            merged_root = uf.union(merged_root, r)

        for r in lower_neighbor_roots:
            component_frontier.pop(r, None)
        component_frontier[uf.find(merged_root)] = saddle_id

    final_roots = sorted({uf.find(i) for i in range(n) if active[i]})
    for root in final_roots:
        peak_v = int(uf.peak[root])
        max_id = builder.add("maximum", peak_v)
        src = component_frontier[root]
        edge_set.add((src, max_id))

    nodes = builder.nodes
    edges = sorted(edge_set)
    kinds = [node.kind for node in nodes]

    return ReebGraphResult(
        shape=(h, w),
        nodes=nodes,
        edges=edges,
        minimum_count=kinds.count("minimum"),
        saddle_count=kinds.count("saddle"),
        maximum_count=kinds.count("maximum"),
    )


def build_demo_field() -> np.ndarray:
    """Create a deterministic scalar field with two minima and one merge saddle."""
    field = np.array(
        [
            [9.5, 8.3, 7.2, 7.5, 8.8, 9.9],
            [8.4, 1.0, 2.4, 3.9, 6.7, 8.6],
            [7.1, 2.2, 4.6, 5.1, 4.3, 7.2],
            [7.6, 3.5, 5.0, 4.7, 2.0, 6.8],
            [8.7, 6.2, 4.1, 3.0, 1.2, 7.9],
            [9.8, 8.4, 7.3, 6.1, 7.0, 9.1],
        ],
        dtype=np.float64,
    )
    return field


def validate_graph_monotonicity(result: ReebGraphResult) -> None:
    """Check that each directed edge follows non-decreasing scalar value."""
    by_id = {node.node_id: node for node in result.nodes}
    for src, dst in result.edges:
        if by_id[src].value > by_id[dst].value + 1e-12:
            raise AssertionError(
                f"non-monotone edge detected: {src}({by_id[src].value}) -> "
                f"{dst}({by_id[dst].value})"
            )


def print_result(field: np.ndarray, result: ReebGraphResult) -> None:
    """Human-readable output for quick verification."""
    np.set_printoptions(precision=2, suppress=True)
    print("Input scalar field:")
    print(field)
    print()

    print("Critical nodes:")
    for n in result.nodes:
        print(
            f"  id={n.node_id:02d} kind={n.kind:7s} "
            f"value={n.value:5.2f} at (r={n.row}, c={n.col})"
        )

    print("\nGraph edges (source -> target):")
    for src, dst in result.edges:
        print(f"  {src:02d} -> {dst:02d}")

    print("\nSummary:")
    print(f"  shape={result.shape}")
    print(f"  nodes={len(result.nodes)}, edges={len(result.edges)}")
    print(
        "  minima/saddles/maxima="
        f"{result.minimum_count}/{result.saddle_count}/{result.maximum_count}"
    )


def main() -> None:
    field = build_demo_field()
    result = compute_reeb_graph(field)

    # Basic sanity checks for this deterministic demo.
    validate_graph_monotonicity(result)
    if result.minimum_count < 1:
        raise AssertionError("expected at least one minimum")
    if result.maximum_count < 1:
        raise AssertionError("expected at least one maximum")
    if len(result.edges) < len(result.nodes) - result.maximum_count:
        raise AssertionError("too few edges for a valid merge skeleton")

    print_result(field, result)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
