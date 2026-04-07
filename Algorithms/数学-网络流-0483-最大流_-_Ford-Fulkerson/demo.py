"""Ford-Fulkerson maximum flow MVP (DFS augmenting path).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


EPS = 1e-12


@dataclass
class ResidualEdge:
    """One residual-network edge."""

    to: int
    rev: int
    cap: float


@dataclass
class AugmentRecord:
    """One augmentation step for explainability."""

    path_nodes: List[int]
    delta: float


@dataclass
class MaxFlowResult:
    """Container for max-flow outputs and trace."""

    max_flow: float
    edge_flows: List[Tuple[int, int, float, float]]
    augmentations: List[AugmentRecord]


def _validate_input(
    n: int,
    edges: Sequence[Tuple[int, int, float]],
    source: int,
    sink: int,
) -> None:
    if n <= 1:
        raise ValueError("n must be >= 2")
    if not (0 <= source < n and 0 <= sink < n):
        raise ValueError("source/sink must be valid vertex ids")
    if source == sink:
        raise ValueError("source and sink must be different")

    for idx, (u, v, c) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge[{idx}] has invalid endpoints: {(u, v)}")
        if c < 0:
            raise ValueError(f"edge[{idx}] has negative capacity: {c}")
        if not math.isfinite(float(c)):
            raise ValueError(f"edge[{idx}] has non-finite capacity: {c}")


def _add_edge(graph: List[List[ResidualEdge]], u: int, v: int, c: float) -> int:
    """Add forward + reverse residual edges and return forward-edge index."""
    fwd = ResidualEdge(to=v, rev=len(graph[v]), cap=float(c))
    rev = ResidualEdge(to=u, rev=len(graph[u]), cap=0.0)
    graph[u].append(fwd)
    graph[v].append(rev)
    return len(graph[u]) - 1


def _find_augmenting_path_dfs(
    graph: List[List[ResidualEdge]], source: int, sink: int
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """Find one augmenting path using DFS over residual graph.

    Returns:
        path_edges: list of (u, edge_idx) from source to sink, or None.
        bottleneck: minimum residual capacity along the path.
    """
    n = len(graph)
    visited = [False] * n
    parent_v = [-1] * n
    parent_e = [-1] * n

    stack = [source]
    visited[source] = True
    found = False

    while stack and not found:
        u = stack.pop()
        for edge_idx, e in enumerate(graph[u]):
            if e.cap <= EPS or visited[e.to]:
                continue
            visited[e.to] = True
            parent_v[e.to] = u
            parent_e[e.to] = edge_idx
            if e.to == sink:
                found = True
                break
            stack.append(e.to)

    if not visited[sink]:
        return None, 0.0

    path_edges: List[Tuple[int, int]] = []
    bottleneck = float("inf")
    cur = sink
    while cur != source:
        u = parent_v[cur]
        edge_idx = parent_e[cur]
        if u < 0 or edge_idx < 0:
            return None, 0.0
        path_edges.append((u, edge_idx))
        bottleneck = min(bottleneck, graph[u][edge_idx].cap)
        cur = u

    path_edges.reverse()
    return path_edges, bottleneck


def ford_fulkerson_max_flow(
    n: int,
    edges: Sequence[Tuple[int, int, float]],
    source: int,
    sink: int,
) -> MaxFlowResult:
    """Compute maximum flow with Ford-Fulkerson (DFS augmenting path)."""
    _validate_input(n, edges, source, sink)

    graph: List[List[ResidualEdge]] = [[] for _ in range(n)]
    original_refs: List[Tuple[int, int, float, int]] = []

    for edge_id, (u, v, c) in enumerate(edges):
        fwd_idx = _add_edge(graph, u, v, c)
        original_refs.append((u, fwd_idx, float(c), edge_id))

    max_flow = 0.0
    augmentations: List[AugmentRecord] = []

    while True:
        path_edges, delta = _find_augmenting_path_dfs(graph, source, sink)
        if path_edges is None or delta <= EPS:
            break

        # Apply augmentation on residual edges.
        for u, edge_idx in path_edges:
            e = graph[u][edge_idx]
            e.cap -= delta
            graph[e.to][e.rev].cap += delta

        max_flow += delta

        # Build node sequence for readable trace.
        path_nodes = [source]
        cur = source
        for u, edge_idx in path_edges:
            if u != cur:
                raise RuntimeError("internal path reconstruction mismatch")
            nxt = graph[u][edge_idx].to
            path_nodes.append(nxt)
            cur = nxt

        augmentations.append(AugmentRecord(path_nodes=path_nodes, delta=delta))

    edge_flows: List[Tuple[int, int, float, float]] = []
    for u, fwd_idx, cap0, _edge_id in original_refs:
        e = graph[u][fwd_idx]
        flow = cap0 - e.cap
        edge_flows.append((u, e.to, cap0, flow))

    return MaxFlowResult(
        max_flow=max_flow,
        edge_flows=edge_flows,
        augmentations=augmentations,
    )


def _fmt_num(x: float) -> str:
    as_int = int(round(x))
    if abs(x - as_int) < 1e-9:
        return str(as_int)
    return f"{x:.6f}"


def run_case(
    title: str,
    n: int,
    edges: Sequence[Tuple[int, int, float]],
    source: int,
    sink: int,
    expected: Optional[float] = None,
) -> None:
    print(f"\n=== {title} ===")
    result = ford_fulkerson_max_flow(n=n, edges=edges, source=source, sink=sink)

    print(f"max_flow = {_fmt_num(result.max_flow)}")
    if expected is not None:
        ok = abs(result.max_flow - expected) < 1e-9
        print(f"expected = {_fmt_num(expected)} | check = {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(
                f"{title}: expected flow {expected}, got {result.max_flow}"
            )

    print("augmentations (path | delta):")
    for i, rec in enumerate(result.augmentations, start=1):
        path_str = " -> ".join(str(v) for v in rec.path_nodes)
        print(f"  {i:02d}. {path_str} | {_fmt_num(rec.delta)}")

    print("edge flows (u -> v | flow/capacity):")
    for (u, v, cap, flow) in result.edge_flows:
        print(f"  {u} -> {v} | {_fmt_num(flow)}/{_fmt_num(cap)}")


def main() -> None:
    # CLRS canonical directed network. Known max-flow answer is 23.
    edges_case1 = [
        (0, 1, 16),
        (0, 2, 13),
        (1, 2, 10),
        (2, 1, 4),
        (1, 3, 12),
        (3, 2, 9),
        (2, 4, 14),
        (4, 3, 7),
        (3, 5, 20),
        (4, 5, 4),
    ]
    run_case(
        title="CLRS directed network",
        n=6,
        edges=edges_case1,
        source=0,
        sink=5,
        expected=23,
    )

    # Small bottleneck graph with multiple feasible routes.
    edges_case2 = [
        (0, 1, 8),
        (0, 2, 5),
        (1, 2, 3),
        (1, 3, 9),
        (2, 3, 4),
        (2, 4, 8),
        (3, 4, 7),
    ]
    run_case(
        title="Small bottleneck network",
        n=5,
        edges=edges_case2,
        source=0,
        sink=4,
        expected=13,
    )

    print("\nAll Ford-Fulkerson checks passed.")


if __name__ == "__main__":
    main()
