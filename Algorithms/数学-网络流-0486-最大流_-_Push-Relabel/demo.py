"""Push-Relabel maximum flow MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple


EPS = 1e-12


@dataclass
class ResidualEdge:
    """Residual edge in adjacency-list residual network."""

    to: int
    rev: int
    cap: float


@dataclass
class MaxFlowResult:
    """Container for max-flow result and explainable artifacts."""

    max_flow: float
    edge_flows: List[Tuple[int, int, float, float]]
    heights: List[int]
    excess: List[float]


def _validate_input(
    n: int, edges: Sequence[Tuple[int, int, float]], source: int, sink: int
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
        if not float(c) < float("inf"):
            raise ValueError(f"edge[{idx}] has non-finite capacity: {c}")


def _add_edge(graph: List[List[ResidualEdge]], u: int, v: int, c: float) -> int:
    """Add one directed edge with residual reverse edge and return forward index."""
    fwd = ResidualEdge(to=v, rev=len(graph[v]), cap=float(c))
    rev = ResidualEdge(to=u, rev=len(graph[u]), cap=0.0)
    graph[u].append(fwd)
    graph[v].append(rev)
    return len(graph[u]) - 1


def push_relabel_max_flow(
    n: int,
    edges: Sequence[Tuple[int, int, float]],
    source: int,
    sink: int,
) -> MaxFlowResult:
    """Compute max flow with Push-Relabel (preflow + discharge).

    Returns both max flow value and per-original-edge flow for inspection.
    """
    _validate_input(n, edges, source, sink)

    graph: List[List[ResidualEdge]] = [[] for _ in range(n)]
    original_refs: List[Tuple[int, int, float, int]] = []
    for edge_id, (u, v, c) in enumerate(edges):
        fwd_idx = _add_edge(graph, u, v, c)
        original_refs.append((u, fwd_idx, float(c), edge_id))

    height = [0] * n
    excess = [0.0] * n
    seen = [0] * n

    def push(u: int, edge_idx: int) -> float:
        e = graph[u][edge_idx]
        if excess[u] <= EPS or e.cap <= EPS:
            return 0.0
        if height[u] != height[e.to] + 1:
            return 0.0
        delta = min(excess[u], e.cap)
        e.cap -= delta
        graph[e.to][e.rev].cap += delta
        excess[u] -= delta
        excess[e.to] += delta
        return delta

    def relabel(u: int) -> None:
        min_h = None
        for e in graph[u]:
            if e.cap > EPS:
                h = height[e.to]
                if min_h is None or h < min_h:
                    min_h = h
        if min_h is None:
            raise RuntimeError("active node has no residual outgoing edge")
        height[u] = min_h + 1

    active = deque()

    # Preflow initialization: saturate outgoing edges from source.
    height[source] = n
    for i, e in enumerate(graph[source]):
        if e.cap <= EPS:
            continue
        delta = e.cap
        e.cap = 0.0
        graph[e.to][e.rev].cap += delta
        excess[source] -= delta
        excess[e.to] += delta
        if e.to != sink:
            active.append(e.to)

    in_queue = [False] * n
    for v in active:
        in_queue[v] = True

    def maybe_activate(v: int) -> None:
        if v == source or v == sink:
            return
        if excess[v] > EPS and not in_queue[v]:
            active.append(v)
            in_queue[v] = True

    def discharge(u: int) -> None:
        while excess[u] > EPS:
            if seen[u] >= len(graph[u]):
                relabel(u)
                seen[u] = 0
                continue
            edge_idx = seen[u]
            e = graph[u][edge_idx]
            before = excess[e.to]
            pushed = push(u, edge_idx)
            if pushed > EPS:
                if before <= EPS and excess[e.to] > EPS:
                    maybe_activate(e.to)
            else:
                seen[u] += 1

    # Process active vertices until no internal excess remains.
    while active:
        u = active.popleft()
        in_queue[u] = False
        if excess[u] <= EPS:
            continue
        old_h = height[u]
        discharge(u)
        if excess[u] > EPS:
            # Relabel-to-front flavor improves practical performance.
            if height[u] > old_h:
                active.appendleft(u)
            else:
                active.append(u)
            in_queue[u] = True

    edge_flows: List[Tuple[int, int, float, float]] = []
    for u, fwd_idx, cap0, _edge_id in original_refs:
        residual_cap = graph[u][fwd_idx].cap
        flow = cap0 - residual_cap
        edge = graph[u][fwd_idx]
        edge_flows.append((u, edge.to, cap0, flow))

    max_flow = excess[sink]
    return MaxFlowResult(
        max_flow=max_flow,
        edge_flows=edge_flows,
        heights=height,
        excess=excess,
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
    expected: float | None = None,
) -> None:
    print(f"\n=== {title} ===")
    result = push_relabel_max_flow(n=n, edges=edges, source=source, sink=sink)
    print(f"max_flow = {_fmt_num(result.max_flow)}")
    if expected is not None:
        ok = abs(result.max_flow - expected) < 1e-9
        print(f"expected = {_fmt_num(expected)} | check = {'PASS' if ok else 'FAIL'}")
    print("edge flows (u -> v | flow/capacity):")
    for (u, v, cap, flow) in result.edge_flows:
        print(f"  {u} -> {v} | {_fmt_num(flow)}/{_fmt_num(cap)}")
    print("vertex heights:", result.heights)


def main() -> None:
    # Canonical CLRS example, max flow should be 23.
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

    # A small graph with parallel routes and a bottleneck.
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


if __name__ == "__main__":
    main()
