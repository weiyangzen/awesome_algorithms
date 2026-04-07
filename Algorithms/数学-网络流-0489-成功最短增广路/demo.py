"""Successive Shortest Augmenting Path (SSAP) MVP.

This demo implements a minimal, auditable min-cost max-flow solver based on
successive shortest augmenting paths in the residual network.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import List, Optional, Sequence, Tuple


@dataclass
class ResidualEdge:
    """Residual edge with reverse-edge pointer."""

    to: int
    rev: int
    cap: int
    cost: int


@dataclass
class AugmentationRecord:
    """One augmentation step for explainability."""

    path: List[int]
    bottleneck: int
    unit_cost: int
    total_flow: int
    total_cost: int


@dataclass
class MinCostFlowResult:
    """Container for min-cost max-flow outputs and debug artifacts."""

    flow: int
    cost: int
    edge_flows: List[Tuple[int, int, int, int, int]]
    augmentations: List[AugmentationRecord]


def _validate_input(
    n: int,
    edges: Sequence[Tuple[int, int, int, int]],
    source: int,
    sink: int,
    max_flow_limit: Optional[int],
) -> None:
    if n <= 1:
        raise ValueError("n must be >= 2")
    if not (0 <= source < n and 0 <= sink < n):
        raise ValueError("source/sink must be valid vertex ids")
    if source == sink:
        raise ValueError("source and sink must be different")
    if max_flow_limit is not None and max_flow_limit < 0:
        raise ValueError("max_flow_limit must be >= 0")

    for idx, (u, v, cap, cost) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge[{idx}] has invalid endpoints: {(u, v)}")
        if cap < 0:
            raise ValueError(f"edge[{idx}] has negative capacity: {cap}")
        if not isinstance(cost, int):
            raise ValueError(f"edge[{idx}] cost must be int for this MVP: {cost}")


def _add_edge(graph: List[List[ResidualEdge]], u: int, v: int, cap: int, cost: int) -> int:
    """Add one directed edge with a residual reverse edge.

    Returns the index of the forward edge in graph[u].
    """
    fwd = ResidualEdge(to=v, rev=len(graph[v]), cap=cap, cost=cost)
    rev = ResidualEdge(to=u, rev=len(graph[u]), cap=0, cost=-cost)
    graph[u].append(fwd)
    graph[v].append(rev)
    return len(graph[u]) - 1


def _build_residual_graph(
    n: int, edges: Sequence[Tuple[int, int, int, int]]
) -> Tuple[List[List[ResidualEdge]], List[Tuple[int, int, int, int, int]]]:
    graph: List[List[ResidualEdge]] = [[] for _ in range(n)]
    refs: List[Tuple[int, int, int, int, int]] = []
    for u, v, cap, cost in edges:
        idx = _add_edge(graph, u, v, cap, cost)
        refs.append((u, idx, v, cap, cost))
    return graph, refs


def _init_potential_with_bellman_ford(
    graph: List[List[ResidualEdge]], source: int
) -> List[int]:
    """Initialize Johnson potentials to support negative edge costs.

    Assumes no negative cycle reachable from source in residual graph.
    """
    n = len(graph)
    dist = [inf] * n
    dist[source] = 0

    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if dist[u] == inf:
                continue
            du = dist[u]
            for e in graph[u]:
                if e.cap <= 0:
                    continue
                v = e.to
                nd = du + e.cost
                if nd < dist[v]:
                    dist[v] = nd
                    updated = True
        if not updated:
            break

    # Unreachable nodes keep potential 0; reachable nodes use BF distance.
    potential = [0] * n
    for v in range(n):
        if dist[v] != inf:
            potential[v] = int(dist[v])
    return potential


def _shortest_path_with_potential(
    graph: List[List[ResidualEdge]],
    source: int,
    potential: List[int],
) -> Tuple[List[float], List[int], List[int]]:
    """Dijkstra on reduced costs: c'(u,v)=c(u,v)+pi[u]-pi[v]."""
    n = len(graph)
    dist = [inf] * n
    parent_v = [-1] * n
    parent_e = [-1] * n
    dist[source] = 0
    heap: List[Tuple[float, int]] = [(0, source)]

    while heap:
        cur_dist, u = heappop(heap)
        if cur_dist != dist[u]:
            continue
        for ei, e in enumerate(graph[u]):
            if e.cap <= 0:
                continue
            v = e.to
            reduced_cost = e.cost + potential[u] - potential[v]
            if reduced_cost < 0:
                # For safety against inconsistent potentials.
                raise RuntimeError("reduced cost became negative; invalid potential state")
            nd = cur_dist + reduced_cost
            if nd < dist[v]:
                dist[v] = nd
                parent_v[v] = u
                parent_e[v] = ei
                heappush(heap, (nd, v))

    return dist, parent_v, parent_e


def successive_shortest_augmenting_path(
    n: int,
    edges: Sequence[Tuple[int, int, int, int]],
    source: int,
    sink: int,
    max_flow_limit: Optional[int] = None,
) -> MinCostFlowResult:
    """Solve min-cost max-flow via successive shortest augmenting paths.

    Args:
        n: Number of vertices.
        edges: (u, v, capacity, cost) directed edges.
        source: Source vertex.
        sink: Sink vertex.
        max_flow_limit: If set, stop after reaching this flow amount.

    Returns:
        MinCostFlowResult containing delivered flow, total cost, per-edge flows,
        and per-augmentation trace.
    """
    _validate_input(n=n, edges=edges, source=source, sink=sink, max_flow_limit=max_flow_limit)

    graph, refs = _build_residual_graph(n=n, edges=edges)
    potential = _init_potential_with_bellman_ford(graph, source)

    target_flow = max_flow_limit
    total_flow = 0
    total_cost = 0
    augmentations: List[AugmentationRecord] = []

    while True:
        if target_flow is not None and total_flow >= target_flow:
            break
        dist, parent_v, parent_e = _shortest_path_with_potential(
            graph=graph,
            source=source,
            potential=potential,
        )
        if parent_v[sink] == -1:
            break

        for v in range(n):
            if dist[v] != inf:
                potential[v] += int(dist[v])

        remaining = (target_flow - total_flow) if target_flow is not None else 10**18
        bottleneck = remaining
        v = sink
        while v != source:
            u = parent_v[v]
            ei = parent_e[v]
            bottleneck = min(bottleneck, graph[u][ei].cap)
            v = u

        path_reversed = [sink]
        unit_cost = 0
        v = sink
        while v != source:
            u = parent_v[v]
            ei = parent_e[v]
            e = graph[u][ei]
            unit_cost += e.cost
            e.cap -= bottleneck
            graph[e.to][e.rev].cap += bottleneck
            v = u
            path_reversed.append(v)

        path = list(reversed(path_reversed))
        total_flow += bottleneck
        total_cost += bottleneck * unit_cost
        augmentations.append(
            AugmentationRecord(
                path=path,
                bottleneck=bottleneck,
                unit_cost=unit_cost,
                total_flow=total_flow,
                total_cost=total_cost,
            )
        )

    edge_flows: List[Tuple[int, int, int, int, int]] = []
    for u, fwd_idx, v, cap0, cost in refs:
        residual_cap = graph[u][fwd_idx].cap
        flow = cap0 - residual_cap
        edge_flows.append((u, v, cap0, cost, flow))

    return MinCostFlowResult(
        flow=total_flow,
        cost=total_cost,
        edge_flows=edge_flows,
        augmentations=augmentations,
    )


def _assert_result_valid(
    n: int,
    source: int,
    sink: int,
    result: MinCostFlowResult,
) -> None:
    """Check capacity, conservation, and cost consistency."""
    in_flow = [0] * n
    out_flow = [0] * n
    recomputed_cost = 0

    for u, v, cap, cost, flow in result.edge_flows:
        if not (0 <= flow <= cap):
            raise AssertionError(f"flow out of bounds on edge {u}->{v}: flow={flow}, cap={cap}")
        out_flow[u] += flow
        in_flow[v] += flow
        recomputed_cost += flow * cost

    for v in range(n):
        if v == source or v == sink:
            continue
        if in_flow[v] != out_flow[v]:
            raise AssertionError(
                f"flow conservation violated at vertex {v}: in={in_flow[v]}, out={out_flow[v]}"
            )

    source_net_out = out_flow[source] - in_flow[source]
    sink_net_in = in_flow[sink] - out_flow[sink]
    if source_net_out != result.flow:
        raise AssertionError(
            f"source flow mismatch: net_out={source_net_out}, recorded={result.flow}"
        )
    if sink_net_in != result.flow:
        raise AssertionError(
            f"sink flow mismatch: net_in={sink_net_in}, recorded={result.flow}"
        )
    if recomputed_cost != result.cost:
        raise AssertionError(
            f"cost mismatch: recomputed={recomputed_cost}, recorded={result.cost}"
        )


def run_case(
    title: str,
    n: int,
    edges: Sequence[Tuple[int, int, int, int]],
    source: int,
    sink: int,
    max_flow_limit: Optional[int],
    expected_flow: int,
    expected_cost: int,
) -> None:
    print(f"\n=== {title} ===")
    result = successive_shortest_augmenting_path(
        n=n,
        edges=edges,
        source=source,
        sink=sink,
        max_flow_limit=max_flow_limit,
    )
    _assert_result_valid(n=n, source=source, sink=sink, result=result)

    print(f"flow = {result.flow}")
    print(f"cost = {result.cost}")
    print(
        "expected = "
        f"(flow={expected_flow}, cost={expected_cost}) | "
        f"check = {'PASS' if (result.flow, result.cost) == (expected_flow, expected_cost) else 'FAIL'}"
    )

    print("augmentations:")
    for i, rec in enumerate(result.augmentations, start=1):
        path_str = " -> ".join(map(str, rec.path))
        print(
            f"  step {i}: path={path_str}, bottleneck={rec.bottleneck}, "
            f"unit_cost={rec.unit_cost}, total_flow={rec.total_flow}, total_cost={rec.total_cost}"
        )

    print("edge flows (u -> v | flow/capacity @ cost):")
    for u, v, cap, cost, flow in result.edge_flows:
        print(f"  {u} -> {v} | {flow}/{cap} @ {cost}")

    if (result.flow, result.cost) != (expected_flow, expected_cost):
        raise AssertionError(f"case {title} failed expected value check")


def main() -> None:
    # Case 1: fixed-demand min-cost flow (no negative costs).
    edges_case1 = [
        (0, 1, 2, 1),
        (0, 2, 2, 2),
        (1, 2, 1, 0),
        (1, 3, 2, 1),
        (2, 3, 2, 1),
    ]
    run_case(
        title="Case 1: demand-limited min-cost flow",
        n=4,
        edges=edges_case1,
        source=0,
        sink=3,
        max_flow_limit=3,
        expected_flow=3,
        expected_cost=7,
    )

    # Case 2: full max-flow with one negative-cost edge (no negative cycle).
    edges_case2 = [
        (0, 1, 2, 2),
        (0, 2, 1, 4),
        (1, 2, 1, -1),
        (1, 3, 1, 2),
        (2, 3, 2, 1),
    ]
    run_case(
        title="Case 2: min-cost max-flow with negative edge",
        n=4,
        edges=edges_case2,
        source=0,
        sink=3,
        max_flow_limit=None,
        expected_flow=3,
        expected_cost=11,
    )

    print("\nAll SSAP checks passed.")


if __name__ == "__main__":
    main()
