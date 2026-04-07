"""最小费用最大流（Min-Cost Max-Flow）MVP.

实现策略：
- 核心求解：逐次最短增广路（Successive Shortest Path, SSP）
- 负权处理：Bellman-Ford 初始化势能（potential）
- 最短路加速：Dijkstra on reduced costs

运行：
    uv run python demo.py
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


EPS = 1e-12


@dataclass
class ResidualEdge:
    """残量网络中的边。"""

    to: int
    rev: int
    cap: float
    cost: float


@dataclass
class MinCostMaxFlowResult:
    """最小费用最大流结果容器。"""

    max_flow: float
    min_cost: float
    edge_flows: List[Tuple[int, int, float, float, float]]


EdgeInput = Tuple[int, int, float, float]  # (u, v, capacity, cost)


def _validate_input(
    n: int,
    edges: Sequence[EdgeInput],
    source: int,
    sink: int,
) -> None:
    if n <= 1:
        raise ValueError("n must be >= 2")
    if source == sink:
        raise ValueError("source and sink must be different")
    if not (0 <= source < n and 0 <= sink < n):
        raise ValueError("source/sink out of range")
    for idx, (u, v, cap, cost) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge[{idx}] has invalid endpoint ({u}, {v})")
        if not (math.isfinite(cap) and math.isfinite(cost)):
            raise ValueError(f"edge[{idx}] has non-finite value")
        if cap < -EPS:
            raise ValueError(f"edge[{idx}] has negative capacity: {cap}")


def _add_edge(
    graph: List[List[ResidualEdge]],
    u: int,
    v: int,
    cap: float,
    cost: float,
) -> int:
    """加入一条前向边和对应反向边，返回前向边下标。"""
    fwd = ResidualEdge(to=v, rev=len(graph[v]), cap=float(cap), cost=float(cost))
    rev = ResidualEdge(to=u, rev=len(graph[u]), cap=0.0, cost=-float(cost))
    graph[u].append(fwd)
    graph[v].append(rev)
    return len(graph[u]) - 1


def _bellman_ford_initial_potential(
    graph: Sequence[Sequence[ResidualEdge]],
    source: int,
) -> List[float]:
    """计算初始势能，允许原始边存在负费用。"""
    n = len(graph)
    dist = [float("inf")] * n
    dist[source] = 0.0

    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if not math.isfinite(dist[u]):
                continue
            base = dist[u]
            for e in graph[u]:
                if e.cap <= EPS:
                    continue
                cand = base + e.cost
                if cand + 1e-15 < dist[e.to]:
                    dist[e.to] = cand
                    updated = True
        if not updated:
            break

    # 若可达区域存在负环，MCMF（最小费用）无下界，本实现直接报错。
    for u in range(n):
        if not math.isfinite(dist[u]):
            continue
        base = dist[u]
        for e in graph[u]:
            if e.cap <= EPS:
                continue
            if base + e.cost + 1e-15 < dist[e.to]:
                raise RuntimeError("negative cost cycle reachable from source")

    return [0.0 if not math.isfinite(d) else d for d in dist]


def min_cost_max_flow(
    n: int,
    edges: Sequence[EdgeInput],
    source: int,
    sink: int,
    flow_limit: Optional[float] = None,
) -> MinCostMaxFlowResult:
    """逐次最短增广路 + 势能重标实现最小费用最大流。

    当 `flow_limit` 为 None 时，求最大可行流下的最小费用。
    当 `flow_limit` 给定时，求发送该流量上限的最小费用流。
    """
    _validate_input(n, edges, source, sink)

    graph: List[List[ResidualEdge]] = [[] for _ in range(n)]
    original_refs: List[Tuple[int, int, float, float]] = []
    for (u, v, cap, cost) in edges:
        fwd_idx = _add_edge(graph, u, v, cap, cost)
        original_refs.append((u, fwd_idx, float(cap), float(cost)))

    potential = _bellman_ford_initial_potential(graph, source)
    total_flow = 0.0
    total_cost = 0.0

    while True:
        if flow_limit is not None and total_flow >= flow_limit - EPS:
            break

        dist = [float("inf")] * n
        prev_v = [-1] * n
        prev_e = [-1] * n
        dist[source] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-15:
                continue
            for ei, e in enumerate(graph[u]):
                if e.cap <= EPS:
                    continue
                reduced_cost = e.cost + potential[u] - potential[e.to]
                nd = d + reduced_cost
                if nd + 1e-15 < dist[e.to]:
                    dist[e.to] = nd
                    prev_v[e.to] = u
                    prev_e[e.to] = ei
                    heapq.heappush(pq, (nd, e.to))

        if not math.isfinite(dist[sink]):
            break

        for v in range(n):
            if math.isfinite(dist[v]):
                potential[v] += dist[v]

        add_flow = float("inf")
        v = sink
        while v != source:
            u = prev_v[v]
            if u < 0:
                raise RuntimeError("failed to reconstruct augmenting path")
            e = graph[u][prev_e[v]]
            add_flow = min(add_flow, e.cap)
            v = u

        if flow_limit is not None:
            add_flow = min(add_flow, flow_limit - total_flow)
        if add_flow <= EPS:
            break

        v = sink
        while v != source:
            u = prev_v[v]
            ei = prev_e[v]
            e = graph[u][ei]
            e.cap -= add_flow
            graph[v][e.rev].cap += add_flow
            total_cost += add_flow * e.cost
            v = u

        total_flow += add_flow

    edge_flows: List[Tuple[int, int, float, float, float]] = []
    for u, fwd_idx, cap0, c0 in original_refs:
        e = graph[u][fwd_idx]
        flow = cap0 - e.cap
        edge_flows.append((u, e.to, cap0, c0, flow))

    return MinCostMaxFlowResult(
        max_flow=total_flow,
        min_cost=total_cost,
        edge_flows=edge_flows,
    )


def _flow_conservation_check(
    n: int,
    edge_flows: Sequence[Tuple[int, int, float, float, float]],
    source: int,
    sink: int,
    expected_flow: float,
) -> None:
    """检查容量约束与流守恒。"""
    balance = np.zeros(n, dtype=float)
    for (u, v, cap, _cost, flow) in edge_flows:
        if flow < -1e-9 or flow > cap + 1e-9:
            raise AssertionError(f"capacity violation on edge {u}->{v}: flow={flow}, cap={cap}")
        balance[u] -= flow
        balance[v] += flow

    if abs(balance[source] + expected_flow) > 1e-8:
        raise AssertionError("source net outflow does not match max_flow")
    if abs(balance[sink] - expected_flow) > 1e-8:
        raise AssertionError("sink net inflow does not match max_flow")
    for v in range(n):
        if v in (source, sink):
            continue
        if abs(balance[v]) > 1e-8:
            raise AssertionError(f"flow conservation violated at node {v}: {balance[v]}")


def _linprog_cost_check(
    n: int,
    edges: Sequence[EdgeInput],
    source: int,
    sink: int,
    target_flow: float,
) -> float:
    """用线性规划交叉验证给定流量目标下的最小费用。"""
    m = len(edges)
    c = np.array([edge[3] for edge in edges], dtype=float)
    bounds = [(0.0, float(edge[2])) for edge in edges]

    a_eq = np.zeros((n, m), dtype=float)
    b_eq = np.zeros(n, dtype=float)
    b_eq[source] = float(target_flow)
    b_eq[sink] = -float(target_flow)

    for j, (u, v, _cap, _cost) in enumerate(edges):
        a_eq[u, j] += 1.0
        a_eq[v, j] -= 1.0

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


def _fmt(x: float) -> str:
    rounded = round(x)
    if abs(x - rounded) < 1e-9:
        return str(int(rounded))
    return f"{x:.6f}"


def run_case(
    name: str,
    n: int,
    edges: Sequence[EdgeInput],
    source: int,
    sink: int,
    expected_flow: float,
    expected_cost: float,
) -> None:
    print(f"\n=== Case: {name} ===")
    result = min_cost_max_flow(n=n, edges=edges, source=source, sink=sink)
    _flow_conservation_check(
        n=n,
        edge_flows=result.edge_flows,
        source=source,
        sink=sink,
        expected_flow=result.max_flow,
    )

    print(f"max_flow = {_fmt(result.max_flow)}")
    print(f"min_cost = {_fmt(result.min_cost)}")
    print(
        f"expected_flow = {_fmt(expected_flow)} | "
        f"check = {'PASS' if abs(result.max_flow - expected_flow) < 1e-9 else 'FAIL'}"
    )
    print(
        f"expected_cost = {_fmt(expected_cost)} | "
        f"check = {'PASS' if abs(result.min_cost - expected_cost) < 1e-9 else 'FAIL'}"
    )
    print("edge flows (u -> v | flow/capacity @ cost):")
    for (u, v, cap, cost, flow) in result.edge_flows:
        print(f"  {u} -> {v} | {_fmt(flow)}/{_fmt(cap)} @ {_fmt(cost)}")

    lp_cost = _linprog_cost_check(
        n=n,
        edges=edges,
        source=source,
        sink=sink,
        target_flow=result.max_flow,
    )
    print(
        f"linprog_cost = {_fmt(lp_cost)} | "
        f"cross-check = {'PASS' if abs(lp_cost - result.min_cost) < 1e-8 else 'FAIL'}"
    )

    assert abs(result.max_flow - expected_flow) < 1e-9, "unexpected max flow"
    assert abs(result.min_cost - expected_cost) < 1e-9, "unexpected min cost"
    assert abs(lp_cost - result.min_cost) < 1e-8, "LP cross-check mismatch"


def main() -> None:
    # Case 1: non-negative costs
    # Max flow = 3, min cost = 9
    case1_edges: List[EdgeInput] = [
        (0, 1, 2, 1),
        (0, 2, 1, 2),
        (1, 2, 1, 0),
        (1, 3, 1, 3),
        (2, 3, 2, 1),
    ]
    run_case(
        name="non_negative_costs",
        n=4,
        edges=case1_edges,
        source=0,
        sink=3,
        expected_flow=3,
        expected_cost=9,
    )

    # Case 2: contains a negative-cost edge but no negative cycle.
    # Max flow = 4, min cost = 21
    case2_edges: List[EdgeInput] = [
        (0, 1, 2, 2),
        (0, 2, 2, 4),
        (1, 2, 1, -3),
        (1, 3, 2, 2),
        (2, 3, 2, 1),
        (3, 4, 3, 1),
        (2, 4, 1, 5),
    ]
    run_case(
        name="with_negative_edge",
        n=5,
        edges=case2_edges,
        source=0,
        sink=4,
        expected_flow=4,
        expected_cost=21,
    )

    print("\nAll min-cost max-flow checks passed.")


if __name__ == "__main__":
    main()
