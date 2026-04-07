"""Minimal runnable MVP for Floyd-Warshall algorithm."""

from __future__ import annotations

from typing import List, Optional, Tuple

INF = float("inf")
NEG_INF = float("-inf")


def floyd_warshall(
    weights: List[List[float]],
) -> Tuple[List[List[float]], List[List[Optional[int]]], bool]:
    """Compute all-pairs shortest paths and next-hop matrix.

    Args:
        weights: Square adjacency matrix. INF means unreachable.

    Returns:
        dist: Shortest-path distances.
        next_hop: Next-hop table for path reconstruction.
        has_negative_cycle: Whether any negative cycle exists.
    """
    n = len(weights)
    if n == 0:
        return [], [], False
    if any(len(row) != n for row in weights):
        raise ValueError("weights must be a square matrix")

    dist = [row[:] for row in weights]
    next_hop: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = min(dist[i][i], 0.0)
        for j in range(n):
            if i != j and dist[i][j] != INF:
                next_hop[i][j] = j

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            dik = dist[i][k]
            for j in range(n):
                if dist[k][j] == INF:
                    continue
                candidate = dik + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate
                    next_hop[i][j] = next_hop[i][k]

    has_negative_cycle = any(dist[v][v] < 0 for v in range(n))

    # Mark pairs that can be reduced indefinitely due to reachable negative cycles.
    if has_negative_cycle:
        neg_nodes = [v for v in range(n) if dist[v][v] < 0]
        for k in neg_nodes:
            for i in range(n):
                if dist[i][k] == INF:
                    continue
                for j in range(n):
                    if dist[k][j] == INF:
                        continue
                    dist[i][j] = NEG_INF
                    next_hop[i][j] = None

    return dist, next_hop, has_negative_cycle


def reconstruct_path(
    next_hop: List[List[Optional[int]]], start: int, end: int
) -> List[int]:
    """Reconstruct one shortest path from start to end using next-hop table."""
    if start == end:
        return [start]
    if not next_hop or next_hop[start][end] is None:
        return []

    path = [start]
    cur = start
    max_steps = len(next_hop) + 1

    for _ in range(max_steps):
        nxt = next_hop[cur][end]
        if nxt is None:
            return []
        path.append(nxt)
        if nxt == end:
            return path
        cur = nxt

    # Should not happen in a valid next-hop table; guards against accidental loops.
    return []


def format_cell(x: float) -> str:
    if x == INF:
        return " inf"
    if x == NEG_INF:
        return "-inf"
    return f"{x:4.0f}"


def print_matrix(matrix: List[List[float]], title: str) -> None:
    print(title)
    for row in matrix:
        print(" ".join(format_cell(v) for v in row))
    print()


def build_graph_without_negative_cycle() -> List[List[float]]:
    return [
        [0, 3, 8, INF, -4],
        [INF, 0, INF, 1, 7],
        [INF, 4, 0, INF, INF],
        [2, INF, -5, 0, INF],
        [INF, INF, INF, 6, 0],
    ]


def build_graph_with_negative_cycle() -> List[List[float]]:
    return [
        [0, 1, INF],
        [INF, 0, -2],
        [-2, INF, 0],
    ]


def run_case(weights: List[List[float]], case_name: str) -> None:
    dist, next_hop, has_neg_cycle = floyd_warshall(weights)
    print(f"=== {case_name} ===")
    print_matrix(dist, "distance matrix:")
    print(f"has_negative_cycle: {has_neg_cycle}")

    if not has_neg_cycle:
        queries = [(0, 3), (3, 4), (4, 2)]
        for s, t in queries:
            path = reconstruct_path(next_hop, s, t)
            if not path:
                print(f"path {s}->{t}: unreachable")
            else:
                print(f"path {s}->{t}: {path}, cost={dist[s][t]:.0f}")
    print()


def main() -> None:
    run_case(build_graph_without_negative_cycle(), "Case A: no negative cycle")
    run_case(build_graph_with_negative_cycle(), "Case B: with negative cycle")


if __name__ == "__main__":
    main()
