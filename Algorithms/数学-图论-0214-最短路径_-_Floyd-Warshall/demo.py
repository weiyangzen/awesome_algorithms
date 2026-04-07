"""Minimal runnable MVP for Floyd-Warshall all-pairs shortest paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FloydWarshallResult:
    dist: np.ndarray
    next_hop: np.ndarray
    has_negative_cycle: bool


def build_weight_matrix(
    n: int,
    edges: Iterable[Tuple[int, int, float]],
    directed: bool = True,
) -> np.ndarray:
    """Build an n x n weighted adjacency matrix with +inf for missing edges."""
    if n <= 0:
        raise ValueError("n must be positive")

    mat = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(mat, 0.0)

    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}, {w}) has invalid node index")
        if not np.isfinite(w):
            raise ValueError(f"edge ({u}, {v}, {w}) has non-finite weight")

        if w < mat[u, v]:
            mat[u, v] = float(w)
        if not directed and w < mat[v, u]:
            mat[v, u] = float(w)

    return mat


def floyd_warshall(weight: np.ndarray) -> FloydWarshallResult:
    """Run Floyd-Warshall and return distance matrix + next-hop matrix."""
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("weight must be a square matrix")
    if not np.all(np.isfinite(np.diag(weight))):
        raise ValueError("diagonal entries must be finite")

    n = weight.shape[0]
    dist = weight.astype(float, copy=True)

    next_hop = np.full((n, n), -1, dtype=int)
    finite_mask = np.isfinite(dist)
    next_hop[finite_mask] = np.tile(np.arange(n, dtype=int), (n, 1))[finite_mask]
    np.fill_diagonal(next_hop, np.arange(n, dtype=int))

    for k in range(n):
        through_k = dist[:, [k]] + dist[[k], :]
        better = through_k < dist

        if np.any(better):
            dist = np.where(better, through_k, dist)
            next_hop = np.where(better, next_hop[:, [k]], next_hop)

    has_negative_cycle = bool(np.any(np.diag(dist) < 0))
    return FloydWarshallResult(dist=dist, next_hop=next_hop, has_negative_cycle=has_negative_cycle)


def reconstruct_path(src: int, dst: int, next_hop: np.ndarray) -> Optional[List[int]]:
    """Reconstruct one shortest path from src to dst using next-hop table."""
    n = next_hop.shape[0]
    if not (0 <= src < n and 0 <= dst < n):
        raise ValueError("src/dst out of range")

    if next_hop[src, dst] == -1:
        return None

    path = [src]
    cur = src
    for _ in range(n + 1):
        if cur == dst:
            return path
        cur = int(next_hop[cur, dst])
        if cur == -1:
            return None
        path.append(cur)

    # A safeguard against loops when a negative cycle contaminates the route.
    return None


def format_matrix(mat: np.ndarray) -> str:
    """Pretty-print matrix with inf markers."""
    lines: List[str] = []
    for row in mat:
        parts: List[str] = []
        for x in row:
            if np.isinf(x):
                parts.append("   inf")
            else:
                parts.append(f"{x:6.1f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def show_paths(
    pairs: Sequence[Tuple[int, int]],
    result: FloydWarshallResult,
) -> None:
    for s, t in pairs:
        path = reconstruct_path(s, t, result.next_hop)
        if path is None:
            print(f"  path {s}->{t}: unreachable or ambiguous (negative-cycle affected)")
        else:
            cost = result.dist[s, t]
            print(f"  path {s}->{t}: {' -> '.join(map(str, path))} (cost={cost:.1f})")


def run_case_no_negative_cycle() -> None:
    print("=== Case A: Directed weighted graph without negative cycle ===")
    n = 5
    edges = [
        (0, 1, 3.0),
        (0, 2, 8.0),
        (0, 4, -4.0),
        (1, 3, 1.0),
        (1, 4, 7.0),
        (2, 1, 4.0),
        (3, 0, 2.0),
        (3, 2, -5.0),
        (4, 3, 6.0),
    ]
    w = build_weight_matrix(n, edges, directed=True)
    result = floyd_warshall(w)

    print("Weight matrix:")
    print(format_matrix(w))
    print("\nAll-pairs shortest distances:")
    print(format_matrix(result.dist))
    print(f"\nHas negative cycle: {result.has_negative_cycle}")

    show_paths(pairs=[(0, 3), (3, 4), (4, 2), (2, 4)], result=result)


def run_case_with_negative_cycle() -> None:
    print("\n=== Case B: Graph with a reachable negative cycle ===")
    n = 4
    edges = [
        (0, 1, 1.0),
        (1, 2, -1.0),
        (2, 3, -1.0),
        (3, 1, -1.0),
    ]
    w = build_weight_matrix(n, edges, directed=True)
    result = floyd_warshall(w)

    print("Weight matrix:")
    print(format_matrix(w))
    print("\nDistance matrix after Floyd-Warshall:")
    print(format_matrix(result.dist))
    print(f"\nHas negative cycle: {result.has_negative_cycle}")
    if result.has_negative_cycle:
        print("  note: shortest paths are undefined for node pairs affected by negative cycles.")


def main() -> None:
    run_case_no_negative_cycle()
    run_case_with_negative_cycle()


if __name__ == "__main__":
    main()
