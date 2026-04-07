"""Minimal runnable MVP for Euler trail/circuit (CS-0034).

This script implements Hierholzer's algorithm from scratch for both
undirected and directed graphs, with existence checks and verification.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


def is_connected_nonzero_undirected(
    n: int, adj: Sequence[Sequence[Tuple[int, int]]], degree: Sequence[int]
) -> bool:
    """Return True if all non-zero-degree vertices are in one component."""
    start = next((v for v in range(n) if degree[v] > 0), None)
    if start is None:
        return True

    seen = [False] * n
    q: deque[int] = deque([start])
    seen[start] = True

    while q:
        v = q.popleft()
        for to, _eid in adj[v]:
            if not seen[to]:
                seen[to] = True
                q.append(to)

    return all(seen[v] for v in range(n) if degree[v] > 0)


def is_weakly_connected_nonzero_directed(
    n: int,
    out_adj: Sequence[Sequence[Tuple[int, int]]],
    indeg: Sequence[int],
    outdeg: Sequence[int],
) -> bool:
    """Return True if all non-zero-degree vertices are weakly connected."""
    undirected_adj: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v, _eid in out_adj[u]:
            undirected_adj[u].append(v)
            undirected_adj[v].append(u)

    start = next((v for v in range(n) if indeg[v] + outdeg[v] > 0), None)
    if start is None:
        return True

    seen = [False] * n
    q: deque[int] = deque([start])
    seen[start] = True

    while q:
        v = q.popleft()
        for to in undirected_adj[v]:
            if not seen[to]:
                seen[to] = True
                q.append(to)

    return all(seen[v] for v in range(n) if indeg[v] + outdeg[v] > 0)


def hierholzer_undirected(n: int, edges: Sequence[Edge], start: int) -> List[int]:
    """Construct an Euler trail/circuit for an undirected multigraph."""
    m = len(edges)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    used = [False] * m
    next_idx = [0] * n
    stack = [start]
    path: List[int] = []

    while stack:
        v = stack[-1]
        while next_idx[v] < len(adj[v]) and used[adj[v][next_idx[v]][1]]:
            next_idx[v] += 1

        if next_idx[v] == len(adj[v]):
            path.append(v)
            stack.pop()
            continue

        to, eid = adj[v][next_idx[v]]
        next_idx[v] += 1
        if used[eid]:
            continue
        used[eid] = True
        stack.append(to)

    path.reverse()
    return path


def hierholzer_directed(n: int, edges: Sequence[Edge], start: int) -> List[int]:
    """Construct an Euler trail/circuit for a directed multigraph."""
    m = len(edges)
    out_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        out_adj[u].append((v, eid))

    used = [False] * m
    next_idx = [0] * n
    stack = [start]
    path: List[int] = []

    while stack:
        v = stack[-1]
        while next_idx[v] < len(out_adj[v]) and used[out_adj[v][next_idx[v]][1]]:
            next_idx[v] += 1

        if next_idx[v] == len(out_adj[v]):
            path.append(v)
            stack.pop()
            continue

        to, eid = out_adj[v][next_idx[v]]
        next_idx[v] += 1
        if used[eid]:
            continue
        used[eid] = True
        stack.append(to)

    path.reverse()
    return path


def find_euler_undirected(n: int, edges: Sequence[Edge]) -> Tuple[str, List[int]]:
    """Return ('none'|'path'|'circuit', vertex_path) for undirected graph."""
    m = len(edges)
    if m == 0:
        return "circuit", [0] if n > 0 else []

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    degree = [0] * n
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        degree[u] += 1
        degree[v] += 1

    if not is_connected_nonzero_undirected(n, adj, degree):
        return "none", []

    odd = [v for v in range(n) if degree[v] % 2 == 1]
    if len(odd) == 0:
        kind = "circuit"
        start = next(v for v in range(n) if degree[v] > 0)
    elif len(odd) == 2:
        kind = "path"
        start = odd[0]
    else:
        return "none", []

    path = hierholzer_undirected(n, edges, start)
    if len(path) != m + 1:
        return "none", []
    if kind == "circuit" and path[0] != path[-1]:
        return "none", []
    return kind, path


def find_euler_directed(n: int, edges: Sequence[Edge]) -> Tuple[str, List[int]]:
    """Return ('none'|'path'|'circuit', vertex_path) for directed graph."""
    m = len(edges)
    if m == 0:
        return "circuit", [0] if n > 0 else []

    indeg = [0] * n
    outdeg = [0] * n
    out_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        outdeg[u] += 1
        indeg[v] += 1
        out_adj[u].append((v, eid))

    if not is_weakly_connected_nonzero_directed(n, out_adj, indeg, outdeg):
        return "none", []

    starts = [v for v in range(n) if outdeg[v] - indeg[v] == 1]
    ends = [v for v in range(n) if indeg[v] - outdeg[v] == 1]
    balanced = [v for v in range(n) if indeg[v] == outdeg[v]]

    if len(starts) == 0 and len(ends) == 0 and len(balanced) == n:
        kind = "circuit"
        start = next(v for v in range(n) if outdeg[v] > 0)
    elif len(starts) == 1 and len(ends) == 1 and len(balanced) == n - 2:
        kind = "path"
        start = starts[0]
    else:
        return "none", []

    path = hierholzer_directed(n, edges, start)
    if len(path) != m + 1:
        return "none", []
    if kind == "circuit" and path[0] != path[-1]:
        return "none", []
    return kind, path


def verify_undirected_path(edges: Sequence[Edge], path: Sequence[int]) -> bool:
    """Check every undirected edge is used exactly once."""
    if len(path) != len(edges) + 1:
        return False
    want = Counter((min(u, v), max(u, v)) for u, v in edges)
    got = Counter((min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1))
    return want == got


def verify_directed_path(edges: Sequence[Edge], path: Sequence[int]) -> bool:
    """Check every directed edge is used exactly once with right direction."""
    if len(path) != len(edges) + 1:
        return False
    want = Counter(edges)
    got = Counter((path[i], path[i + 1]) for i in range(len(path) - 1))
    return want == got


def run_demo_cases() -> None:
    """Run deterministic demo cases and assert expected behavior."""
    print("Euler Trail/Circuit MVP (CS-0034)")
    print("=" * 72)

    undirected_circuit = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    undirected_path_only = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 2]], dtype=np.int64)
    undirected_none = np.array([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=np.int64)

    directed_circuit = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 0]], dtype=np.int64)
    directed_path_only = np.array([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=np.int64)

    undirected_cases = [
        ("undirected-circuit", 4, [tuple(e) for e in undirected_circuit.tolist()], "circuit"),
        ("undirected-path", 4, [tuple(e) for e in undirected_path_only.tolist()], "path"),
        ("undirected-none", 4, [tuple(e) for e in undirected_none.tolist()], "none"),
    ]

    for name, n, edges, expected in undirected_cases:
        kind, path = find_euler_undirected(n, edges)
        ok = kind == "none" or verify_undirected_path(edges, path)
        print(f"[{name}] kind={kind:7s} path={path}")
        if not ok:
            raise RuntimeError(f"undirected verify failed: {name}")
        if kind != expected:
            raise RuntimeError(f"unexpected kind for {name}: got={kind}, expected={expected}")

    directed_cases = [
        ("directed-circuit", 4, [tuple(e) for e in directed_circuit.tolist()], "circuit"),
        ("directed-path", 4, [tuple(e) for e in directed_path_only.tolist()], "path"),
        ("directed-none", 4, [(0, 1), (1, 0), (2, 3)], "none"),
    ]

    for name, n, edges, expected in directed_cases:
        kind, path = find_euler_directed(n, edges)
        ok = kind == "none" or verify_directed_path(edges, path)
        print(f"[{name}] kind={kind:7s} path={path}")
        if not ok:
            raise RuntimeError(f"directed verify failed: {name}")
        if kind != expected:
            raise RuntimeError(f"unexpected kind for {name}: got={kind}, expected={expected}")

    print("=" * 72)
    print("All demo cases passed.")


def main() -> None:
    run_demo_cases()


if __name__ == "__main__":
    main()
