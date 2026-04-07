"""Minimal runnable MVP for planar embedding (combinatorial embedding).

This MVP targets small/medium graphs and demonstrates the core idea of
planar embedding via rotation systems:
1) enumerate cyclic neighbor orders around vertices,
2) trace faces induced by each rotation system,
3) apply Euler criterion to confirm genus 0 for each connected component.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import prod
from typing import Dict, List, Sequence, Set, Tuple


Edge = Tuple[int, int]
Rotation = Dict[int, Tuple[int, ...]]


@dataclass
class EmbeddingResult:
    is_planar: bool
    rotation: Rotation
    faces: List[List[int]]
    states_explored: int
    message: str


def _build_simple_undirected_graph(n: int, edges: Sequence[Edge]) -> Tuple[List[List[int]], List[Edge]]:
    if n < 0:
        raise ValueError("n must be non-negative")

    edge_set: Set[Edge] = set()
    adj_sets: List[Set[int]] = [set() for _ in range(n)]

    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range for n={n}")
        if u == v:
            raise ValueError("self-loop is not supported in this MVP")

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue

        edge_set.add((a, b))
        adj_sets[a].add(b)
        adj_sets[b].add(a)

    adj = [sorted(neis) for neis in adj_sets]
    unique_edges = sorted(edge_set)
    return adj, unique_edges


def _connected_components(adj: Sequence[Sequence[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []

    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))

    return comps


def _is_bipartite_component(vertices: Sequence[int], adj: Sequence[Sequence[int]]) -> bool:
    color: Dict[int, int] = {}

    for s in vertices:
        if s in color:
            continue
        color[s] = 0
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    return False

    return True


def _cyclic_orders(neighbors: Sequence[int]) -> List[Tuple[int, ...]]:
    """Return unique cyclic orders (modulo rotation only).

    Important: local reversal (clockwise/counterclockwise choice at one vertex)
    must be kept as an independent choice. Removing it can incorrectly discard
    valid embeddings (e.g., cube graph).
    """
    neis = sorted(neighbors)
    d = len(neis)

    if d <= 2:
        return [tuple(neis)]

    anchor = neis[0]
    others = neis[1:]
    unique: Set[Tuple[int, ...]] = set()
    for perm in permutations(others):
        unique.add((anchor,) + perm)

    return sorted(unique)


def _trace_faces(component_edges: Sequence[Edge], rotation: Rotation) -> List[List[int]]:
    """Trace dart cycles induced by rotation system.

    Dart transition:
    (u -> v)  -->  (v -> next_v(u))
    where next_v(u) is the next neighbor after u in v's cyclic order.
    """
    successor: Dict[Tuple[int, int], int] = {}
    for v, order in rotation.items():
        if not order:
            continue
        d = len(order)
        for i, incoming in enumerate(order):
            successor[(v, incoming)] = order[(i + 1) % d]

    darts: List[Tuple[int, int]] = []
    for u, v in component_edges:
        darts.append((u, v))
        darts.append((v, u))

    visited: Set[Tuple[int, int]] = set()
    faces: List[List[int]] = []

    for start in darts:
        if start in visited:
            continue

        cur = start
        face: List[int] = []
        while cur not in visited:
            visited.add(cur)
            u, v = cur
            face.append(u)
            if (v, u) not in successor:
                raise RuntimeError(
                    f"incomplete rotation while tracing face at dart ({u}->{v}); missing key ({v}, {u})"
                )
            w = successor[(v, u)]
            cur = (v, w)

        faces.append(face)

    return faces


def _canonicalize_face(face: Sequence[int]) -> List[int]:
    if len(face) <= 1:
        return list(face)

    tup = tuple(face)

    def min_rotation(seq: Tuple[int, ...]) -> Tuple[int, ...]:
        best = seq
        for i in range(1, len(seq)):
            cand = seq[i:] + seq[:i]
            if cand < best:
                best = cand
        return best

    a = min_rotation(tup)
    b = min_rotation(tup[::-1])
    return list(min(a, b))


def _find_component_embedding(
    component_vertices: Sequence[int],
    component_edges: Sequence[Edge],
    adj: Sequence[Sequence[int]],
    max_states: int,
) -> EmbeddingResult:
    n = len(component_vertices)
    m = len(component_edges)

    # Isolated vertex component: trivially planar, one outer face.
    if m == 0:
        v = component_vertices[0]
        return EmbeddingResult(
            is_planar=True,
            rotation={v: tuple()},
            faces=[[v]],
            states_explored=0,
            message="isolated vertex component",
        )

    if n >= 3 and m > 3 * n - 6:
        return EmbeddingResult(
            is_planar=False,
            rotation={},
            faces=[],
            states_explored=0,
            message=f"fails Euler necessary bound: m={m} > 3n-6={3 * n - 6}",
        )

    if n >= 3 and _is_bipartite_component(component_vertices, adj) and m > 2 * n - 4:
        return EmbeddingResult(
            is_planar=False,
            rotation={},
            faces=[],
            states_explored=0,
            message=f"fails bipartite planar bound: m={m} > 2n-4={2 * n - 4}",
        )

    comp_set = set(component_vertices)
    options: Dict[int, List[Tuple[int, ...]]] = {}
    for v in component_vertices:
        local_neighbors = [u for u in adj[v] if u in comp_set]
        options[v] = _cyclic_orders(local_neighbors)

    estimate = prod(len(opts) for opts in options.values())
    order = sorted(component_vertices, key=lambda v: (len(options[v]), -len(adj[v]), v))

    states = 0
    cutoff = False
    rotation: Rotation = {}

    def dfs(i: int) -> Tuple[Rotation, List[List[int]]] | None:
        nonlocal states, cutoff

        if i == len(order):
            faces = _trace_faces(component_edges, rotation)
            # Connected component Euler criterion on sphere.
            if n - m + len(faces) == 2:
                return dict(rotation), faces
            return None

        v = order[i]
        for cyc in options[v]:
            states += 1
            if states > max_states:
                cutoff = True
                return None
            rotation[v] = cyc
            found = dfs(i + 1)
            if found is not None:
                return found
            del rotation[v]

        return None

    found = dfs(0)
    if found is not None:
        found_rotation, raw_faces = found
        faces = [_canonicalize_face(face) for face in raw_faces]
        faces.sort()
        return EmbeddingResult(
            is_planar=True,
            rotation=found_rotation,
            faces=faces,
            states_explored=states,
            message=f"embedding found (search-space estimate={estimate})",
        )

    if cutoff:
        return EmbeddingResult(
            is_planar=False,
            rotation={},
            faces=[],
            states_explored=states,
            message=(
                "search cutoff before decision; "
                f"increase max_states (current={max_states}, estimate={estimate})"
            ),
        )

    return EmbeddingResult(
        is_planar=False,
        rotation={},
        faces=[],
        states_explored=states,
        message=f"no genus-0 rotation found after exhaustive search (estimate={estimate})",
    )


def planar_embedding(n: int, edges: Sequence[Edge], max_states_per_component: int = 200_000) -> EmbeddingResult:
    """Find a combinatorial planar embedding for an undirected simple graph.

    Returns:
        EmbeddingResult with cyclic neighbor orders (`rotation`) and traced `faces`
        when planar; otherwise returns non-planar or undecided (cutoff) diagnostics.
    """
    adj, unique_edges = _build_simple_undirected_graph(n, edges)
    comps = _connected_components(adj)

    all_rotation: Rotation = {}
    all_faces: List[List[int]] = []
    total_states = 0

    for comp in comps:
        comp_set = set(comp)
        comp_edges = [e for e in unique_edges if e[0] in comp_set and e[1] in comp_set]

        result = _find_component_embedding(comp, comp_edges, adj, max_states=max_states_per_component)
        total_states += result.states_explored
        if not result.is_planar:
            return EmbeddingResult(
                is_planar=False,
                rotation={},
                faces=[],
                states_explored=total_states,
                message=f"component {comp}: {result.message}",
            )

        all_rotation.update(result.rotation)
        all_faces.extend(result.faces)

    all_faces.sort()
    return EmbeddingResult(
        is_planar=True,
        rotation=all_rotation,
        faces=all_faces,
        states_explored=total_states,
        message="all connected components embedded on sphere",
    )


def _pretty_rotation(rotation: Rotation) -> List[str]:
    lines: List[str] = []
    for v in sorted(rotation):
        lines.append(f"  {v}: {list(rotation[v])}")
    return lines


def run_demo_case(name: str, n: int, edges: Sequence[Edge]) -> None:
    print(f"\n=== {name} ===")
    print(f"n={n}, m={len(edges)}")
    result = planar_embedding(n, edges)
    print(f"planar={result.is_planar}, states={result.states_explored}")
    print(f"message: {result.message}")

    if result.is_planar:
        print("rotation system (cyclic order at each vertex):")
        for line in _pretty_rotation(result.rotation):
            print(line)
        print("faces:")
        for i, face in enumerate(result.faces):
            print(f"  F{i}: {face}")


def main() -> None:
    # Planar example 1: cube graph (8 vertices, 12 edges).
    cube_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Non-planar example 1: complete bipartite K3,3.
    k33_edges = [
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
    ]

    # Non-planar example 2: complete graph K5.
    k5_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]

    run_demo_case("Planar graph: Cube", 8, cube_edges)
    run_demo_case("Non-planar graph: K3,3", 6, k33_edges)
    run_demo_case("Non-planar graph: K5", 5, k5_edges)


if __name__ == "__main__":
    main()
