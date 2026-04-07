"""Steiner Tree 2-approximation MVP (KMB-style) for weighted undirected graphs."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Set, Tuple


Node = str
Adj = Dict[Node, Dict[Node, float]]
Edge = Tuple[Node, Node, float]


def norm_edge(u: Node, v: Node) -> Tuple[Node, Node]:
    return (u, v) if u <= v else (v, u)


def add_undirected_edge(graph: Adj, u: Node, v: Node, w: float) -> None:
    if w < 0:
        raise ValueError(f"Negative edge weight not supported by Dijkstra: {(u, v, w)}")
    graph.setdefault(u, {})[v] = float(w)
    graph.setdefault(v, {})[u] = float(w)


def all_undirected_edges(graph: Adj) -> List[Edge]:
    seen: Set[Tuple[Node, Node]] = set()
    out: List[Edge] = []
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            key = norm_edge(u, v)
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1], float(w)))
    return out


@dataclass
class UnionFind:
    parent: Dict[Node, Node]
    rank: Dict[Node, int]

    @classmethod
    def from_nodes(cls, nodes: Iterable[Node]) -> "UnionFind":
        parent = {x: x for x in nodes}
        rank = {x: 0 for x in nodes}
        return cls(parent=parent, rank=rank)

    def find(self, x: Node) -> Node:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: Node, b: Node) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_mst(nodes: Iterable[Node], edges: Iterable[Edge]) -> List[Edge]:
    node_list = sorted(set(nodes))
    if not node_list:
        return []
    if len(node_list) == 1:
        return []

    uf = UnionFind.from_nodes(node_list)
    sorted_edges = sorted((float(w), u, v) for (u, v, w) in edges)
    mst: List[Edge] = []

    for w, u, v in sorted_edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == len(node_list) - 1:
                break

    if len(mst) != len(node_list) - 1:
        raise ValueError("Input edges are not enough to connect all requested nodes.")
    return mst


def dijkstra(graph: Adj, source: Node) -> Tuple[Dict[Node, float], Dict[Node, Node]]:
    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Node] = {}
    heap: List[Tuple[float, Node]] = [(0.0, source)]

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, {}).items():
            nd = cur_dist + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[Node, Node], source: Node, target: Node) -> List[Node]:
    if source == target:
        return [source]
    if target not in prev:
        return []
    path = [target]
    cur = target
    while cur != source:
        parent = prev.get(cur)
        if parent is None:
            return []
        path.append(parent)
        cur = parent
    path.reverse()
    return path


def metric_closure_shortest_paths(
    graph: Adj, terminals: List[Node]
) -> Tuple[Dict[Node, Dict[Node, float]], Dict[Tuple[Node, Node], List[Node]]]:
    dist_map: Dict[Node, Dict[Node, float]] = {}
    path_map: Dict[Tuple[Node, Node], List[Node]] = {}

    for s in terminals:
        dist, prev = dijkstra(graph, s)
        dist_map[s] = dist
        for t in terminals:
            if s == t:
                continue
            if t not in dist:
                raise ValueError(f"Terminal {t} is unreachable from terminal {s}.")
            path_map[(s, t)] = reconstruct_path(prev, s, t)

    return dist_map, path_map


def prune_non_terminal_leaves(edges: List[Edge], terminals: Set[Node]) -> List[Edge]:
    edge_map: Dict[Tuple[Node, Node], float] = {}
    for u, v, w in edges:
        edge_map[norm_edge(u, v)] = float(w)

    while True:
        degree: Dict[Node, int] = {}
        incident: Dict[Node, List[Tuple[Node, Node]]] = {}
        for (u, v), _w in edge_map.items():
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            incident.setdefault(u, []).append((u, v))
            incident.setdefault(v, []).append((u, v))

        leaves = [x for x, d in degree.items() if d == 1 and x not in terminals]
        if not leaves:
            break

        for leaf in leaves:
            for e in incident.get(leaf, []):
                edge_map.pop(norm_edge(e[0], e[1]), None)

    pruned = [(u, v, w) for (u, v), w in edge_map.items()]
    pruned.sort(key=lambda x: (x[0], x[1], x[2]))
    return pruned


def kmb_steiner_tree(graph: Adj, terminals: Iterable[Node]) -> Tuple[List[Edge], float]:
    terminal_list = sorted(set(terminals))
    if len(terminal_list) < 2:
        return [], 0.0
    missing = [t for t in terminal_list if t not in graph]
    if missing:
        raise ValueError(f"Terminals not in graph: {missing}")

    dist_map, path_map = metric_closure_shortest_paths(graph, terminal_list)

    closure_edges: List[Edge] = []
    for i, u in enumerate(terminal_list):
        for v in terminal_list[i + 1 :]:
            closure_edges.append((u, v, dist_map[u][v]))
    closure_mst = kruskal_mst(terminal_list, closure_edges)

    expanded_edge_map: Dict[Tuple[Node, Node], float] = {}
    for u, v, _w in closure_mst:
        path = path_map[(u, v)]
        if len(path) < 2:
            continue
        for a, b in zip(path, path[1:]):
            key = norm_edge(a, b)
            ew = graph[a][b]
            expanded_edge_map[key] = min(expanded_edge_map.get(key, ew), ew)

    expanded_edges: List[Edge] = [(u, v, w) for (u, v), w in expanded_edge_map.items()]
    expanded_nodes: Set[Node] = set()
    for u, v, _w in expanded_edges:
        expanded_nodes.add(u)
        expanded_nodes.add(v)
    if not set(terminal_list).issubset(expanded_nodes):
        raise RuntimeError("Expanded graph does not contain all terminals.")

    tree_edges = kruskal_mst(expanded_nodes, expanded_edges)
    pruned_tree = prune_non_terminal_leaves(tree_edges, set(terminal_list))
    total_weight = sum(w for _u, _v, w in pruned_tree)
    return pruned_tree, total_weight


def validate_steiner_tree(
    original_graph: Adj, tree_edges: List[Edge], terminals: Set[Node], eps: float = 1e-9
) -> None:
    if not tree_edges:
        raise AssertionError("Tree edges are empty.")
    for u, v, w in tree_edges:
        if u not in original_graph or v not in original_graph[u]:
            raise AssertionError(f"Tree edge {(u, v)} not in original graph.")
        if abs(original_graph[u][v] - w) > eps:
            raise AssertionError(f"Edge weight mismatch on {(u, v)}.")
        if w < -eps:
            raise AssertionError("Negative weight in tree.")

    used_nodes: Set[Node] = set()
    for u, v, _w in tree_edges:
        used_nodes.add(u)
        used_nodes.add(v)
    if not terminals.issubset(used_nodes):
        raise AssertionError("Not all terminals appear in the result tree.")

    if len(tree_edges) != len(used_nodes) - 1:
        raise AssertionError("Result is not a tree (edges != nodes-1).")

    # Connectivity check among used nodes.
    adj: Dict[Node, List[Node]] = {x: [] for x in used_nodes}
    for u, v, _w in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    root = next(iter(used_nodes))
    stack = [root]
    seen: Set[Node] = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(adj[x])
    if seen != used_nodes:
        raise AssertionError("Tree is disconnected.")


def brute_force_optimal_steiner(graph: Adj, terminals: Set[Node]) -> Tuple[List[Edge], float]:
    # Exact search for tiny graphs only: enumerate all acyclic edge subsets.
    # Used here just for demo quality check.
    edges = all_undirected_edges(graph)
    m = len(edges)
    if len(terminals) < 2:
        return [], 0.0
    if m > 24:
        raise ValueError("Brute force guard: graph too large for exhaustive search.")

    best = float("inf")
    best_edges: List[Edge] = []
    vertices = set(graph.keys())

    for mask in range(1, 1 << m):
        if bin(mask).count("1") < len(terminals) - 1:
            continue

        uf = UnionFind.from_nodes(vertices)
        chosen: List[Edge] = []
        endpoints: Set[Node] = set()
        weight = 0.0
        bad = False

        for i in range(m):
            if ((mask >> i) & 1) == 0:
                continue
            u, v, w = edges[i]
            if not uf.union(u, v):
                bad = True  # cycle
                break
            chosen.append((u, v, w))
            endpoints.update((u, v))
            weight += w
            if weight >= best:
                bad = True
                break
        if bad:
            continue
        if not terminals.issubset(endpoints):
            continue

        root = uf.find(next(iter(terminals)))
        if any(uf.find(t) != root for t in terminals):
            continue
        if any(uf.find(x) != root for x in endpoints):
            continue

        # Tree check on the selected component.
        if len(chosen) != len(endpoints) - 1:
            continue

        if weight < best:
            best = weight
            best_edges = chosen[:]

    if best == float("inf"):
        raise RuntimeError("No feasible Steiner tree found by brute force.")
    best_edges.sort(key=lambda x: (x[0], x[1], x[2]))
    return best_edges, best


def build_demo_graph() -> Tuple[Adj, Set[Node]]:
    graph: Adj = {}
    for u, v, w in [
        ("A", "B", 2.0),
        ("A", "C", 3.0),
        ("B", "C", 1.0),
        ("B", "D", 4.0),
        ("C", "D", 2.0),
        ("C", "E", 3.0),
        ("D", "E", 1.0),
        ("D", "F", 5.0),
        ("E", "F", 2.0),
        ("E", "G", 4.0),
        ("F", "G", 1.0),
        ("E", "H", 3.0),
        ("G", "H", 2.0),
        ("C", "H", 6.0),
        ("B", "E", 5.0),
    ]:
        add_undirected_edge(graph, u, v, w)
    terminals = {"A", "D", "G", "H"}
    return graph, terminals


def format_edges(edges: List[Edge]) -> str:
    return ", ".join([f"{u}-{v}:{w:.1f}" for u, v, w in sorted(edges)])


def main() -> None:
    graph, terminals = build_demo_graph()
    approx_edges, approx_weight = kmb_steiner_tree(graph, terminals)
    validate_steiner_tree(graph, approx_edges, terminals)

    # Exact solution is only for this tiny demo graph.
    optimal_edges, optimal_weight = brute_force_optimal_steiner(graph, terminals)

    ratio = approx_weight / optimal_weight
    if ratio > 2.0 + 1e-9:
        raise AssertionError(f"Approximation ratio violated: {ratio:.6f}")

    print("Steiner Tree Approximation Demo (KMB-style)")
    print(f"Nodes={len(graph)}, UndirectedEdges={len(all_undirected_edges(graph))}")
    print(f"Terminals={sorted(terminals)}")
    print(f"Approx tree weight={approx_weight:.3f}")
    print(f"Approx edges: {format_edges(approx_edges)}")
    print(f"Optimal tree weight (brute force on toy graph)={optimal_weight:.3f}")
    print(f"Optimal edges: {format_edges(optimal_edges)}")
    print(f"Approx ratio={ratio:.6f} (guarantee <= 2)")


if __name__ == "__main__":
    main()
