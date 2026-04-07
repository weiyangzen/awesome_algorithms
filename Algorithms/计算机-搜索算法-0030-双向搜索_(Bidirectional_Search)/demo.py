"""Bidirectional Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

Node = Hashable
Graph = Dict[Node, List[Node]]


@dataclass
class SearchResult:
    found: bool
    path: List[Node]
    expanded_nodes: int
    meeting_node: Optional[Node]


def _normalize_graph(graph: MutableMapping[Node, Iterable[Node]]) -> Graph:
    """Convert input mapping to a concrete adjacency list graph."""
    normalized: Graph = {node: list(neighbors) for node, neighbors in graph.items()}

    # Ensure every referenced neighbor exists as a key.
    for node, neighbors in list(normalized.items()):
        for nxt in neighbors:
            if nxt not in normalized:
                normalized[nxt] = []
    return normalized


def _build_reverse_graph(graph: Graph) -> Graph:
    """Build reverse adjacency list for directed bidirectional search."""
    reverse: Graph = {node: [] for node in graph}
    for u, neighbors in graph.items():
        for v in neighbors:
            reverse[v].append(u)
    return reverse


def _expand_one_layer(
    queue: Deque[Node],
    own_visited: Set[Node],
    other_visited: Set[Node],
    parent: Dict[Node, Optional[Node]],
    adjacency: Graph,
) -> Tuple[Optional[Node], int]:
    """Expand one BFS layer and return (meeting_node, expanded_count)."""
    layer_size = len(queue)
    expanded = 0

    for _ in range(layer_size):
        current = queue.popleft()
        expanded += 1

        if current in other_visited:
            return current, expanded

        for nxt in adjacency[current]:
            if nxt in own_visited:
                continue
            own_visited.add(nxt)
            parent[nxt] = current
            if nxt in other_visited:
                return nxt, expanded
            queue.append(nxt)

    return None, expanded


def _construct_path(
    meet: Node,
    start: Node,
    goal: Node,
    front_parent: Dict[Node, Optional[Node]],
    back_parent: Dict[Node, Optional[Node]],
) -> List[Node]:
    """Reconstruct path start -> meet -> goal from parent maps."""
    left: List[Node] = []
    node: Optional[Node] = meet
    while node is not None:
        left.append(node)
        node = front_parent[node]
    left.reverse()  # start -> meet

    right: List[Node] = []
    node = back_parent[meet]
    while node is not None:
        right.append(node)
        node = back_parent[node]

    path = left + right
    if not path or path[0] != start or path[-1] != goal:
        raise AssertionError(
            f"Path reconstruction failed: path={path}, start={start}, goal={goal}, meet={meet}"
        )
    return path


def bidirectional_search(
    graph: MutableMapping[Node, Iterable[Node]],
    start: Node,
    goal: Node,
    *,
    directed: bool = False,
) -> SearchResult:
    """Find a shortest path between start and goal in an unweighted graph."""
    g = _normalize_graph(graph)

    if start not in g:
        raise KeyError(f"start node {start!r} not found in graph")
    if goal not in g:
        raise KeyError(f"goal node {goal!r} not found in graph")

    if start == goal:
        return SearchResult(found=True, path=[start], expanded_nodes=0, meeting_node=start)

    forward_adj = g
    backward_adj = _build_reverse_graph(g) if directed else g

    front_queue: Deque[Node] = deque([start])
    back_queue: Deque[Node] = deque([goal])

    front_visited: Set[Node] = {start}
    back_visited: Set[Node] = {goal}

    front_parent: Dict[Node, Optional[Node]] = {start: None}
    back_parent: Dict[Node, Optional[Node]] = {goal: None}

    expanded_total = 0

    while front_queue and back_queue:
        if len(front_queue) <= len(back_queue):
            meet, expanded = _expand_one_layer(
                front_queue, front_visited, back_visited, front_parent, forward_adj
            )
        else:
            meet, expanded = _expand_one_layer(
                back_queue, back_visited, front_visited, back_parent, backward_adj
            )

        expanded_total += expanded

        if meet is not None:
            path = _construct_path(meet, start, goal, front_parent, back_parent)
            return SearchResult(found=True, path=path, expanded_nodes=expanded_total, meeting_node=meet)

    return SearchResult(found=False, path=[], expanded_nodes=expanded_total, meeting_node=None)


def _validate_path(
    graph: MutableMapping[Node, Iterable[Node]],
    path: Sequence[Node],
    *,
    directed: bool,
) -> bool:
    """Return True if every consecutive edge in path is valid."""
    if len(path) <= 1:
        return True

    g = _normalize_graph(graph)
    edge_set = {(u, v) for u, neighbors in g.items() for v in neighbors}

    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        if directed:
            if (u, v) not in edge_set:
                return False
        else:
            if (u, v) not in edge_set and (v, u) not in edge_set:
                return False
    return True


def main() -> None:
    undirected_graph: Graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B", "G"],
        "E": ["B", "G", "H"],
        "F": ["C", "H"],
        "G": ["D", "E", "I"],
        "H": ["E", "F", "I"],
        "I": ["G", "H"],
        "X": [],
    }

    directed_graph: Graph = {
        "S": ["A", "B"],
        "A": ["C"],
        "B": ["D"],
        "C": ["E"],
        "D": ["E", "F"],
        "E": ["T"],
        "F": ["T"],
        "T": [],
        "Z": ["S"],
    }

    cases = [
        {
            "name": "undirected-reachable",
            "graph": undirected_graph,
            "start": "A",
            "goal": "I",
            "directed": False,
            "expect_found": True,
            "expect_len": 5,
        },
        {
            "name": "undirected-unreachable",
            "graph": undirected_graph,
            "start": "A",
            "goal": "X",
            "directed": False,
            "expect_found": False,
            "expect_len": 0,
        },
        {
            "name": "directed-reachable",
            "graph": directed_graph,
            "start": "S",
            "goal": "T",
            "directed": True,
            "expect_found": True,
            "expect_len": 5,
        },
        {
            "name": "directed-unreachable",
            "graph": directed_graph,
            "start": "T",
            "goal": "S",
            "directed": True,
            "expect_found": False,
            "expect_len": 0,
        },
        {
            "name": "start-equals-goal",
            "graph": undirected_graph,
            "start": "H",
            "goal": "H",
            "directed": False,
            "expect_found": True,
            "expect_len": 1,
        },
    ]

    print("Bidirectional Search MVP demo")
    print("-" * 42)

    for idx, case in enumerate(cases, start=1):
        result = bidirectional_search(
            case["graph"],
            case["start"],
            case["goal"],
            directed=case["directed"],
        )

        if result.found != case["expect_found"]:
            raise AssertionError(
                f"Case {case['name']}: found={result.found}, expected={case['expect_found']}"
            )

        if len(result.path) != case["expect_len"]:
            raise AssertionError(
                f"Case {case['name']}: path length={len(result.path)}, expected={case['expect_len']}"
            )

        if result.found:
            if result.path[0] != case["start"] or result.path[-1] != case["goal"]:
                raise AssertionError(
                    f"Case {case['name']}: invalid endpoints in path={result.path}"
                )
            if not _validate_path(case["graph"], result.path, directed=case["directed"]):
                raise AssertionError(
                    f"Case {case['name']}: invalid edge sequence in path={result.path}"
                )

        print(
            f"Case {idx:02d} [{case['name']}]: found={result.found}, "
            f"meeting={result.meeting_node}, expanded={result.expanded_nodes}, path={result.path}"
        )

    try:
        bidirectional_search(undirected_graph, "NOT_IN_GRAPH", "A")
        raise AssertionError("Missing-node check failed: KeyError was expected")
    except KeyError:
        print("Missing-node check: passed (KeyError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()
