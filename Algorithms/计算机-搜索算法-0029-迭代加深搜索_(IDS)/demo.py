"""Iterative Deepening Search (IDS) MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

Node = str
Graph = dict[Node, list[Node]]


@dataclass(frozen=True)
class DLSResult:
    """Result of one depth-limited DFS run."""

    found: bool
    path: list[Node]
    cutoff: bool
    expanded_nodes: int


@dataclass(frozen=True)
class IterationLog:
    """Per-depth diagnostics for IDS."""

    depth_limit: int
    found: bool
    cutoff: bool
    expanded_nodes: int


@dataclass(frozen=True)
class SearchResult:
    """Final IDS result."""

    found: bool
    path: list[Node]
    cost: int
    depth_limit: int | None
    total_expanded: int
    iterations: list[IterationLog]


def _validate_graph(graph: Graph, start: Node, goal: Node) -> None:
    if not graph:
        raise ValueError("graph must be non-empty")

    for name, node in (("start", start), ("goal", goal)):
        if node not in graph:
            raise ValueError(f"{name} node {node!r} is missing from graph")

    for node, neighbors in graph.items():
        if not isinstance(neighbors, list):
            raise ValueError(f"neighbors of node {node!r} must be a list")
        for nbr in neighbors:
            if nbr not in graph:
                raise ValueError(
                    f"graph references undefined node {nbr!r} from {node!r}; all neighbors must exist"
                )


def depth_limited_dfs(graph: Graph, start: Node, goal: Node, depth_limit: int) -> DLSResult:
    """Run one DFS with maximum depth `depth_limit`."""
    if depth_limit < 0:
        raise ValueError("depth_limit must be >= 0")

    path: list[Node] = [start]
    on_path: set[Node] = {start}
    expanded_nodes = 0
    cutoff_happened = False

    def _visit(node: Node, remaining_depth: int) -> list[Node] | None:
        nonlocal expanded_nodes, cutoff_happened

        expanded_nodes += 1
        if node == goal:
            return path.copy()

        if remaining_depth == 0:
            cutoff_happened = True
            return None

        for nxt in graph[node]:
            # Avoid cycles on the current DFS branch.
            if nxt in on_path:
                continue

            path.append(nxt)
            on_path.add(nxt)
            result_path = _visit(nxt, remaining_depth - 1)
            if result_path is not None:
                return result_path
            on_path.remove(nxt)
            path.pop()

        return None

    found_path = _visit(start, depth_limit)
    return DLSResult(
        found=found_path is not None,
        path=found_path or [],
        cutoff=cutoff_happened,
        expanded_nodes=expanded_nodes,
    )


def iterative_deepening_search(graph: Graph, start: Node, goal: Node, max_depth: int) -> SearchResult:
    """Run IDS by repeatedly increasing the depth bound from 0 to max_depth."""
    _validate_graph(graph, start, goal)
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    iteration_logs: list[IterationLog] = []
    total_expanded = 0

    for limit in range(max_depth + 1):
        run = depth_limited_dfs(graph, start, goal, limit)
        total_expanded += run.expanded_nodes
        iteration_logs.append(
            IterationLog(
                depth_limit=limit,
                found=run.found,
                cutoff=run.cutoff,
                expanded_nodes=run.expanded_nodes,
            )
        )

        if run.found:
            return SearchResult(
                found=True,
                path=run.path,
                cost=len(run.path) - 1,
                depth_limit=limit,
                total_expanded=total_expanded,
                iterations=iteration_logs,
            )

        # No cutoff means this depth already exhausts all reachable nodes.
        if not run.cutoff:
            break

    return SearchResult(
        found=False,
        path=[],
        cost=-1,
        depth_limit=None,
        total_expanded=total_expanded,
        iterations=iteration_logs,
    )


def _bfs_shortest_distance(graph: Graph, start: Node, goal: Node) -> int | None:
    """Reference shortest path length for unit-cost edges."""
    if start == goal:
        return 0

    queue: deque[tuple[Node, int]] = deque([(start, 0)])
    seen: set[Node] = {start}

    while queue:
        node, depth = queue.popleft()
        for nxt in graph[node]:
            if nxt in seen:
                continue
            if nxt == goal:
                return depth + 1
            seen.add(nxt)
            queue.append((nxt, depth + 1))

    return None


def _is_valid_path(graph: Graph, path: Sequence[Node], start: Node, goal: Node) -> bool:
    if not path:
        return False
    if path[0] != start or path[-1] != goal:
        return False

    for i in range(1, len(path)):
        if path[i] not in graph[path[i - 1]]:
            return False

    return True


def _format_iterations(logs: Sequence[IterationLog]) -> str:
    rows = ["depth | expanded | cutoff | found", "------|----------|--------|------"]
    for log in logs:
        rows.append(
            f"{log.depth_limit:>5} | {log.expanded_nodes:>8} | {str(log.cutoff):>6} | {str(log.found):>5}"
        )
    return "\n".join(rows)


def main() -> None:
    graph_reachable: Graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F", "A"],
        "D": ["G"],
        "E": ["G", "H"],
        "F": ["I"],
        "G": ["J"],
        "H": ["J"],
        "I": ["J"],
        "J": [],
    }

    graph_unreachable: Graph = {
        "S": ["A"],
        "A": ["B", "S"],
        "B": [],
        "T": [],
    }

    cases = [
        ("reachable", graph_reachable, "A", "J", 10, True),
        ("unreachable", graph_unreachable, "S", "T", 6, False),
        ("start_is_goal", graph_reachable, "E", "E", 5, True),
    ]

    print("Iterative Deepening Search (IDS) MVP demo")
    print("=" * 48)

    for idx, (name, graph, start, goal, max_depth, should_find) in enumerate(cases, start=1):
        result = iterative_deepening_search(graph, start, goal, max_depth)
        bfs_dist = _bfs_shortest_distance(graph, start, goal)

        if result.found != should_find:
            raise AssertionError(
                f"Case {idx} ({name}): expected found={should_find}, got found={result.found}"
            )

        if result.found:
            if not _is_valid_path(graph, result.path, start, goal):
                raise AssertionError(f"Case {idx} ({name}): invalid path {result.path}")
            if result.cost != len(result.path) - 1:
                raise AssertionError(
                    f"Case {idx} ({name}): cost mismatch, cost={result.cost}, path_len={len(result.path)}"
                )
            if bfs_dist is None or result.cost != bfs_dist:
                raise AssertionError(
                    f"Case {idx} ({name}): IDS cost={result.cost}, BFS shortest={bfs_dist}"
                )
            if result.depth_limit is None or result.depth_limit > max_depth:
                raise AssertionError(
                    f"Case {idx} ({name}): invalid depth_limit={result.depth_limit}"
                )

            print(
                f"Case {idx:02d} [{name}] -> found=True, cost={result.cost}, "
                f"depth_limit={result.depth_limit}, total_expanded={result.total_expanded}"
            )
            print("path:", " -> ".join(result.path))
        else:
            if result.path or result.cost != -1:
                raise AssertionError(
                    f"Case {idx} ({name}): expected empty path and cost=-1, got {result.path}, {result.cost}"
                )
            if bfs_dist is not None:
                raise AssertionError(
                    f"Case {idx} ({name}): BFS found a path ({bfs_dist}), IDS said unreachable"
                )
            print(
                f"Case {idx:02d} [{name}] -> found=False, "
                f"depth_limit_reached={len(result.iterations) - 1}, total_expanded={result.total_expanded}"
            )

        print(_format_iterations(result.iterations))
        print("-" * 48)

    # Input validation checks.
    try:
        iterative_deepening_search({"A": ["B"]}, "A", "B", 3)
        raise AssertionError("Undefined-neighbor validation failed: expected ValueError")
    except ValueError:
        print("Undefined-neighbor validation: passed")

    try:
        iterative_deepening_search(graph_reachable, "A", "J", -1)
        raise AssertionError("Negative-depth validation failed: expected ValueError")
    except ValueError:
        print("Negative-depth validation: passed")

    try:
        iterative_deepening_search(graph_reachable, "X", "J", 5)
        raise AssertionError("Missing-start validation failed: expected ValueError")
    except ValueError:
        print("Missing-start validation: passed")

    print("All checks passed.")


if __name__ == "__main__":
    main()
