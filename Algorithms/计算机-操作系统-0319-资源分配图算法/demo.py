"""资源分配图算法 MVP：RAG -> 等待图 -> DFS 环检测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHITE = 0
GRAY = 1
BLACK = 2


@dataclass(frozen=True)
class RAGCase:
    """单个资源分配图测试用例（单实例资源模型）。"""

    name: str
    processes: tuple[str, ...]
    resources: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...]  # (resource, owner_process)
    requests: tuple[tuple[str, str], ...]  # (request_process, resource)
    expected_deadlock: bool


@dataclass
class AnalysisResult:
    case_name: str
    has_deadlock: bool
    cycle: list[str] | None
    edge_count: int
    matrix: np.ndarray
    process_order: list[str]


def validate_case(case: RAGCase) -> None:
    proc_set = set(case.processes)
    res_set = set(case.resources)

    if len(proc_set) != len(case.processes):
        raise ValueError(f"{case.name}: duplicated process name")
    if len(res_set) != len(case.resources):
        raise ValueError(f"{case.name}: duplicated resource name")

    owner_by_resource: dict[str, str] = {}
    for resource, owner in case.assignments:
        if resource not in res_set:
            raise ValueError(f"{case.name}: unknown resource in assignment: {resource}")
        if owner not in proc_set:
            raise ValueError(f"{case.name}: unknown owner process in assignment: {owner}")
        if resource in owner_by_resource:
            raise ValueError(f"{case.name}: resource assigned twice: {resource}")
        owner_by_resource[resource] = owner

    for proc, resource in case.requests:
        if proc not in proc_set:
            raise ValueError(f"{case.name}: unknown process in request: {proc}")
        if resource not in res_set:
            raise ValueError(f"{case.name}: unknown resource in request: {resource}")


def build_wait_for_graph(case: RAGCase) -> dict[str, set[str]]:
    """将 RAG 折叠为等待图 WFG：P_i 请求被 P_j 占有的资源，则连边 P_i -> P_j。"""
    validate_case(case)

    owner_by_resource = {resource: owner for resource, owner in case.assignments}
    graph: dict[str, set[str]] = {proc: set() for proc in case.processes}

    for requester, resource in case.requests:
        owner = owner_by_resource.get(resource)
        if owner is None:
            # 请求空闲资源，不形成等待依赖边。
            continue
        if owner == requester:
            # 进程请求自己已占有资源时，忽略该边（按输入异常处理亦可）。
            continue
        graph[requester].add(owner)
    return graph


def find_cycle_dfs(graph: dict[str, set[str]]) -> list[str] | None:
    """三色 DFS 找任意一个环；若存在返回环路节点序列，首尾相同。"""
    color = {node: WHITE for node in graph}
    stack: list[str] = []
    index_on_stack: dict[str, int] = {}

    def dfs(node: str) -> list[str] | None:
        color[node] = GRAY
        index_on_stack[node] = len(stack)
        stack.append(node)

        for nxt in sorted(graph[node]):
            if color[nxt] == WHITE:
                cycle = dfs(nxt)
                if cycle is not None:
                    return cycle
            elif color[nxt] == GRAY:
                start = index_on_stack[nxt]
                return stack[start:] + [nxt]

        color[node] = BLACK
        stack.pop()
        index_on_stack.pop(node, None)
        return None

    for node in sorted(graph):
        if color[node] == WHITE:
            cycle = dfs(node)
            if cycle is not None:
                return cycle
    return None


def adjacency_matrix(
    graph: dict[str, set[str]], process_order: list[str]
) -> np.ndarray:
    idx = {proc: i for i, proc in enumerate(process_order)}
    mat = np.zeros((len(process_order), len(process_order)), dtype=np.int64)
    for src, dst_set in graph.items():
        for dst in dst_set:
            mat[idx[src], idx[dst]] = 1
    return mat


def analyze_case(case: RAGCase) -> AnalysisResult:
    graph = build_wait_for_graph(case)
    cycle = find_cycle_dfs(graph)
    process_order = list(case.processes)
    matrix = adjacency_matrix(graph, process_order)
    edge_count = int(matrix.sum())
    has_deadlock = cycle is not None
    return AnalysisResult(
        case_name=case.name,
        has_deadlock=has_deadlock,
        cycle=cycle,
        edge_count=edge_count,
        matrix=matrix,
        process_order=process_order,
    )


def format_cycle(cycle: list[str] | None) -> str:
    if cycle is None:
        return "None"
    return " -> ".join(cycle)


def print_result(case: RAGCase, result: AnalysisResult) -> None:
    print(f"\n=== {result.case_name} ===")
    print(f"Processes: {', '.join(case.processes)}")
    print(f"Resources: {', '.join(case.resources)}")
    print(f"Expected deadlock: {case.expected_deadlock}")
    print(f"Detected deadlock: {result.has_deadlock}")
    print(f"Edge count in WFG: {result.edge_count}")
    print(f"Cycle: {format_cycle(result.cycle)}")
    print("Process order:", ", ".join(result.process_order))
    print("Adjacency matrix (rows=from, cols=to):")
    print(result.matrix)


def build_demo_cases() -> list[RAGCase]:
    deadlock_case = RAGCase(
        name="Case-Deadlock-3Cycle",
        processes=("P1", "P2", "P3"),
        resources=("R1", "R2", "R3"),
        assignments=(("R1", "P1"), ("R2", "P2"), ("R3", "P3")),
        requests=(("P1", "R2"), ("P2", "R3"), ("P3", "R1")),
        expected_deadlock=True,
    )
    safe_case = RAGCase(
        name="Case-Safe-Chain",
        processes=("P1", "P2", "P3"),
        resources=("R1", "R2", "R3"),
        assignments=(("R1", "P1"), ("R2", "P2")),
        requests=(("P1", "R2"), ("P2", "R3")),  # R3 空闲，不产生 P2->* 等待边
        expected_deadlock=False,
    )
    return [deadlock_case, safe_case]


def main() -> None:
    print("Resource Allocation Graph Deadlock Detection Demo")
    cases = build_demo_cases()

    results: list[AnalysisResult] = []
    for case in cases:
        result = analyze_case(case)
        print_result(case, result)
        results.append(result)

        assert result.has_deadlock == case.expected_deadlock, (
            f"{case.name}: expected {case.expected_deadlock}, got {result.has_deadlock}"
        )
        if result.has_deadlock:
            assert result.cycle is not None
            assert len(result.cycle) >= 3
            assert result.cycle[0] == result.cycle[-1]
        else:
            assert result.cycle is None

    deadlock_count = sum(1 for r in results if r.has_deadlock)
    total_edges = int(np.sum([r.edge_count for r in results]))
    print("\n=== Summary ===")
    print(f"Cases: {len(results)}")
    print(f"Deadlock cases detected: {deadlock_count}")
    print(f"Total WFG edges across cases: {total_edges}")
    print("All assertions passed.")


if __name__ == "__main__":
    main()
