"""基本群计算 MVP: 在有限连通图(1 维 CW 复形)上计算 π1 的自由群表示.

核心思想:
- 对连通图 G 选定基点 root;
- 先构造一棵生成树 T;
- 每条非树边对应一个自由群生成元;
- 生成元对应的基点回路为: root->u(树路径) + 非树边(u->v) + v->root(树路径逆).

该离散模型是代数拓扑中的标准结论:
π1(G, root) ≅ F_{m-n+1}, 其中 m 为边数, n 为点数.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


OrientedEdge = Tuple[int, int]  # (edge_id, sign) where sign in {+1, -1}


@dataclass(frozen=True)
class Edge:
    """无向边，存储一个参考方向 u -> v。"""

    edge_id: int
    u: int
    v: int
    label: str


@dataclass(frozen=True)
class GeneratorLoop:
    """一个自由群生成元及其在原图中的基点回路表示。"""

    name: str
    non_tree_edge: int
    loop_word: Tuple[OrientedEdge, ...]


class UndirectedGraph:
    """多重边/自环友好的无向图容器。"""

    def __init__(self, edges: Sequence[Tuple[int, int, str]]) -> None:
        self.edges: List[Edge] = []
        self.vertices: set[int] = set()
        self.adj: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        # adj[u] item = (neighbor, edge_id, sign) where sign indicates orientation from u to neighbor

        for edge_id, (u, v, label) in enumerate(edges):
            e = Edge(edge_id=edge_id, u=u, v=v, label=label)
            self.edges.append(e)
            self.vertices.add(u)
            self.vertices.add(v)

            # 从 u 沿参考方向到 v
            self.adj[u].append((v, edge_id, +1))
            # 从 v 逆参考方向到 u
            self.adj[v].append((u, edge_id, -1))

        self.vertex_list = sorted(self.vertices)

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def n(self) -> int:
        return len(self.vertex_list)

    def oriented_token(self, oe: OrientedEdge) -> str:
        edge_id, sign = oe
        edge = self.edges[edge_id]
        base = f"{edge.label}_{edge.edge_id}"
        return base if sign == 1 else f"{base}^-1"

    def format_word(self, word: Sequence[OrientedEdge]) -> str:
        if not word:
            return "e"
        return " ".join(self.oriented_token(x) for x in word)


def invert_word(word: Sequence[OrientedEdge]) -> List[OrientedEdge]:
    """回路逆元: 顺序反转 + 每条边方向翻转。"""

    return [(edge_id, -sign) for edge_id, sign in reversed(word)]


def reduce_word(word: Sequence[OrientedEdge]) -> List[OrientedEdge]:
    """自由群词约化: 连续逆元对消。"""

    stack: List[OrientedEdge] = []
    for edge_id, sign in word:
        if stack and stack[-1][0] == edge_id and stack[-1][1] == -sign:
            stack.pop()
        else:
            stack.append((edge_id, sign))
    return stack


def build_spanning_tree(
    graph: UndirectedGraph, root: int
) -> Tuple[Dict[int, Optional[int]], Dict[int, OrientedEdge], set[int]]:
    """BFS 生成树.

    返回:
    - parent_vertex[v] = 父顶点(root 的父为 None)
    - parent_arc[v] = (eid, sign), 表示父 -> v 的有向边
    - tree_edges = 生成树中的边 id 集合
    """

    if root not in graph.vertices:
        raise ValueError(f"root={root} 不在图顶点集合中")

    parent_vertex: Dict[int, Optional[int]] = {root: None}
    parent_arc: Dict[int, OrientedEdge] = {}
    tree_edges: set[int] = set()

    q: deque[int] = deque([root])
    while q:
        u = q.popleft()
        for v, edge_id, sign in graph.adj[u]:
            if v in parent_vertex:
                continue
            parent_vertex[v] = u
            parent_arc[v] = (edge_id, sign)
            tree_edges.add(edge_id)
            q.append(v)

    if len(parent_vertex) != graph.n:
        raise ValueError("当前 MVP 仅处理连通图；检测到图不连通")

    return parent_vertex, parent_arc, tree_edges


def path_from_root(
    vertex: int, root: int, parent_vertex: Dict[int, Optional[int]], parent_arc: Dict[int, OrientedEdge]
) -> List[OrientedEdge]:
    """返回 root -> vertex 的树路径(有向边序列)."""

    if vertex == root:
        return []

    rev: List[OrientedEdge] = []
    cur = vertex
    while cur != root:
        arc = parent_arc[cur]
        rev.append(arc)
        parent = parent_vertex[cur]
        if parent is None:
            raise RuntimeError("树路径回溯失败")
        cur = parent

    rev.reverse()
    return rev


def incidence_matrix(graph: UndirectedGraph) -> np.ndarray:
    """构造顶点-边关联矩阵 B (n x m)."""

    idx = {v: i for i, v in enumerate(graph.vertex_list)}
    B = np.zeros((graph.n, graph.m), dtype=np.int64)

    for e in graph.edges:
        if e.u == e.v:
            # 自环列为 0，不影响 B 的秩，但会把环路维数 +1。
            continue
        B[idx[e.u], e.edge_id] -= 1
        B[idx[e.v], e.edge_id] += 1

    return B


def compute_fundamental_group(
    graph: UndirectedGraph, root: int
) -> Tuple[int, List[GeneratorLoop], int, int]:
    """计算图模型下的基本群自由表示.

    返回:
    - rank_formula: 由 m-n+1 得到的自由群秩
    - generators: 每个生成元对应一个基点回路词
    - cycle_rank_numeric: 用关联矩阵秩得到的回路空间维数
    - tree_non_tree_count: 非树边数量
    """

    parent_vertex, parent_arc, tree_edges = build_spanning_tree(graph, root=root)

    generators: List[GeneratorLoop] = []
    non_tree_edges = [e for e in graph.edges if e.edge_id not in tree_edges]

    for i, e in enumerate(non_tree_edges, start=1):
        path_to_u = path_from_root(e.u, root, parent_vertex, parent_arc)
        path_to_v = path_from_root(e.v, root, parent_vertex, parent_arc)

        raw_loop = path_to_u + [(e.edge_id, +1)] + invert_word(path_to_v)
        reduced = tuple(reduce_word(raw_loop))

        generators.append(
            GeneratorLoop(
                name=f"g{i}",
                non_tree_edge=e.edge_id,
                loop_word=reduced,
            )
        )

    rank_formula = graph.m - graph.n + 1

    B = incidence_matrix(graph)
    rank_B = int(np.linalg.matrix_rank(B.astype(np.float64)))
    cycle_rank_numeric = graph.m - rank_B

    return rank_formula, generators, cycle_rank_numeric, len(non_tree_edges)


def run_case(name: str, edges: Sequence[Tuple[int, int, str]], root: int) -> None:
    graph = UndirectedGraph(edges)
    rank_formula, generators, cycle_rank_numeric, non_tree_count = compute_fundamental_group(
        graph, root=root
    )

    print(f"\n=== {name} ===")
    print(f"顶点数 n = {graph.n}, 边数 m = {graph.m}, 基点 = {root}")
    print(f"秩公式 m-n+1 = {rank_formula}")
    print(f"生成树非树边数量 = {non_tree_count}")
    print(f"关联矩阵法回路维数 = {cycle_rank_numeric}")

    if not generators:
        print("π1 ≅ {e}（平凡群）")
        return

    presentation = "<" + ", ".join(g.name for g in generators) + " | >"
    print(f"π1 表示（自由群）: {presentation}")

    table = pd.DataFrame(
        [
            {
                "generator": g.name,
                "non_tree_edge": g.non_tree_edge,
                "loop_word": graph.format_word(g.loop_word),
                "word_len": len(g.loop_word),
            }
            for g in generators
        ]
    )
    print("生成元对应的基点回路:")
    print(table.to_string(index=False))



def main() -> None:
    # Case 1: 一点双圈, π1 ≅ F2
    case1 = [
        (0, 0, "a"),
        (0, 0, "b"),
    ]

    # Case 2: theta 图(两个点三条平行边), π1 ≅ F2
    case2 = [
        (0, 1, "x"),
        (0, 1, "y"),
        (0, 1, "z"),
    ]

    # Case 3: 单圈 C4, π1 ≅ Z(作为自由群 F1)
    case3 = [
        (0, 1, "e01"),
        (1, 2, "e12"),
        (2, 3, "e23"),
        (3, 0, "e30"),
    ]

    print("基本群计算 MVP（图离散模型）")
    run_case("Case 1: bouquet of 2 circles", case1, root=0)
    run_case("Case 2: theta graph", case2, root=0)
    run_case("Case 3: cycle C4", case3, root=0)


if __name__ == "__main__":
    main()
