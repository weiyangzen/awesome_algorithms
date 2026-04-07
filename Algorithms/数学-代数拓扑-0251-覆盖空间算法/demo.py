"""覆盖空间算法 MVP: 有限图的截断万有覆盖构造.

该示例把“覆盖空间”离散化到图论语境：
1) 基空间使用连通无向图 G；
2) 覆盖空间使用“从根出发的约化路径”组成的树（万有覆盖的截断近似）；
3) 覆盖映射 p 将每条约化路径映到其在 G 中的终点。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DirectedEdge:
    """无向边的一个定向副本."""

    arc_id: int
    edge_id: int
    tail: int
    head: int
    symbol: str


class UndirectedGraph:
    """带定向弧展开的无向图."""

    def __init__(self, edges: Sequence[Tuple[int, int, str]]) -> None:
        self.edge_count = len(edges)
        self._directed: List[DirectedEdge] = []
        self.inverse: Dict[int, int] = {}
        self.outgoing: Dict[int, List[int]] = defaultdict(list)
        self.vertices: set[int] = set()

        for edge_id, (u, v, label) in enumerate(edges):
            self.vertices.add(u)
            self.vertices.add(v)
            a = 2 * edge_id
            b = a + 1
            arc_forward = DirectedEdge(a, edge_id, u, v, f"{label}")
            arc_backward = DirectedEdge(b, edge_id, v, u, f"{label}^-1")
            self._directed.append(arc_forward)
            self._directed.append(arc_backward)
            self.inverse[a] = b
            self.inverse[b] = a
            self.outgoing[u].append(a)
            self.outgoing[v].append(b)

        self.vertex_count = len(self.vertices)

    def arc(self, arc_id: int) -> DirectedEdge:
        return self._directed[arc_id]

    def arc_symbols(self, arc_ids: Sequence[int]) -> str:
        if not arc_ids:
            return "e"
        return " ".join(self.arc(a).symbol for a in arc_ids)


@dataclass(frozen=True)
class CoverNode:
    """覆盖图中的节点: 一条从根开始的约化弧序列."""

    node_id: int
    word: Tuple[int, ...]
    base_vertex: int
    depth: int
    parent: Optional[int]
    arc_from_parent: Optional[int]


@dataclass
class TruncatedUniversalCover:
    """截断万有覆盖."""

    nodes: List[CoverNode]
    adjacency: Dict[int, List[Tuple[int, int]]]  # node -> [(neighbor, outgoing_arc_at_node)]
    max_depth: int

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2


def cyclomatic_rank(graph: UndirectedGraph, components: int = 1) -> int:
    """连通图的基本群秩（自由群秩）: m - n + c."""

    return graph.edge_count - graph.vertex_count + components


def build_truncated_universal_cover(
    graph: UndirectedGraph, root_vertex: int, max_depth: int
) -> TruncatedUniversalCover:
    """构造“以 root 为根，深度 max_depth”的万有覆盖截断树."""

    root = CoverNode(
        node_id=0,
        word=tuple(),
        base_vertex=root_vertex,
        depth=0,
        parent=None,
        arc_from_parent=None,
    )
    nodes: List[CoverNode] = [root]
    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    word_to_id: Dict[Tuple[int, ...], int] = {tuple(): 0}
    queue: deque[int] = deque([0])

    while queue:
        cur_id = queue.popleft()
        cur = nodes[cur_id]
        if cur.depth >= max_depth:
            continue

        forbidden = None
        if cur.word:
            forbidden = graph.inverse[cur.word[-1]]

        for arc_id in graph.outgoing[cur.base_vertex]:
            # 约化词约束: 禁止“马上走回头边”的消去对.
            if forbidden is not None and arc_id == forbidden:
                continue

            new_word = cur.word + (arc_id,)
            if new_word in word_to_id:
                # 理论上约化词构造不会发生冲突, 此处仅做保护.
                nxt_id = word_to_id[new_word]
            else:
                nxt_id = len(nodes)
                word_to_id[new_word] = nxt_id
                arc = graph.arc(arc_id)
                nxt = CoverNode(
                    node_id=nxt_id,
                    word=new_word,
                    base_vertex=arc.head,
                    depth=cur.depth + 1,
                    parent=cur_id,
                    arc_from_parent=arc_id,
                )
                nodes.append(nxt)
                queue.append(nxt_id)

            # 添加无向边（两端分别记录局部“出射弧”）
            if not any(nei == nxt_id for nei, _ in adjacency[cur_id]):
                adjacency[cur_id].append((nxt_id, arc_id))
                adjacency[nxt_id].append((cur_id, graph.inverse[arc_id]))

    return TruncatedUniversalCover(nodes=nodes, adjacency=adjacency, max_depth=max_depth)


def check_local_covering_property(
    graph: UndirectedGraph, cover: TruncatedUniversalCover
) -> Tuple[bool, List[str]]:
    """检查截断内部节点的局部星状邻域是否与基图完全对应."""

    errors: List[str] = []
    for node in cover.nodes:
        if node.depth >= cover.max_depth:
            continue
        got = sorted(arc_id for _, arc_id in cover.adjacency.get(node.node_id, []))
        expected = sorted(graph.outgoing[node.base_vertex])
        if got != expected:
            errors.append(
                f"node={node.node_id}, depth={node.depth}, "
                f"base_v={node.base_vertex}, got={got}, expected={expected}"
            )
    return (len(errors) == 0, errors)


def summarize_levels(cover: TruncatedUniversalCover) -> List[Tuple[int, int]]:
    """使用 numpy 汇总每层节点数量."""

    depths = np.array([n.depth for n in cover.nodes], dtype=np.int64)
    values, counts = np.unique(depths, return_counts=True)
    return list(zip(values.tolist(), counts.tolist()))


def main() -> None:
    # 示例基图: 一点双圈（bouquet of two circles），基本群秩为 2 的自由群.
    base_graph = UndirectedGraph(
        edges=[
            (0, 0, "a"),
            (0, 0, "b"),
        ]
    )

    max_depth = 5
    cover = build_truncated_universal_cover(base_graph, root_vertex=0, max_depth=max_depth)
    ok, violations = check_local_covering_property(base_graph, cover)
    rank = cyclomatic_rank(base_graph)

    print("=== 覆盖空间算法 MVP: 截断万有覆盖 ===")
    print(f"基图顶点数: {base_graph.vertex_count}")
    print(f"基图边数: {base_graph.edge_count}")
    print(f"基本群秩(自由群秩): {rank}")
    print(f"截断深度: {max_depth}")
    print(f"覆盖图节点数: {len(cover.nodes)}")
    print(f"覆盖图边数: {cover.edge_count}")
    print(f"内部节点局部覆盖性质: {'PASS' if ok else 'FAIL'}")

    if not ok:
        print("局部覆盖失败节点:")
        for line in violations[:8]:
            print(" -", line)

    print("\n每层节点数:")
    for d, c in summarize_levels(cover):
        print(f"depth={d}: count={c}")

    print("\n前 10 个覆盖节点（node_id: 约化词 -> 基图顶点）:")
    for node in cover.nodes[:10]:
        print(
            f"{node.node_id:>2}: {base_graph.arc_symbols(node.word):<20} -> {node.base_vertex}"
        )


if __name__ == "__main__":
    main()
