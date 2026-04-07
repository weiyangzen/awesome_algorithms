"""Mapper算法 MVP.

实现目标:
1) 生成一个带环结构的二维点云（noisy circles）；
2) 用 1D lens 函数 + 重叠区间覆盖（cover）切片；
3) 在每个切片内做局部聚类，聚类块作为 Mapper 节点；
4) 节点成员集合有交集则连边，得到 Mapper 图。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles


@dataclass(frozen=True)
class CoverInterval:
    """1D lens 上的一个重叠区间."""

    interval_id: int
    left: float
    right: float


@dataclass(frozen=True)
class MapperNode:
    """Mapper 图节点: 某个区间中的一个聚类块."""

    node_id: int
    interval_id: int
    cluster_label: int
    members: Tuple[int, ...]


@dataclass
class MapperGraph:
    """Mapper 图结果容器."""

    nodes: List[MapperNode]
    edges: List[Tuple[int, int]]
    intervals: List[CoverInterval]
    interval_point_counts: List[int]


def compute_lens_torch_pca1(x: np.ndarray) -> np.ndarray:
    """用 PyTorch SVD 计算第一主方向投影作为 lens."""

    xt = torch.from_numpy(x).to(torch.float64)
    xc = xt - xt.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(xc, full_matrices=False)
    principal_direction = vt[0]
    lens = (xc @ principal_direction).cpu().numpy()
    return lens.astype(np.float64)


def build_cover_intervals(
    lens: np.ndarray, n_intervals: int, overlap_ratio: float
) -> List[CoverInterval]:
    """在 lens 的数值范围上构造等宽重叠区间."""

    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    lo = float(np.min(lens))
    hi = float(np.max(lens))
    denom = n_intervals - (n_intervals - 1) * overlap_ratio
    if denom <= 0:
        raise ValueError("invalid n_intervals/overlap_ratio combination")

    width = (hi - lo) / denom if hi > lo else 1.0
    step = width * (1.0 - overlap_ratio)

    intervals: List[CoverInterval] = []
    for i in range(n_intervals):
        left = lo + i * step
        right = left + width
        intervals.append(CoverInterval(interval_id=i, left=left, right=right))
    return intervals


def points_in_interval(
    lens: np.ndarray, interval: CoverInterval, is_last: bool
) -> np.ndarray:
    """返回落在区间内的样本下标."""

    if is_last:
        mask = (lens >= interval.left) & (lens <= interval.right)
    else:
        mask = (lens >= interval.left) & (lens < interval.right)
    return np.where(mask)[0]


def build_mapper_graph(
    x: np.ndarray,
    lens: np.ndarray,
    n_intervals: int = 8,
    overlap_ratio: float = 0.35,
    dbscan_eps: float = 0.11,
    dbscan_min_samples: int = 5,
) -> MapperGraph:
    """构建 Mapper 图."""

    intervals = build_cover_intervals(lens, n_intervals=n_intervals, overlap_ratio=overlap_ratio)

    nodes: List[MapperNode] = []
    interval_point_counts: List[int] = []

    for i, interval in enumerate(intervals):
        idx = points_in_interval(lens, interval, is_last=(i == len(intervals) - 1))
        interval_point_counts.append(int(idx.size))
        if idx.size == 0:
            continue

        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(x[idx])
        for cluster_label in sorted(set(labels.tolist())):
            if cluster_label < 0:
                continue  # DBSCAN 噪声点
            members = idx[labels == cluster_label]
            node = MapperNode(
                node_id=len(nodes),
                interval_id=interval.interval_id,
                cluster_label=int(cluster_label),
                members=tuple(sorted(members.tolist())),
            )
            nodes.append(node)

    member_sets = [set(node.members) for node in nodes]
    edges_set: set[Tuple[int, int]] = set()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i].interval_id == nodes[j].interval_id:
                continue
            if member_sets[i].intersection(member_sets[j]):
                edges_set.add((i, j))

    edges = sorted(edges_set)
    return MapperGraph(
        nodes=nodes,
        edges=edges,
        intervals=intervals,
        interval_point_counts=interval_point_counts,
    )


def graph_connected_components(node_count: int, edges: Sequence[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
    """用 scipy.sparse.csgraph 计算连通分量."""

    if node_count == 0:
        return 0, np.array([], dtype=np.int64)
    if not edges:
        return node_count, np.arange(node_count, dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])

    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    return int(n_components), labels.astype(np.int64)


def main() -> None:
    np.random.seed(7)

    x, _ = make_circles(n_samples=500, factor=0.35, noise=0.04, random_state=7)
    lens = compute_lens_torch_pca1(x)

    mapper = build_mapper_graph(
        x=x,
        lens=lens,
        n_intervals=8,
        overlap_ratio=0.35,
        dbscan_eps=0.11,
        dbscan_min_samples=5,
    )

    n_components, component_labels = graph_connected_components(
        node_count=len(mapper.nodes), edges=mapper.edges
    )

    interval_table = pd.DataFrame(
        {
            "interval_id": [it.interval_id for it in mapper.intervals],
            "left": [it.left for it in mapper.intervals],
            "right": [it.right for it in mapper.intervals],
            "point_count": mapper.interval_point_counts,
            "node_count": [sum(1 for n in mapper.nodes if n.interval_id == it.interval_id) for it in mapper.intervals],
        }
    )

    node_rows = []
    for node in mapper.nodes:
        members = np.array(node.members, dtype=np.int64)
        node_rows.append(
            {
                "node_id": node.node_id,
                "interval_id": node.interval_id,
                "cluster_label": node.cluster_label,
                "size": int(members.size),
                "lens_min": float(np.min(lens[members])) if members.size else np.nan,
                "lens_max": float(np.max(lens[members])) if members.size else np.nan,
                "component": int(component_labels[node.node_id]) if component_labels.size else -1,
            }
        )
    node_table = pd.DataFrame(node_rows)

    print("=== Mapper算法 MVP ===")
    print(f"样本数: {x.shape[0]}, 维度: {x.shape[1]}")
    print(f"lens范围: [{lens.min():.4f}, {lens.max():.4f}]")
    print(f"区间数: {len(mapper.intervals)}, 节点数: {len(mapper.nodes)}, 边数: {len(mapper.edges)}")
    print(f"连通分量数: {n_components}")

    print("\n[区间统计]")
    print(interval_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[节点统计（前12行）]")
    if node_table.empty:
        print("(empty)")
    else:
        print(node_table.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[边列表（前20条）]")
    if not mapper.edges:
        print("(no edges)")
    else:
        for edge in mapper.edges[:20]:
            print(edge)


if __name__ == "__main__":
    main()
