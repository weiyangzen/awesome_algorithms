"""持久同调算法 MVP（Vietoris-Rips + Z2 边界矩阵约化）.

实现目标：
1) 从二维点云构建 Vietoris-Rips 过滤（到 2-单形）；
2) 显式构造边界矩阵并做 Z2 列约化；
3) 产出 H0/H1 的持久区间摘要（birth, death, persistence）。

说明：
- 该实现不依赖专用 TDA 库，便于追踪算法细节；
- 通过最多 2-单形可稳定得到 H0/H1（H2 及以上未实现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Simplex:
    """过滤中的单形."""

    vertices: Tuple[int, ...]
    dim: int
    filtration: float


@dataclass(frozen=True)
class Interval:
    """持久区间 [birth, death). death=inf 表示未被杀死."""

    dim: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        if math.isinf(self.death):
            return math.inf
        return self.death - self.birth


def make_point_cloud(n_samples: int = 28, noise: float = 0.01) -> np.ndarray:
    """生成双环点云并做标准化."""

    points, _ = make_circles(
        n_samples=n_samples, factor=0.45, noise=noise, random_state=7
    )
    points = StandardScaler().fit_transform(points)
    return points.astype(np.float64)


def pairwise_distance(points: np.ndarray) -> np.ndarray:
    """使用 scipy 计算欧氏距离矩阵."""

    return squareform(pdist(points, metric="euclidean"))


def torch_distance_error(points: np.ndarray, dist_ref: np.ndarray) -> float:
    """用 torch.cdist 复算距离并给出与 scipy 结果的最大误差."""

    tensor = torch.from_numpy(points)
    dist_torch = torch.cdist(tensor, tensor, p=2).cpu().numpy()
    return float(np.max(np.abs(dist_torch - dist_ref)))


def build_vr_complex(
    dist_mat: np.ndarray, max_edge_length: float, max_dim: int = 2
) -> Tuple[List[Simplex], Dict[Tuple[int, int], float]]:
    """构造 Vietoris-Rips 过滤（最多到 max_dim）.

    过滤值定义：
    - 顶点: 0
    - 边: 边长
    - 三角形: 三条边长度的最大值
    """

    n = dist_mat.shape[0]
    simplices: List[Simplex] = [Simplex((i,), 0, 0.0) for i in range(n)]
    edge_weight: Dict[Tuple[int, int], float] = {}

    # 1-单形
    for i in range(n - 1):
        for j in range(i + 1, n):
            dij = float(dist_mat[i, j])
            if dij <= max_edge_length:
                edge_weight[(i, j)] = dij
                simplices.append(Simplex((i, j), 1, dij))

    # 2-单形
    if max_dim >= 2:
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                dij = edge_weight.get((i, j))
                if dij is None:
                    continue
                for k in range(j + 1, n):
                    dik = edge_weight.get((i, k))
                    djk = edge_weight.get((j, k))
                    if dik is None or djk is None:
                        continue
                    filtration = max(dij, dik, djk)
                    simplices.append(Simplex((i, j, k), 2, filtration))

    # 关键排序：先按过滤值，再按维度（保证同过滤值下先有面后有体）
    simplices.sort(key=lambda s: (s.filtration, s.dim, s.vertices))
    return simplices, edge_weight


def build_boundary_columns(simplices: Sequence[Simplex]) -> List[Set[int]]:
    """构造 Z2 边界矩阵的列（稀疏集合表示）."""

    simplex_to_index = {simplex.vertices: i for i, simplex in enumerate(simplices)}
    columns: List[Set[int]] = []

    for simplex in simplices:
        if simplex.dim == 0:
            columns.append(set())
            continue

        boundary_faces: Set[int] = set()
        verts = simplex.vertices
        for t in range(len(verts)):
            face = verts[:t] + verts[t + 1 :]
            face_idx = simplex_to_index.get(face)
            if face_idx is None:
                raise RuntimeError(f"缺失面单形: {face}")
            boundary_faces.add(face_idx)
        columns.append(boundary_faces)

    return columns


def reduce_boundary_matrix_z2(columns: Sequence[Set[int]]) -> Tuple[List[Set[int]], Dict[int, int]]:
    """标准列约化算法（系数域 Z2）.

    返回：
    - reduced_columns: 每列约化后的稀疏集合
    - pivot_column_of_low: low(row_index) -> column_index
    """

    reduced_columns: List[Set[int]] = [set() for _ in range(len(columns))]
    pivot_column_of_low: Dict[int, int] = {}

    for j, original_col in enumerate(columns):
        col = set(original_col)

        while col:
            low = max(col)
            pivot_col = pivot_column_of_low.get(low)
            if pivot_col is None:
                break
            # Z2 上加法等于对称差
            col ^= reduced_columns[pivot_col]

        reduced_columns[j] = col
        if col:
            pivot_column_of_low[max(col)] = j

    return reduced_columns, pivot_column_of_low


def extract_intervals(
    simplices: Sequence[Simplex],
    reduced_columns: Sequence[Set[int]],
    pivot_column_of_low: Dict[int, int],
) -> List[Interval]:
    """从约化结果提取持久区间."""

    intervals: List[Interval] = []
    paired_birth_indices: Set[int] = set()

    # finite intervals
    for birth_idx, death_idx in pivot_column_of_low.items():
        birth_simplex = simplices[birth_idx]
        death_simplex = simplices[death_idx]
        intervals.append(
            Interval(
                dim=birth_simplex.dim,
                birth=birth_simplex.filtration,
                death=death_simplex.filtration,
            )
        )
        paired_birth_indices.add(birth_idx)

    # infinite intervals（未被配对且本列约化后为空）
    for idx, simplex in enumerate(simplices):
        if idx in paired_birth_indices:
            continue
        if reduced_columns[idx]:
            continue
        intervals.append(Interval(dim=simplex.dim, birth=simplex.filtration, death=math.inf))

    return intervals


def intervals_to_dataframe(
    intervals: Sequence[Interval], dim: int, eps: float = 1e-10, top_k: int = 8
) -> pd.DataFrame:
    """将指定维度区间转为表格并按持久度排序."""

    rows = []
    for itv in intervals:
        if itv.dim != dim:
            continue
        if math.isfinite(itv.death) and (itv.death - itv.birth) <= eps:
            continue
        rows.append(
            {
                "dim": itv.dim,
                "birth": itv.birth,
                "death": itv.death,
                "persistence": itv.persistence,
                "is_infinite": math.isinf(itv.death),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["dim", "birth", "death", "persistence", "is_infinite"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["persistence", "birth"], ascending=[False, True]).head(top_k)
    return df.reset_index(drop=True)


def betti_at_scale(intervals: Sequence[Interval], dim: int, scale: float) -> int:
    """给定过滤阈值下的 Betti 数估计（按区间存活判定）."""

    alive = 0
    for itv in intervals:
        if itv.dim != dim:
            continue
        if itv.birth <= scale and (scale < itv.death or math.isinf(itv.death)):
            alive += 1
    return alive


def main() -> None:
    points = make_point_cloud(n_samples=28, noise=0.01)
    dist_mat = pairwise_distance(points)
    max_error = torch_distance_error(points, dist_mat)

    max_edge_length = 1.6
    simplices, edge_weight = build_vr_complex(
        dist_mat=dist_mat, max_edge_length=max_edge_length, max_dim=2
    )
    columns = build_boundary_columns(simplices)
    reduced_columns, pivot_column_of_low = reduce_boundary_matrix_z2(columns)
    intervals = extract_intervals(simplices, reduced_columns, pivot_column_of_low)

    h0_df = intervals_to_dataframe(intervals, dim=0, top_k=8)
    h1_df = intervals_to_dataframe(intervals, dim=1, top_k=8)

    tri_count = sum(1 for s in simplices if s.dim == 2)
    print("=== 持久同调算法 MVP（H0/H1, Vietoris-Rips） ===")
    print(f"点数: {points.shape[0]}")
    print(f"最大边阈值: {max_edge_length:.3f}")
    print(f"1-单形(边)数量: {len(edge_weight)}")
    print(f"2-单形(三角形)数量: {tri_count}")
    print(f"过滤单形总数: {len(simplices)}")
    print(f"torch 与 scipy 距离矩阵最大绝对误差: {max_error:.3e}")

    if h0_df.empty:
        print("\nH0 区间: (empty)")
    else:
        print("\nH0 主要区间（按持久度降序）:")
        print(h0_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if h1_df.empty:
        print("\nH1 区间: (empty)")
    else:
        print("\nH1 主要区间（按持久度降序）:")
        print(h1_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    probe_scales = [0.6, 1.0, 1.4]
    print("\nBetti 数探针（由区间直接计数）:")
    for s in probe_scales:
        b0 = betti_at_scale(intervals, dim=0, scale=s)
        b1 = betti_at_scale(intervals, dim=1, scale=s)
        print(f"scale={s:.2f}: beta0={b0}, beta1={b1}")


if __name__ == "__main__":
    main()
