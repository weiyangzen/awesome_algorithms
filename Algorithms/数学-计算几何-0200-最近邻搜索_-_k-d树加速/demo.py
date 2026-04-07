"""最近邻搜索 MVP（k-d tree 加速）。

运行方式:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple

import numpy as np


@dataclass
class KDNode:
    """二维 k-d tree 节点。"""

    index: int
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


def build_kdtree(points: np.ndarray, indices: np.ndarray, depth: int = 0) -> Optional[KDNode]:
    """递归构建 k-d tree。"""
    if len(indices) == 0:
        return None

    axis = depth % 2
    order = np.argsort(points[indices, axis], kind="mergesort")
    sorted_indices = indices[order]
    mid = len(sorted_indices) // 2

    root_idx = int(sorted_indices[mid])
    node = KDNode(index=root_idx, axis=axis)
    node.left = build_kdtree(points, sorted_indices[:mid], depth + 1)
    node.right = build_kdtree(points, sorted_indices[mid + 1 :], depth + 1)
    return node


def _nearest_search(
    node: Optional[KDNode],
    points: np.ndarray,
    query: np.ndarray,
    best_idx: int,
    best_dist2: float,
    visited: int,
) -> Tuple[int, float, int]:
    """在 k-d tree 上做一次最近邻搜索（带回溯剪枝）。"""
    if node is None:
        return best_idx, best_dist2, visited

    visited += 1
    point = points[node.index]
    diff_vec = point - query
    dist2 = float(np.dot(diff_vec, diff_vec))

    if dist2 < best_dist2:
        best_idx = node.index
        best_dist2 = dist2

    axis = node.axis
    axis_diff = float(query[axis] - point[axis])

    # 先访问更可能包含最近点的一侧，再按超平面距离决定是否回溯另一侧。
    near_child = node.left if axis_diff <= 0 else node.right
    far_child = node.right if axis_diff <= 0 else node.left

    best_idx, best_dist2, visited = _nearest_search(
        near_child,
        points,
        query,
        best_idx,
        best_dist2,
        visited,
    )

    if axis_diff * axis_diff < best_dist2:
        best_idx, best_dist2, visited = _nearest_search(
            far_child,
            points,
            query,
            best_idx,
            best_dist2,
            visited,
        )

    return best_idx, best_dist2, visited


def kd_nearest_neighbor(root: KDNode, points: np.ndarray, query: np.ndarray) -> Tuple[int, float, int]:
    """返回 (最近点索引, 最近平方距离, 访问节点数)。"""
    root_point = points[root.index]
    init_diff = root_point - query
    init_dist2 = float(np.dot(init_diff, init_diff))
    return _nearest_search(root, points, query, root.index, init_dist2, 0)


def brute_force_nearest(points: np.ndarray, query: np.ndarray) -> Tuple[int, float]:
    """暴力扫描基线：O(n) 计算最近邻。"""
    diff = points - query
    dist2 = np.einsum("ij,ij->i", diff, diff)
    idx = int(np.argmin(dist2))
    return idx, float(dist2[idx])


def main() -> None:
    rng = np.random.default_rng(2026)
    n_points = 10000
    points = rng.uniform(-100.0, 100.0, size=(n_points, 2))

    build_start = perf_counter()
    root = build_kdtree(points, np.arange(n_points))
    build_time = perf_counter() - build_start
    if root is None:
        raise RuntimeError("k-d tree 构建失败")

    # 固定查询 + 随机查询正确性验证（比较最近距离，避免并列最近点引起索引歧义）。
    fixed_query = np.array([12.5, -18.0], dtype=np.float64)
    kd_idx, kd_d2, fixed_visited = kd_nearest_neighbor(root, points, fixed_query)
    bf_idx, bf_d2 = brute_force_nearest(points, fixed_query)
    assert np.isclose(kd_d2, bf_d2, atol=1e-12), "固定查询最近距离与暴力法不一致"

    for _ in range(200):
        query = rng.uniform(-100.0, 100.0, size=2)
        _, kd_dist2, _ = kd_nearest_neighbor(root, points, query)
        _, bf_dist2 = brute_force_nearest(points, query)
        assert np.isclose(kd_dist2, bf_dist2, atol=1e-12), "随机查询最近距离与暴力法不一致"

    # 批量性能对比
    n_queries = 500
    queries = rng.uniform(-100.0, 100.0, size=(n_queries, 2))

    kd_total_visited = 0
    kd_time_start = perf_counter()
    kd_dist_checksum = 0.0
    for q in queries:
        _, d2, visited = kd_nearest_neighbor(root, points, q)
        kd_dist_checksum += d2
        kd_total_visited += visited
    kd_time = perf_counter() - kd_time_start

    bf_time_start = perf_counter()
    bf_dist_checksum = 0.0
    for q in queries:
        _, d2 = brute_force_nearest(points, q)
        bf_dist_checksum += d2
    bf_time = perf_counter() - bf_time_start

    assert np.isclose(kd_dist_checksum, bf_dist_checksum, atol=1e-9), "批量查询校验失败"

    avg_visited = kd_total_visited / n_queries
    visited_ratio = avg_visited / n_points
    speedup = bf_time / kd_time if kd_time > 0 else float("inf")

    print("=== 最近邻搜索 - k-d树加速 MVP ===")
    print(f"点数量: {n_points}")
    print(f"构建耗时: {build_time:.6f} s")
    print(f"固定查询: {fixed_query.tolist()}")
    print(f"固定查询最近点索引(kd): {kd_idx}, 距离平方: {kd_d2:.8f}")
    print(f"固定查询最近点索引(brute): {bf_idx}, 距离平方: {bf_d2:.8f}")
    print(f"固定查询访问节点数(kd): {fixed_visited}")
    print("200 次随机查询正确性校验: 通过")
    print(f"{n_queries} 次查询总距离平方校验: 通过")
    print(f"k-d tree 查询耗时: {kd_time:.6f} s")
    print(f"暴力扫描耗时: {bf_time:.6f} s")
    print(f"平均访问节点数(kd): {avg_visited:.2f} ({visited_ratio:.2%} of 全点数)")
    print(f"查询加速比(暴力/kd): {speedup:.2f}x")


if __name__ == "__main__":
    main()
