"""k-d 树最近邻查询 MVP（二维，静态点集）。

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
    """k-d tree 节点。"""

    split_index: int
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


def build_kdtree(points: np.ndarray, indices: np.ndarray, depth: int = 0) -> Optional[KDNode]:
    """递归构建 k-d tree。"""
    if len(indices) == 0:
        return None

    dim = points.shape[1]
    axis = depth % dim

    # 稳定排序，保证可复现实验。
    order = np.argsort(points[indices, axis], kind="mergesort")
    sorted_indices = indices[order]
    mid = len(sorted_indices) // 2

    pivot = int(sorted_indices[mid])
    node = KDNode(split_index=pivot, axis=axis)
    node.left = build_kdtree(points, sorted_indices[:mid], depth + 1)
    node.right = build_kdtree(points, sorted_indices[mid + 1 :], depth + 1)
    return node


def _nn_search(
    node: Optional[KDNode],
    points: np.ndarray,
    target: np.ndarray,
    best_idx: int,
    best_dist2: float,
) -> Tuple[int, float]:
    """递归分支定界最近邻搜索。"""
    if node is None:
        return best_idx, best_dist2

    pivot_point = points[node.split_index]
    dist2 = float(np.sum((pivot_point - target) ** 2))
    if dist2 < best_dist2:
        best_idx, best_dist2 = node.split_index, dist2

    axis = node.axis
    diff = float(target[axis] - pivot_point[axis])

    # 先访问更可能包含最近邻的一侧子树。
    near_child = node.left if diff <= 0.0 else node.right
    far_child = node.right if diff <= 0.0 else node.left

    best_idx, best_dist2 = _nn_search(near_child, points, target, best_idx, best_dist2)

    # 若切分超平面与当前最优球相交，则远侧子树仍有机会改写最优值。
    if diff * diff < best_dist2:
        best_idx, best_dist2 = _nn_search(far_child, points, target, best_idx, best_dist2)

    return best_idx, best_dist2


def nearest_neighbor(root: KDNode, points: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
    """返回最近邻索引及欧氏距离。"""
    seed_point = points[root.split_index]
    seed_dist2 = float(np.sum((seed_point - target) ** 2))
    idx, dist2 = _nn_search(root, points, target, root.split_index, seed_dist2)
    return idx, float(np.sqrt(dist2))


def brute_force_nearest(points: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
    """暴力线性扫描最近邻，用于正确性校验。"""
    dist2 = np.sum((points - target) ** 2, axis=1)
    idx = int(np.argmin(dist2))
    return idx, float(np.sqrt(float(dist2[idx])))


def main() -> None:
    rng = np.random.default_rng(195)
    n_points = 8000
    dim = 2
    points = rng.uniform(-200.0, 200.0, size=(n_points, dim))

    root = build_kdtree(points, np.arange(n_points))
    if root is None:
        raise RuntimeError("k-d tree 构建失败")

    fixed_target = np.array([33.3, -19.7])
    kd_idx, kd_dist = nearest_neighbor(root, points, fixed_target)
    bf_idx, bf_dist = brute_force_nearest(points, fixed_target)

    kd_dist2 = float(np.sum((points[kd_idx] - fixed_target) ** 2))
    bf_dist2 = float(np.sum((points[bf_idx] - fixed_target) ** 2))
    assert np.isclose(kd_dist2, bf_dist2, atol=1e-10), "固定查询最近邻距离不一致"

    for _ in range(200):
        q = rng.uniform(-200.0, 200.0, size=dim)
        kd_i, _ = nearest_neighbor(root, points, q)
        bf_i, _ = brute_force_nearest(points, q)

        kd_d2 = float(np.sum((points[kd_i] - q) ** 2))
        bf_d2 = float(np.sum((points[bf_i] - q) ** 2))
        assert np.isclose(kd_d2, bf_d2, atol=1e-10), "随机查询最近邻距离不一致"

    queries = rng.uniform(-200.0, 200.0, size=(500, dim))

    t0 = perf_counter()
    kd_acc = 0.0
    for q in queries:
        _, dist = nearest_neighbor(root, points, q)
        kd_acc += dist
    t1 = perf_counter()

    bf_acc = 0.0
    for q in queries:
        _, dist = brute_force_nearest(points, q)
        bf_acc += dist
    t2 = perf_counter()

    print("=== k-d树 最近邻查询 MVP ===")
    print(f"点数量: {n_points}, 维度: {dim}")
    print(f"固定查询点: {fixed_target.tolist()}")
    print(f"固定查询最近邻索引(k-d): {kd_idx}, 距离: {kd_dist:.6f}")
    print(f"固定查询最近邻索引(暴力): {bf_idx}, 距离: {bf_dist:.6f}")
    print("200 次随机查询正确性校验: 通过")
    print(f"500 次查询累计距离(k-d): {kd_acc:.6f}, 用时: {t1 - t0:.6f} s")
    print(f"500 次查询累计距离(暴力): {bf_acc:.6f}, 用时: {t2 - t1:.6f} s")


if __name__ == "__main__":
    main()
