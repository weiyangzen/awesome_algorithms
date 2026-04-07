"""二维范围搜索算法 MVP（k-d tree + 轴对齐矩形查询）。

运行方式:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

import numpy as np


@dataclass
class KDNode:
    """k-d tree 节点。"""

    point: np.ndarray
    index: int
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


def build_kdtree(points: np.ndarray, indices: np.ndarray, depth: int = 0) -> Optional[KDNode]:
    """递归构建二维 k-d tree。"""
    if len(indices) == 0:
        return None

    axis = depth % 2
    order = np.argsort(points[indices, axis], kind="mergesort")
    sorted_indices = indices[order]
    mid = len(sorted_indices) // 2

    idx = int(sorted_indices[mid])
    node = KDNode(point=points[idx], index=idx, axis=axis)
    node.left = build_kdtree(points, sorted_indices[:mid], depth + 1)
    node.right = build_kdtree(points, sorted_indices[mid + 1 :], depth + 1)
    return node


def range_search(
    node: Optional[KDNode],
    rect_min: np.ndarray,
    rect_max: np.ndarray,
    out_indices: List[int],
) -> None:
    """在 k-d tree 上执行轴对齐矩形范围搜索。"""
    if node is None:
        return

    p = node.point
    if np.all(p >= rect_min) and np.all(p <= rect_max):
        out_indices.append(node.index)

    axis = node.axis
    if rect_min[axis] <= p[axis]:
        range_search(node.left, rect_min, rect_max, out_indices)
    if rect_max[axis] >= p[axis]:
        range_search(node.right, rect_min, rect_max, out_indices)


def brute_force_range(points: np.ndarray, rect_min: np.ndarray, rect_max: np.ndarray) -> np.ndarray:
    """线性扫描基线实现，用于正确性校验。"""
    mask = np.all((points >= rect_min) & (points <= rect_max), axis=1)
    return np.where(mask)[0]


def one_query(root: KDNode, rect_min: np.ndarray, rect_max: np.ndarray) -> np.ndarray:
    """执行一次查询并返回按升序排列的点索引。"""
    hits: List[int] = []
    range_search(root, rect_min, rect_max, hits)
    return np.array(sorted(hits), dtype=np.int64)


def main() -> None:
    rng = np.random.default_rng(42)
    n_points = 5000
    points = rng.uniform(-100.0, 100.0, size=(n_points, 2))

    root = build_kdtree(points, np.arange(n_points))
    if root is None:
        raise RuntimeError("k-d tree 构建失败")

    # 固定一次可复现实验
    rect_min = np.array([-20.0, -10.0])
    rect_max = np.array([35.0, 40.0])
    kd_ans = one_query(root, rect_min, rect_max)
    bf_ans = brute_force_range(points, rect_min, rect_max)
    assert np.array_equal(kd_ans, bf_ans), "固定查询结果与暴力法不一致"

    # 多次随机查询校验正确性
    for _ in range(100):
        a = rng.uniform(-100.0, 100.0, size=2)
        b = rng.uniform(-100.0, 100.0, size=2)
        qmin = np.minimum(a, b)
        qmax = np.maximum(a, b)
        kd = one_query(root, qmin, qmax)
        bf = brute_force_range(points, qmin, qmax)
        assert np.array_equal(kd, bf), "随机查询结果与暴力法不一致"

    # 粗略性能对比（同一批查询）
    queries = []
    for _ in range(200):
        a = rng.uniform(-100.0, 100.0, size=2)
        b = rng.uniform(-100.0, 100.0, size=2)
        queries.append((np.minimum(a, b), np.maximum(a, b)))

    t0 = perf_counter()
    kd_total = 0
    for qmin, qmax in queries:
        kd_total += len(one_query(root, qmin, qmax))
    t1 = perf_counter()

    bf_total = 0
    for qmin, qmax in queries:
        bf_total += len(brute_force_range(points, qmin, qmax))
    t2 = perf_counter()

    print("=== 范围搜索算法 (k-d tree) MVP ===")
    print(f"点数量: {n_points}")
    print(f"固定矩形: min={rect_min.tolist()}, max={rect_max.tolist()}")
    print(f"固定查询命中数: {len(kd_ans)}")
    print("100 次随机查询正确性校验: 通过")
    print(f"200 次查询总命中(k-d tree): {kd_total}, 用时: {t1 - t0:.6f} s")
    print(f"200 次查询总命中(暴力扫描): {bf_total}, 用时: {t2 - t1:.6f} s")


if __name__ == "__main__":
    main()
