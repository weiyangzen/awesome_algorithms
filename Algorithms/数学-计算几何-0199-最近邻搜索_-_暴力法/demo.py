"""最近邻搜索 MVP（暴力法）。

运行方式:
    python3 demo.py
"""

from __future__ import annotations

from time import perf_counter
from typing import Tuple

import numpy as np


def brute_force_nearest_python(points: np.ndarray, query: np.ndarray) -> Tuple[int, float]:
    """纯 Python 循环实现的最近邻暴力搜索。"""
    best_idx = -1
    best_dist2 = float("inf")

    for idx in range(points.shape[0]):
        px = float(points[idx, 0])
        py = float(points[idx, 1])
        dx = px - float(query[0])
        dy = py - float(query[1])
        dist2 = dx * dx + dy * dy

        if dist2 < best_dist2:
            best_dist2 = dist2
            best_idx = idx

    if best_idx < 0:
        raise ValueError("点集为空，无法执行最近邻搜索")

    return best_idx, best_dist2


def brute_force_nearest_numpy(points: np.ndarray, query: np.ndarray) -> Tuple[int, float]:
    """NumPy 向量化实现的单查询暴力最近邻。"""
    diff = points - query
    dist2 = np.einsum("ij,ij->i", diff, diff)
    idx = int(np.argmin(dist2))
    return idx, float(dist2[idx])


def brute_force_batch_numpy(
    points: np.ndarray,
    queries: np.ndarray,
    chunk_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy 分块批量查询，避免一次性占用过高内存。"""
    n_queries = queries.shape[0]
    all_indices = np.empty(n_queries, dtype=np.int64)
    all_dist2 = np.empty(n_queries, dtype=np.float64)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        query_chunk = queries[start:end]

        diff = query_chunk[:, None, :] - points[None, :, :]
        dist2_matrix = np.einsum("bij,bij->bi", diff, diff)

        local_idx = np.argmin(dist2_matrix, axis=1)
        row = np.arange(dist2_matrix.shape[0])

        all_indices[start:end] = local_idx
        all_dist2[start:end] = dist2_matrix[row, local_idx]

    return all_indices, all_dist2


def main() -> None:
    rng = np.random.default_rng(2026)

    n_points = 12000
    points = rng.uniform(-200.0, 200.0, size=(n_points, 2)).astype(np.float64)

    fixed_query = np.array([25.5, -13.0], dtype=np.float64)
    idx_np, d2_np = brute_force_nearest_numpy(points, fixed_query)
    idx_py, d2_py = brute_force_nearest_python(points, fixed_query)

    assert np.isclose(d2_np, d2_py, atol=1e-12), "固定查询最近距离不一致"

    # 随机查询一致性校验
    for _ in range(200):
        query = rng.uniform(-200.0, 200.0, size=2)
        _, d2_a = brute_force_nearest_numpy(points, query)
        _, d2_b = brute_force_nearest_python(points, query)
        assert np.isclose(d2_a, d2_b, atol=1e-12), "随机查询最近距离不一致"

    # NumPy 批量路径性能
    n_queries_numpy = 800
    queries = rng.uniform(-200.0, 200.0, size=(n_queries_numpy, 2)).astype(np.float64)

    t0 = perf_counter()
    idx_batch_np, d2_batch_np = brute_force_batch_numpy(points, queries, chunk_size=128)
    time_numpy = perf_counter() - t0

    # 在较小子集上对照 Python 循环耗时与结果
    n_queries_python = 120
    queries_small = queries[:n_queries_python]

    t1 = perf_counter()
    py_dist_checksum = 0.0
    py_idx_checksum = 0
    for q in queries_small:
        idx, d2 = brute_force_nearest_python(points, q)
        py_dist_checksum += d2
        py_idx_checksum += idx
    time_python = perf_counter() - t1

    np_dist_checksum = float(np.sum(d2_batch_np[:n_queries_python]))
    np_idx_checksum = int(np.sum(idx_batch_np[:n_queries_python]))

    assert np.isclose(py_dist_checksum, np_dist_checksum, atol=1e-9), "批量距离校验失败"
    assert py_idx_checksum == np_idx_checksum, "批量索引校验失败"

    speedup = time_python / time_numpy if time_numpy > 0 else float("inf")

    print("=== 最近邻搜索 - 暴力法 MVP ===")
    print(f"点数量: {n_points}")
    print(f"固定查询点: {fixed_query.tolist()}")
    print(f"固定查询最近邻索引(np): {idx_np}, 距离平方: {d2_np:.8f}")
    print(f"固定查询最近邻索引(python): {idx_py}, 距离平方: {d2_py:.8f}")
    print("200 次随机查询一致性校验: 通过")
    print(f"NumPy 批量查询: {n_queries_numpy} 次, 耗时 {time_numpy:.6f} s")
    print(f"Python 循环查询: {n_queries_python} 次, 耗时 {time_python:.6f} s")
    print("批量子集距离/索引校验: 通过")
    print(f"同批加速比(python/numpy, 仅供参考): {speedup:.2f}x")


if __name__ == "__main__":
    main()
