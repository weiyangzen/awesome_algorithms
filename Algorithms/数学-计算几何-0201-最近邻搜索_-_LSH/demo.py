"""最近邻搜索 MVP（LSH: Random Hyperplane for cosine similarity）。

运行方式:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def matmul_no_warn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """在当前环境中抑制底层 BLAS 的无害浮点告警。"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，零向量保持为零。"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return x / safe_norms


def normalize_vector(x: np.ndarray) -> np.ndarray:
    """向量 L2 归一化，零向量保持为零。"""
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return x.copy()
    return x / norm


@dataclass
class RandomHyperplaneLSH:
    """基于随机超平面的多表 LSH 索引（近似余弦最近邻）。"""

    num_tables: int = 12
    num_bits: int = 10
    min_candidates: int = 80
    enable_multiprobe: bool = True
    seed: int = 2026

    data_unit: Optional[np.ndarray] = None
    hyperplanes: Optional[np.ndarray] = None
    tables: Optional[List[Dict[int, List[int]]]] = None
    bit_weights: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "RandomHyperplaneLSH":
        """构建 LSH 索引。"""
        if data.ndim != 2:
            raise ValueError("data 必须是二维数组")

        n_points, dim = data.shape
        if n_points == 0:
            raise ValueError("data 不能为空")
        if self.num_bits <= 0 or self.num_bits > 62:
            raise ValueError("num_bits 需在 1..62 之间，便于整数哈希")

        self.data_unit = normalize_rows(data.astype(np.float64, copy=False))
        self.bit_weights = (1 << np.arange(self.num_bits, dtype=np.uint64))

        rng = np.random.default_rng(self.seed)
        self.hyperplanes = rng.standard_normal((self.num_tables, self.num_bits, dim))
        self.tables = []

        for table_id in range(self.num_tables):
            proj = matmul_no_warn(self.data_unit, self.hyperplanes[table_id].T)
            bits = (proj >= 0.0).astype(np.uint64)
            hashes = (bits * self.bit_weights).sum(axis=1, dtype=np.uint64)

            buckets: Dict[int, List[int]] = {}
            for idx, hv in enumerate(hashes):
                key = int(hv)
                if key not in buckets:
                    buckets[key] = [idx]
                else:
                    buckets[key].append(idx)
            self.tables.append(buckets)

        return self

    def _check_ready(self) -> None:
        if self.data_unit is None or self.hyperplanes is None or self.tables is None or self.bit_weights is None:
            raise RuntimeError("LSH 索引尚未构建，请先调用 fit")

    def _hash_one(self, query_unit: np.ndarray, table_id: int) -> int:
        """计算 query 在某个哈希表中的桶 ID。"""
        assert self.hyperplanes is not None and self.bit_weights is not None
        proj = self.hyperplanes[table_id] @ query_unit
        bits = (proj >= 0.0).astype(np.uint64)
        return int(np.dot(bits, self.bit_weights))

    def _collect_candidates(self, query_unit: np.ndarray) -> np.ndarray:
        """收集候选集合：主桶 + 可选的一跳 multiprobe 桶。"""
        assert self.tables is not None

        candidates: set[int] = set()
        primary_hashes: List[int] = []

        for table_id in range(self.num_tables):
            hv = self._hash_one(query_unit, table_id)
            primary_hashes.append(hv)
            bucket = self.tables[table_id].get(hv)
            if bucket is not None:
                candidates.update(bucket)

        if self.enable_multiprobe and len(candidates) < self.min_candidates:
            for table_id, hv in enumerate(primary_hashes):
                for bit in range(self.num_bits):
                    alt_hv = hv ^ (1 << bit)
                    bucket = self.tables[table_id].get(alt_hv)
                    if bucket is not None:
                        candidates.update(bucket)
                if len(candidates) >= self.min_candidates:
                    break

        if not candidates:
            return np.empty((0,), dtype=np.int64)

        return np.fromiter(candidates, dtype=np.int64)

    def query(self, query: np.ndarray) -> Tuple[int, float, int]:
        """返回 (近似最近邻索引, 余弦相似度, 候选数)。"""
        self._check_ready()
        q_unit = normalize_vector(query.astype(np.float64, copy=False))
        return self.query_unit(q_unit)

    def query_unit(self, query_unit: np.ndarray) -> Tuple[int, float, int]:
        """输入已归一化向量，返回近似最近邻。"""
        self._check_ready()
        assert self.data_unit is not None

        candidates = self._collect_candidates(query_unit)
        if candidates.size == 0:
            # 极端情况兜底：退化成全量扫描，保证可返回结果。
            candidates = np.arange(self.data_unit.shape[0], dtype=np.int64)

        sims = matmul_no_warn(self.data_unit[candidates], query_unit)
        local_best = int(np.argmax(sims))
        best_idx = int(candidates[local_best])
        best_sim = float(sims[local_best])
        return best_idx, best_sim, int(candidates.size)


def exact_nearest_cosine(data_unit: np.ndarray, query_unit: np.ndarray) -> Tuple[int, float]:
    """暴力法精确最近邻（余弦相似度最大）。"""
    sims = matmul_no_warn(data_unit, query_unit)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def main() -> None:
    rng = np.random.default_rng(2026)

    n_points = 100000
    dim = 64
    n_queries = 250

    data = rng.standard_normal((n_points, dim))
    queries = rng.standard_normal((n_queries, dim))
    queries_unit = normalize_rows(queries)

    lsh = RandomHyperplaneLSH(
        num_tables=20,
        num_bits=11,
        min_candidates=80,
        enable_multiprobe=False,
        seed=2026,
    )

    build_start = perf_counter()
    lsh.fit(data)
    build_time = perf_counter() - build_start

    assert lsh.data_unit is not None
    data_unit = lsh.data_unit

    # 基础正确性烟雾测试：查询数据集中某个已知向量，应该检索到自己（或相同向量）。
    probe_idx = 777
    got_idx, got_sim, probe_candidates = lsh.query(data[probe_idx])
    assert got_idx == probe_idx or np.isclose(got_sim, 1.0, atol=1e-12), "自查询未命中预期结果"

    exact_indices = np.empty(n_queries, dtype=np.int64)
    exact_sims = np.empty(n_queries, dtype=np.float64)

    exact_start = perf_counter()
    for i, q in enumerate(queries_unit):
        idx, sim = exact_nearest_cosine(data_unit, q)
        exact_indices[i] = idx
        exact_sims[i] = sim
    exact_time = perf_counter() - exact_start

    lsh_indices = np.empty(n_queries, dtype=np.int64)
    lsh_sims = np.empty(n_queries, dtype=np.float64)
    candidate_counts = np.empty(n_queries, dtype=np.int64)

    lsh_start = perf_counter()
    for i, q in enumerate(queries_unit):
        idx, sim, cand = lsh.query_unit(q)
        lsh_indices[i] = idx
        lsh_sims[i] = sim
        candidate_counts[i] = cand
    lsh_time = perf_counter() - lsh_start

    recall_at_1 = float(np.mean(lsh_indices == exact_indices))
    mean_sim_gap = float(np.mean(exact_sims - lsh_sims))
    speedup = exact_time / lsh_time if lsh_time > 0 else float("inf")

    assert np.all((lsh_indices >= 0) & (lsh_indices < n_points)), "LSH 返回了越界索引"

    fixed_query = np.array([0.5, -1.2, 0.3, 0.7] + [0.0] * (dim - 4), dtype=np.float64)
    fixed_query = normalize_vector(fixed_query)
    fixed_exact_idx, fixed_exact_sim = exact_nearest_cosine(data_unit, fixed_query)
    fixed_lsh_idx, fixed_lsh_sim, fixed_cands = lsh.query_unit(fixed_query)

    print("=== 最近邻搜索 - LSH(Random Hyperplane) MVP ===")
    print(f"点数量: {n_points}, 维度: {dim}, 查询数: {n_queries}")
    print(
        f"LSH 参数: tables={lsh.num_tables}, bits={lsh.num_bits}, "
        f"min_candidates={lsh.min_candidates}, multiprobe={lsh.enable_multiprobe}"
    )
    print(f"索引构建耗时: {build_time:.6f} s")
    print(f"固定探针(数据内)命中: idx={got_idx}, sim={got_sim:.6f}, candidates={probe_candidates}")
    print(
        "固定查询结果: "
        f"exact(idx={fixed_exact_idx}, sim={fixed_exact_sim:.6f}) | "
        f"lsh(idx={fixed_lsh_idx}, sim={fixed_lsh_sim:.6f}, candidates={fixed_cands})"
    )
    print(f"暴力精确查询耗时: {exact_time:.6f} s")
    print(f"LSH 近似查询耗时: {lsh_time:.6f} s")
    print(f"查询加速比(精确/LSH): {speedup:.2f}x")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"平均相似度损失(exact - lsh): {mean_sim_gap:.6f}")
    print(
        "候选规模统计: "
        f"mean={candidate_counts.mean():.2f}, median={np.median(candidate_counts):.0f}, "
        f"min={candidate_counts.min()}, max={candidate_counts.max()}"
    )


if __name__ == "__main__":
    main()
