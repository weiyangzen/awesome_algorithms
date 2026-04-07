"""Minimal runnable MVP for nearest neighbor search with a Random Projection Tree."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class RPTreeNode:
    """A node in the random projection tree."""

    indices: IntArray
    normal: FloatArray | None = None
    threshold: float | None = None
    left: "RPTreeNode | None" = None
    right: "RPTreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class RandomProjectionTree:
    """A simple random projection tree for approximate 1-NN search."""

    def __init__(
        self,
        leaf_size: int = 32,
        max_depth: int = 24,
        random_state: int = 0,
    ) -> None:
        if leaf_size <= 0:
            raise ValueError("leaf_size must be > 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._x: FloatArray | None = None
        self._root: RPTreeNode | None = None

    def fit(self, x: FloatArray) -> "RandomProjectionTree":
        """Build the tree from data matrix x with shape (n_samples, n_features)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D matrix")
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sample")

        self._x = x
        all_indices = np.arange(x.shape[0], dtype=np.int64)
        self._root = self._build(all_indices, depth=0)
        return self

    def _build(self, indices: IntArray, depth: int) -> RPTreeNode:
        """Recursively build tree nodes using random projection splits."""
        if indices.size <= self.leaf_size or depth >= self.max_depth:
            return RPTreeNode(indices=indices)

        assert self._x is not None
        n_features = self._x.shape[1]

        normal = self._rng.standard_normal(n_features)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            return RPTreeNode(indices=indices)
        normal = normal / norm

        # Some BLAS builds may emit spurious FP warnings for safe dot products.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            proj = self._x[indices] @ normal
        if not np.all(np.isfinite(proj)):
            proj = np.nan_to_num(proj, nan=0.0, posinf=1e308, neginf=-1e308)
        threshold = float(np.median(proj))

        left_mask = proj <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if left_indices.size == 0 or right_indices.size == 0:
            order = np.argsort(proj)
            sorted_indices = indices[order]
            mid = sorted_indices.size // 2
            left_indices = sorted_indices[:mid]
            right_indices = sorted_indices[mid:]
            if left_indices.size == 0 or right_indices.size == 0:
                return RPTreeNode(indices=indices)
            sorted_proj = proj[order]
            threshold = float((sorted_proj[mid - 1] + sorted_proj[mid]) / 2.0)

        left_node = self._build(left_indices, depth + 1)
        right_node = self._build(right_indices, depth + 1)
        return RPTreeNode(
            indices=np.empty(0, dtype=np.int64),
            normal=normal,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def query(self, q: FloatArray, max_leaves: int = 16) -> tuple[int, float, int]:
        """Approximate 1-NN query.

        Returns
        -------
        (best_index, best_distance, visited_leaf_count)
        """
        if max_leaves <= 0:
            raise ValueError("max_leaves must be > 0")
        if self._root is None or self._x is None:
            raise RuntimeError("call fit before query")

        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] != self._x.shape[1]:
            raise ValueError("q must be a 1D vector with matching feature dimension")

        best_idx = -1
        best_dist2 = float("inf")
        visited_leaves = 0

        counter = itertools.count()
        heap: list[tuple[float, int, RPTreeNode]] = [(0.0, next(counter), self._root)]

        while heap and visited_leaves < max_leaves:
            lb, _, node = heapq.heappop(heap)
            if lb > best_dist2:
                break

            if node.is_leaf:
                if node.indices.size == 0:
                    continue
                visited_leaves += 1
                points = self._x[node.indices]
                diff = points - q
                dist2 = np.einsum("ij,ij->i", diff, diff)
                local = int(np.argmin(dist2))
                local_dist2 = float(dist2[local])
                if local_dist2 < best_dist2:
                    best_dist2 = local_dist2
                    best_idx = int(node.indices[local])
                continue

            assert node.normal is not None and node.threshold is not None
            assert node.left is not None and node.right is not None

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                margin = float(q @ node.normal - node.threshold)
            if not np.isfinite(margin):
                margin = 0.0
            if margin <= 0.0:
                near_child, far_child = node.left, node.right
            else:
                near_child, far_child = node.right, node.left

            heapq.heappush(heap, (lb, next(counter), near_child))

            far_lb = max(lb, margin * margin)
            if far_lb <= best_dist2:
                heapq.heappush(heap, (far_lb, next(counter), far_child))

        return best_idx, float(np.sqrt(best_dist2)), visited_leaves


def brute_force_nn(x: FloatArray, q: FloatArray) -> tuple[int, float]:
    """Exact 1-NN by exhaustive scan."""
    diff = x - q
    dist2 = np.einsum("ij,ij->i", diff, diff)
    idx = int(np.argmin(dist2))
    return idx, float(np.sqrt(dist2[idx]))


def run_demo() -> None:
    rng = np.random.default_rng(2026)
    n_samples = 12_000
    n_features = 24
    n_queries = 300

    x = rng.normal(size=(n_samples, n_features))
    # Add weak cluster structure so nearest neighbors are non-trivial yet meaningful.
    x[:, :4] += 0.8 * rng.normal(size=(n_samples, 4))
    queries = rng.normal(size=(n_queries, n_features))

    tree = RandomProjectionTree(leaf_size=40, max_depth=26, random_state=7)

    t0 = time.perf_counter()
    tree.fit(x)
    build_time = time.perf_counter() - t0

    approx_indices: list[int] = []
    approx_dists: list[float] = []
    leaf_visits: list[int] = []

    t1 = time.perf_counter()
    for q in queries:
        idx, dist, visited = tree.query(q, max_leaves=24)
        approx_indices.append(idx)
        approx_dists.append(dist)
        leaf_visits.append(visited)
    rp_query_time = time.perf_counter() - t1

    exact_indices: list[int] = []
    exact_dists: list[float] = []
    t2 = time.perf_counter()
    for q in queries:
        idx, dist = brute_force_nn(x, q)
        exact_indices.append(idx)
        exact_dists.append(dist)
    brute_query_time = time.perf_counter() - t2

    approx_arr = np.asarray(approx_indices, dtype=np.int64)
    exact_arr = np.asarray(exact_indices, dtype=np.int64)
    recall_at_1 = float(np.mean(approx_arr == exact_arr))

    dist_ratio = np.asarray(approx_dists) / np.maximum(np.asarray(exact_dists), 1e-12)
    avg_leaf_visits = float(np.mean(np.asarray(leaf_visits)))
    speedup = brute_query_time / max(rp_query_time, 1e-12)

    print("=== Random Projection Tree MVP Demo ===")
    print(f"dataset: n_samples={n_samples}, n_features={n_features}, n_queries={n_queries}")
    print(f"build_time: {build_time:.4f} s")
    print(f"rp_query_time: {rp_query_time:.4f} s")
    print(f"brute_query_time: {brute_query_time:.4f} s")
    print(f"speedup (brute / rp): {speedup:.2f}x")
    print(f"recall@1: {recall_at_1:.4f}")
    print(f"mean(approx_dist / exact_dist): {float(np.mean(dist_ratio)):.4f}")
    print(f"avg_visited_leaves: {avg_leaf_visits:.2f}")

    print("\nSample query results (first 5):")
    for i in range(5):
        print(
            f"  q{i}: approx_idx={approx_indices[i]}, exact_idx={exact_indices[i]}, "
            f"approx_dist={approx_dists[i]:.4f}, exact_dist={exact_dists[i]:.4f}, "
            f"leaf_visits={leaf_visits[i]}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
