"""Minimal runnable MVP for H-matrix / H^2-style matrix compression (MATH-0111)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Numerically transparent matmul helper using einsum to avoid BLAS warning noise."""
    if a.ndim != 2:
        raise ValueError(f"left operand must be 2D, got shape={a.shape}")
    if b.ndim == 1:
        return np.einsum("ij,j->i", a, b, optimize=True)
    if b.ndim == 2:
        return np.einsum("ik,kj->ij", a, b, optimize=True)
    raise ValueError(f"right operand must be 1D/2D, got shape={b.shape}")


@dataclass
class ClusterNode:
    """Binary geometric cluster tree node."""

    node_id: int
    level: int
    indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: Optional["ClusterNode"] = None
    right: Optional["ClusterNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))


@dataclass
class HBlock:
    """H-matrix block: dense near field or low-rank far field."""

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_node_id: int
    col_node_id: int
    kind: str  # "dense" | "lr"
    dense: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None

    @property
    def rank(self) -> int:
        if self.kind != "lr" or self.U is None:
            return 0
        return int(self.U.shape[1])


@dataclass
class H2LikeBlock:
    """Shared-basis block used by a simplified H^2-style representation."""

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_node_id: int
    col_node_id: int
    kind: str  # "dense" | "coupling"
    dense: Optional[np.ndarray] = None
    coupling: Optional[np.ndarray] = None


def build_kernel_matrix(points: np.ndarray, delta: float = 2e-2) -> np.ndarray:
    """Construct a dense kernel matrix K_ij = 1 / sqrt(||xi-xj||^2 + delta^2)."""
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return 1.0 / np.sqrt(dist2 + delta * delta)


def bbox_distance(a: ClusterNode, b: ClusterNode) -> float:
    """Euclidean distance between two axis-aligned bounding boxes."""
    gap = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(gap))


def admissible(a: ClusterNode, b: ClusterNode, eta: float) -> bool:
    """Standard geometric admissibility for H-matrices."""
    dist = bbox_distance(a, b)
    if dist <= 0.0:
        return False
    return max(a.diameter, b.diameter) <= eta * dist


def build_cluster_tree(
    points: np.ndarray,
    indices: np.ndarray,
    leaf_size: int,
    level: int,
    next_id: List[int],
) -> ClusterNode:
    """Build a binary geometric cluster tree by median split."""
    node_id = next_id[0]
    next_id[0] += 1

    pts = points[indices]
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    node = ClusterNode(
        node_id=node_id,
        level=level,
        indices=indices,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    if indices.size <= leaf_size:
        return node

    spreads = bbox_max - bbox_min
    split_dim = int(np.argmax(spreads))
    order = indices[np.argsort(points[indices, split_dim])]
    mid = order.size // 2
    if mid == 0 or mid == order.size:
        return node

    node.left = build_cluster_tree(points, order[:mid], leaf_size, level + 1, next_id)
    node.right = build_cluster_tree(points, order[mid:], leaf_size, level + 1, next_id)
    return node


def choose_rank_from_singular_values(s: np.ndarray, rel_tol: float, rank_max: int) -> int:
    """Select truncated SVD rank by relative Frobenius energy."""
    if s.size == 0 or s[0] <= 0.0:
        return 0
    rank_cap = min(rank_max, s.size)
    sq = s * s
    total = float(np.sum(sq))
    prefix = np.cumsum(sq)
    target_tail = (rel_tol * rel_tol) * total
    for r in range(1, rank_cap + 1):
        tail = total - float(prefix[r - 1])
        if tail <= target_tail:
            return r
    return rank_cap


class HMatrix:
    """Simple geometric H-matrix using dense/low-rank blocks."""

    def __init__(
        self,
        points: np.ndarray,
        dense_matrix: np.ndarray,
        leaf_size: int = 24,
        eta: float = 1.2,
        svd_rel_tol: float = 1e-4,
        rank_max: int = 20,
        min_block_area_for_lr: int = 256,
    ) -> None:
        self.points = points
        self.n = points.shape[0]
        self.leaf_size = leaf_size
        self.eta = eta
        self.svd_rel_tol = svd_rel_tol
        self.rank_max = rank_max
        self.min_block_area_for_lr = min_block_area_for_lr
        self.blocks: List[HBlock] = []

        root_idx = np.arange(self.n, dtype=int)
        self.root = build_cluster_tree(points, root_idx, leaf_size, level=0, next_id=[0])
        self._build_blocks(dense_matrix, self.root, self.root)

    def _append_dense(self, matrix: np.ndarray, row: ClusterNode, col: ClusterNode) -> None:
        sub = matrix[np.ix_(row.indices, col.indices)]
        self.blocks.append(
            HBlock(
                row_idx=row.indices,
                col_idx=col.indices,
                row_node_id=row.node_id,
                col_node_id=col.node_id,
                kind="dense",
                dense=sub,
            )
        )

    def _try_low_rank(self, matrix: np.ndarray, row: ClusterNode, col: ClusterNode) -> bool:
        m = row.indices.size
        n = col.indices.size
        if m * n < self.min_block_area_for_lr:
            return False
        sub = matrix[np.ix_(row.indices, col.indices)]
        u, s, vt = np.linalg.svd(sub, full_matrices=False)
        rank = choose_rank_from_singular_values(s, self.svd_rel_tol, self.rank_max)
        if rank == 0 or rank >= min(m, n):
            return False

        dense_params = m * n
        lr_params = m * rank + rank * n
        if lr_params >= dense_params:
            return False

        u_scaled = u[:, :rank] * s[:rank]
        v = vt[:rank, :]
        self.blocks.append(
            HBlock(
                row_idx=row.indices,
                col_idx=col.indices,
                row_node_id=row.node_id,
                col_node_id=col.node_id,
                kind="lr",
                U=u_scaled,
                V=v,
            )
        )
        return True

    def _build_blocks(self, matrix: np.ndarray, row: ClusterNode, col: ClusterNode) -> None:
        if admissible(row, col, self.eta):
            if self._try_low_rank(matrix, row, col):
                return

        if row.is_leaf and col.is_leaf:
            self._append_dense(matrix, row, col)
            return

        if not row.is_leaf and not col.is_leaf:
            assert row.left is not None and row.right is not None
            assert col.left is not None and col.right is not None
            self._build_blocks(matrix, row.left, col.left)
            self._build_blocks(matrix, row.left, col.right)
            self._build_blocks(matrix, row.right, col.left)
            self._build_blocks(matrix, row.right, col.right)
            return

        if not row.is_leaf:
            assert row.left is not None and row.right is not None
            self._build_blocks(matrix, row.left, col)
            self._build_blocks(matrix, row.right, col)
            return

        if not col.is_leaf:
            assert col.left is not None and col.right is not None
            self._build_blocks(matrix, row, col.left)
            self._build_blocks(matrix, row, col.right)
            return

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.n, dtype=float)
        for block in self.blocks:
            x_sub = x[block.col_idx]
            if block.kind == "dense":
                assert block.dense is not None
                y[block.row_idx] += safe_matmul(block.dense, x_sub)
            else:
                assert block.U is not None and block.V is not None
                y[block.row_idx] += safe_matmul(block.U, safe_matmul(block.V, x_sub))
        return y

    def reconstruct_dense(self) -> np.ndarray:
        out = np.zeros((self.n, self.n), dtype=float)
        for block in self.blocks:
            if block.kind == "dense":
                assert block.dense is not None
                out[np.ix_(block.row_idx, block.col_idx)] = block.dense
            else:
                assert block.U is not None and block.V is not None
                out[np.ix_(block.row_idx, block.col_idx)] = safe_matmul(block.U, block.V)
        return out

    def stats(self) -> Dict[str, float]:
        dense_blocks = [b for b in self.blocks if b.kind == "dense"]
        lr_blocks = [b for b in self.blocks if b.kind == "lr"]
        dense_params = int(sum(int(b.dense.size) for b in dense_blocks if b.dense is not None))
        lr_params = int(sum(int(b.U.size + b.V.size) for b in lr_blocks if b.U is not None and b.V is not None))
        raw_params = self.n * self.n
        ranks = [b.rank for b in lr_blocks]
        return {
            "n": float(self.n),
            "raw_params": float(raw_params),
            "compressed_params": float(dense_params + lr_params),
            "compression_ratio": float(raw_params / max(1, dense_params + lr_params)),
            "dense_block_count": float(len(dense_blocks)),
            "lr_block_count": float(len(lr_blocks)),
            "avg_lr_rank": float(np.mean(ranks) if ranks else 0.0),
            "max_lr_rank": float(max(ranks) if ranks else 0.0),
        }


class H2LikeMatrix:
    """A simplified shared-basis compression built on top of H-matrix low-rank blocks."""

    def __init__(
        self,
        n: int,
        row_bases: Dict[int, np.ndarray],
        col_bases: Dict[int, np.ndarray],
        blocks: List[H2LikeBlock],
    ) -> None:
        self.n = n
        self.row_bases = row_bases
        self.col_bases = col_bases
        self.blocks = blocks

    @classmethod
    def from_hmatrix(
        cls,
        hmat: HMatrix,
        basis_rel_tol: float = 1e-5,
        basis_rank_max: int = 24,
    ) -> "H2LikeMatrix":
        row_factors: Dict[int, List[np.ndarray]] = {}
        col_factors: Dict[int, List[np.ndarray]] = {}
        for b in hmat.blocks:
            if b.kind != "lr":
                continue
            assert b.U is not None and b.V is not None
            row_factors.setdefault(b.row_node_id, []).append(b.U)
            col_factors.setdefault(b.col_node_id, []).append(b.V.T)

        row_bases = {
            node_id: cls._compress_basis(np.hstack(factors), basis_rel_tol, basis_rank_max)
            for node_id, factors in row_factors.items()
        }
        col_bases = {
            node_id: cls._compress_basis(np.hstack(factors), basis_rel_tol, basis_rank_max)
            for node_id, factors in col_factors.items()
        }

        out_blocks: List[H2LikeBlock] = []
        for b in hmat.blocks:
            if b.kind == "dense":
                out_blocks.append(
                    H2LikeBlock(
                        row_idx=b.row_idx,
                        col_idx=b.col_idx,
                        row_node_id=b.row_node_id,
                        col_node_id=b.col_node_id,
                        kind="dense",
                        dense=b.dense,
                    )
                )
                continue

            assert b.U is not None and b.V is not None
            row_basis = row_bases[b.row_node_id]
            col_basis = col_bases[b.col_node_id]
            uv = safe_matmul(b.U, b.V)
            coupling = safe_matmul(row_basis.T, safe_matmul(uv, col_basis))
            out_blocks.append(
                H2LikeBlock(
                    row_idx=b.row_idx,
                    col_idx=b.col_idx,
                    row_node_id=b.row_node_id,
                    col_node_id=b.col_node_id,
                    kind="coupling",
                    coupling=coupling,
                )
            )

        return cls(n=hmat.n, row_bases=row_bases, col_bases=col_bases, blocks=out_blocks)

    @staticmethod
    def _compress_basis(factor_matrix: np.ndarray, rel_tol: float, rank_max: int) -> np.ndarray:
        u, s, _ = np.linalg.svd(factor_matrix, full_matrices=False)
        rank = choose_rank_from_singular_values(s, rel_tol, rank_max)
        if rank <= 0:
            rank = 1
        return u[:, :rank]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.n, dtype=float)
        for b in self.blocks:
            x_sub = x[b.col_idx]
            if b.kind == "dense":
                assert b.dense is not None
                y[b.row_idx] += safe_matmul(b.dense, x_sub)
            else:
                assert b.coupling is not None
                row_basis = self.row_bases[b.row_node_id]
                col_basis = self.col_bases[b.col_node_id]
                col_proj = safe_matmul(col_basis.T, x_sub)
                y[b.row_idx] += safe_matmul(row_basis, safe_matmul(b.coupling, col_proj))
        return y

    def stats(self) -> Dict[str, float]:
        dense_blocks = [b for b in self.blocks if b.kind == "dense"]
        coupling_blocks = [b for b in self.blocks if b.kind == "coupling"]
        dense_params = int(sum(int(b.dense.size) for b in dense_blocks if b.dense is not None))
        coupling_params = int(
            sum(int(b.coupling.size) for b in coupling_blocks if b.coupling is not None)
        )
        basis_params = int(sum(int(mat.size) for mat in self.row_bases.values()))
        basis_params += int(sum(int(mat.size) for mat in self.col_bases.values()))
        total_params = dense_params + coupling_params + basis_params
        raw_params = self.n * self.n
        return {
            "raw_params": float(raw_params),
            "compressed_params": float(total_params),
            "compression_ratio": float(raw_params / max(1, total_params)),
            "dense_block_count": float(len(dense_blocks)),
            "coupling_block_count": float(len(coupling_blocks)),
            "row_basis_count": float(len(self.row_bases)),
            "col_basis_count": float(len(self.col_bases)),
        }


def relative_error(reference: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(reference - approx) / denom)


def main() -> None:
    rng = np.random.default_rng(42)
    n = 192
    points = rng.random((n, 2))
    dense_kernel = build_kernel_matrix(points, delta=2e-2)

    hmat = HMatrix(
        points=points,
        dense_matrix=dense_kernel,
        leaf_size=16,
        eta=2.0,
        svd_rel_tol=1e-3,
        rank_max=20,
        min_block_area_for_lr=64,
    )

    h2_like = H2LikeMatrix.from_hmatrix(hmat, basis_rel_tol=5e-4, basis_rank_max=20)

    reconstructed_h = hmat.reconstruct_dense()
    rel_frob_h = relative_error(dense_kernel, reconstructed_h)

    test_vectors = rng.normal(size=(5, n))
    h_matvec_errors: List[float] = []
    h2_matvec_errors: List[float] = []
    for x in test_vectors:
        exact = safe_matmul(dense_kernel, x)
        approx_h = hmat.matvec(x)
        approx_h2 = h2_like.matvec(x)
        h_matvec_errors.append(relative_error(exact, approx_h))
        h2_matvec_errors.append(relative_error(exact, approx_h2))

    h_stats = hmat.stats()
    h2_stats = h2_like.stats()

    print("H-matrix / H^2-style MVP demo (MATH-0111)")
    print("=" * 72)
    print(f"matrix_size n: {n}")
    print(f"H blocks: dense={int(h_stats['dense_block_count'])}, low_rank={int(h_stats['lr_block_count'])}")
    print(
        "H compression ratio (raw/compressed): "
        f"{h_stats['compression_ratio']:.2f}x | avg_rank={h_stats['avg_lr_rank']:.2f}, "
        f"max_rank={int(h_stats['max_lr_rank'])}"
    )
    print(f"H reconstruction relative Frobenius error: {rel_frob_h:.3e}")
    print(
        "H matvec relative error (mean/max over 5 vectors): "
        f"{np.mean(h_matvec_errors):.3e} / {np.max(h_matvec_errors):.3e}"
    )
    print("-" * 72)
    print(
        "H^2-like compression ratio (raw/compressed): "
        f"{h2_stats['compression_ratio']:.2f}x | "
        f"bases(row/col)={int(h2_stats['row_basis_count'])}/{int(h2_stats['col_basis_count'])}"
    )
    print(
        "H^2-like matvec relative error (mean/max over 5 vectors): "
        f"{np.mean(h2_matvec_errors):.3e} / {np.max(h2_matvec_errors):.3e}"
    )
    print("=" * 72)

    # Conservative checks: enough to catch obvious regressions while staying robust.
    assert h_stats["lr_block_count"] > 0, "H-matrix failed to generate low-rank far-field blocks."
    assert rel_frob_h < 2e-2, f"H reconstruction error too large: {rel_frob_h}"
    assert np.max(h_matvec_errors) < 3e-2, f"H matvec error too large: {np.max(h_matvec_errors)}"
    assert np.max(h2_matvec_errors) < 8e-2, f"H^2-like matvec error too large: {np.max(h2_matvec_errors)}"
    print("All checks passed.")


if __name__ == "__main__":
    main()
