"""Black-box Fast Multipole Method (2D inverse-distance kernel) MVP.

This demo implements a kernel-independent/treecode-style FMM approximation:
- Quadtree partitioning of source points
- Chebyshev interpolation nodes inside each box
- Upward pass that builds equivalent source coefficients on interpolation nodes
- Adaptive target evaluation: far boxes use interpolation proxy evaluation,
  near boxes are computed directly

The implementation is intentionally compact and explicit for study purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Tuple

import numpy as np


@dataclass
class QuadNode:
    center: np.ndarray
    half_size: float
    depth: int
    indices: np.ndarray
    children: list["QuadNode"] = field(default_factory=list)
    alpha: np.ndarray | None = None
    proxy_points: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class BlackBoxFMM2D:
    """Kernel-independent FMM-style solver for K(x,y)=1/||x-y|| in 2D coordinates."""

    def __init__(
        self,
        interp_order: int = 4,
        leaf_size: int = 32,
        max_depth: int = 8,
        theta: float = 2.0,
        eps: float = 1e-12,
    ) -> None:
        if interp_order < 2:
            raise ValueError("interp_order must be >= 2")
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if theta <= 1.0:
            raise ValueError("theta should be > 1.0 for meaningful separation")

        self.p = interp_order
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.theta = theta
        self.eps = eps

        self.nodes_1d = self._chebyshev_lobatto_nodes(self.p)
        self.weights_1d = self._barycentric_weights_lobatto(self.p)
        gx, gy = np.meshgrid(self.nodes_1d, self.nodes_1d, indexing="ij")
        self.ref_grid = np.column_stack([gx.ravel(), gy.ravel()])

        # Cache transfer matrices for the 4 child relative positions.
        self._transfer_cache: Dict[Tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _chebyshev_lobatto_nodes(p: int) -> np.ndarray:
        k = np.arange(p)
        return np.cos(np.pi * k / (p - 1))

    @staticmethod
    def _barycentric_weights_lobatto(p: int) -> np.ndarray:
        w = np.ones(p, dtype=float)
        w[0] = 0.5
        w[-1] = 0.5
        w *= (-1.0) ** np.arange(p)
        return w

    def _lagrange_basis_1d(self, x: float) -> np.ndarray:
        diff = x - self.nodes_1d
        hit = np.where(np.abs(diff) <= 1e-14)[0]
        if hit.size:
            basis = np.zeros(self.p, dtype=float)
            basis[hit[0]] = 1.0
            return basis
        tmp = self.weights_1d / diff
        return tmp / np.sum(tmp)

    def _tensor_basis(self, x_local: float, y_local: float) -> np.ndarray:
        lx = self._lagrange_basis_1d(x_local)
        ly = self._lagrange_basis_1d(y_local)
        return np.outer(lx, ly).ravel()

    def _transfer_matrix(self, sign_x: int, sign_y: int) -> np.ndarray:
        key = (sign_x, sign_y)
        cached = self._transfer_cache.get(key)
        if cached is not None:
            return cached

        # Child center in parent local coordinates is (+/-0.5, +/-0.5).
        tx_nodes = 0.5 * float(sign_x) + 0.5 * self.nodes_1d
        ty_nodes = 0.5 * float(sign_y) + 0.5 * self.nodes_1d

        mx = np.column_stack([self._lagrange_basis_1d(v) for v in tx_nodes])
        my = np.column_stack([self._lagrange_basis_1d(v) for v in ty_nodes])

        # Flatten order is (ix, iy) -> ix * p + iy.
        mat = np.kron(mx, my)
        self._transfer_cache[key] = mat
        return mat

    def _build_tree(
        self,
        points: np.ndarray,
        indices: np.ndarray,
        center: np.ndarray,
        half_size: float,
        depth: int,
    ) -> QuadNode:
        node = QuadNode(
            center=center,
            half_size=half_size,
            depth=depth,
            indices=indices,
            children=[],
        )

        if depth >= self.max_depth or indices.size <= self.leaf_size:
            return node

        px = points[indices, 0]
        py = points[indices, 1]
        east = px >= center[0]
        north = py >= center[1]

        child_masks = [
            (~east) & (~north),  # SW
            east & (~north),  # SE
            (~east) & north,  # NW
            east & north,  # NE
        ]
        child_offsets = [
            np.array([-0.5, -0.5]),
            np.array([0.5, -0.5]),
            np.array([-0.5, 0.5]),
            np.array([0.5, 0.5]),
        ]

        child_half = half_size * 0.5
        for mask, offset in zip(child_masks, child_offsets):
            child_idx = indices[mask]
            if child_idx.size == 0:
                continue
            child_center = center + half_size * offset
            child = self._build_tree(
                points=points,
                indices=child_idx,
                center=child_center,
                half_size=child_half,
                depth=depth + 1,
            )
            node.children.append(child)

        return node

    def _upward_pass(self, node: QuadNode, points: np.ndarray, charges: np.ndarray) -> None:
        node.proxy_points = node.center[None, :] + node.half_size * self.ref_grid
        p2 = self.p * self.p

        if node.is_leaf:
            alpha = np.zeros(p2, dtype=float)
            if node.indices.size:
                for idx in node.indices:
                    local = (points[idx] - node.center) / node.half_size
                    basis = self._tensor_basis(float(local[0]), float(local[1]))
                    alpha += charges[idx] * basis
            node.alpha = alpha
            return

        for child in node.children:
            self._upward_pass(child, points, charges)

        alpha = np.zeros(p2, dtype=float)
        for child in node.children:
            sign_x = 1 if child.center[0] >= node.center[0] else -1
            sign_y = 1 if child.center[1] >= node.center[1] else -1
            tmat = self._transfer_matrix(sign_x, sign_y)
            alpha += tmat @ child.alpha

        node.alpha = alpha

    def _well_separated(self, target: np.ndarray, node: QuadNode) -> bool:
        dist_inf = np.max(np.abs(target - node.center))
        return bool(dist_inf >= self.theta * node.half_size)

    def _direct_leaf_eval(
        self,
        target: np.ndarray,
        target_idx: int,
        node: QuadNode,
        points: np.ndarray,
        charges: np.ndarray,
    ) -> float:
        idx = node.indices
        if idx.size == 0:
            return 0.0

        vec = points[idx] - target[None, :]
        r = np.linalg.norm(vec, axis=1)

        mask = idx != target_idx
        if not np.any(mask):
            return 0.0

        rr = np.maximum(r[mask], self.eps)
        qq = charges[idx[mask]]
        return float(np.dot(qq, 1.0 / rr))

    def _eval_point_recursive(
        self,
        node: QuadNode,
        target: np.ndarray,
        target_idx: int,
        points: np.ndarray,
        charges: np.ndarray,
    ) -> float:
        if node.is_leaf:
            return self._direct_leaf_eval(target, target_idx, node, points, charges)

        if self._well_separated(target, node):
            r = np.linalg.norm(node.proxy_points - target[None, :], axis=1)
            rr = np.maximum(r, self.eps)
            return float(np.dot(node.alpha, 1.0 / rr))

        val = 0.0
        for child in node.children:
            val += self._eval_point_recursive(child, target, target_idx, points, charges)
        return val

    def build(self, points: np.ndarray, charges: np.ndarray) -> QuadNode:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if charges.ndim != 1 or charges.shape[0] != points.shape[0]:
            raise ValueError("charges must have shape (N,)")

        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        center = 0.5 * (mins + maxs)
        half_size = 0.5 * float(np.max(maxs - mins))
        half_size = max(half_size * 1.000001, 1e-6)

        root = self._build_tree(
            points=points,
            indices=np.arange(points.shape[0], dtype=int),
            center=center,
            half_size=half_size,
            depth=0,
        )
        self._upward_pass(root, points, charges)
        return root

    def evaluate(self, root: QuadNode, points: np.ndarray, charges: np.ndarray) -> np.ndarray:
        out = np.empty(points.shape[0], dtype=float)
        for i in range(points.shape[0]):
            out[i] = self._eval_point_recursive(root, points[i], i, points, charges)
        return out


def direct_potential(points: np.ndarray, charges: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    inv_r = 1.0 / np.maximum(r, eps)
    return np.sum(inv_r * charges[None, :], axis=1)


def count_nodes(node: QuadNode) -> int:
    total = 1
    for child in node.children:
        total += count_nodes(child)
    return total


def max_depth(node: QuadNode) -> int:
    if node.is_leaf:
        return node.depth
    return max(max_depth(c) for c in node.children)


def main() -> None:
    rng = np.random.default_rng(42)

    n = 1000
    points = rng.uniform(-1.0, 1.0, size=(n, 2))
    charges = rng.normal(0.0, 1.0, size=n)

    solver = BlackBoxFMM2D(
        interp_order=4,
        leaf_size=32,
        max_depth=8,
        theta=2.0,
        eps=1e-12,
    )

    t0 = perf_counter()
    root = solver.build(points, charges)
    t1 = perf_counter()
    phi_bb = solver.evaluate(root, points, charges)
    t2 = perf_counter()

    phi_direct = direct_potential(points, charges)
    t3 = perf_counter()

    rel_l2 = np.linalg.norm(phi_bb - phi_direct) / max(np.linalg.norm(phi_direct), 1e-12)
    max_abs = float(np.max(np.abs(phi_bb - phi_direct)))

    print("=== Black-box FMM MVP (2D, K=1/r) ===")
    print(f"N points                : {n}")
    print(f"Interpolation order     : {solver.p} (per axis)")
    print(f"Leaf size               : {solver.leaf_size}")
    print(f"Separation theta        : {solver.theta}")
    print(f"Tree nodes              : {count_nodes(root)}")
    print(f"Tree max depth          : {max_depth(root)}")
    print(f"Build time (s)          : {t1 - t0:.6f}")
    print(f"BBFMM eval time (s)     : {t2 - t1:.6f}")
    print(f"Direct eval time (s)    : {t3 - t2:.6f}")
    print(f"Relative L2 error       : {rel_l2:.6e}")
    print(f"Max abs error           : {max_abs:.6e}")


if __name__ == "__main__":
    main()
