"""A minimal but honest 1D Fast Multipole Method (FMM) MVP.

Kernel:
    K(x, y) = 1 / (x - y)

The script builds a uniform binary tree, runs classic FMM passes
(P2M, M2M, M2L, L2L, L2P) and compares against direct summation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cell:
    """Single cell in a uniform 1D binary tree."""

    level: int
    index: int
    center: float
    half_width: float
    parent: tuple[int, int] | None
    children: list[tuple[int, int]] = field(default_factory=list)
    source_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    target_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    multipole: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    local: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def translate_m2m(child_m: np.ndarray, shift: float, p: int) -> np.ndarray:
    """Translate child multipole moments to parent center."""
    out = np.zeros(p + 1, dtype=np.float64)
    for n in range(p + 1):
        acc = 0.0
        for k in range(n + 1):
            acc += math.comb(n, k) * (shift ** (n - k)) * child_m[k]
        out[n] = acc
    return out


def translate_m2l(source_m: np.ndarray, d: float, p: int) -> np.ndarray:
    """Convert source multipole at distance d into target local coefficients."""
    out = np.zeros(p + 1, dtype=np.float64)
    for n in range(p + 1):
        sign = -1.0 if (n % 2 == 1) else 1.0
        acc = 0.0
        for k in range(p + 1):
            coeff = math.comb(n + k, k)
            acc += source_m[k] * sign * coeff / (d ** (n + k + 1))
        out[n] = acc
    return out


def translate_l2l(parent_l: np.ndarray, shift: float, p: int) -> np.ndarray:
    """Translate local coefficients from parent center to child center."""
    out = np.zeros(p + 1, dtype=np.float64)
    for k in range(p + 1):
        acc = 0.0
        for n in range(k, p + 1):
            acc += math.comb(n, k) * (shift ** (n - k)) * parent_l[n]
        out[k] = acc
    return out


def eval_local(local: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Evaluate local polynomial at offsets dx."""
    result = np.zeros_like(dx)
    power = np.ones_like(dx)
    for coeff in local:
        result += coeff * power
        power *= dx
    return result


def build_uniform_tree(xmin: float, xmax: float, max_level: int, p: int) -> dict[tuple[int, int], Cell]:
    """Create complete binary tree up to max_level."""
    cells: dict[tuple[int, int], Cell] = {}
    width = xmax - xmin

    for level in range(max_level + 1):
        count = 1 << level
        cell_width = width / count
        half = 0.5 * cell_width
        for idx in range(count):
            center = xmin + (idx + 0.5) * cell_width
            parent = None if level == 0 else (level - 1, idx // 2)
            cells[(level, idx)] = Cell(
                level=level,
                index=idx,
                center=center,
                half_width=half,
                parent=parent,
                multipole=np.zeros(p + 1, dtype=np.float64),
                local=np.zeros(p + 1, dtype=np.float64),
            )

    for level in range(max_level):
        for idx in range(1 << level):
            cells[(level, idx)].children = [(level + 1, 2 * idx), (level + 1, 2 * idx + 1)]

    return cells


def bucket_points(x: np.ndarray, xmin: float, xmax: float, max_level: int) -> np.ndarray:
    """Map points to leaf indices."""
    leaf_count = 1 << max_level
    t = (x - xmin) / (xmax - xmin)
    idx = np.floor(t * leaf_count).astype(np.int64)
    return np.clip(idx, 0, leaf_count - 1)


def assign_points_to_leaves(
    cells: dict[tuple[int, int], Cell],
    xs: np.ndarray,
    xt: np.ndarray,
    xmin: float,
    xmax: float,
    max_level: int,
) -> None:
    """Attach source and target point indices to leaf cells."""
    leaf_count = 1 << max_level
    source_leaf_idx = bucket_points(xs, xmin, xmax, max_level)
    target_leaf_idx = bucket_points(xt, xmin, xmax, max_level)

    src_buckets: list[list[int]] = [[] for _ in range(leaf_count)]
    tgt_buckets: list[list[int]] = [[] for _ in range(leaf_count)]

    for i, li in enumerate(source_leaf_idx):
        src_buckets[li].append(i)
    for i, li in enumerate(target_leaf_idx):
        tgt_buckets[li].append(i)

    for leaf in range(leaf_count):
        cell = cells[(max_level, leaf)]
        cell.source_ids = np.asarray(src_buckets[leaf], dtype=np.int64)
        cell.target_ids = np.asarray(tgt_buckets[leaf], dtype=np.int64)


def upward_pass(cells: dict[tuple[int, int], Cell], xs: np.ndarray, q: np.ndarray, max_level: int, p: int) -> None:
    """Compute multipole coefficients bottom-up (P2M + M2M)."""
    for leaf in range(1 << max_level):
        cell = cells[(max_level, leaf)]
        if cell.source_ids.size == 0:
            continue
        dx = xs[cell.source_ids] - cell.center
        w = q[cell.source_ids]
        for k in range(p + 1):
            cell.multipole[k] = np.dot(w, dx ** k)

    for level in range(max_level - 1, -1, -1):
        for idx in range(1 << level):
            parent = cells[(level, idx)]
            parent.multipole.fill(0.0)
            for child_key in parent.children:
                child = cells[child_key]
                shift = child.center - parent.center
                parent.multipole += translate_m2m(child.multipole, shift, p)


def interaction_list_1d(level: int, idx: int) -> list[int]:
    """Return same-level interaction list in 1D FMM style.

    Build from the parent's neighbor cells' children, excluding adjacent cells.
    """
    if level == 0:
        return []

    parent_idx = idx // 2
    parent_count = 1 << (level - 1)
    level_count = 1 << level
    result: set[int] = set()

    for pn in range(max(0, parent_idx - 1), min(parent_count - 1, parent_idx + 1) + 1):
        for child in (2 * pn, 2 * pn + 1):
            if 0 <= child < level_count and abs(child - idx) > 1:
                result.add(child)

    return sorted(result)


def downward_pass(cells: dict[tuple[int, int], Cell], max_level: int, p: int) -> None:
    """Compute local coefficients top-down (M2L + L2L)."""
    cells[(0, 0)].local.fill(0.0)

    for level in range(1, max_level + 1):
        for idx in range(1 << level):
            cell = cells[(level, idx)]
            parent = cells[cell.parent]  # type: ignore[index]
            shift = cell.center - parent.center
            cell.local = translate_l2l(parent.local, shift, p)

        for idx in range(1 << level):
            target = cells[(level, idx)]
            for src_idx in interaction_list_1d(level, idx):
                source = cells[(level, src_idx)]
                d = target.center - source.center
                if abs(d) < 1e-15:
                    continue
                target.local += translate_m2l(source.multipole, d, p)


def evaluate_potential(
    cells: dict[tuple[int, int], Cell],
    xs: np.ndarray,
    q: np.ndarray,
    xt: np.ndarray,
    max_level: int,
) -> np.ndarray:
    """Evaluate potential at targets (L2P + near-field direct sum)."""
    out = np.zeros_like(xt)
    leaf_count = 1 << max_level

    for leaf in range(leaf_count):
        cell = cells[(max_level, leaf)]
        if cell.target_ids.size == 0:
            continue

        tx = xt[cell.target_ids]
        far = eval_local(cell.local, tx - cell.center)

        near = np.zeros_like(tx)
        for neigh in range(max(0, leaf - 1), min(leaf_count - 1, leaf + 1) + 1):
            src_cell = cells[(max_level, neigh)]
            if src_cell.source_ids.size == 0:
                continue
            sx = xs[src_cell.source_ids]
            sq = q[src_cell.source_ids]
            diff = tx[:, None] - sx[None, :]
            near += np.sum(sq[None, :] / diff, axis=1)

        out[cell.target_ids] = far + near

    return out


def fmm_1d(
    xs: np.ndarray,
    q: np.ndarray,
    xt: np.ndarray,
    p: int = 6,
    max_level: int = 6,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> np.ndarray:
    """Compute potentials using a small 1D FMM implementation."""
    cells = build_uniform_tree(xmin=xmin, xmax=xmax, max_level=max_level, p=p)
    assign_points_to_leaves(cells, xs, xt, xmin=xmin, xmax=xmax, max_level=max_level)
    upward_pass(cells, xs=xs, q=q, max_level=max_level, p=p)
    downward_pass(cells, max_level=max_level, p=p)
    return evaluate_potential(cells, xs=xs, q=q, xt=xt, max_level=max_level)


def direct_sum(xs: np.ndarray, q: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """Reference O(NM) direct summation."""
    diff = xt[:, None] - xs[None, :]
    return np.sum(q[None, :] / diff, axis=1)


def make_dataset(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic non-interactive test data."""
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    xs = np.clip(grid + 0.12 / n * np.sin(6.0 * math.pi * grid), 1e-12, 1.0 - 1e-12)
    xt = (grid + 0.37 / n + 0.09 / n * np.cos(4.0 * math.pi * grid)) % 1.0
    xt = np.clip(xt, 1e-12, 1.0 - 1e-12)
    q = np.sin(5.0 * math.pi * grid) + 0.5 * np.cos(11.0 * math.pi * grid)
    return xs, q, xt


def main() -> None:
    n = 2000
    p = 6
    max_level = 6

    xs, q, xt = make_dataset(n)

    t0 = time.perf_counter()
    phi_direct = direct_sum(xs, q, xt)
    t1 = time.perf_counter()

    phi_fmm = fmm_1d(xs, q, xt, p=p, max_level=max_level)
    t2 = time.perf_counter()

    abs_err = np.abs(phi_fmm - phi_direct)
    rel_l2 = np.linalg.norm(phi_fmm - phi_direct) / np.linalg.norm(phi_direct)

    print("FMM MVP for kernel K(x,y)=1/(x-y)")
    print(f"N(source)=N(target)={n}, p={p}, max_level={max_level}")
    print(f"Direct time : {t1 - t0:.4f} s")
    print(f"FMM time    : {t2 - t1:.4f} s")
    print(f"Speedup     : {(t1 - t0) / max(t2 - t1, 1e-12):.2f}x")
    print(f"Relative L2 error : {rel_l2:.3e}")
    print(f"Max abs error     : {np.max(abs_err):.3e}")


if __name__ == "__main__":
    main()
