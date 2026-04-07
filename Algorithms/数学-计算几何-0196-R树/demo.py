"""Minimal runnable MVP for R-tree (MATH-0196).

This demo implements a compact 2D R-tree from scratch with:
- insertion
- quadratic node split
- axis-aligned range query
- k-nearest-neighbor (branch-and-bound over MBR lower bounds)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle (minimum bounding rectangle, MBR)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_point(pt: np.ndarray) -> "Rect":
        x = float(pt[0])
        y = float(pt[1])
        return Rect(x, y, x, y)

    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )

    def enlargement_needed(self, other: "Rect") -> float:
        return self.union(other).area() - self.area()

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.xmax < other.xmin
            or other.xmax < self.xmin
            or self.ymax < other.ymin
            or other.ymax < self.ymin
        )

    def contains_point(self, pt: np.ndarray) -> bool:
        x = float(pt[0])
        y = float(pt[1])
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def min_dist2_to_point(self, pt: np.ndarray) -> float:
        """Squared min distance from a point to this rectangle."""
        x = float(pt[0])
        y = float(pt[1])

        dx = 0.0
        if x < self.xmin:
            dx = self.xmin - x
        elif x > self.xmax:
            dx = x - self.xmax

        dy = 0.0
        if y < self.ymin:
            dy = self.ymin - y
        elif y > self.ymax:
            dy = y - self.ymax

        return dx * dx + dy * dy


@dataclass
class Entry:
    """Node entry: either child pointer (internal) or point payload (leaf)."""

    mbr: Rect
    child: Optional["RTreeNode"] = None
    point: Optional[np.ndarray] = None
    payload: Optional[int] = None


@dataclass
class RTreeNode:
    """R-tree node."""

    is_leaf: bool
    entries: List[Entry] = field(default_factory=list)
    parent: Optional["RTreeNode"] = None

    def mbr(self) -> Rect:
        if not self.entries:
            raise ValueError("Cannot compute MBR of an empty node.")
        out = self.entries[0].mbr
        for e in self.entries[1:]:
            out = out.union(e.mbr)
        return out


class RTree:
    """A small 2D R-tree with quadratic split (Guttman-style MVP)."""

    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None) -> None:
        if max_entries < 3:
            raise ValueError("max_entries must be >= 3.")
        if min_entries is None:
            min_entries = max_entries // 2
        if min_entries < 1 or min_entries > max_entries // 2:
            raise ValueError("min_entries must satisfy 1 <= min_entries <= max_entries//2.")

        self.max_entries = int(max_entries)
        self.min_entries = int(min_entries)
        self.root = RTreeNode(is_leaf=True)

    def insert(self, point: np.ndarray, payload: int) -> None:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,):
            raise ValueError("point must have shape (2,).")

        leaf_entry = Entry(mbr=Rect.from_point(p), point=p, payload=int(payload))
        leaf = self._choose_leaf(self.root, leaf_entry.mbr)
        leaf.entries.append(leaf_entry)

        split_node = self._split_node(leaf) if len(leaf.entries) > self.max_entries else None
        self._adjust_tree_after_insert(leaf, split_node)

    def range_query(self, query: Rect) -> List[int]:
        out: List[int] = []

        def dfs(node: RTreeNode) -> None:
            for e in node.entries:
                if not e.mbr.overlaps(query):
                    continue
                if node.is_leaf:
                    out.append(int(e.payload))
                else:
                    assert e.child is not None
                    dfs(e.child)

        if self.root.entries:
            dfs(self.root)
        return out

    def k_nearest(self, query_point: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """Branch-and-bound kNN over MBR lower bounds."""
        if k <= 0:
            raise ValueError("k must be positive.")
        q = np.asarray(query_point, dtype=float)
        if q.shape != (2,):
            raise ValueError("query_point must have shape (2,).")
        if not self.root.entries:
            return []

        # Min-heap for traversal candidates: (lower_bound_dist2, tie_id, kind, object)
        queue: List[Tuple[float, int, str, object]] = []
        counter = 0

        root_bound = self.root.mbr().min_dist2_to_point(q)
        heapq.heappush(queue, (root_bound, counter, "node", self.root))
        counter += 1

        # Max-heap for current best neighbors: (-dist2, payload)
        best: List[Tuple[float, int]] = []

        while queue:
            bound2, _, kind, obj = heapq.heappop(queue)

            # If current lower bound already worse than worst best, we can stop.
            if len(best) == k and bound2 > -best[0][0]:
                break

            if kind == "node":
                node = obj
                assert isinstance(node, RTreeNode)

                if node.is_leaf:
                    for e in node.entries:
                        assert e.point is not None and e.payload is not None
                        d2 = float(np.sum((e.point - q) ** 2))
                        if len(best) < k:
                            heapq.heappush(best, (-d2, int(e.payload)))
                        elif d2 < -best[0][0]:
                            heapq.heapreplace(best, (-d2, int(e.payload)))
                else:
                    for e in node.entries:
                        assert e.child is not None
                        child_bound2 = e.mbr.min_dist2_to_point(q)
                        if len(best) < k or child_bound2 <= -best[0][0]:
                            heapq.heappush(queue, (child_bound2, counter, "node", e.child))
                            counter += 1
            else:
                raise RuntimeError(f"Unexpected queue item kind: {kind}")

        best_sorted = sorted([(-neg_d2, payload) for neg_d2, payload in best], key=lambda t: t[0])
        return [(payload, float(np.sqrt(dist2))) for dist2, payload in best_sorted]

    def height(self) -> int:
        h = 1
        node = self.root
        while node.entries and not node.is_leaf:
            child = node.entries[0].child
            assert child is not None
            node = child
            h += 1
        return h

    def count_nodes(self) -> Tuple[int, int]:
        """Return (total_nodes, leaf_nodes)."""
        if not self.root.entries:
            return (1, 1)

        total = 0
        leaf = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            total += 1
            if n.is_leaf:
                leaf += 1
            else:
                for e in n.entries:
                    assert e.child is not None
                    stack.append(e.child)
        return total, leaf

    def _choose_leaf(self, node: RTreeNode, new_rect: Rect) -> RTreeNode:
        if node.is_leaf:
            return node

        best_entry: Optional[Entry] = None
        best_enlargement = float("inf")
        best_area = float("inf")

        for e in node.entries:
            enlargement = e.mbr.enlargement_needed(new_rect)
            area = e.mbr.area()
            if (
                enlargement < best_enlargement
                or (enlargement == best_enlargement and area < best_area)
            ):
                best_enlargement = enlargement
                best_area = area
                best_entry = e

        assert best_entry is not None and best_entry.child is not None
        return self._choose_leaf(best_entry.child, new_rect)

    def _adjust_tree_after_insert(
        self,
        node: RTreeNode,
        split_node: Optional[RTreeNode],
    ) -> None:
        """Propagate MBR changes and splits upward."""
        current = node
        new_sibling = split_node

        while True:
            if current is self.root:
                if new_sibling is not None:
                    new_root = RTreeNode(is_leaf=False)
                    current.parent = new_root
                    new_sibling.parent = new_root
                    new_root.entries = [
                        Entry(mbr=current.mbr(), child=current),
                        Entry(mbr=new_sibling.mbr(), child=new_sibling),
                    ]
                    self.root = new_root
                return

            parent = current.parent
            assert parent is not None

            # Refresh parent's MBR for current.
            for e in parent.entries:
                if e.child is current:
                    e.mbr = current.mbr()
                    break

            if new_sibling is not None:
                new_sibling.parent = parent
                parent.entries.append(Entry(mbr=new_sibling.mbr(), child=new_sibling))

            if len(parent.entries) > self.max_entries:
                current = parent
                new_sibling = self._split_node(parent)
            else:
                current = parent
                new_sibling = None

    def _split_node(self, node: RTreeNode) -> RTreeNode:
        """Quadratic split: return new sibling node, mutate original node in place."""
        entries = list(node.entries)
        if len(entries) <= self.max_entries:
            raise ValueError("Split called on non-overflow node.")

        i, j = self._pick_seeds_quadratic(entries)
        seed_a = entries[i]
        seed_b = entries[j]

        remaining = [e for idx, e in enumerate(entries) if idx not in (i, j)]

        node.entries = [seed_a]
        sibling = RTreeNode(is_leaf=node.is_leaf, parent=node.parent, entries=[seed_b])

        self._reattach_child_parent(seed_a, node)
        self._reattach_child_parent(seed_b, sibling)

        while remaining:
            # Force assignment to satisfy minimum fill.
            if len(node.entries) + len(remaining) == self.min_entries:
                for e in remaining:
                    node.entries.append(e)
                    self._reattach_child_parent(e, node)
                break
            if len(sibling.entries) + len(remaining) == self.min_entries:
                for e in remaining:
                    sibling.entries.append(e)
                    self._reattach_child_parent(e, sibling)
                break

            next_idx = self._pick_next_entry(node.entries, sibling.entries, remaining)
            e = remaining.pop(next_idx)
            self._assign_entry(node, sibling, e)

        return sibling

    def _pick_seeds_quadratic(self, entries: List[Entry]) -> Tuple[int, int]:
        max_waste = -1.0
        seed_i, seed_j = 0, 1

        for i in range(len(entries) - 1):
            for j in range(i + 1, len(entries)):
                a = entries[i].mbr
                b = entries[j].mbr
                waste = a.union(b).area() - a.area() - b.area()
                if waste > max_waste:
                    max_waste = waste
                    seed_i, seed_j = i, j

        return seed_i, seed_j

    def _pick_next_entry(
        self,
        group_a: List[Entry],
        group_b: List[Entry],
        remaining: List[Entry],
    ) -> int:
        mbr_a = self._mbr_of_entries(group_a)
        mbr_b = self._mbr_of_entries(group_b)

        best_idx = 0
        best_diff = -1.0

        for idx, e in enumerate(remaining):
            d_a = mbr_a.enlargement_needed(e.mbr)
            d_b = mbr_b.enlargement_needed(e.mbr)
            diff = abs(d_a - d_b)
            if diff > best_diff:
                best_diff = diff
                best_idx = idx

        return best_idx

    def _assign_entry(self, node: RTreeNode, sibling: RTreeNode, e: Entry) -> None:
        mbr_node = self._mbr_of_entries(node.entries)
        mbr_sib = self._mbr_of_entries(sibling.entries)

        enlarge_node = mbr_node.enlargement_needed(e.mbr)
        enlarge_sib = mbr_sib.enlargement_needed(e.mbr)

        if enlarge_node < enlarge_sib:
            target = node
        elif enlarge_sib < enlarge_node:
            target = sibling
        else:
            area_node = mbr_node.area()
            area_sib = mbr_sib.area()
            if area_node < area_sib:
                target = node
            elif area_sib < area_node:
                target = sibling
            else:
                target = node if len(node.entries) <= len(sibling.entries) else sibling

        target.entries.append(e)
        self._reattach_child_parent(e, target)

    @staticmethod
    def _mbr_of_entries(entries: Iterable[Entry]) -> Rect:
        seq = list(entries)
        if not seq:
            raise ValueError("Cannot compute MBR of empty entry group.")
        out = seq[0].mbr
        for e in seq[1:]:
            out = out.union(e.mbr)
        return out

    @staticmethod
    def _reattach_child_parent(entry: Entry, parent_node: RTreeNode) -> None:
        if entry.child is not None:
            entry.child.parent = parent_node


def brute_force_range(points: np.ndarray, query: Rect) -> List[int]:
    mask = (
        (points[:, 0] >= query.xmin)
        & (points[:, 0] <= query.xmax)
        & (points[:, 1] >= query.ymin)
        & (points[:, 1] <= query.ymax)
    )
    return np.where(mask)[0].tolist()


def brute_force_knn(points: np.ndarray, query: np.ndarray, k: int) -> List[int]:
    d2 = np.sum((points - query[None, :]) ** 2, axis=1)
    return np.argsort(d2)[:k].tolist()


def main() -> None:
    rng = np.random.default_rng(196)
    points = rng.uniform(0.0, 100.0, size=(500, 2))

    tree = RTree(max_entries=8, min_entries=4)
    for idx, p in enumerate(points):
        tree.insert(p, idx)

    queries = [
        Rect(10.0, 10.0, 40.0, 40.0),
        Rect(35.0, 20.0, 80.0, 70.0),
        Rect(0.0, 0.0, 100.0, 100.0),
    ]

    print("R-tree MVP report (MATH-0196)")
    print("=" * 64)
    print(f"points_indexed                 : {points.shape[0]}")
    print(f"max_entries / min_entries      : {tree.max_entries} / {tree.min_entries}")

    total_nodes, leaf_nodes = tree.count_nodes()
    print(f"tree_height                    : {tree.height()}")
    print(f"total_nodes                    : {total_nodes}")
    print(f"leaf_nodes                     : {leaf_nodes}")

    for i, q in enumerate(queries, start=1):
        tree_ids = sorted(tree.range_query(q))
        brute_ids = sorted(brute_force_range(points, q))
        if tree_ids != brute_ids:
            raise AssertionError(f"Range query mismatch on query #{i}.")
        print(f"range_query_{i}_hit_count       : {len(tree_ids)}")

    q_knn = np.array([52.0, 48.0], dtype=float)
    k = 5
    knn_tree = tree.k_nearest(q_knn, k=k)
    knn_tree_ids = [pid for pid, _ in knn_tree]
    knn_brut_ids = brute_force_knn(points, q_knn, k=k)

    if knn_tree_ids != knn_brut_ids:
        raise AssertionError("kNN mismatch between R-tree and brute-force result.")

    print(f"knn_query_point                : {q_knn.tolist()}")
    print(f"knn_topk_ids                   : {knn_tree_ids}")
    print(
        "knn_topk_distances             : "
        f"{[round(d, 6) for _, d in knn_tree]}"
    )

    print("=" * 64)
    print("All checks passed.")


if __name__ == "__main__":
    main()
