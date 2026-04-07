"""BSP 树（2D 线段）最小可运行 MVP。

运行:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

EPS = 1e-9


@dataclass
class Segment2D:
    """二维线段。"""

    p0: np.ndarray
    p1: np.ndarray
    label: str

    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class BSPNode:
    """BSP 树节点。"""

    splitter: Segment2D
    coplanar: List[Segment2D]
    front: Optional["BSPNode"] = None
    back: Optional["BSPNode"] = None


def signed_side(splitter: Segment2D, point: np.ndarray) -> float:
    """点相对 splitter 支撑线的有向面积值。"""
    direction = splitter.p1 - splitter.p0
    rel = point - splitter.p0
    return float(direction[0] * rel[1] - direction[1] * rel[0])


def classify_segment(
    splitter: Segment2D, seg: Segment2D
) -> Tuple[str, float, float]:
    """将线段分类为 front/back/coplanar/spanning。"""
    s0 = signed_side(splitter, seg.p0)
    s1 = signed_side(splitter, seg.p1)

    if abs(s0) <= EPS and abs(s1) <= EPS:
        return "coplanar", s0, s1
    if s0 >= -EPS and s1 >= -EPS:
        return "front", s0, s1
    if s0 <= EPS and s1 <= EPS:
        return "back", s0, s1
    return "spanning", s0, s1


def split_spanning_segment(
    splitter: Segment2D, seg: Segment2D, s0: float, s1: float
) -> Tuple[Segment2D, Segment2D]:
    """将跨越 splitter 支撑线的线段切分为前后两段。"""
    # s(t)=s0+t*(s1-s0)=0 -> t=s0/(s0-s1)
    t = float(s0 / (s0 - s1))
    t = min(1.0, max(0.0, t))
    mid = seg.p0 + t * (seg.p1 - seg.p0)

    if s0 > 0.0:
        front_piece = Segment2D(seg.p0.copy(), mid.copy(), f"{seg.label}|F")
        back_piece = Segment2D(mid.copy(), seg.p1.copy(), f"{seg.label}|B")
    else:
        back_piece = Segment2D(seg.p0.copy(), mid.copy(), f"{seg.label}|B")
        front_piece = Segment2D(mid.copy(), seg.p1.copy(), f"{seg.label}|F")
    return front_piece, back_piece


def build_bsp_tree(segments: List[Segment2D], depth: int = 0) -> Optional[BSPNode]:
    """递归构建 BSP 树。"""
    if not segments:
        return None

    # 使用最长线段作为分割线，减少极端不平衡。
    splitter_idx = int(np.argmax([s.length() for s in segments]))
    splitter = segments[splitter_idx]
    rest = segments[:splitter_idx] + segments[splitter_idx + 1 :]

    coplanar: List[Segment2D] = [splitter]
    front_bucket: List[Segment2D] = []
    back_bucket: List[Segment2D] = []

    for seg in rest:
        cls, s0, s1 = classify_segment(splitter, seg)

        if cls == "coplanar":
            coplanar.append(seg)
        elif cls == "front":
            front_bucket.append(seg)
        elif cls == "back":
            back_bucket.append(seg)
        else:
            # 跨越支撑线: 做几何切分，分别放入前后子树。
            if abs(s0) <= EPS:
                if s1 > 0.0:
                    front_bucket.append(seg)
                else:
                    back_bucket.append(seg)
                continue
            if abs(s1) <= EPS:
                if s0 > 0.0:
                    front_bucket.append(seg)
                else:
                    back_bucket.append(seg)
                continue

            front_piece, back_piece = split_spanning_segment(splitter, seg, s0, s1)
            front_bucket.append(front_piece)
            back_bucket.append(back_piece)

    node = BSPNode(splitter=splitter, coplanar=coplanar)
    node.front = build_bsp_tree(front_bucket, depth + 1)
    node.back = build_bsp_tree(back_bucket, depth + 1)
    return node


def classify_point(splitter: Segment2D, p: np.ndarray) -> int:
    """点相对分割线的位置: 1(front), -1(back), 0(on line)。"""
    v = signed_side(splitter, p)
    if v > EPS:
        return 1
    if v < -EPS:
        return -1
    return 0


def painter_order(node: Optional[BSPNode], eye: np.ndarray) -> List[Segment2D]:
    """按视点返回 back-to-front 的线段绘制顺序。"""
    if node is None:
        return []

    side = classify_point(node.splitter, eye)
    if side >= 0:
        return painter_order(node.back, eye) + node.coplanar + painter_order(node.front, eye)
    return painter_order(node.front, eye) + node.coplanar + painter_order(node.back, eye)


def tree_stats(node: Optional[BSPNode]) -> Tuple[int, int, int]:
    """返回 (节点数, 最大深度, 存储线段数)。"""
    if node is None:
        return 0, 0, 0

    ln, ld, ls = tree_stats(node.front)
    rn, rd, rs = tree_stats(node.back)
    nodes = 1 + ln + rn
    depth = 1 + max(ld, rd)
    segs = len(node.coplanar) + ls + rs
    return nodes, depth, segs


def collect_labels(node: Optional[BSPNode]) -> List[str]:
    """收集树内全部线段标签。"""
    if node is None:
        return []
    return collect_labels(node.front) + [s.label for s in node.coplanar] + collect_labels(node.back)


def make_demo_segments() -> List[Segment2D]:
    """构造一组包含相交关系的线段，便于演示分割效果。"""
    raw = [
        ((-5.0, 0.0), (5.0, 0.0), "H0"),
        ((0.0, -5.0), (0.0, 5.0), "V0"),
        ((-5.0, -4.0), (5.0, 4.0), "D0"),
        ((-5.0, 4.0), (5.0, -4.0), "D1"),
        ((-4.0, -2.0), (4.0, -2.0), "H1"),
        ((-4.0, 2.0), (4.0, 2.0), "H2"),
        ((-2.0, -4.5), (-2.0, 4.5), "V1"),
        ((2.0, -4.5), (2.0, 4.5), "V2"),
    ]
    segments: List[Segment2D] = []
    for p0, p1, label in raw:
        segments.append(
            Segment2D(
                p0=np.array(p0, dtype=float),
                p1=np.array(p1, dtype=float),
                label=label,
            )
        )
    return segments


def main() -> None:
    segments = make_demo_segments()
    root = build_bsp_tree(segments)
    if root is None:
        raise RuntimeError("BSP 树构建失败")

    node_count, max_depth, stored_segments = tree_stats(root)
    labels_in_tree = collect_labels(root)

    eye_a = np.array([6.0, 6.0], dtype=float)
    eye_b = np.array([-6.0, -6.0], dtype=float)
    order_a = painter_order(root, eye_a)
    order_b = painter_order(root, eye_b)

    order_a_labels = [s.label for s in order_a]
    order_b_labels = [s.label for s in order_b]

    # 正确性与一致性检查。
    assert len(order_a_labels) == stored_segments
    assert len(order_b_labels) == stored_segments
    assert set(order_a_labels) == set(labels_in_tree)
    assert set(order_b_labels) == set(labels_in_tree)
    assert order_a_labels != order_b_labels, "不同视点应得到不同绘制顺序"

    print("=== BSP树 MVP (2D 线段) ===")
    print(f"原始线段数: {len(segments)}")
    print(f"树节点数: {node_count}, 最大深度: {max_depth}, 树中线段片段数: {stored_segments}")
    print(f"视点 A: {eye_a.tolist()}, 绘制顺序前 12 项: {order_a_labels[:12]}")
    print(f"视点 B: {eye_b.tolist()}, 绘制顺序前 12 项: {order_b_labels[:12]}")
    print("断言校验: 通过")


if __name__ == "__main__":
    main()
