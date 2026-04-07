"""Multipoint polynomial evaluation MVP.

Algorithm:
1) Build a product tree over factors (x - x_i).
2) Push polynomial remainders top-down to leaves.
3) Leaf remainders are the evaluation values.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Sequence

import numpy as np

EPS = 1e-12


def poly_trim(poly: List[float]) -> List[float]:
    """Remove near-zero high-order coefficients (ascending order)."""
    p = poly[:]
    while len(p) > 1 and abs(p[-1]) < EPS:
        p.pop()
    return p


def poly_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Naive O(len(a)*len(b)) multiplication in ascending-order coefficients."""
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return poly_trim(out)


def poly_mod(dividend: Sequence[float], divisor: Sequence[float]) -> List[float]:
    """Return dividend mod divisor (ascending-order coefficients)."""
    d = poly_trim(list(dividend))
    v = poly_trim(list(divisor))
    if len(v) == 1 and abs(v[0]) < EPS:
        raise ZeroDivisionError("zero polynomial divisor")
    if len(d) < len(v):
        return d

    rem = d[:]
    q_deg = len(rem) - len(v)
    lead = v[-1]
    for k in range(q_deg, -1, -1):
        coeff = rem[len(v) - 1 + k] / lead
        if abs(coeff) < EPS:
            continue
        for j in range(len(v)):
            rem[j + k] -= coeff * v[j]
    return poly_trim(rem[: len(v) - 1] if len(v) > 1 else [0.0])


def horner_eval(poly: Sequence[float], x: float) -> float:
    """Evaluate polynomial at x with Horner method (ascending coefficients)."""
    acc = 0.0
    for c in reversed(poly):
        acc = acc * x + c
    return acc


@dataclass
class ProductNode:
    poly: List[float]
    left: "ProductNode | None" = None
    right: "ProductNode | None" = None
    leaf_index: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf_index is not None


def build_product_tree(points: Sequence[float], indices: Sequence[int]) -> ProductNode:
    if len(indices) == 1:
        i = indices[0]
        return ProductNode(poly=[-points[i], 1.0], leaf_index=i)

    mid = len(indices) // 2
    left = build_product_tree(points, indices[:mid])
    right = build_product_tree(points, indices[mid:])
    return ProductNode(poly=poly_mul(left.poly, right.poly), left=left, right=right)


def multipoint_eval(poly: Sequence[float], points: Sequence[float]) -> List[float]:
    """Evaluate poly at many points by product-tree / remainder-tree method."""
    if not points:
        return []

    idx = list(range(len(points)))
    root = build_product_tree(points, idx)
    results = [0.0] * len(points)

    def descend(node: ProductNode, rem: List[float]) -> None:
        if node.is_leaf:
            results[node.leaf_index] = rem[0] if rem else 0.0
            return

        assert node.left is not None and node.right is not None
        left_rem = poly_mod(rem, node.left.poly)
        right_rem = poly_mod(rem, node.right.poly)
        descend(node.left, left_rem)
        descend(node.right, right_rem)

    descend(root, poly_trim(list(poly)))
    return results


def naive_eval(poly: Sequence[float], points: Sequence[float]) -> List[float]:
    return [horner_eval(poly, x) for x in points]


def main() -> None:
    # Deterministic example polynomial: 2 - x + 3x^2 + 0x^3 + 5x^4 - 2x^5 + x^6
    poly = [2.0, -1.0, 3.0, 0.0, 5.0, -2.0, 1.0]
    points = np.linspace(-3.0, 3.0, 25).tolist()

    t0 = perf_counter()
    fast_vals = multipoint_eval(poly, points)
    t1 = perf_counter()

    naive_vals = naive_eval(poly, points)
    t2 = perf_counter()

    # Numpy uses descending-order coefficients for polyval.
    np_vals = np.polyval(list(reversed(poly)), np.array(points))

    err_fast_vs_naive = max(abs(a - b) for a, b in zip(fast_vals, naive_vals))
    err_fast_vs_numpy = max(abs(a - b) for a, b in zip(fast_vals, np_vals))

    print("=== Multipoint Evaluation MVP ===")
    print(f"Polynomial degree: {len(poly) - 1}")
    print(f"Point count: {len(points)}")
    print(f"Max |fast-naive|: {err_fast_vs_naive:.3e}")
    print(f"Max |fast-numpy|: {err_fast_vs_numpy:.3e}")
    print(f"Fast method time: {(t1 - t0) * 1e3:.3f} ms")
    print(f"Naive method time: {(t2 - t1) * 1e3:.3f} ms")

    print("\nSample values (x, P(x)):")
    for i in [0, 6, 12, 18, 24]:
        print(f"  x={points[i]:>6.3f}, P(x)={fast_vals[i]:>12.6f}")

    ok = err_fast_vs_naive < 1e-8 and err_fast_vs_numpy < 1e-8
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
