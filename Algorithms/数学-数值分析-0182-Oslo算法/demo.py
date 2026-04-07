"""Oslo algorithm minimal runnable MVP.

This script demonstrates B-spline knot refinement via an Oslo-style
transformation matrix (implemented by equivalent repeated single-knot insertion).
It verifies geometric invariance between coarse and refined representations.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional pretty print dependency
    pd = None


def ensure_nondecreasing(values: np.ndarray, name: str) -> None:
    """Validate nondecreasing knot sequence."""
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(np.diff(values) < 0.0):
        raise ValueError(f"{name} must be nondecreasing")


def multiset_difference_sorted(target: Sequence[float], source: Sequence[float], ndigits: int = 12) -> List[float]:
    """Return sorted values that appear in target but not in source (with multiplicity)."""
    tgt = Counter(round(float(v), ndigits) for v in target)
    src = Counter(round(float(v), ndigits) for v in source)

    for key, cnt in src.items():
        if tgt[key] < cnt:
            raise ValueError("target knot vector is not a refinement of source")

    extras: List[float] = []
    for key in sorted(tgt):
        extras.extend([float(key)] * (tgt[key] - src.get(key, 0)))
    return extras


def single_knot_insertion_matrix(knots: np.ndarray, degree: int, xi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build one-step knot insertion matrix A and new knot vector.

    If old control points are P (shape: n x d), new control points are:
        Q = A @ P
    where A has shape ((n+1) x n).
    """
    u = np.asarray(knots, dtype=float)
    ensure_nondecreasing(u, "knots")

    p = int(degree)
    if p < 1:
        raise ValueError("degree must be >= 1")

    ctrl_count = u.size - p - 1
    if ctrl_count <= 1:
        raise ValueError("invalid knots/degree combination")

    left = float(u[p])
    right = float(u[-p - 1])
    x = float(xi)
    if x < left or x > right:
        raise ValueError(f"inserted knot {x} outside valid range [{left}, {right}]")

    # Span index k such that u[k] <= x < u[k+1] (right endpoint handled by clipping).
    k = int(np.searchsorted(u, x, side="right") - 1)
    k = min(max(k, p), ctrl_count - 1)

    new_knots = np.insert(u, k + 1, x)

    n = ctrl_count - 1
    a = np.zeros((ctrl_count + 1, ctrl_count), dtype=float)

    # Unaffected prefix.
    for i in range(0, k - p + 1):
        a[i, i] = 1.0

    # Local convex blend rows.
    for i in range(k - p + 1, k + 1):
        denom = u[i + p] - u[i]
        alpha = 0.0 if denom == 0.0 else (x - u[i]) / denom
        a[i, i - 1] = 1.0 - alpha
        a[i, i] = alpha

    # Unaffected suffix.
    for i in range(k + 1, n + 2):
        a[i, i - 1] = 1.0

    return new_knots, a


def oslo_refinement_matrix(
    coarse_knots: np.ndarray,
    refined_knots: np.ndarray,
    degree: int,
    ndigits: int = 12,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Compute refinement matrix T from coarse basis to refined basis.

    This MVP uses the constructive equivalent of Oslo: repeated single-knot insertion.
    If P are coarse control points, refined control points are Q = T @ P.
    """
    u = np.asarray(coarse_knots, dtype=float)
    v = np.asarray(refined_knots, dtype=float)
    ensure_nondecreasing(u, "coarse_knots")
    ensure_nondecreasing(v, "refined_knots")

    p = int(degree)
    if u.size < p + 2 or v.size < p + 2:
        raise ValueError("knot vector too short for given degree")

    extras = multiset_difference_sorted(v.tolist(), u.tolist(), ndigits=ndigits)

    current_knots = u.copy()
    ctrl_count = current_knots.size - p - 1
    transform = np.eye(ctrl_count, dtype=float)

    for x in extras:
        current_knots, a = single_knot_insertion_matrix(current_knots, p, x)
        transform = a @ transform

    if current_knots.size != v.size or np.max(np.abs(current_knots - v)) > 1e-10:
        raise ValueError("failed to reach target refined knot vector")

    return current_knots, transform, extras


def bspline_basis_all(knots: np.ndarray, degree: int, x: float) -> np.ndarray:
    """Evaluate all nonzero/order-p B-spline basis values at one x."""
    u = np.asarray(knots, dtype=float)
    p = int(degree)
    m = u.size - 1  # number of knot spans

    if m <= p:
        raise ValueError("invalid knot vector")

    # Degree-0 basis on spans.
    n0 = np.zeros(m, dtype=float)
    for i in range(m):
        if (u[i] <= x < u[i + 1]) or (x == u[-1] and i == m - 1):
            n0[i] = 1.0

    n = n0
    for k in range(1, p + 1):
        nxt = np.zeros(m - k, dtype=float)
        for i in range(m - k):
            left_denom = u[i + k] - u[i]
            right_denom = u[i + k + 1] - u[i + 1]

            left = 0.0 if left_denom == 0.0 else (x - u[i]) / left_denom * n[i]
            right = 0.0 if right_denom == 0.0 else (u[i + k + 1] - x) / right_denom * n[i + 1]
            nxt[i] = left + right
        n = nxt

    return n


def evaluate_bspline_curve(knots: np.ndarray, degree: int, control_points: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Evaluate B-spline curve points at xs using explicit basis expansion."""
    cps = np.asarray(control_points, dtype=float)
    expected_count = np.asarray(knots).size - degree - 1
    if cps.shape[0] != expected_count:
        raise ValueError("control point count does not match knot/degree")

    out = np.zeros((xs.size, cps.shape[1]), dtype=float)
    for idx, x in enumerate(xs):
        basis = bspline_basis_all(knots, degree, float(x))
        out[idx] = basis @ cps
    return out


def format_table(rows: List[Dict[str, float]]) -> str:
    """Format rows with pandas if available, else plain text table."""
    if pd is not None:
        return pd.DataFrame(rows).to_string(index=False)

    if not rows:
        return "(empty)"

    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    header = " | ".join(h.ljust(widths[h]) for h in headers)
    split = "-+-".join("-" * widths[h] for h in headers)
    body = [" | ".join(str(r[h]).ljust(widths[h]) for h in headers) for r in rows]
    return "\n".join([header, split, *body])


def main() -> None:
    degree = 3

    coarse_knots = np.array([0.0, 0.0, 0.0, 0.0, 0.30, 0.60, 1.0, 1.0, 1.0, 1.0], dtype=float)
    refined_knots = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )

    coarse_ctrl = np.array(
        [
            [0.00, 0.00],
            [0.12, 0.92],
            [0.40, 1.18],
            [0.72, 0.42],
            [0.88, -0.18],
            [1.00, 0.05],
        ],
        dtype=float,
    )

    reached_knots, transform, inserted_knots = oslo_refinement_matrix(
        coarse_knots=coarse_knots,
        refined_knots=refined_knots,
        degree=degree,
    )

    refined_ctrl = transform @ coarse_ctrl

    x_min = float(coarse_knots[degree])
    x_max = float(coarse_knots[-degree - 1])
    xs = np.linspace(x_min, x_max, 401)

    curve_coarse = evaluate_bspline_curve(coarse_knots, degree, coarse_ctrl, xs)
    curve_refined = evaluate_bspline_curve(reached_knots, degree, refined_ctrl, xs)

    diff = curve_coarse - curve_refined
    point_errors = np.linalg.norm(diff, axis=1)
    max_geom_error = float(np.max(point_errors))
    mean_geom_error = float(np.mean(point_errors))

    row_sums = transform.sum(axis=1)
    nnz_per_row = np.count_nonzero(np.abs(transform) > 0.0, axis=1)

    print("Oslo algorithm MVP (B-spline knot refinement)")
    print(f"degree={degree}")
    print(f"coarse_ctrl_count={coarse_ctrl.shape[0]}, refined_ctrl_count={refined_ctrl.shape[0]}")
    print(f"inserted_knots={inserted_knots}")
    print()

    summary_rows: List[Dict[str, float]] = [
        {
            "metric": "max_geom_error",
            "value": round(max_geom_error, 12),
        },
        {
            "metric": "mean_geom_error",
            "value": round(mean_geom_error, 12),
        },
        {
            "metric": "transform_rows",
            "value": transform.shape[0],
        },
        {
            "metric": "transform_cols",
            "value": transform.shape[1],
        },
        {
            "metric": "row_sum_min",
            "value": round(float(np.min(row_sums)), 12),
        },
        {
            "metric": "row_sum_max",
            "value": round(float(np.max(row_sums)), 12),
        },
    ]

    print("Summary:")
    print(format_table(summary_rows))
    print()

    row_rows: List[Dict[str, float]] = []
    for i in range(transform.shape[0]):
        row_rows.append(
            {
                "row": i,
                "nnz": int(nnz_per_row[i]),
                "row_sum": round(float(row_sums[i]), 8),
            }
        )

    print("Transform row sparsity:")
    print(format_table(row_rows))
    print()

    ctrl_rows: List[Dict[str, float]] = []
    for i, pt in enumerate(refined_ctrl):
        ctrl_rows.append(
            {
                "idx": i,
                "x": round(float(pt[0]), 6),
                "y": round(float(pt[1]), 6),
            }
        )

    print("Refined control points:")
    print(format_table(ctrl_rows))


if __name__ == "__main__":
    main()
