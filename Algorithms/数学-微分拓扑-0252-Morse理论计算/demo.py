"""Morse theory computation MVP on a 2D torus.

This script computes critical points of the Morse function
f(x, y) = cos(x) + cos(y) on T^2 = (R / 2pi Z)^2,
classifies Morse indices using the Hessian, and verifies
Morse inequalities against known Betti numbers of the torus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import root as scipy_root
except Exception:  # pragma: no cover
    scipy_root = None


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class CriticalPoint:
    x: float
    y: float
    value: float
    index: int
    eigvals: Tuple[float, float]


def normalize_angle(theta: float, tol: float = 1e-8) -> float:
    """Map an angle to [0, 2pi), snapping points near 0/pi/2pi."""
    mapped = float(np.mod(theta, TWOPI))
    if abs(mapped) < tol or abs(mapped - TWOPI) < tol:
        return 0.0
    if abs(mapped - np.pi) < tol:
        return float(np.pi)
    return mapped


def normalize_point(point: Sequence[float]) -> np.ndarray:
    return np.array([normalize_angle(float(point[0])), normalize_angle(float(point[1]))])


def morse_function(point: Sequence[float]) -> float:
    x, y = point
    return float(np.cos(x) + np.cos(y))


def gradient(point: Sequence[float]) -> np.ndarray:
    x, y = point
    return np.array([-np.sin(x), -np.sin(y)], dtype=float)


def hessian(point: Sequence[float]) -> np.ndarray:
    x, y = point
    return np.array([[-np.cos(x), 0.0], [0.0, -np.cos(y)]], dtype=float)


def deduplicate_points(points: List[np.ndarray], tol: float = 1e-6) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for p in points:
        if all(np.linalg.norm(p - q) > tol for q in unique):
            unique.append(p)
    return unique


def find_critical_points(grid_size: int = 9, grad_tol: float = 1e-8) -> List[np.ndarray]:
    """Find critical points by solving grad(f)=0 from multiple seeds.

    If SciPy is unavailable, fall back to the analytic critical set.
    """
    if scipy_root is None:
        return [
            np.array([0.0, 0.0]),
            np.array([0.0, np.pi]),
            np.array([np.pi, 0.0]),
            np.array([np.pi, np.pi]),
        ]

    seeds = np.linspace(0.0, TWOPI, grid_size, endpoint=False)
    raw_points: List[np.ndarray] = []

    for sx in seeds:
        for sy in seeds:
            result = scipy_root(lambda z: gradient(z), x0=np.array([sx, sy]), method="hybr")
            if not result.success:
                continue
            p = normalize_point(result.x)
            if np.linalg.norm(gradient(p)) <= grad_tol:
                raw_points.append(p)

    unique_points = deduplicate_points(raw_points)
    if not unique_points:
        raise RuntimeError("No critical points were found; check solver setup.")
    return unique_points


def classify_critical_points(points: List[np.ndarray], eig_tol: float = 1e-10) -> List[CriticalPoint]:
    classified: List[CriticalPoint] = []
    for p in points:
        hess = hessian(p)
        eigvals_arr = np.linalg.eigvalsh(hess)
        if float(np.min(np.abs(eigvals_arr))) <= eig_tol:
            raise ValueError(f"Degenerate critical point detected at {p}.")
        index = int(np.sum(eigvals_arr < 0.0))
        classified.append(
            CriticalPoint(
                x=float(p[0]),
                y=float(p[1]),
                value=morse_function(p),
                index=index,
                eigvals=(float(eigvals_arr[0]), float(eigvals_arr[1])),
            )
        )
    return classified


def morse_counts(points: List[CriticalPoint], dim: int = 2) -> Dict[int, int]:
    counts = {k: 0 for k in range(dim + 1)}
    for cp in points:
        counts[cp.index] += 1
    return counts


def check_morse_inequalities(m_counts: Dict[int, int], betti: Dict[int, int]) -> pd.DataFrame:
    rows = []
    max_k = max(max(m_counts), max(betti))

    for k in range(max_k + 1):
        mk = m_counts.get(k, 0)
        bk = betti.get(k, 0)
        rows.append({"kind": "weak", "k": k, "lhs": mk, "rhs": bk, "holds": mk >= bk})

    for p in range(max_k + 1):
        lhs = sum(((-1) ** (p - k)) * m_counts.get(k, 0) for k in range(p + 1))
        rhs = sum(((-1) ** (p - k)) * betti.get(k, 0) for k in range(p + 1))
        rows.append({"kind": "strong", "k": p, "lhs": lhs, "rhs": rhs, "holds": lhs >= rhs})

    return pd.DataFrame(rows)


def euler_characteristic_from_counts(counts: Dict[int, int]) -> int:
    return int(sum(((-1) ** k) * v for k, v in counts.items()))


def format_morse_polynomial(counts: Dict[int, int]) -> str:
    terms = [f"{counts[k]}*t^{k}" for k in sorted(counts.keys())]
    return " + ".join(terms)


def main() -> None:
    points = find_critical_points(grid_size=11)
    classified = classify_critical_points(points)

    cps_df = pd.DataFrame(
        [
            {
                "x": cp.x,
                "y": cp.y,
                "f(x,y)": cp.value,
                "index": cp.index,
                "lambda_min": cp.eigvals[0],
                "lambda_max": cp.eigvals[1],
            }
            for cp in classified
        ]
    ).sort_values(by=["index", "f(x,y)", "x", "y"], ignore_index=True)

    counts = morse_counts(classified, dim=2)
    betti_torus = {0: 1, 1: 2, 2: 1}
    inequalities_df = check_morse_inequalities(counts, betti_torus)

    chi_morse = euler_characteristic_from_counts(counts)
    chi_betti = euler_characteristic_from_counts(betti_torus)

    print("=== Morse Theory Computation MVP (T^2) ===")
    print(f"scipy_available: {scipy_root is not None}")
    print(f"critical_point_count: {len(classified)}")
    print()

    print("[Critical Points]")
    print(cps_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    print("[Morse Counts]")
    for k in sorted(counts.keys()):
        print(f"m_{k} = {counts[k]}")
    print(f"M(t) = {format_morse_polynomial(counts)}")
    print()

    print("[Morse Inequalities vs Betti(T^2)]")
    print(inequalities_df.to_string(index=False))
    print()

    print("[Euler Characteristic Check]")
    print(f"chi_from_morse = {chi_morse}")
    print(f"chi_from_betti = {chi_betti}")
    print(f"chi_consistent = {chi_morse == chi_betti}")


if __name__ == "__main__":
    main()
