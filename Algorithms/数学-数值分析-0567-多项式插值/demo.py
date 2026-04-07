"""Polynomial interpolation MVP (barycentric Lagrange form)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    from scipy.interpolate import BarycentricInterpolator  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BarycentricInterpolator = None


ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExperimentResult:
    case_name: str
    node_kind: str
    n_nodes: int
    node_consistency: float
    max_abs_error_barycentric: float
    max_abs_error_polyfit: float
    max_abs_diff_vs_scipy: Optional[float]


def build_nodes(a: float, b: float, n_nodes: int, node_kind: str) -> np.ndarray:
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2 for polynomial interpolation.")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Interval endpoints must be finite.")

    if node_kind == "equidistant":
        return np.linspace(a, b, n_nodes, dtype=float)
    if node_kind == "chebyshev":
        # Chebyshev-Lobatto nodes mapped from [-1, 1] to [a, b].
        k = np.arange(n_nodes, dtype=float)
        x_ref = np.cos(np.pi * k / (n_nodes - 1))
        x_mapped = 0.5 * (a + b) + 0.5 * (b - a) * x_ref
        return np.sort(x_mapped)
    raise ValueError(f"Unsupported node_kind: {node_kind}")


def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < 2:
        raise ValueError("At least 2 nodes are required.")

    sorted_x = np.sort(x)
    if np.any(np.isclose(np.diff(sorted_x), 0.0, atol=1e-14, rtol=0.0)):
        raise ValueError("Interpolation nodes must be distinct.")


def compute_barycentric_weights(x: np.ndarray) -> np.ndarray:
    n = x.size
    w = np.empty(n, dtype=float)
    for i in range(n):
        prod = 1.0
        xi = x[i]
        for j in range(n):
            if i != j:
                prod *= xi - x[j]
        w[i] = 1.0 / prod
    return w


def barycentric_interpolate(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, xq: np.ndarray
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float).reshape(-1)
    result = np.empty_like(xq, dtype=float)
    for idx, q in enumerate(xq):
        hit = np.where(np.isclose(q, x, atol=1e-12, rtol=0.0))[0]
        if hit.size > 0:
            result[idx] = y[hit[0]]
            continue

        terms = w / (q - x)
        scale = np.max(np.abs(terms))
        if scale == 0.0:
            result[idx] = 0.0
            continue

        terms_scaled = terms / scale
        result[idx] = float(np.dot(terms_scaled, y) / np.sum(terms_scaled))

    return result


def run_experiment(
    case_name: str,
    func: ArrayFn,
    a: float,
    b: float,
    n_nodes: int,
    n_eval: int,
    node_kind: str,
) -> ExperimentResult:
    x = build_nodes(a, b, n_nodes, node_kind=node_kind)
    y = func(x)
    validate_inputs(x, y)

    w = compute_barycentric_weights(x)
    y_on_nodes = barycentric_interpolate(x, y, w, x)
    node_consistency = float(np.max(np.abs(y_on_nodes - y)))

    xq = np.linspace(a, b, n_eval, dtype=float)
    y_true = func(xq)

    y_bary = barycentric_interpolate(x, y, w, xq)
    max_abs_error_barycentric = float(np.max(np.abs(y_bary - y_true)))

    coeff = np.polyfit(x, y, deg=n_nodes - 1)
    y_polyfit = np.polyval(coeff, xq)
    max_abs_error_polyfit = float(np.max(np.abs(y_polyfit - y_true)))

    max_abs_diff_vs_scipy: Optional[float] = None
    if BarycentricInterpolator is not None:
        scipy_interp = BarycentricInterpolator(x, y)
        y_scipy = np.asarray(scipy_interp(xq), dtype=float)
        max_abs_diff_vs_scipy = float(np.max(np.abs(y_scipy - y_bary)))

    return ExperimentResult(
        case_name=case_name,
        node_kind=node_kind,
        n_nodes=n_nodes,
        node_consistency=node_consistency,
        max_abs_error_barycentric=max_abs_error_barycentric,
        max_abs_error_polyfit=max_abs_error_polyfit,
        max_abs_diff_vs_scipy=max_abs_diff_vs_scipy,
    )


def format_float_or_na(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3e}"


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    cases: list[tuple[str, ArrayFn, float, float]] = [
        ("Runge: 1/(1+25x^2)", lambda t: 1.0 / (1.0 + 25.0 * t * t), -1.0, 1.0),
        ("Smooth: exp(x)*cos(3x)", lambda t: np.exp(t) * np.cos(3.0 * t), -1.0, 1.0),
    ]

    n_nodes = 15
    n_eval = 2000
    node_kinds = ("equidistant", "chebyshev")

    results: list[ExperimentResult] = []
    for case_name, func, a, b in cases:
        for node_kind in node_kinds:
            result = run_experiment(
                case_name=case_name,
                func=func,
                a=a,
                b=b,
                n_nodes=n_nodes,
                n_eval=n_eval,
                node_kind=node_kind,
            )
            results.append(result)

    print("Polynomial Interpolation Demo (Barycentric Lagrange)")
    print(
        f"{'Case':30} {'Nodes':11} {'N':>3} {'NodeConsistency':>16} "
        f"{'MaxErr(Bary)':>14} {'MaxErr(polyfit)':>16} {'Max|Bary-SciPy|':>16}"
    )
    print("-" * 118)

    for r in results:
        print(
            f"{r.case_name:30} {r.node_kind:11} {r.n_nodes:>3d} "
            f"{r.node_consistency:>16.3e} {r.max_abs_error_barycentric:>14.3e} "
            f"{r.max_abs_error_polyfit:>16.3e} {format_float_or_na(r.max_abs_diff_vs_scipy):>16}"
        )


if __name__ == "__main__":
    main()
