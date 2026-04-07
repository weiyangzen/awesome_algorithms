"""Minimal runnable MVP: Gauss-Laguerre quadrature via Golub-Welsch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class IntegrationResult:
    """Container for one Gauss-Laguerre quadrature run."""

    n: int
    estimate: float
    nodes: np.ndarray
    weights: np.ndarray
    fx: np.ndarray


def gauss_laguerre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Laguerre nodes/weights on [0, +inf).

    This is the alpha=0 case for weight w(x)=exp(-x), implemented by
    Golub-Welsch using the Laguerre Jacobi matrix:
      - diagonal: d_k = 2k + 1
      - offdiag:  e_k = k + 1
    where k starts from 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        return np.array([1.0], dtype=float), np.array([1.0], dtype=float)

    k = np.arange(n, dtype=float)
    diag = 2.0 * k + 1.0

    offdiag = np.arange(1, n, dtype=float)

    jacobi = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    jacobi[idx, idx] = diag
    upper_lower = np.arange(n - 1)
    jacobi[upper_lower, upper_lower + 1] = offdiag
    jacobi[upper_lower + 1, upper_lower] = offdiag

    # Nodes are eigenvalues; for alpha=0, mu0 = int_0^inf exp(-x) dx = 1,
    # so weights are the square of first-row eigenvector components.
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = eigenvectors[0, :] ** 2
    return nodes, weights


def gauss_laguerre_integrate(
    func: Callable[[float], float],
    n: int,
) -> IntegrationResult:
    """Approximate integral_0^inf exp(-x) * func(x) dx with n nodes."""
    nodes, weights = gauss_laguerre_nodes_weights(n)
    fx = np.array([float(func(float(x))) for x in nodes], dtype=float)

    if not np.all(np.isfinite(fx)):
        raise ValueError("function evaluation produced non-finite values")

    estimate = float(np.dot(weights, fx))
    return IntegrationResult(n=n, estimate=estimate, nodes=nodes, weights=weights, fx=fx)


def print_result_summary(
    title: str,
    result: IntegrationResult,
    reference: float,
    preview_k: int = 4,
) -> None:
    """Print one run result with concise node/weight preview."""
    abs_error = abs(result.estimate - reference)
    k = min(preview_k, result.n)

    nodes_preview = ", ".join(f"{v:.6f}" for v in result.nodes[:k])
    weights_preview = ", ".join(f"{v:.6f}" for v in result.weights[:k])

    print(f"[{title}] n={result.n}")
    print(f"  estimate        = {result.estimate:.16e}")
    print(f"  reference       = {reference:.16e}")
    print(f"  abs_error       = {abs_error:.3e}")
    print(f"  sample_nodes    = [{nodes_preview}]")
    print(f"  sample_weights  = [{weights_preview}]")


def run_examples() -> None:
    """Run deterministic examples without interactive input."""
    examples = [
        {
            "title": "poly x^5 (exactness demo)",
            "func": lambda x: x**5,
            "reference": math.factorial(5),  # int_0^inf e^{-x} x^5 dx = 5!
            "n_values": [2, 3, 4, 8],
        },
        {
            "title": "sin(x)",
            "func": lambda x: math.sin(x),
            "reference": 0.5,  # int_0^inf e^{-x} sin(x) dx = 1/(1^2+1^2)
            "n_values": [2, 4, 8, 16],
        },
        {
            "title": "exp(-x)",
            "func": lambda x: math.exp(-x),
            "reference": 0.5,  # int_0^inf e^{-x} e^{-x} dx = 1/2
            "n_values": [2, 4, 8, 16],
        },
    ]

    print("=" * 84)
    print("Gauss-Laguerre Quadrature MVP (Golub-Welsch implementation)")
    print("Computes I = integral_0^inf exp(-x) * f(x) dx")
    print("=" * 84)

    for item in examples:
        print("-" * 84)
        print(f"Example: {item['title']}")
        print("-" * 84)

        func = item["func"]
        reference = float(item["reference"])

        for n in item["n_values"]:
            result = gauss_laguerre_integrate(func=func, n=int(n))
            print_result_summary(
                title=item["title"],
                result=result,
                reference=reference,
                preview_k=4,
            )
            print()


def main() -> None:
    run_examples()


if __name__ == "__main__":
    main()
