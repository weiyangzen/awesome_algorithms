"""Minimal runnable MVP: Gauss-Legendre quadrature via Golub-Welsch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class IntegrationResult:
    """Container for one quadrature run."""

    n: int
    a: float
    b: float
    estimate: float
    nodes_std: np.ndarray
    weights_std: np.ndarray
    nodes_mapped: np.ndarray
    weights_mapped: np.ndarray
    fx: np.ndarray


def check_interval(a: float, b: float) -> None:
    """Validate integration interval."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f"interval endpoints must be finite, got a={a!r}, b={b!r}")
    if not a < b:
        raise ValueError(f"interval must satisfy a < b, got a={a}, b={b}")


def gauss_legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute nodes and weights on [-1, 1] with Golub-Welsch for Legendre."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        return np.array([0.0], dtype=float), np.array([2.0], dtype=float)

    k = np.arange(1, n, dtype=float)
    beta = k / np.sqrt(4.0 * k * k - 1.0)

    jacobi = np.zeros((n, n), dtype=float)
    idx = np.arange(n - 1)
    jacobi[idx, idx + 1] = beta
    jacobi[idx + 1, idx] = beta

    # For Legendre weight function w(t)=1 on [-1,1], nodes are eigenvalues of J,
    # and weights are 2*(first component of normalized eigenvector)^2.
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = 2.0 * (eigenvectors[0, :] ** 2)
    return nodes, weights


def map_nodes_weights(
    nodes_std: np.ndarray,
    weights_std: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map nodes/weights from [-1,1] to [a,b]."""
    half_width = 0.5 * (b - a)
    center = 0.5 * (a + b)
    nodes_mapped = half_width * nodes_std + center
    weights_mapped = half_width * weights_std
    return nodes_mapped, weights_mapped


def gauss_legendre_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> IntegrationResult:
    """Integrate func on [a,b] with n-point Gauss-Legendre quadrature."""
    check_interval(a, b)
    nodes_std, weights_std = gauss_legendre_nodes_weights(n)
    nodes_mapped, weights_mapped = map_nodes_weights(nodes_std, weights_std, a, b)

    fx = np.array([float(func(float(x))) for x in nodes_mapped], dtype=float)
    if not np.all(np.isfinite(fx)):
        raise ValueError("function evaluation produced non-finite values")

    estimate = float(np.dot(weights_mapped, fx))
    return IntegrationResult(
        n=n,
        a=a,
        b=b,
        estimate=estimate,
        nodes_std=nodes_std,
        weights_std=weights_std,
        nodes_mapped=nodes_mapped,
        weights_mapped=weights_mapped,
        fx=fx,
    )


def print_result_summary(
    title: str,
    result: IntegrationResult,
    reference: float,
    preview_k: int = 4,
) -> None:
    """Print one run result with concise node/weight preview."""
    abs_error = abs(result.estimate - reference)
    k = min(preview_k, result.n)

    nodes_preview = ", ".join(f"{v:+.6f}" for v in result.nodes_mapped[:k])
    weights_preview = ", ".join(f"{v:.6f}" for v in result.weights_mapped[:k])

    print(f"[{title}] n={result.n}, interval=[{result.a}, {result.b}]")
    print(f"  estimate   = {result.estimate:.16e}")
    print(f"  reference  = {reference:.16e}")
    print(f"  abs_error  = {abs_error:.3e}")
    print(f"  sample_nodes   = [{nodes_preview}]")
    print(f"  sample_weights = [{weights_preview}]")


def run_examples() -> None:
    """Run a few deterministic examples without interactive input."""

    def poly(x: float) -> float:
        return x**5 - 2.0 * x**3 + x + 1.0

    examples = [
        {
            "title": "poly x^5-2x^3+x+1 (exactness demo)",
            "func": poly,
            "a": -1.0,
            "b": 1.0,
            "reference": 2.0,
            "n_values": [2, 3, 4, 8],
        },
        {
            "title": "exp(x)",
            "func": math.exp,
            "a": 0.0,
            "b": 1.0,
            "reference": math.e - 1.0,
            "n_values": [2, 3, 4, 8, 16],
        },
        {
            "title": "cos(5x)",
            "func": lambda x: math.cos(5.0 * x),
            "a": 0.0,
            "b": 1.0,
            "reference": math.sin(5.0) / 5.0,
            "n_values": [2, 3, 4, 8, 16],
        },
    ]

    print("=" * 84)
    print("Gauss-Legendre Quadrature MVP (Golub-Welsch implementation)")
    print("=" * 84)

    for item in examples:
        print("-" * 84)
        print(f"Example: {item['title']}")
        print("-" * 84)

        func = item["func"]
        a = float(item["a"])
        b = float(item["b"])
        reference = float(item["reference"])

        for n in item["n_values"]:
            result = gauss_legendre_integrate(func=func, a=a, b=b, n=int(n))
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
