"""Minimal runnable MVP: Gauss-Hermite quadrature via Golub-Welsch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class IntegrationResult:
    """Container for one weighted Gauss-Hermite quadrature run."""

    n: int
    estimate: float
    nodes: np.ndarray
    weights: np.ndarray
    fx: np.ndarray


def gauss_hermite_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute nodes/weights for integral exp(-x^2) f(x) on (-inf, inf)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        return np.array([0.0], dtype=float), np.array([math.sqrt(math.pi)], dtype=float)

    k = np.arange(1, n, dtype=float)
    beta = np.sqrt(k / 2.0)

    jacobi = np.zeros((n, n), dtype=float)
    idx = np.arange(n - 1)
    jacobi[idx, idx + 1] = beta
    jacobi[idx + 1, idx] = beta

    # Golub-Welsch for Hermite (weight exp(-x^2)).
    # nodes: eigenvalues of Jacobi matrix
    # weights: mu0 * (first component of normalized eigenvector)^2, mu0=sqrt(pi)
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = math.sqrt(math.pi) * (eigenvectors[0, :] ** 2)
    return nodes, weights


def gauss_hermite_integrate(
    func: Callable[[float], float],
    n: int,
) -> IntegrationResult:
    """Approximate I = integral exp(-x^2) f(x) dx using n-point Gauss-Hermite."""
    nodes, weights = gauss_hermite_nodes_weights(n)

    fx = np.array([float(func(float(x))) for x in nodes], dtype=float)
    if not np.all(np.isfinite(fx)):
        raise ValueError("function evaluation produced non-finite values")

    estimate = float(np.dot(weights, fx))
    return IntegrationResult(
        n=n,
        estimate=estimate,
        nodes=nodes,
        weights=weights,
        fx=fx,
    )


def gauss_hermite_normal_expectation(
    func: Callable[[float], float],
    n: int,
) -> float:
    """Approximate E[func(Z)] for Z~N(0,1) with Gauss-Hermite."""

    def transformed(x: float) -> float:
        z = math.sqrt(2.0) * x
        return float(func(z))

    weighted = gauss_hermite_integrate(transformed, n=n)
    return weighted.estimate / math.sqrt(math.pi)


def print_weighted_integral_summary(
    title: str,
    result: IntegrationResult,
    reference: float,
    preview_k: int = 4,
) -> None:
    """Print one weighted integral result with concise node/weight preview."""
    abs_error = abs(result.estimate - reference)
    k = min(preview_k, result.n)

    nodes_preview = ", ".join(f"{v:+.6f}" for v in result.nodes[:k])
    weights_preview = ", ".join(f"{v:.6f}" for v in result.weights[:k])

    print(f"[{title}] n={result.n}")
    print(f"  estimate   = {result.estimate:.16e}")
    print(f"  reference  = {reference:.16e}")
    print(f"  abs_error  = {abs_error:.3e}")
    print(f"  sample_nodes   = [{nodes_preview}]")
    print(f"  sample_weights = [{weights_preview}]")


def print_normal_expectation_summary(
    title: str,
    estimate: float,
    reference: float,
    n: int,
) -> None:
    """Print one normal-expectation approximation result."""
    abs_error = abs(estimate - reference)
    print(f"[{title}] n={n}")
    print(f"  estimate   = {estimate:.16e}")
    print(f"  reference  = {reference:.16e}")
    print(f"  abs_error  = {abs_error:.3e}")


def run_examples() -> None:
    """Run deterministic examples without interactive input."""

    # Example A: weighted polynomial integral
    # integral exp(-x^2) x^4 dx = 3*sqrt(pi)/4
    weighted_examples = [
        {
            "title": "weighted integral: f(x)=x^4",
            "func": lambda x: x**4,
            "reference": 0.75 * math.sqrt(math.pi),
            "n_values": [1, 2, 3, 6],
        },
        {
            "title": "weighted integral: f(x)=cos(x)",
            "func": math.cos,
            "reference": math.sqrt(math.pi) * math.exp(-0.25),
            "n_values": [2, 3, 4, 8, 16],
        },
    ]

    # Example B: standard normal expectation
    # E[Z^4]=3, Z~N(0,1)
    expectation_example = {
        "title": "normal expectation: E[Z^4], Z~N(0,1)",
        "func": lambda z: z**4,
        "reference": 3.0,
        "n_values": [2, 3, 4, 8],
    }

    print("=" * 84)
    print("Gauss-Hermite Quadrature MVP (Golub-Welsch implementation)")
    print("=" * 84)

    for item in weighted_examples:
        print("-" * 84)
        print(f"Example: {item['title']}")
        print("-" * 84)

        func = item["func"]
        reference = float(item["reference"])

        for n in item["n_values"]:
            result = gauss_hermite_integrate(func=func, n=int(n))
            print_weighted_integral_summary(
                title=item["title"],
                result=result,
                reference=reference,
                preview_k=4,
            )
            print()

    print("-" * 84)
    print(f"Example: {expectation_example['title']}")
    print("-" * 84)

    func = expectation_example["func"]
    reference = float(expectation_example["reference"])

    for n in expectation_example["n_values"]:
        estimate = gauss_hermite_normal_expectation(func=func, n=int(n))
        print_normal_expectation_summary(
            title=expectation_example["title"],
            estimate=estimate,
            reference=reference,
            n=int(n),
        )
        print()


def main() -> None:
    run_examples()


if __name__ == "__main__":
    main()
