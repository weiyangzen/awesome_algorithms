"""Minimal runnable MVP: Monte Carlo integration on a finite interval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MonteCarloResult:
    """Container for one Monte Carlo integration run."""

    n: int
    a: float
    b: float
    seed: int
    estimate: float
    std_error: float
    ci95_low: float
    ci95_high: float
    sample_mean: float
    sample_var: float
    sample_x_head: np.ndarray
    sample_fx_head: np.ndarray


def check_interval(a: float, b: float) -> None:
    """Validate a finite interval with strict order."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f"interval endpoints must be finite, got a={a!r}, b={b!r}")
    if not a < b:
        raise ValueError(f"interval must satisfy a < b, got a={a}, b={b}")


def monte_carlo_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    n: int,
    seed: int,
    preview_k: int = 5,
) -> MonteCarloResult:
    """Estimate integral of func over [a,b] with uniform-sampling Monte Carlo."""
    check_interval(a, b)
    if n < 2:
        raise ValueError(f"n must be >= 2 to estimate sample variance, got n={n}")

    rng = np.random.default_rng(seed)
    samples_x = rng.uniform(a, b, size=n)
    samples_fx = np.array([float(func(float(x))) for x in samples_x], dtype=float)

    if not np.all(np.isfinite(samples_fx)):
        raise ValueError("function evaluation produced non-finite values")

    width = b - a
    sample_mean = float(np.mean(samples_fx))
    sample_var = float(np.var(samples_fx, ddof=1))

    estimate = width * sample_mean
    std_error = width * math.sqrt(sample_var / n)
    ci_radius = 1.96 * std_error

    k = max(0, min(preview_k, n))
    return MonteCarloResult(
        n=n,
        a=a,
        b=b,
        seed=seed,
        estimate=estimate,
        std_error=std_error,
        ci95_low=estimate - ci_radius,
        ci95_high=estimate + ci_radius,
        sample_mean=sample_mean,
        sample_var=sample_var,
        sample_x_head=samples_x[:k].copy(),
        sample_fx_head=samples_fx[:k].copy(),
    )


def print_result_summary(title: str, result: MonteCarloResult, reference: float) -> None:
    """Print one run with concise, deterministic formatting."""
    abs_error = abs(result.estimate - reference)
    x_preview = ", ".join(f"{v:+.6f}" for v in result.sample_x_head)
    fx_preview = ", ".join(f"{v:+.6f}" for v in result.sample_fx_head)

    print(f"[{title}] n={result.n}, seed={result.seed}, interval=[{result.a}, {result.b}]")
    print(f"  estimate    = {result.estimate:.16e}")
    print(f"  reference   = {reference:.16e}")
    print(f"  abs_error   = {abs_error:.3e}")
    print(f"  std_error   = {result.std_error:.3e}")
    print(f"  ci95        = [{result.ci95_low:.16e}, {result.ci95_high:.16e}]")
    print(f"  sample_x    = [{x_preview}]")
    print(f"  sample_f(x) = [{fx_preview}]")


def run_examples() -> None:
    """Run deterministic examples without interactive input."""

    def poly(x: float) -> float:
        return x * x

    examples = [
        {
            "title": "x^2 on [0,1]",
            "func": poly,
            "a": 0.0,
            "b": 1.0,
            "reference": 1.0 / 3.0,
            "n_values": [500, 5_000, 50_000],
        },
        {
            "title": "sin(x) on [0,pi]",
            "func": math.sin,
            "a": 0.0,
            "b": math.pi,
            "reference": 2.0,
            "n_values": [500, 5_000, 50_000],
        },
        {
            "title": "exp(-x^2) on [0,1]",
            "func": lambda x: math.exp(-(x * x)),
            "a": 0.0,
            "b": 1.0,
            "reference": 0.5 * math.sqrt(math.pi) * math.erf(1.0),
            "n_values": [500, 5_000, 50_000],
        },
    ]

    seed_base = 20260407

    print("=" * 88)
    print("Monte Carlo Integration MVP (uniform sampling on finite interval)")
    print("=" * 88)

    for e_idx, item in enumerate(examples):
        print("-" * 88)
        print(f"Example: {item['title']}")
        print("-" * 88)

        func = item["func"]
        a = float(item["a"])
        b = float(item["b"])
        reference = float(item["reference"])

        for n_idx, n in enumerate(item["n_values"]):
            seed = seed_base + 10_000 * e_idx + n_idx
            result = monte_carlo_integrate(
                func=func,
                a=a,
                b=b,
                n=int(n),
                seed=seed,
                preview_k=5,
            )
            print_result_summary(item["title"], result, reference)
            print()


def main() -> None:
    run_examples()


if __name__ == "__main__":
    main()
