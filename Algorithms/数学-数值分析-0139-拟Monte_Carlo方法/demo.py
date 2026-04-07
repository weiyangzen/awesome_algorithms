"""Minimal runnable MVP for Quasi-Monte Carlo integration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import qmc as scipy_qmc

    HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback branch depends on environment
    scipy_qmc = None
    HAS_SCIPY = False


@dataclass
class LevelMetrics:
    """Summary metrics for one sample-size level."""

    n: int
    exact: float
    mc_mean: float
    qmc_mean: float
    mc_std: float
    qmc_std: float
    mc_rmse: float
    qmc_rmse: float


def integrand(points: np.ndarray) -> np.ndarray:
    """Vectorized test integrand on [0,1]^d."""
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array, got ndim={points.ndim}")

    values = np.exp(-points) * (1.0 + 0.1 * np.cos(2.0 * math.pi * points))
    out = np.prod(values, axis=1)

    if not np.all(np.isfinite(out)):
        raise ValueError("integrand produced non-finite values")
    return out


def exact_integral(dim: int) -> float:
    """Analytic integral of the chosen separable integrand on [0,1]^dim."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    a = 1.0
    w = 2.0 * math.pi
    i0 = (1.0 - math.exp(-a)) / a
    icos = a * (1.0 - math.exp(-a)) / (a * a + w * w)
    one_dim = i0 + 0.1 * icos
    return float(one_dim**dim)


def mc_estimate_once(n: int, dim: int, seed: int) -> float:
    """Single Monte Carlo estimate."""
    if n <= 0 or dim <= 0:
        raise ValueError(f"n and dim must be positive, got n={n}, dim={dim}")

    rng = np.random.default_rng(seed)
    points = rng.random((n, dim))
    return float(np.mean(integrand(points)))


def is_power_of_two(n: int) -> bool:
    """Return whether n is power of two."""
    return n > 0 and (n & (n - 1) == 0)


def first_primes(k: int) -> list[int]:
    """Get first k prime numbers (small-k utility for Halton fallback)."""
    if k <= 0:
        return []

    primes: list[int] = []
    candidate = 2
    while len(primes) < k:
        is_prime = True
        limit = int(math.isqrt(candidate))
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def radical_inverse(index: int, base: int) -> float:
    """Compute radical inverse in a given base."""
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    if base <= 1:
        raise ValueError(f"base must be > 1, got {base}")

    value = 0.0
    factor = 1.0 / base
    i = index
    while i > 0:
        digit = i % base
        value += digit * factor
        i //= base
        factor /= base
    return value


def halton_sequence(n: int, dim: int, start_index: int = 1) -> np.ndarray:
    """Generate n points from Halton sequence in [0,1]^dim."""
    if n <= 0 or dim <= 0:
        raise ValueError(f"n and dim must be positive, got n={n}, dim={dim}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")

    primes = first_primes(dim)
    points = np.empty((n, dim), dtype=float)

    for j, base in enumerate(primes):
        for i in range(n):
            points[i, j] = radical_inverse(start_index + i, base)

    return points


def qmc_estimate_once(n: int, dim: int, seed: int) -> float:
    """Single randomized QMC estimate (Sobol if available, else Halton fallback)."""
    if n <= 0 or dim <= 0:
        raise ValueError(f"n and dim must be positive, got n={n}, dim={dim}")

    if HAS_SCIPY:
        if not is_power_of_two(n):
            raise ValueError(
                f"Sobol random_base2 requires power-of-two n, got n={n}"
            )
        m = n.bit_length() - 1
        sampler = scipy_qmc.Sobol(d=dim, scramble=True, seed=seed)
        points = sampler.random_base2(m=m)
    else:
        base_points = halton_sequence(n=n, dim=dim, start_index=1)
        rng = np.random.default_rng(seed)
        shift = rng.random(dim)
        points = (base_points + shift) % 1.0

    return float(np.mean(integrand(points)))


def evaluate_level(n: int, dim: int, repeats: int, base_seed: int, exact: float) -> LevelMetrics:
    """Evaluate MC and QMC on one sample size level."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    mc_values = np.empty(repeats, dtype=float)
    qmc_values = np.empty(repeats, dtype=float)

    for r in range(repeats):
        mc_seed = base_seed + 100_000 + 13 * n + r
        qmc_seed = base_seed + 200_000 + 17 * n + r
        mc_values[r] = mc_estimate_once(n=n, dim=dim, seed=mc_seed)
        qmc_values[r] = qmc_estimate_once(n=n, dim=dim, seed=qmc_seed)

    mc_mean = float(np.mean(mc_values))
    qmc_mean = float(np.mean(qmc_values))
    mc_std = float(np.std(mc_values, ddof=1)) if repeats > 1 else 0.0
    qmc_std = float(np.std(qmc_values, ddof=1)) if repeats > 1 else 0.0
    mc_rmse = float(np.sqrt(np.mean((mc_values - exact) ** 2)))
    qmc_rmse = float(np.sqrt(np.mean((qmc_values - exact) ** 2)))

    return LevelMetrics(
        n=n,
        exact=exact,
        mc_mean=mc_mean,
        qmc_mean=qmc_mean,
        mc_std=mc_std,
        qmc_std=qmc_std,
        mc_rmse=mc_rmse,
        qmc_rmse=qmc_rmse,
    )


def fit_loglog_rate(n_values: list[int], errors: list[float]) -> float:
    """Fit slope of log(error) versus log(n)."""
    x = np.asarray(n_values, dtype=float)
    y = np.asarray(errors, dtype=float)

    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")

    coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    return float(coeff[0])


def print_report(metrics: list[LevelMetrics], dim: int, repeats: int) -> None:
    """Pretty-print experiment results."""
    backend = "Sobol(scrambled, scipy)" if HAS_SCIPY else "Halton + random shift (fallback)"
    exact = metrics[0].exact if metrics else float("nan")

    print("=" * 96)
    print("Quasi-Monte Carlo MVP")
    print("=" * 96)
    print(f"dimension={dim}, repeats={repeats}, backend={backend}")
    print(f"exact integral = {exact:.16e}")
    print("-" * 96)
    print(
        f"{'n':>8} {'MC_RMSE':>14} {'QMC_RMSE':>14} "
        f"{'MC_STD':>14} {'QMC_STD':>14} {'MC_MEAN':>14} {'QMC_MEAN':>14}"
    )
    print("-" * 96)

    for row in metrics:
        print(
            f"{row.n:8d} {row.mc_rmse:14.6e} {row.qmc_rmse:14.6e} "
            f"{row.mc_std:14.6e} {row.qmc_std:14.6e} "
            f"{row.mc_mean:14.6e} {row.qmc_mean:14.6e}"
        )

    mc_slope = fit_loglog_rate(
        n_values=[m.n for m in metrics],
        errors=[m.mc_rmse for m in metrics],
    )
    qmc_slope = fit_loglog_rate(
        n_values=[m.n for m in metrics],
        errors=[m.qmc_rmse for m in metrics],
    )

    print("-" * 96)
    print(f"empirical log-log slope (MC)  = {mc_slope:.4f}")
    print(f"empirical log-log slope (QMC) = {qmc_slope:.4f}")
    print("(more negative slope means faster error decay)")


def main() -> None:
    dimension = 8
    n_values = [2**k for k in range(5, 13)]
    repeats = 16
    base_seed = 20260407

    exact = exact_integral(dimension)
    metrics: list[LevelMetrics] = []

    for n in n_values:
        metrics.append(
            evaluate_level(
                n=n,
                dim=dimension,
                repeats=repeats,
                base_seed=base_seed,
                exact=exact,
            )
        )

    print_report(metrics=metrics, dim=dimension, repeats=repeats)


if __name__ == "__main__":
    main()
