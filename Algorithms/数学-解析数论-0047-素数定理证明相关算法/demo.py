"""PNT-related algorithm MVP.

This script builds arithmetic functions used in prime-number-theorem contexts:
- pi(x): prime counting function
- theta(x): sum_{p<=x} log p
- psi(x): sum_{n<=x} Lambda(n)

It then compares these values against x/log(x) and li(x) on fixed sample points.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None


@dataclass(frozen=True)
class PNTRecord:
    """Single sampling record for PNT-related quantities."""

    x: int
    pi_x: int
    x_over_logx: float
    li_x: float
    theta_x: float
    psi_x: float
    rel_err_pi_vs_xlogx: float
    rel_err_pi_vs_li: float
    theta_over_x: float
    psi_over_x: float


def sieve_primes(limit: int) -> List[int]:
    """Return all primes <= limit using the Sieve of Eratosthenes."""
    if limit < 2:
        return []

    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[:2] = False

    bound = int(limit**0.5)
    for p in range(2, bound + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = False

    return np.flatnonzero(is_prime).tolist()


def build_prefix_tables(limit: int, primes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build prefix tables for pi(x), theta(x), psi(x)."""
    pi_indicator = np.zeros(limit + 1, dtype=np.int64)
    theta_point = np.zeros(limit + 1, dtype=np.float64)
    lambda_point = np.zeros(limit + 1, dtype=np.float64)

    for p in primes:
        pi_indicator[p] = 1
        logp = log(p)
        theta_point[p] = logp

        power = p
        while power <= limit:
            lambda_point[power] = logp
            if power > limit // p:
                break
            power *= p

    pi_prefix = np.cumsum(pi_indicator, dtype=np.int64)
    theta_prefix = np.cumsum(theta_point, dtype=np.float64)
    psi_prefix = np.cumsum(lambda_point, dtype=np.float64)
    return pi_prefix, theta_prefix, psi_prefix


def approximate_li_on_points(points: Sequence[int], grid_size: int = 50_000) -> Dict[int, float]:
    """Approximate li(x)=integral_2^x dt/log(t) for many points via one shared grid."""
    if not points:
        return {}

    max_x = max(points)
    if max_x <= 2:
        return {x: 0.0 for x in points}

    used_grid = max(4_000, grid_size)
    grid = np.linspace(2.0, float(max_x), used_grid + 1, dtype=np.float64)
    integrand = 1.0 / np.log(grid)

    dx = np.diff(grid)
    trapezoids = 0.5 * (integrand[:-1] + integrand[1:]) * dx
    cumulative = np.concatenate(([0.0], np.cumsum(trapezoids, dtype=np.float64)))

    li_values = np.interp(np.asarray(points, dtype=np.float64), grid, cumulative)
    return {int(x): float(v) for x, v in zip(points, li_values)}


def relative_error(observed: float, target: float) -> float:
    """Compute absolute relative error with safe zero handling."""
    if observed == 0.0:
        return 0.0 if target == 0.0 else float("inf")
    return abs(observed - target) / abs(observed)


def evaluate_pnt_records(
    sample_points: Sequence[int],
    pi_prefix: np.ndarray,
    theta_prefix: np.ndarray,
    psi_prefix: np.ndarray,
    li_map: Dict[int, float],
) -> List[PNTRecord]:
    """Create structured records across fixed sample points."""
    records: List[PNTRecord] = []
    for x in sample_points:
        x_float = float(x)
        pi_x = int(pi_prefix[x])
        x_over_logx = x_float / log(x_float)
        li_x = li_map[x]
        theta_x = float(theta_prefix[x])
        psi_x = float(psi_prefix[x])

        records.append(
            PNTRecord(
                x=x,
                pi_x=pi_x,
                x_over_logx=x_over_logx,
                li_x=li_x,
                theta_x=theta_x,
                psi_x=psi_x,
                rel_err_pi_vs_xlogx=relative_error(float(pi_x), x_over_logx),
                rel_err_pi_vs_li=relative_error(float(pi_x), li_x),
                theta_over_x=theta_x / x_float,
                psi_over_x=psi_x / x_float,
            )
        )
    return records


def render_table(records: Sequence[PNTRecord]) -> None:
    """Render record table using pandas if available, otherwise plain text."""
    if not records:
        print("No records to display.")
        return

    if pd is not None:
        df = pd.DataFrame([r.__dict__ for r in records])
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            240,
            "display.float_format",
            lambda v: f"{v:,.6f}",
        ):
            print(df.to_string(index=False))
        return

    header = (
        " x | pi(x) | x/log(x) | li(x) | theta(x)/x | psi(x)/x "
        "| rel_err(pi,x/logx) | rel_err(pi,li)"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r.x:>7} | {r.pi_x:>6} | {r.x_over_logx:>8.3f} | {r.li_x:>8.3f} | "
            f"{r.theta_over_x:>10.6f} | {r.psi_over_x:>8.6f} | "
            f"{r.rel_err_pi_vs_xlogx:>16.6f} | {r.rel_err_pi_vs_li:>13.6f}"
        )


def main() -> None:
    """Run fixed non-interactive MVP for PNT-related quantities."""
    limit = 100_000
    sample_points = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

    primes = sieve_primes(limit)
    pi_prefix, theta_prefix, psi_prefix = build_prefix_tables(limit, primes)
    li_map = approximate_li_on_points(sample_points)
    records = evaluate_pnt_records(sample_points, pi_prefix, theta_prefix, psi_prefix, li_map)

    print("Prime Number Theorem Related Algorithm MVP")
    print(f"Upper bound N: {limit}")
    print(f"Prime count pi(N): {int(pi_prefix[limit])}")
    print(f"theta(N): {theta_prefix[limit]:.6f}")
    print(f"psi(N): {psi_prefix[limit]:.6f}")
    print(f"theta(N)/N: {theta_prefix[limit] / limit:.6f}")
    print(f"psi(N)/N: {psi_prefix[limit] / limit:.6f}")

    max_err_xlogx = max(r.rel_err_pi_vs_xlogx for r in records)
    max_err_li = max(r.rel_err_pi_vs_li for r in records)
    print(f"Max relative error of pi(x) vs x/log(x) over samples: {max_err_xlogx:.6f}")
    print(f"Max relative error of pi(x) vs li(x) over samples: {max_err_li:.6f}")
    print("\nDetailed sample table:")
    render_table(records)


if __name__ == "__main__":
    main()
