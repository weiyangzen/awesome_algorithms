"""Catalan number calculation MVP.

This script implements multiple ways to compute Catalan numbers and cross-checks
results for a finite range without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy.special import comb as scipy_comb


@dataclass
class CatalanRecord:
    n: int
    dp: int
    binomial: int
    multiplicative: int
    scipy_exact: int

    @property
    def all_equal(self) -> bool:
        return self.dp == self.binomial == self.multiplicative == self.scipy_exact


def catalan_dp(n: int) -> int:
    """Compute C_n with the classic O(n^2) dynamic programming recurrence."""
    if n < 0:
        raise ValueError("n must be >= 0")

    values = [0] * (n + 1)
    values[0] = 1

    for k in range(1, n + 1):
        total = 0
        for i in range(k):
            total += values[i] * values[k - 1 - i]
        values[k] = total

    return values[n]


def catalan_binomial(n: int) -> int:
    """Compute C_n = binom(2n, n) / (n + 1) using exact integer arithmetic."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return comb(2 * n, n) // (n + 1)


def catalan_multiplicative(n: int) -> int:
    """Compute C_n through integer-safe multiplicative recurrence.

    C_0 = 1
    C_{k+1} = C_k * 2*(2k+1)/(k+2)
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    c = 1
    for k in range(n):
        c = (c * 2 * (2 * k + 1)) // (k + 2)
    return c


def catalan_scipy_exact(n: int) -> int:
    """Compute C_n using scipy.special.comb(exact=True) as an independent check."""
    if n < 0:
        raise ValueError("n must be >= 0")

    binom_2n_n = int(scipy_comb(2 * n, n, exact=True))
    return binom_2n_n // (n + 1)


def build_records(max_n: int) -> list[CatalanRecord]:
    if max_n < 0:
        raise ValueError("max_n must be >= 0")

    records: list[CatalanRecord] = []
    for n in range(max_n + 1):
        records.append(
            CatalanRecord(
                n=n,
                dp=catalan_dp(n),
                binomial=catalan_binomial(n),
                multiplicative=catalan_multiplicative(n),
                scipy_exact=catalan_scipy_exact(n),
            )
        )
    return records


def records_to_dataframe(records: list[CatalanRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "n": record.n,
                "Catalan": record.dp,
                "digits": int(np.floor(np.log10(record.dp))) + 1,
                "dp==binomial": record.dp == record.binomial,
                "dp==multiplicative": record.dp == record.multiplicative,
                "dp==scipy": record.dp == record.scipy_exact,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    max_n = 25
    records = build_records(max_n=max_n)

    if not all(record.all_equal for record in records):
        failed = [record.n for record in records if not record.all_equal]
        raise RuntimeError(f"Catalan cross-check failed for n={failed}")

    df = records_to_dataframe(records)

    print("Catalan Number MVP (MATH-0551)")
    print(f"Validated 4 methods on n=0..{max_n}.")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
