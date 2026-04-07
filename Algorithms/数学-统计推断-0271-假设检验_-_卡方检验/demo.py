"""Chi-square hypothesis test MVP (MATH-0271).

This script implements chi-square tests from formulas (non-black-box):
1) Goodness-of-fit chi-square test.
2) Chi-square test of independence for contingency tables.

SciPy is used only to validate numerical consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GoodnessOfFitResult:
    """Outputs for one goodness-of-fit chi-square test."""

    scenario: str
    chi2_stat: float
    df: int
    p_value: float
    alpha: float
    reject_h0: bool
    phi: float


@dataclass
class IndependenceResult:
    """Outputs for one independence chi-square test."""

    scenario: str
    chi2_stat: float
    df: int
    p_value: float
    alpha: float
    reject_h0: bool
    cramers_v: float
    max_abs_std_resid: float


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")


def _as_1d_nonnegative(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two categories")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _as_2d_nonnegative(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if min(arr.shape) < 2:
        raise ValueError(f"{name} must be at least 2x2")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _prepare_expected_counts(
    observed: np.ndarray,
    expected_counts: np.ndarray | None,
    expected_probs: np.ndarray | None,
) -> np.ndarray:
    """Build expected counts from expected_counts or expected_probs."""
    if (expected_counts is None) == (expected_probs is None):
        raise ValueError("Provide exactly one of expected_counts or expected_probs")

    total = float(observed.sum())
    if total <= 0.0:
        raise ValueError("observed total must be positive")

    if expected_counts is not None:
        expected = _as_1d_nonnegative(expected_counts, "expected_counts")
        if expected.shape != observed.shape:
            raise ValueError("expected_counts shape must match observed")
        expected_total = float(expected.sum())
        if expected_total <= 0.0:
            raise ValueError("expected_counts sum must be positive")
        # Scale to match observed total to avoid accidental mismatch.
        expected = expected * (total / expected_total)
    else:
        probs = _as_1d_nonnegative(expected_probs, "expected_probs")
        if probs.shape != observed.shape:
            raise ValueError("expected_probs shape must match observed")
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            raise ValueError("expected_probs sum must be positive")
        probs = probs / prob_sum
        expected = total * probs

    if np.any(expected <= 0.0):
        raise ValueError("all expected counts must be > 0 for chi-square approximation")
    return expected


def goodness_of_fit_chi_square_manual(
    observed: np.ndarray,
    expected_counts: np.ndarray | None = None,
    expected_probs: np.ndarray | None = None,
    ddof: int = 0,
    alpha: float = 0.05,
    scenario: str = "GOF",
) -> tuple[GoodnessOfFitResult, np.ndarray]:
    """Manual chi-square goodness-of-fit test."""
    _validate_alpha(alpha)
    obs = _as_1d_nonnegative(observed, "observed")
    exp = _prepare_expected_counts(obs, expected_counts=expected_counts, expected_probs=expected_probs)

    df = int(obs.size - 1 - ddof)
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")

    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
    p_value = float(stats.chi2.sf(chi2_stat, df=df))
    phi = float(np.sqrt(chi2_stat / np.sum(obs)))

    result = GoodnessOfFitResult(
        scenario=scenario,
        chi2_stat=chi2_stat,
        df=df,
        p_value=p_value,
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        phi=phi,
    )
    return result, exp


def independence_chi_square_manual(
    contingency_table: np.ndarray,
    alpha: float = 0.05,
    scenario: str = "Independence",
) -> tuple[IndependenceResult, np.ndarray, np.ndarray]:
    """Manual chi-square independence test for an r x c contingency table."""
    _validate_alpha(alpha)
    table = _as_2d_nonnegative(contingency_table, "contingency_table")

    total = float(np.sum(table))
    if total <= 0.0:
        raise ValueError("contingency_table total must be positive")

    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total

    if np.any(expected <= 0.0):
        raise ValueError("all expected counts must be > 0 for chi-square approximation")

    chi2_stat = float(np.sum((table - expected) ** 2 / expected))
    df = int((table.shape[0] - 1) * (table.shape[1] - 1))
    p_value = float(stats.chi2.sf(chi2_stat, df=df))

    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(np.sqrt((chi2_stat / total) / min_dim))
    std_residuals = (table - expected) / np.sqrt(expected)

    result = IndependenceResult(
        scenario=scenario,
        chi2_stat=chi2_stat,
        df=df,
        p_value=p_value,
        alpha=float(alpha),
        reject_h0=bool(p_value < alpha),
        cramers_v=cramers_v,
        max_abs_std_resid=float(np.max(np.abs(std_residuals))),
    )
    return result, expected, std_residuals


def _assert_close(a: float, b: float, name: str, atol: float = 1e-10) -> None:
    if not np.isclose(a, b, atol=atol):
        raise RuntimeError(f"Mismatch in {name}: manual={a}, scipy={b}")


def compare_with_scipy(
    gof_observed: np.ndarray,
    gof_expected: np.ndarray,
    indep_table: np.ndarray,
    ddof: int = 0,
) -> None:
    """Numerically validate manual implementations against SciPy."""
    manual_gof, _ = goodness_of_fit_chi_square_manual(
        observed=gof_observed,
        expected_counts=gof_expected,
        ddof=ddof,
        alpha=0.05,
        scenario="GOF-check",
    )
    scipy_gof = stats.chisquare(f_obs=gof_observed, f_exp=gof_expected, ddof=ddof)

    _assert_close(manual_gof.chi2_stat, float(scipy_gof.statistic), "GOF chi2 statistic")
    _assert_close(manual_gof.p_value, float(scipy_gof.pvalue), "GOF p-value")

    manual_ind, _, _ = independence_chi_square_manual(
        contingency_table=indep_table,
        alpha=0.05,
        scenario="Independence-check",
    )
    scipy_chi2, scipy_p, scipy_df, scipy_expected = stats.chi2_contingency(
        indep_table,
        correction=False,
    )

    _assert_close(manual_ind.chi2_stat, float(scipy_chi2), "independence chi2 statistic")
    _assert_close(manual_ind.p_value, float(scipy_p), "independence p-value")
    if manual_ind.df != int(scipy_df):
        raise RuntimeError(f"Mismatch in independence df: manual={manual_ind.df}, scipy={scipy_df}")
    if not np.allclose(scipy_expected, (indep_table.sum(axis=1, keepdims=True) @ indep_table.sum(axis=0, keepdims=True)) / indep_table.sum(), atol=1e-10):
        raise RuntimeError("Mismatch in expected contingency counts")


def _gof_row(result: GoodnessOfFitResult) -> dict[str, str | float | int | bool]:
    return {
        "scenario": result.scenario,
        "chi2_stat": result.chi2_stat,
        "df": result.df,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "phi": result.phi,
    }


def _ind_row(result: IndependenceResult) -> dict[str, str | float | int | bool]:
    return {
        "scenario": result.scenario,
        "chi2_stat": result.chi2_stat,
        "df": result.df,
        "p_value": result.p_value,
        "alpha": result.alpha,
        "reject_h0": result.reject_h0,
        "cramers_v": result.cramers_v,
        "max_abs_std_resid": result.max_abs_std_resid,
    }


def main() -> None:
    print("Chi-square test MVP (MATH-0271)")
    print("=" * 88)

    alpha = 0.05

    # Goodness-of-fit scenarios (dice frequencies)
    fair_like_observed = np.array([18, 17, 16, 15, 17, 17], dtype=np.float64)
    biased_observed = np.array([8, 10, 12, 18, 24, 28], dtype=np.float64)
    fair_probs = np.full(6, 1.0 / 6.0, dtype=np.float64)

    gof_result_1, gof_expected_1 = goodness_of_fit_chi_square_manual(
        observed=fair_like_observed,
        expected_probs=fair_probs,
        alpha=alpha,
        scenario="GOF: fair-like dice sample vs fair die",
    )
    gof_result_2, _ = goodness_of_fit_chi_square_manual(
        observed=biased_observed,
        expected_probs=fair_probs,
        alpha=alpha,
        scenario="GOF: biased dice sample vs fair die",
    )

    # Independence scenarios (rows: channel, cols: purchase)
    weak_assoc_table = np.array(
        [
            [45, 55],
            [40, 60],
            [42, 58],
        ],
        dtype=np.float64,
    )
    strong_assoc_table = np.array(
        [
            [70, 30],
            [45, 55],
            [20, 80],
        ],
        dtype=np.float64,
    )

    ind_result_1, _, _ = independence_chi_square_manual(
        contingency_table=weak_assoc_table,
        alpha=alpha,
        scenario="Independence: weak association table",
    )
    ind_result_2, _, _ = independence_chi_square_manual(
        contingency_table=strong_assoc_table,
        alpha=alpha,
        scenario="Independence: strong association table",
    )

    # SciPy alignment checks on representative datasets.
    compare_with_scipy(
        gof_observed=fair_like_observed,
        gof_expected=gof_expected_1,
        indep_table=strong_assoc_table,
        ddof=0,
    )

    gof_df = pd.DataFrame([_gof_row(gof_result_1), _gof_row(gof_result_2)])
    ind_df = pd.DataFrame([_ind_row(ind_result_1), _ind_row(ind_result_2)])

    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 80)

    print("[Goodness-of-Fit Chi-square]")
    print(gof_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("[Chi-square Test of Independence]")
    print(ind_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("SciPy consistency checks: PASSED")


if __name__ == "__main__":
    main()
