"""One-way ANOVA MVP.

This script implements one-way ANOVA from scratch (sum-of-squares decomposition,
F statistic, p-value, effect size), then compares results with SciPy's
`f_oneway` for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

Array1D = np.ndarray
GroupDict = dict[str, Array1D]


@dataclass(frozen=True)
class ANOVAResult:
    """Container for one-way ANOVA components."""

    k: int
    total_n: int
    ss_between: float
    ss_within: float
    ss_total: float
    df_between: int
    df_within: int
    df_total: int
    ms_between: float
    ms_within: float
    f_stat: float
    p_value: float
    eta_squared: float


def validate_groups(groups: dict[str, np.ndarray]) -> GroupDict:
    """Validate input groups and return clean 1D float arrays."""
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    clean: GroupDict = {}
    for name, values in groups.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Group {name!r} must be 1D, got ndim={arr.ndim}")
        if arr.size < 2:
            raise ValueError(f"Group {name!r} must contain at least 2 samples")
        if not np.isfinite(arr).all():
            raise ValueError(f"Group {name!r} contains NaN or Inf")
        clean[name] = arr

    total_n = sum(arr.size for arr in clean.values())
    if total_n <= len(clean):
        raise ValueError("Need total_n > number_of_groups to have positive residual df")
    return clean


def manual_one_way_anova(groups: GroupDict) -> ANOVAResult:
    """Compute one-way ANOVA quantities from first principles."""
    clean = validate_groups(groups)
    k = len(clean)
    all_values = np.concatenate(list(clean.values()))
    total_n = int(all_values.size)

    group_sizes = np.array([arr.size for arr in clean.values()], dtype=float)
    group_means = np.array([arr.mean() for arr in clean.values()], dtype=float)
    grand_mean = float(all_values.mean())

    ss_between = float(np.sum(group_sizes * (group_means - grand_mean) ** 2))
    ss_within = float(sum(np.sum((arr - arr.mean()) ** 2) for arr in clean.values()))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))

    df_between = k - 1
    df_within = total_n - k
    df_total = total_n - 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    p_value = float(scipy_stats.f.sf(f_stat, df_between, df_within))
    eta_squared = ss_between / ss_total if ss_total > 0.0 else 0.0

    return ANOVAResult(
        k=k,
        total_n=total_n,
        ss_between=ss_between,
        ss_within=ss_within,
        ss_total=ss_total,
        df_between=df_between,
        df_within=df_within,
        df_total=df_total,
        ms_between=ms_between,
        ms_within=ms_within,
        f_stat=f_stat,
        p_value=p_value,
        eta_squared=eta_squared,
    )


def group_summary_frame(groups: GroupDict) -> pd.DataFrame:
    """Return per-group descriptive statistics."""
    records: list[dict[str, float | int | str]] = []
    for name, arr in groups.items():
        records.append(
            {
                "group": name,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std(ddof=1)": float(np.std(arr, ddof=1)),
            }
        )
    return pd.DataFrame(records)


def make_anova_table(result: ANOVAResult) -> pd.DataFrame:
    """Format ANOVA decomposition as a table."""
    return pd.DataFrame(
        [
            {
                "source": "between_groups",
                "SS": result.ss_between,
                "df": result.df_between,
                "MS": result.ms_between,
                "F": result.f_stat,
                "p_value": result.p_value,
            },
            {
                "source": "within_groups",
                "SS": result.ss_within,
                "df": result.df_within,
                "MS": result.ms_within,
                "F": float("nan"),
                "p_value": float("nan"),
            },
            {
                "source": "total",
                "SS": result.ss_total,
                "df": result.df_total,
                "MS": float("nan"),
                "F": float("nan"),
                "p_value": float("nan"),
            },
        ]
    )


def assumption_checks(groups: GroupDict) -> tuple[pd.DataFrame, float]:
    """Run lightweight assumptions diagnostics.

    - Per-group normality: Shapiro-Wilk (p-value)
    - Homogeneity of variance: Levene test (median-centered)
    """
    normality_rows: list[dict[str, float | int | str]] = []
    for name, arr in groups.items():
        if 3 <= arr.size <= 5000:
            shapiro_p = float(scipy_stats.shapiro(arr).pvalue)
        else:
            shapiro_p = float("nan")
        normality_rows.append(
            {
                "group": name,
                "n": int(arr.size),
                "shapiro_p": shapiro_p,
            }
        )

    levene_p = float(scipy_stats.levene(*groups.values(), center="median").pvalue)
    return pd.DataFrame(normality_rows), levene_p


def compare_with_scipy_f_oneway(groups: GroupDict) -> tuple[float, float]:
    """Return SciPy one-way ANOVA results for consistency checking."""
    scipy_res = scipy_stats.f_oneway(*groups.values())
    return float(scipy_res.statistic), float(scipy_res.pvalue)


def run_experiment(
    scenario: str,
    groups: dict[str, np.ndarray],
    alpha: float,
) -> tuple[dict[str, float | bool | str | int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a full ANOVA experiment and return report components."""
    clean = validate_groups(groups)
    result = manual_one_way_anova(clean)
    scipy_f, scipy_p = compare_with_scipy_f_oneway(clean)
    assumption_df, levene_p = assumption_checks(clean)

    record: dict[str, float | bool | str | int] = {
        "scenario": scenario,
        "k": result.k,
        "N": result.total_n,
        "manual_F": result.f_stat,
        "scipy_F": scipy_f,
        "|F_diff|": abs(result.f_stat - scipy_f),
        "manual_p": result.p_value,
        "scipy_p": scipy_p,
        "|p_diff|": abs(result.p_value - scipy_p),
        "eta_squared": result.eta_squared,
        "levene_p": levene_p,
        "reject_H0_at_0.05": bool(result.p_value < alpha),
    }
    return record, group_summary_frame(clean), make_anova_table(result), assumption_df


def main() -> None:
    alpha = 0.05

    # Deterministic datasets to keep validation stable.
    scenarios: dict[str, dict[str, np.ndarray]] = {
        "A_equal_means(H0_true)": {
            "G1": np.array([10, 12, 9, 11, 8, 10, 11, 9], dtype=float),
            "G2": np.array([9, 11, 10, 12, 8, 10, 9, 11], dtype=float),
            "G3": np.array([11, 9, 10, 8, 12, 10, 11, 9], dtype=float),
        },
        "B_shifted_means(H1_true)": {
            "G1": np.array([8, 9, 10, 11, 9, 10, 8, 11], dtype=float),
            "G2": np.array([12, 13, 14, 12, 13, 14, 12, 13], dtype=float),
            "G3": np.array([16, 17, 15, 16, 17, 15, 16, 17], dtype=float),
        },
    }

    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 100)

    summary_rows: list[dict[str, float | bool | str | int]] = []

    print("=" * 90)
    print("One-way ANOVA MVP")
    print(f"alpha={alpha}")
    print("Manual ANOVA is validated against scipy.stats.f_oneway")
    print("=" * 90)

    for scenario_name, groups in scenarios.items():
        summary, group_df, anova_df, assumption_df = run_experiment(scenario_name, groups, alpha=alpha)
        summary_rows.append(summary)

        print(f"\n[Scenario] {scenario_name}")
        print("Group summary:")
        print(group_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

        print("ANOVA table (manual):")
        print(anova_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

        print("Assumption checks:")
        print(assumption_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print(f"Levene p-value: {summary['levene_p']:.6f}")

    summary_df = pd.DataFrame(summary_rows)
    print("\n[Summary and SciPy cross-check]")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Validation: manual ANOVA must match SciPy numerically.
    assert np.allclose(summary_df["manual_F"], summary_df["scipy_F"], atol=1e-12)
    assert np.allclose(summary_df["manual_p"], summary_df["scipy_p"], atol=1e-12)

    # Scenario sanity checks: equal-means should not reject; shifted-means should reject.
    equal_case_p = float(summary_df.loc[summary_df["scenario"] == "A_equal_means(H0_true)", "manual_p"].iloc[0])
    shifted_case_p = float(summary_df.loc[summary_df["scenario"] == "B_shifted_means(H1_true)", "manual_p"].iloc[0])
    assert equal_case_p > alpha
    assert shifted_case_p < alpha


if __name__ == "__main__":
    main()
