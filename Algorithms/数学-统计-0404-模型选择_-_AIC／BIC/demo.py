"""AIC/BIC model selection MVP on polynomial regression candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class ModelScore:
    degree: int
    n: int
    k: int
    rss: float
    sigma2_hat: float
    log_likelihood: float
    aic: float
    bic: float


def simulate_polynomial_data(
    n_samples: int = 220,
    noise_std: float = 1.0,
    seed: int = 404,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate reproducible data from a cubic ground-truth model."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=n_samples)
    y_true = 1.5 - 2.0 * x + 0.8 * x**2 - 0.55 * x**3
    y = y_true + rng.normal(0.0, noise_std, size=n_samples)
    return x, y, 3


def build_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Build [x, x^2, ..., x^degree] without intercept column."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([x**power for power in range(1, degree + 1)])


def fit_ols_gaussian_mle(X: np.ndarray, y: np.ndarray, degree: int) -> ModelScore:
    """Fit OLS, then compute Gaussian log-likelihood, AIC and BIC."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the same number of samples")

    n = y.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])

    coef, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    y_hat = X_with_intercept @ coef
    residual = y - y_hat

    rss = float(residual @ residual)
    rss = max(rss, EPS)
    sigma2_hat = rss / n

    log_likelihood = -0.5 * n * (
        np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_hat)
    )

    # Number of estimated parameters: regression coefficients (incl. intercept)
    # plus one variance parameter sigma^2.
    k = X_with_intercept.shape[1] + 1
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = np.log(n) * k - 2.0 * log_likelihood

    return ModelScore(
        degree=degree,
        n=n,
        k=k,
        rss=rss,
        sigma2_hat=sigma2_hat,
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
    )


def evaluate_candidates(
    x: np.ndarray,
    y: np.ndarray,
    degrees: Iterable[int],
) -> list[ModelScore]:
    """Evaluate all candidate polynomial degrees."""
    scores: list[ModelScore] = []
    for degree in degrees:
        X = build_design_matrix(x, degree)
        score = fit_ols_gaussian_mle(X, y, degree)
        scores.append(score)
    return scores


def scores_to_dataframe(scores: list[ModelScore]) -> pd.DataFrame:
    """Convert scores to sorted dataframe for reporting."""
    records = [
        {
            "degree": s.degree,
            "n": s.n,
            "k": s.k,
            "rss": s.rss,
            "sigma2_hat": s.sigma2_hat,
            "log_likelihood": s.log_likelihood,
            "aic": s.aic,
            "bic": s.bic,
        }
        for s in scores
    ]
    return pd.DataFrame.from_records(records).sort_values("degree").reset_index(drop=True)


def pick_best(df: pd.DataFrame, criterion: str) -> pd.Series:
    """Pick the row with the minimum criterion value."""
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")
    idx = int(df[criterion].idxmin())
    return df.loc[idx]


def main() -> None:
    x, y, true_degree = simulate_polynomial_data()
    candidate_degrees = range(1, 9)

    scores = evaluate_candidates(x, y, candidate_degrees)
    df = scores_to_dataframe(scores)

    best_aic = pick_best(df, "aic")
    best_bic = pick_best(df, "bic")

    print("=== AIC/BIC Model Selection Demo (Polynomial Regression) ===")
    print(f"n_samples={len(x)}, candidate_degrees={list(candidate_degrees)}, true_degree={true_degree}")
    print()

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.4f}".format,
    ):
        print(df[["degree", "k", "rss", "sigma2_hat", "log_likelihood", "aic", "bic"]].to_string(index=False))

    print()
    print(
        "Best by AIC: "
        f"degree={int(best_aic['degree'])}, AIC={best_aic['aic']:.4f}, BIC={best_aic['bic']:.4f}"
    )
    print(
        "Best by BIC: "
        f"degree={int(best_bic['degree'])}, AIC={best_bic['aic']:.4f}, BIC={best_bic['bic']:.4f}"
    )

    # Quality gates for this deterministic demo.
    assert df[["rss", "aic", "bic"]].isna().sum().sum() == 0, "NaN detected in score table"

    rss_diff = np.diff(df["rss"].to_numpy())
    assert np.all(rss_diff <= 1e-6), "RSS should be non-increasing for nested OLS models"

    best_aic_degree = int(best_aic["degree"])
    best_bic_degree = int(best_bic["degree"])
    assert 2 <= best_aic_degree <= 5, "AIC-picked degree is unexpectedly far from cubic truth"
    assert 2 <= best_bic_degree <= 5, "BIC-picked degree is unexpectedly far from cubic truth"

    baseline_aic = float(df.loc[df["degree"] == 1, "aic"].iloc[0])
    baseline_bic = float(df.loc[df["degree"] == 1, "bic"].iloc[0])
    assert float(best_aic["aic"]) <= baseline_aic, "AIC selection failed to beat linear baseline"
    assert float(best_bic["bic"]) <= baseline_bic, "BIC selection failed to beat linear baseline"

    print("All checks passed.")


if __name__ == "__main__":
    main()
