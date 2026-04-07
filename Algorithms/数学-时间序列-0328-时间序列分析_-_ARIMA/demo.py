"""Minimal ARIMA MVP implemented with conditional least squares (CSS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-10


@dataclass(frozen=True)
class ArimaFitResult:
    order: tuple[int, int, int]
    n_obs: int
    n_eff: int
    intercept: float
    ar_params: np.ndarray
    ma_params: np.ndarray
    differenced_series: np.ndarray
    fitted_diff: np.ndarray
    residuals: np.ndarray
    sigma2: float
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    iterations: int


def difference_series(series: np.ndarray, d: int) -> np.ndarray:
    """Apply d-order differencing to a 1D series."""
    if d < 0:
        raise ValueError("d must be >= 0")

    out = np.asarray(series, dtype=float).reshape(-1)
    for _ in range(d):
        if out.size < 2:
            raise ValueError("series is too short for requested differencing order")
        out = np.diff(out)
    return out


def invert_differences(history: np.ndarray, diff_forecast: np.ndarray, d: int) -> np.ndarray:
    """Restore level forecasts from d-order differenced forecasts."""
    history = np.asarray(history, dtype=float).reshape(-1)
    diff_forecast = np.asarray(diff_forecast, dtype=float).reshape(-1)

    if d < 0:
        raise ValueError("d must be >= 0")
    if history.size == 0:
        raise ValueError("history must be non-empty")
    if d == 0:
        return diff_forecast.copy()
    if history.size <= d:
        raise ValueError("history is too short for inverse differencing")

    # states[k] stores the last value of the k-th order difference.
    # k=0 -> level, k=1 -> first difference, ...
    states = [float(history[-1])]
    current = history.copy()
    for _ in range(1, d):
        current = np.diff(current)
        states.append(float(current[-1]))

    restored: list[float] = []
    for value in diff_forecast:
        carry = float(value)
        for idx in range(d - 1, 0, -1):
            states[idx] += carry
            carry = states[idx]
        states[0] += carry
        restored.append(states[0])

    return np.asarray(restored, dtype=float)


def _compute_one_step_predictions(
    x: np.ndarray,
    intercept: float,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute one-step fitted values and residuals for ARMA on x."""
    x = np.asarray(x, dtype=float).reshape(-1)
    ar_params = np.asarray(ar_params, dtype=float).reshape(-1)
    ma_params = np.asarray(ma_params, dtype=float).reshape(-1)

    p = int(ar_params.size)
    q = int(ma_params.size)
    max_lag = max(p, q)
    n = x.size

    fitted = np.full(n, np.nan, dtype=float)
    residuals = np.zeros(n, dtype=float)

    for t in range(max_lag, n):
        pred = intercept
        for i in range(1, p + 1):
            pred += ar_params[i - 1] * x[t - i]
        for j in range(1, q + 1):
            pred += ma_params[j - 1] * residuals[t - j]
        fitted[t] = pred
        residuals[t] = x[t] - pred

    return fitted, residuals, max_lag


def fit_arma_css(
    x: np.ndarray,
    p: int,
    q: int,
    max_iter: int = 120,
    tol: float = 1e-7,
) -> dict[str, object]:
    """Fit ARMA(p, q) on x using conditional least squares."""
    if p < 0 or q < 0:
        raise ValueError("p and q must be >= 0")

    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    max_lag = max(p, q)

    if n <= max_lag + 8:
        raise ValueError("not enough samples for requested ARMA order")

    k = 1 + p + q  # intercept + AR + MA
    params = np.zeros(k, dtype=float)
    params[0] = float(np.mean(x[max_lag:]))
    residuals = np.zeros(n, dtype=float)

    converged = False
    iterations = max_iter

    for it in range(1, max_iter + 1):
        rows: list[list[float]] = []
        targets: list[float] = []

        for t in range(max_lag, n):
            row = [1.0]
            for i in range(1, p + 1):
                row.append(float(x[t - i]))
            for j in range(1, q + 1):
                row.append(float(residuals[t - j]))
            rows.append(row)
            targets.append(float(x[t]))

        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)

        new_params, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(new_params[0])
        ar_params = new_params[1 : 1 + p]
        ma_params = new_params[1 + p :]

        fitted, new_residuals, _ = _compute_one_step_predictions(
            x=x,
            intercept=intercept,
            ar_params=ar_params,
            ma_params=ma_params,
        )

        delta = float(np.max(np.abs(new_params - params)))
        params = new_params
        residuals = new_residuals

        if delta < tol:
            converged = True
            iterations = it
            break

    intercept = float(params[0])
    ar_params = params[1 : 1 + p].copy()
    ma_params = params[1 + p :].copy()

    fitted, residuals, max_lag = _compute_one_step_predictions(
        x=x,
        intercept=intercept,
        ar_params=ar_params,
        ma_params=ma_params,
    )

    effective_residuals = residuals[max_lag:]
    n_eff = int(effective_residuals.size)
    sse = float(effective_residuals @ effective_residuals)
    sigma2 = max(sse / max(n_eff, 1), EPS)

    log_likelihood = -0.5 * n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0)
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = np.log(max(n_eff, 1)) * k - 2.0 * log_likelihood

    return {
        "n_obs": int(n),
        "n_eff": n_eff,
        "intercept": intercept,
        "ar_params": ar_params,
        "ma_params": ma_params,
        "fitted": fitted,
        "residuals": residuals,
        "sigma2": float(sigma2),
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "converged": converged,
        "iterations": iterations,
    }


def fit_arima_css(
    series: np.ndarray,
    order: tuple[int, int, int],
    max_iter: int = 120,
    tol: float = 1e-7,
) -> ArimaFitResult:
    """Fit ARIMA(p, d, q) by fitting ARMA(p, q) on differenced series."""
    p, d, q = order
    if d < 0:
        raise ValueError("d must be >= 0")

    series = np.asarray(series, dtype=float).reshape(-1)
    diff_series = difference_series(series, d)

    arma = fit_arma_css(diff_series, p=p, q=q, max_iter=max_iter, tol=tol)

    return ArimaFitResult(
        order=order,
        n_obs=int(arma["n_obs"]),
        n_eff=int(arma["n_eff"]),
        intercept=float(arma["intercept"]),
        ar_params=np.asarray(arma["ar_params"], dtype=float),
        ma_params=np.asarray(arma["ma_params"], dtype=float),
        differenced_series=diff_series,
        fitted_diff=np.asarray(arma["fitted"], dtype=float),
        residuals=np.asarray(arma["residuals"], dtype=float),
        sigma2=float(arma["sigma2"]),
        log_likelihood=float(arma["log_likelihood"]),
        aic=float(arma["aic"]),
        bic=float(arma["bic"]),
        converged=bool(arma["converged"]),
        iterations=int(arma["iterations"]),
    )


def forecast_arima(model: ArimaFitResult, history: np.ndarray, steps: int) -> np.ndarray:
    """Forecast future level values from a fitted ARIMA model."""
    if steps <= 0:
        return np.empty(0, dtype=float)

    p, d, q = model.order
    _ = p  # explicit to show order unpacking intent

    diff_history = list(model.differenced_series.astype(float))
    resid_history = list(model.residuals.astype(float))

    diff_forecast: list[float] = []
    for _step in range(steps):
        pred = model.intercept
        for i in range(1, model.ar_params.size + 1):
            pred += model.ar_params[i - 1] * diff_history[-i]
        for j in range(1, model.ma_params.size + 1):
            pred += model.ma_params[j - 1] * resid_history[-j]

        diff_forecast.append(float(pred))
        diff_history.append(float(pred))
        resid_history.append(0.0)  # expected future innovation is zero

    diff_forecast_arr = np.asarray(diff_forecast, dtype=float)
    return invert_differences(history=np.asarray(history, dtype=float), diff_forecast=diff_forecast_arr, d=d)


def simulate_arima_111(
    n_samples: int = 320,
    drift: float = 0.35,
    phi: float = 0.55,
    theta: float = -0.30,
    noise_std: float = 0.8,
    seed: int = 328,
) -> np.ndarray:
    """Generate an ARIMA(1,1,1)-like series for deterministic demo."""
    rng = np.random.default_rng(seed)
    burn_in = 100
    total = n_samples + burn_in

    eps = rng.normal(loc=0.0, scale=noise_std, size=total)
    diff_series = np.zeros(total, dtype=float)

    for t in range(1, total):
        diff_series[t] = (
            drift
            + phi * diff_series[t - 1]
            + eps[t]
            + theta * eps[t - 1]
        )

    diff_series = diff_series[burn_in:]
    level_series = 100.0 + np.cumsum(diff_series)
    return level_series


def evaluate_candidate_orders(
    train_series: np.ndarray,
    orders: Iterable[tuple[int, int, int]],
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], ArimaFitResult]]:
    """Fit candidate ARIMA orders and return table + fitted models."""
    results: dict[tuple[int, int, int], ArimaFitResult] = {}
    records: list[dict[str, object]] = []

    for order in orders:
        fit = fit_arima_css(train_series, order)
        results[order] = fit

        records.append(
            {
                "order": order,
                "n_eff": fit.n_eff,
                "k": 1 + order[0] + order[2],
                "sigma2": fit.sigma2,
                "log_likelihood": fit.log_likelihood,
                "aic": fit.aic,
                "bic": fit.bic,
                "converged": fit.converged,
                "iterations": fit.iterations,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("aic").reset_index(drop=True)
    return df, results


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    series = simulate_arima_111()
    train_size = 260
    train = series[:train_size]
    test = series[train_size:]

    candidate_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
    score_df, fit_map = evaluate_candidate_orders(train, candidate_orders)

    best_row = score_df.iloc[0]
    best_order = tuple(best_row["order"])
    best_model = fit_map[best_order]

    forecast = forecast_arima(best_model, history=train, steps=test.size)
    baseline = np.full(test.shape, train[-1], dtype=float)

    arima_rmse = rmse(test, forecast)
    arima_mae = mae(test, forecast)
    baseline_rmse = rmse(test, baseline)
    baseline_mae = mae(test, baseline)

    print("=== ARIMA CSS Demo (from-scratch MVP) ===")
    print(f"train_size={train.size}, test_size={test.size}, candidates={candidate_orders}")
    print()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        140,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(score_df.to_string(index=False))

    print()
    print(f"Best AIC order: {best_order}")
    print(f"intercept={best_model.intercept:.6f}")
    print(f"ar_params={np.array2string(best_model.ar_params, precision=6)}")
    print(f"ma_params={np.array2string(best_model.ma_params, precision=6)}")
    print()
    print("Test metrics:")
    print(f"ARIMA   -> RMSE={arima_rmse:.6f}, MAE={arima_mae:.6f}")
    print(f"LastVal -> RMSE={baseline_rmse:.6f}, MAE={baseline_mae:.6f}")

    # Quality gates.
    assert not score_df[["sigma2", "aic", "bic"]].isna().any().any(), "NaN found in score table"
    assert np.isfinite(score_df[["sigma2", "aic", "bic"]].to_numpy()).all(), "Non-finite score found"
    assert bool(score_df["converged"].any()), "No candidate model converged"

    assert np.isfinite(best_model.intercept), "Best model intercept is not finite"
    assert np.isfinite(best_model.aic), "Best model AIC is not finite"
    assert np.isfinite(best_model.bic), "Best model BIC is not finite"

    assert forecast.size == test.size, "Forecast horizon mismatch"
    assert np.isfinite(forecast).all(), "Forecast contains non-finite value"
    assert arima_rmse < baseline_rmse, "ARIMA should beat last-value baseline on this dataset"

    print("All checks passed.")


if __name__ == "__main__":
    main()
