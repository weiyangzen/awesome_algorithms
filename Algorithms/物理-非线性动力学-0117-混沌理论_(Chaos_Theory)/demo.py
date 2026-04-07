"""Chaos Theory MVP via source-level logistic-map analysis.

This script avoids black-box chaos packages and implements core dynamics directly:
1) Logistic-map iteration x_{n+1} = r x_n (1 - x_n)
2) Lyapunov exponent estimation from map derivative
3) Initial-condition sensitivity fit in log-distance space
4) Autocorrelation-based period hint detection
5) NumPy/Torch trajectory consistency checks
6) Bifurcation-sample table generation

No interactive input is required.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def check_unit_interval(name: str, value: float, strict: bool = True) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite.")
    if strict:
        if not (0.0 < x < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {x}.")
    else:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {x}.")
    return x


def check_logistic_parameter(r: float) -> float:
    rv = float(r)
    if not np.isfinite(rv) or not (0.0 <= rv <= 4.0):
        raise ValueError(f"r must be finite and in [0, 4], got {r}.")
    return rv


def check_positive_int(name: str, value: int) -> int:
    iv = int(value)
    if iv <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return iv


def logistic_step(x: float, r: float) -> float:
    return float(r * x * (1.0 - x))


def simulate_logistic_numpy(r: float, x0: float, steps: int, burn_in: int = 0) -> np.ndarray:
    """Return post-burn-in trajectory for the logistic map."""
    rv = check_logistic_parameter(r)
    x = check_unit_interval("x0", x0, strict=True)
    n_steps = check_positive_int("steps", steps)
    n_burn = int(burn_in)
    if n_burn < 0:
        raise ValueError("burn_in must be >= 0.")

    trajectory = np.empty(n_steps, dtype=float)
    t = 0
    for i in range(n_steps + n_burn):
        x = logistic_step(x, rv)
        if i >= n_burn:
            trajectory[t] = x
            t += 1

    return trajectory


def simulate_logistic_torch(r_values: np.ndarray, x0: float, steps: int, burn_in: int = 0) -> np.ndarray:
    """Batch logistic trajectories for multiple r values using PyTorch."""
    if r_values.ndim != 1:
        raise ValueError(f"r_values must be 1D, got shape={r_values.shape}.")
    if not np.all(np.isfinite(r_values)):
        raise ValueError("r_values contains non-finite entries.")
    if np.any((r_values < 0.0) | (r_values > 4.0)):
        raise ValueError("all r values must be within [0, 4].")

    x_init = check_unit_interval("x0", x0, strict=True)
    n_steps = check_positive_int("steps", steps)
    n_burn = int(burn_in)
    if n_burn < 0:
        raise ValueError("burn_in must be >= 0.")

    r_t = torch.as_tensor(r_values, dtype=torch.float64)
    x_t = torch.full_like(r_t, fill_value=x_init, dtype=torch.float64)

    records = []
    for i in range(n_steps + n_burn):
        x_t = r_t * x_t * (1.0 - x_t)
        if i >= n_burn:
            records.append(x_t.clone())

    stacked = torch.stack(records, dim=0)
    return stacked.cpu().numpy()


def lyapunov_exponent_logistic(series: np.ndarray, r: float, eps: float = 1e-15) -> float:
    """Estimate lambda = mean(log|f'(x_n)|) for f(x)=r x (1-x)."""
    if series.ndim != 1 or series.size == 0:
        raise ValueError("series must be a non-empty 1D array.")
    if not np.all(np.isfinite(series)):
        raise ValueError("series contains non-finite entries.")

    rv = check_logistic_parameter(r)
    deriv_abs = np.abs(rv * (1.0 - 2.0 * series))
    deriv_abs = np.clip(deriv_abs, eps, None)
    return float(np.mean(np.log(deriv_abs)))


def estimate_period_autocorr(series: np.ndarray, max_lag: int = 100, prominence: float = 0.08) -> int:
    """Return a dominant lag from autocorrelation peaks; -1 means no clear period."""
    if series.ndim != 1 or series.size < 20:
        raise ValueError("series must be 1D with at least 20 points.")
    if not np.all(np.isfinite(series)):
        raise ValueError("series contains non-finite entries.")

    lag = check_positive_int("max_lag", max_lag)
    centered = series - float(np.mean(series))
    var = float(np.var(centered))
    if var < 1e-15:
        return 1

    corr_full = np.correlate(centered, centered, mode="full")
    base = float(corr_full[series.size - 1])
    if abs(base) < 1e-15:
        return -1

    acf = corr_full[series.size - 1 : series.size + lag] / base
    peaks, props = find_peaks(acf[1:], prominence=prominence)
    if peaks.size == 0:
        return -1

    best = int(peaks[int(np.argmax(props["prominences"]))] + 1)
    return best


def finite_time_divergence_rate(
    r: float,
    x0: float,
    delta0: float,
    steps: int,
    fit_points: int = 25,
) -> Tuple[float, float, float, np.ndarray]:
    """Fit log distance growth between two nearby trajectories."""
    rv = check_logistic_parameter(r)
    x_a = check_unit_interval("x0", x0, strict=True)
    d0 = float(delta0)
    if not np.isfinite(d0) or d0 <= 0.0:
        raise ValueError("delta0 must be finite and > 0.")

    x_b = x_a + d0
    if x_b >= 1.0:
        raise ValueError("x0 + delta0 must stay in (0, 1).")

    n_steps = check_positive_int("steps", steps)
    n_fit = check_positive_int("fit_points", fit_points)

    dists = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        x_a = logistic_step(x_a, rv)
        x_b = logistic_step(x_b, rv)
        d = abs(x_b - x_a)
        dists[i] = max(d, np.finfo(float).tiny)

    usable = np.where((dists > np.finfo(float).tiny) & (dists < 1e-2))[0]
    if usable.size >= 5:
        idx = usable[: min(n_fit, usable.size)]
    else:
        idx = np.arange(min(n_fit, n_steps))

    x_fit = idx.reshape(-1, 1).astype(float)
    y_fit = np.log(dists[idx])

    model = LinearRegression()
    model.fit(x_fit, y_fit)
    pred = model.predict(x_fit)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    rmse = float(np.sqrt(mean_squared_error(y_fit, pred)))
    return slope, intercept, rmse, dists


def bifurcation_dataframe(
    r_values: Iterable[float],
    x0: float,
    steps: int,
    burn_in: int,
    tail_keep: int,
) -> pd.DataFrame:
    rows = []
    keep = check_positive_int("tail_keep", tail_keep)

    for r in r_values:
        traj = simulate_logistic_numpy(r=float(r), x0=x0, steps=steps, burn_in=burn_in)
        tail = traj[-keep:]
        for x in tail:
            rows.append({"r": float(r), "x": float(x)})

    return pd.DataFrame(rows)


def classify_regime(lyapunov: float, period_hint: int) -> str:
    if lyapunov > 0.0:
        return "chaotic"
    if period_hint > 0:
        return f"periodic_or_quasiperiodic(period~{period_hint})"
    return "undetermined_nonchaotic"


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    x0 = 0.217
    cases = [3.50, 3.57, 3.90]

    case_rows = []
    for r in cases:
        series = simulate_logistic_numpy(r=r, x0=x0, steps=1600, burn_in=400)
        lyap = lyapunov_exponent_logistic(series, r)
        period_hint = estimate_period_autocorr(series[-500:], max_lag=120, prominence=0.08)
        hist_counts, _ = np.histogram(series, bins=25, range=(0.0, 1.0), density=True)
        shannon_entropy = float(entropy(hist_counts + 1e-12))

        case_rows.append(
            {
                "r": r,
                "lyapunov": lyap,
                "period_hint": period_hint,
                "hist_entropy": shannon_entropy,
                "regime": classify_regime(lyap, period_hint),
            }
        )

    case_df = pd.DataFrame(case_rows)

    slope, intercept, fit_rmse, dists = finite_time_divergence_rate(
        r=3.9,
        x0=x0,
        delta0=1e-10,
        steps=80,
        fit_points=25,
    )

    r_grid = np.linspace(3.5, 4.0, 64)
    np_batch = np.column_stack(
        [simulate_logistic_numpy(r=float(r), x0=x0, steps=300, burn_in=100) for r in r_grid]
    )
    torch_batch = simulate_logistic_torch(r_values=r_grid, x0=x0, steps=300, burn_in=100)

    rmse_np_vs_torch = float(
        np.sqrt(mean_squared_error(np_batch.reshape(-1), torch_batch.reshape(-1)))
    )
    max_abs_np_vs_torch = float(np.max(np.abs(np_batch - torch_batch)))

    bif_df = bifurcation_dataframe(
        r_values=np.linspace(3.5, 4.0, 90),
        x0=x0,
        steps=1200,
        burn_in=700,
        tail_keep=12,
    )

    bif_summary = (
        bif_df.groupby("r", as_index=False)
        .agg(x_min=("x", "min"), x_max=("x", "max"), x_mean=("x", "mean"), x_std=("x", "std"))
        .round(6)
    )

    lyap_periodic = float(case_df.loc[np.isclose(case_df["r"], 3.50), "lyapunov"].iloc[0])
    lyap_chaotic = float(case_df.loc[np.isclose(case_df["r"], 3.90), "lyapunov"].iloc[0])

    checks: Dict[str, bool] = {
        "periodic_case_lyapunov_negative": lyap_periodic < 0.0,
        "chaotic_case_lyapunov_positive": lyap_chaotic > 0.0,
        "sensitive_dependence_slope_positive": slope > 0.0,
        "torch_numpy_close": max_abs_np_vs_torch < 1e-12,
        "bifurcation_values_in_unit_interval": bool(
            np.all((bif_df["x"].to_numpy() >= 0.0) & (bif_df["x"].to_numpy() <= 1.0))
        ),
        "all_outputs_finite": bool(
            np.all(np.isfinite(case_df[["lyapunov", "hist_entropy"]].to_numpy()))
            and np.all(np.isfinite(bif_df.to_numpy()))
            and np.all(np.isfinite(dists))
        ),
    }

    print("Chaos Theory MVP (Logistic Map)")
    print(f"x0={x0}")
    print()

    print("Regime diagnostics by r:")
    print(case_df.round(6).to_string(index=False))
    print()

    print("Initial-condition sensitivity (r=3.9):")
    print(f"log_distance_slope={slope:.6f}")
    print(f"log_distance_intercept={intercept:.6f}")
    print(f"linear_fit_rmse={fit_rmse:.6e}")
    print(
        "distance_first10="
        + np.array2string(
            dists[:10],
            formatter={"float_kind": lambda v: f"{v:.3e}"},
        )
    )
    print()

    print("NumPy vs Torch trajectory consistency:")
    print(f"rmse_np_vs_torch={rmse_np_vs_torch:.6e}")
    print(f"max_abs_np_vs_torch={max_abs_np_vs_torch:.6e}")
    print()

    print("Bifurcation sample head (r, x):")
    print(bif_df.head(20).round(6).to_string(index=False))
    print()

    print("Bifurcation aggregated summary head:")
    print(bif_summary.head(12).to_string(index=False))
    print()

    print("Checks:")
    for key, value in checks.items():
        print(f"- {key}={value}")
    print(f"all_core_checks_pass={all(checks.values())}")


if __name__ == "__main__":
    main()
