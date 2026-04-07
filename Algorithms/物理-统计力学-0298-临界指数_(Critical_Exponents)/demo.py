"""MVP: estimate critical exponents from near-critical synthetic observations.

The script demonstrates practical estimation for power-law behavior:
    y ~ A * |t|^p,  t = (T - Tc) / Tc
where p maps to critical exponents (beta, gamma, alpha, nu) depending on the observable.

Run:
    uv run python Algorithms/物理-统计力学-0298-临界指数_(Critical_Exponents)/demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class ObservableData:
    name: str
    temperatures: np.ndarray
    values: np.ndarray
    side: str  # "below" or "above"
    diverges: bool
    true_exponent: float


def _make_power_law_samples(
    rng: np.random.Generator,
    tc: float,
    n: int,
    side: str,
    amplitude: float,
    exponent_p: float,
    t_min: float,
    t_max: float,
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positive synthetic observations y=A*t^p with multiplicative noise."""
    t = np.linspace(t_min, t_max, n)
    if side == "below":
        temperatures = tc * (1.0 - t)
    elif side == "above":
        temperatures = tc * (1.0 + t)
    else:
        raise ValueError(f"Unknown side: {side}")

    clean = amplitude * np.power(t, exponent_p)
    noise = np.exp(rng.normal(loc=0.0, scale=noise_sigma, size=n))
    values = clean * noise
    return temperatures, values


def generate_synthetic_dataset(seed: int = 7) -> Dict[str, ObservableData]:
    """Create near-critical synthetic data using 3D Ising-like exponents."""
    rng = np.random.default_rng(seed)
    tc = 2.269

    # Typical universal exponents for 3D Ising universality class (approximate).
    beta = 0.326
    gamma = 1.237
    alpha = 0.110
    nu = 0.630

    dataset: Dict[str, ObservableData] = {}

    t_m, y_m = _make_power_law_samples(
        rng,
        tc=tc,
        n=60,
        side="below",
        amplitude=1.7,
        exponent_p=beta,
        t_min=0.01,
        t_max=0.25,
        noise_sigma=0.05,
    )
    dataset["magnetization"] = ObservableData(
        name="magnetization",
        temperatures=t_m,
        values=y_m,
        side="below",
        diverges=False,
        true_exponent=beta,
    )

    t_x, y_x = _make_power_law_samples(
        rng,
        tc=tc,
        n=60,
        side="above",
        amplitude=0.9,
        exponent_p=-gamma,
        t_min=0.01,
        t_max=0.25,
        noise_sigma=0.06,
    )
    dataset["susceptibility"] = ObservableData(
        name="susceptibility",
        temperatures=t_x,
        values=y_x,
        side="above",
        diverges=True,
        true_exponent=gamma,
    )

    t_c, y_c = _make_power_law_samples(
        rng,
        tc=tc,
        n=60,
        side="above",
        amplitude=2.8,
        exponent_p=-alpha,
        t_min=0.01,
        t_max=0.25,
        noise_sigma=0.05,
    )
    dataset["specific_heat"] = ObservableData(
        name="specific_heat",
        temperatures=t_c,
        values=y_c,
        side="above",
        diverges=True,
        true_exponent=alpha,
    )

    t_xi, y_xi = _make_power_law_samples(
        rng,
        tc=tc,
        n=60,
        side="above",
        amplitude=1.3,
        exponent_p=-nu,
        t_min=0.01,
        t_max=0.25,
        noise_sigma=0.04,
    )
    dataset["correlation_length"] = ObservableData(
        name="correlation_length",
        temperatures=t_xi,
        values=y_xi,
        side="above",
        diverges=True,
        true_exponent=nu,
    )

    return dataset


def _prepare_log_features(
    temperatures: np.ndarray,
    values: np.ndarray,
    tc: float,
    side: str,
    t_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_t = (temperatures - tc) / tc

    if side == "below":
        side_mask = reduced_t < 0.0
    elif side == "above":
        side_mask = reduced_t > 0.0
    else:
        raise ValueError(f"Unknown side: {side}")

    t_abs = np.abs(reduced_t)
    t_min, t_max = t_window
    mask = side_mask & (t_abs >= t_min) & (t_abs <= t_max) & (values > 0.0)
    if np.count_nonzero(mask) < 8:
        raise ValueError("Not enough points after filtering.")

    t_use = t_abs[mask]
    y_use = values[mask]
    x_log = np.log(t_use)
    y_log = np.log(y_use)
    return t_use, x_log, y_log


def fit_exponent_sklearn(
    temperatures: np.ndarray,
    values: np.ndarray,
    tc: float,
    side: str,
    diverges: bool,
    t_window: tuple[float, float] = (0.015, 0.22),
) -> dict:
    t_use, x_log, y_log = _prepare_log_features(temperatures, values, tc, side, t_window)

    model = LinearRegression()
    model.fit(x_log.reshape(-1, 1), y_log)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    pred_log = model.predict(x_log.reshape(-1, 1))
    exponent = -slope if diverges else slope

    return {
        "method": "sklearn_linear_logfit",
        "tc": tc,
        "amplitude": float(np.exp(intercept)),
        "raw_slope_p": slope,
        "estimated_exponent": exponent,
        "r2_logspace": float(r2_score(y_log, pred_log)),
        "n_points": int(t_use.size),
    }


def _power_model(t: np.ndarray, amplitude: float, p: float) -> np.ndarray:
    return amplitude * np.power(t, p)


def fit_exponent_scipy(
    temperatures: np.ndarray,
    values: np.ndarray,
    tc: float,
    side: str,
    diverges: bool,
    t_window: tuple[float, float] = (0.015, 0.22),
) -> dict:
    t_use, _, _ = _prepare_log_features(temperatures, values, tc, side, t_window)
    y_use = values[
        ((temperatures - tc) / tc < 0.0 if side == "below" else (temperatures - tc) / tc > 0.0)
        & (np.abs((temperatures - tc) / tc) >= t_window[0])
        & (np.abs((temperatures - tc) / tc) <= t_window[1])
        & (values > 0.0)
    ]

    p0 = (float(np.median(y_use)), -1.0 if diverges else 0.3)
    bounds = ([1e-12, -8.0], [1e8, 8.0])
    params, _ = curve_fit(_power_model, t_use, y_use, p0=p0, bounds=bounds, maxfev=12000)

    amplitude, slope_p = float(params[0]), float(params[1])
    pred = _power_model(t_use, amplitude, slope_p)

    exponent = -slope_p if diverges else slope_p
    r2 = r2_score(np.log(y_use), np.log(pred))

    return {
        "method": "scipy_curve_fit",
        "tc": tc,
        "amplitude": amplitude,
        "raw_slope_p": slope_p,
        "estimated_exponent": exponent,
        "r2_logspace": float(r2),
        "n_points": int(t_use.size),
    }


def fit_exponent_torch(
    temperatures: np.ndarray,
    values: np.ndarray,
    tc: float,
    side: str,
    diverges: bool,
    t_window: tuple[float, float] = (0.015, 0.22),
    steps: int = 1200,
    lr: float = 0.05,
) -> dict:
    _, x_log, y_log = _prepare_log_features(temperatures, values, tc, side, t_window)

    x_t = torch.tensor(x_log, dtype=torch.float32)
    y_t = torch.tensor(y_log, dtype=torch.float32)

    log_amplitude = torch.nn.Parameter(torch.tensor(0.0))
    slope_p = torch.nn.Parameter(torch.tensor(-0.5 if diverges else 0.3))

    optimizer = torch.optim.Adam([log_amplitude, slope_p], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = log_amplitude + slope_p * x_t
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = log_amplitude + slope_p * x_t
        residual = torch.sum((y_t - pred) ** 2)
        total = torch.sum((y_t - torch.mean(y_t)) ** 2)
        r2 = 1.0 - float(residual / total)

    slope = float(slope_p.item())
    exponent = -slope if diverges else slope

    return {
        "method": "torch_gradient_descent_logfit",
        "tc": tc,
        "amplitude": float(torch.exp(log_amplitude).item()),
        "raw_slope_p": slope,
        "estimated_exponent": exponent,
        "r2_logspace": r2,
        "n_points": int(x_t.numel()),
    }


def scan_tc_by_best_r2(
    temperatures: np.ndarray,
    values: np.ndarray,
    side: str,
    diverges: bool,
    tc_candidates: Iterable[float],
    t_window: tuple[float, float] = (0.015, 0.22),
) -> dict:
    best: Optional[dict] = None

    for tc in tc_candidates:
        try:
            result = fit_exponent_sklearn(temperatures, values, tc, side, diverges, t_window)
        except ValueError:
            continue

        if best is None or result["r2_logspace"] > best["r2_logspace"]:
            best = result

    if best is None:
        raise RuntimeError("No valid Tc candidate produced a fit.")

    return best


def main() -> None:
    true_tc = 2.269
    dataset = generate_synthetic_dataset(seed=7)

    rows = []
    tc_rows = []

    for obs in dataset.values():
        fit1 = fit_exponent_sklearn(
            obs.temperatures,
            obs.values,
            tc=true_tc,
            side=obs.side,
            diverges=obs.diverges,
        )
        fit2 = fit_exponent_scipy(
            obs.temperatures,
            obs.values,
            tc=true_tc,
            side=obs.side,
            diverges=obs.diverges,
        )
        fit3 = fit_exponent_torch(
            obs.temperatures,
            obs.values,
            tc=true_tc,
            side=obs.side,
            diverges=obs.diverges,
        )

        for fit in (fit1, fit2, fit3):
            rows.append(
                {
                    "observable": obs.name,
                    "method": fit["method"],
                    "true_exponent": obs.true_exponent,
                    "estimated_exponent": fit["estimated_exponent"],
                    "abs_error": abs(fit["estimated_exponent"] - obs.true_exponent),
                    "r2_logspace": fit["r2_logspace"],
                    "n_points": fit["n_points"],
                }
            )

        tc_grid = np.linspace(2.20, 2.34, 120)
        tc_fit = scan_tc_by_best_r2(
            obs.temperatures,
            obs.values,
            side=obs.side,
            diverges=obs.diverges,
            tc_candidates=tc_grid,
        )
        tc_rows.append(
            {
                "observable": obs.name,
                "true_tc": true_tc,
                "estimated_tc": tc_fit["tc"],
                "estimated_exponent": tc_fit["estimated_exponent"],
                "r2_logspace": tc_fit["r2_logspace"],
            }
        )

    exponent_df = pd.DataFrame(rows)
    tc_df = pd.DataFrame(tc_rows)

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:0.5f}")

    print("=== Critical Exponents MVP (synthetic near-critical data) ===")
    print(f"Reference Tc = {true_tc:.6f}")
    print()

    print("[Exponent estimation with known Tc]")
    print(exponent_df.sort_values(["observable", "method"]).to_string(index=False))
    print()

    print("[Joint Tc scan + exponent fit (best log-space R^2)]")
    print(tc_df.sort_values("observable").to_string(index=False))


if __name__ == "__main__":
    main()
