"""Minimal runnable MVP for Doppler Effect (sound waves).

This script validates 1D acoustic Doppler shift by computing wave-crest
arrival times from a propagation equation and comparing measured frequencies
against the theoretical formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class Scenario:
    """One-dimensional source/observer motion setup.

    Sign convention:
    - +x points from source to observer at t=0.
    - observer is initially to the right of source.
    """

    name: str
    c: float
    f0: float
    x_source0: float
    x_observer0: float
    v_source: float
    v_observer: float
    n_crests: int = 360

    def validate(self) -> None:
        if self.c <= 0.0:
            raise ValueError(f"{self.name}: sound speed must be positive")
        if self.f0 <= 0.0:
            raise ValueError(f"{self.name}: source frequency must be positive")
        if self.n_crests < 3:
            raise ValueError(f"{self.name}: n_crests must be >= 3")
        if abs(self.v_source) >= self.c:
            raise ValueError(f"{self.name}: |v_source| must be < c")
        if abs(self.v_observer) >= self.c:
            raise ValueError(f"{self.name}: |v_observer| must be < c")
        if self.x_observer0 <= self.x_source0:
            raise ValueError(f"{self.name}: observer must start ahead of source")

        # Ensure observer stays ahead during all emissions in this MVP geometry.
        t_emit_max = (self.n_crests - 1) / self.f0
        gap_at_last_emission = (
            (self.x_observer0 - self.x_source0)
            + (self.v_observer - self.v_source) * t_emit_max
        )
        if gap_at_last_emission <= 0.0:
            raise ValueError(
                f"{self.name}: observer no longer ahead during emission window"
            )


def theoretical_received_frequency(s: Scenario) -> float:
    """Acoustic Doppler formula under this sign convention.

    f' = f0 * (c - v_observer) / (c - v_source)
    """

    return s.f0 * (s.c - s.v_observer) / (s.c - s.v_source)


def arrival_equation(t_arrival: float, t_emit: float, s: Scenario) -> float:
    """Implicit arrival-time equation for one crest.

    x_o(t_arr) - x_s(t_emit) = c * (t_arr - t_emit)
    """

    x_obs = s.x_observer0 + s.v_observer * t_arrival
    x_src_emit = s.x_source0 + s.v_source * t_emit
    return x_obs - x_src_emit - s.c * (t_arrival - t_emit)


def solve_arrival_time(t_emit: float, s: Scenario) -> float:
    """Solve one crest arrival time via robust bracketing + Brent root finder."""

    left = t_emit
    g_left = arrival_equation(left, t_emit, s)
    if g_left <= 0.0:
        raise RuntimeError(
            f"{s.name}: expected positive residual at t_emit, got {g_left:.6e}"
        )

    # Monotone decreasing in t_arrival because v_observer - c < 0.
    step = 0.1
    right = left + step
    g_right = arrival_equation(right, t_emit, s)
    expand_count = 0
    while g_right > 0.0:
        step *= 2.0
        right = left + step
        g_right = arrival_equation(right, t_emit, s)
        expand_count += 1
        if expand_count > 80:
            raise RuntimeError(f"{s.name}: failed to bracket arrival root")

    root = brentq(
        f=lambda t: arrival_equation(t, t_emit, s),
        a=left,
        b=right,
        xtol=1e-14,
        rtol=1e-12,
        maxiter=200,
    )
    return float(root)


def simulate_arrivals(s: Scenario) -> Tuple[np.ndarray, np.ndarray]:
    """Generate emission and arrival time arrays for all crests."""

    t_emit = np.arange(s.n_crests, dtype=np.float64) / s.f0
    t_arrive = np.array([solve_arrival_time(t, s) for t in t_emit], dtype=np.float64)

    dt_arrive = np.diff(t_arrive)
    if np.any(dt_arrive <= 0.0):
        raise RuntimeError(f"{s.name}: non-increasing arrival times detected")

    return t_emit, t_arrive


def closed_form_arrivals(t_emit: np.ndarray, s: Scenario) -> np.ndarray:
    """Analytical arrival times for this linear-motion 1D geometry."""

    d0 = s.x_observer0 - s.x_source0
    return (d0 + (s.c - s.v_source) * t_emit) / (s.c - s.v_observer)


def estimate_frequency_from_arrivals(arrivals: np.ndarray) -> Dict[str, float]:
    """Estimate received frequency from crest arrival intervals."""

    dt = np.diff(arrivals)
    inst_freq = 1.0 / dt

    freq_t = torch.tensor(inst_freq, dtype=torch.float64)
    mean_hz = float(torch.mean(freq_t).item())
    std_hz = float(torch.std(freq_t, unbiased=False).item())
    ptp_hz = float((torch.max(freq_t) - torch.min(freq_t)).item())

    return {
        "mean_inst_hz": mean_hz,
        "std_inst_hz": std_hz,
        "ptp_inst_hz": ptp_hz,
    }


def evaluate_scenarios(scenarios: Iterable[Scenario]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run all scenarios and collect per-scenario and global metrics."""

    rows: List[Dict[str, float | str]] = []

    for s in scenarios:
        s.validate()
        t_emit, t_arrive_num = simulate_arrivals(s)
        t_arrive_closed = closed_form_arrivals(t_emit, s)

        arrival_mae_s = float(mean_absolute_error(t_arrive_closed, t_arrive_num))

        freq_stats = estimate_frequency_from_arrivals(t_arrive_num)
        f_theory = theoretical_received_frequency(s)
        f_measured = float(freq_stats["mean_inst_hz"])
        rel_error = float(abs(f_measured - f_theory) / f_theory)

        rows.append(
            {
                "scenario": s.name,
                "c_m_per_s": s.c,
                "f0_hz": s.f0,
                "v_source_m_per_s": s.v_source,
                "v_observer_m_per_s": s.v_observer,
                "f_theory_hz": f_theory,
                "f_measured_hz": f_measured,
                "mean_inst_hz": float(freq_stats["mean_inst_hz"]),
                "std_inst_hz": float(freq_stats["std_inst_hz"]),
                "ptp_inst_hz": float(freq_stats["ptp_inst_hz"]),
                "arrival_mae_s": arrival_mae_s,
                "rel_error": rel_error,
            }
        )

    df = pd.DataFrame(rows)
    theory = df["f_theory_hz"].to_numpy(dtype=np.float64)
    measured = df["f_measured_hz"].to_numpy(dtype=np.float64)

    metrics: Dict[str, float] = {
        "global_r2": float(r2_score(theory, measured)),
        "global_mae_hz": float(mean_absolute_error(theory, measured)),
        "max_rel_error": float(df["rel_error"].max()),
        "max_arrival_mae_s": float(df["arrival_mae_s"].max()),
    }

    measured_t = torch.tensor(measured, dtype=torch.float64)
    metrics["torch_global_span_hz"] = float(
        (torch.max(measured_t) - torch.min(measured_t)).item()
    )

    return df, metrics


def make_default_scenarios() -> List[Scenario]:
    """Construct deterministic benchmark scenarios."""

    c = 343.0
    f0 = 440.0
    x_source0 = 0.0
    x_observer0 = 80.0
    n_crests = 360

    return [
        Scenario("baseline_static", c, f0, x_source0, x_observer0, 0.0, 0.0, n_crests),
        Scenario("source_toward_observer", c, f0, x_source0, x_observer0, 35.0, 0.0, n_crests),
        Scenario("source_away_from_observer", c, f0, x_source0, x_observer0, -30.0, 0.0, n_crests),
        Scenario("observer_toward_source", c, f0, x_source0, x_observer0, 0.0, -25.0, n_crests),
        Scenario("observer_away_from_source", c, f0, x_source0, x_observer0, 0.0, 22.0, n_crests),
        Scenario("both_toward_each_other", c, f0, x_source0, x_observer0, 28.0, -18.0, n_crests),
        Scenario("both_away_each_other", c, f0, x_source0, x_observer0, -20.0, 16.0, n_crests),
    ]


def main() -> None:
    scenarios = make_default_scenarios()
    df, metrics = evaluate_scenarios(scenarios)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("=== Doppler Effect (Sound) Scenario Table ===")
    print(
        df[
            [
                "scenario",
                "v_source_m_per_s",
                "v_observer_m_per_s",
                "f_theory_hz",
                "f_measured_hz",
                "rel_error",
                "arrival_mae_s",
            ]
        ].to_string(index=False)
    )

    print("\n=== Global Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.12e}")

    # Deterministic acceptance thresholds for this MVP.
    assert metrics["global_r2"] > 0.999999999999, "R^2 too low"
    assert metrics["global_mae_hz"] < 1e-8, "MAE too high"
    assert metrics["max_rel_error"] < 1e-11, "Relative error too high"
    assert metrics["max_arrival_mae_s"] < 1e-12, "Arrival-time MAE too high"

    print("\nAssertions passed: numerical arrivals match Doppler theory.")


if __name__ == "__main__":
    main()
