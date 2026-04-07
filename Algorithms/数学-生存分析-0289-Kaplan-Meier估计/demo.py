"""Kaplan-Meier estimator MVP (runnable demo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KaplanMeierEstimator:
    """Minimal right-censored Kaplan-Meier estimator."""

    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

    def fit(self, durations: np.ndarray, events: np.ndarray) -> "KaplanMeierEstimator":
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)

        if durations.ndim != 1 or events.ndim != 1:
            raise ValueError("durations and events must both be 1D arrays")
        if durations.shape[0] != events.shape[0]:
            raise ValueError("durations and events must have same length")
        if durations.size == 0:
            raise ValueError("input arrays cannot be empty")
        if np.any(durations < 0.0):
            raise ValueError("durations must be non-negative")
        if not np.all(np.isin(events, [0, 1])):
            raise ValueError("events must be binary values in {0, 1}")

        order = np.argsort(durations)
        t_sorted = durations[order]
        e_sorted = events[order]

        unique_times = np.unique(t_sorted)
        n_times = unique_times.size

        n_at_risk = np.zeros(n_times, dtype=float)
        n_events = np.zeros(n_times, dtype=float)
        n_censored = np.zeros(n_times, dtype=float)
        survival = np.zeros(n_times, dtype=float)
        greenwood_var = np.zeros(n_times, dtype=float)
        ci_lower = np.zeros(n_times, dtype=float)
        ci_upper = np.zeros(n_times, dtype=float)

        n_current = float(t_sorted.size)
        s_current = 1.0
        greenwood_sum = 0.0
        z_value = 1.959963984540054  # Standard-normal 97.5% quantile for alpha=0.05.
        if abs(self.alpha - 0.05) > 1e-12:
            from scipy.stats import norm

            z_value = float(norm.ppf(1.0 - self.alpha / 2.0))

        for i, t_i in enumerate(unique_times):
            mask = t_sorted == t_i
            d_i = float(np.sum(e_sorted[mask] == 1))
            c_i = float(np.sum(e_sorted[mask] == 0))

            n_at_risk[i] = n_current
            n_events[i] = d_i
            n_censored[i] = c_i

            if d_i > 0.0:
                s_current *= 1.0 - d_i / n_current
                if n_current - d_i > 0.0:
                    greenwood_sum += d_i / (n_current * (n_current - d_i))

            survival[i] = s_current
            greenwood_var[i] = (s_current**2) * greenwood_sum

            if s_current <= 0.0:
                ci_lower[i] = 0.0
                ci_upper[i] = 0.0
            elif s_current >= 1.0:
                ci_lower[i] = 1.0
                ci_upper[i] = 1.0
            else:
                se_loglog = np.sqrt(greenwood_sum) / abs(np.log(s_current))
                loglog = np.log(-np.log(s_current))
                lower = np.exp(-np.exp(loglog + z_value * se_loglog))
                upper = np.exp(-np.exp(loglog - z_value * se_loglog))
                ci_lower[i] = float(np.clip(lower, 0.0, 1.0))
                ci_upper[i] = float(np.clip(upper, 0.0, 1.0))

            n_current -= d_i + c_i

        self.timeline_ = unique_times
        self.n_at_risk_ = n_at_risk
        self.n_events_ = n_events
        self.n_censored_ = n_censored
        self.survival_ = survival
        self.greenwood_var_ = greenwood_var
        self.ci_lower_ = ci_lower
        self.ci_upper_ = ci_upper
        self.n_samples_ = int(durations.size)
        self.n_observed_events_ = int(np.sum(events == 1))
        return self

    def predict(self, query_times: np.ndarray) -> np.ndarray:
        self._check_fitted()
        query_times = np.asarray(query_times, dtype=float)
        if query_times.ndim != 1:
            raise ValueError("query_times must be a 1D array")

        indices = np.searchsorted(self.timeline_, query_times, side="right") - 1
        preds = np.ones_like(query_times, dtype=float)
        valid = indices >= 0
        preds[valid] = self.survival_[indices[valid]]
        return preds

    def predict_interval(self, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        query_times = np.asarray(query_times, dtype=float)
        if query_times.ndim != 1:
            raise ValueError("query_times must be a 1D array")

        indices = np.searchsorted(self.timeline_, query_times, side="right") - 1
        lower = np.ones_like(query_times, dtype=float)
        upper = np.ones_like(query_times, dtype=float)
        valid = indices >= 0
        lower[valid] = self.ci_lower_[indices[valid]]
        upper[valid] = self.ci_upper_[indices[valid]]
        return lower, upper

    def median_survival_time(self) -> float:
        self._check_fitted()
        below = np.where(self.survival_ <= 0.5)[0]
        if below.size == 0:
            return float("inf")
        return float(self.timeline_[below[0]])

    def event_table(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(
            {
                "time": self.timeline_,
                "n_at_risk": self.n_at_risk_,
                "n_events": self.n_events_,
                "n_censored": self.n_censored_,
                "survival": self.survival_,
                "ci_lower": self.ci_lower_,
                "ci_upper": self.ci_upper_,
            }
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "timeline_"):
            raise RuntimeError("model must be fitted before prediction")


def simulate_right_censored_data(n_samples: int = 300, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic right-censored cohort with two risk groups."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n_samples)  # 0: low-risk, 1: high-risk

    # High-risk group has larger hazard -> shorter survival.
    hazard = np.where(group == 1, 0.22, 0.12)
    event_time = rng.exponential(scale=1.0 / hazard)

    censor_time = rng.exponential(scale=1.0 / 0.07, size=n_samples)
    duration = np.minimum(event_time, censor_time)
    observed_event = (event_time <= censor_time).astype(int)
    return duration, observed_event, group


def summarize_model(name: str, model: KaplanMeierEstimator, query_times: np.ndarray) -> None:
    surv = model.predict(query_times)
    lo, hi = model.predict_interval(query_times)
    median_t = model.median_survival_time()
    event_rate = model.n_observed_events_ / model.n_samples_

    print(f"\n=== {name} ===")
    print(f"n_samples={model.n_samples_}, observed_events={model.n_observed_events_} ({event_rate:.2%})")
    if np.isfinite(median_t):
        print(f"median_survival_time={median_t:.3f}")
    else:
        print("median_survival_time=inf (curve never crossed 0.5)")

    print("time | survival | 95% CI")
    for t, s, l, u in zip(query_times, surv, lo, hi):
        print(f"{t:>4.1f} | {s:>8.4f} | [{l:>7.4f}, {u:>7.4f}]")


def main() -> None:
    durations, events, groups = simulate_right_censored_data(n_samples=320, seed=42)
    query_times = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

    km_all = KaplanMeierEstimator(alpha=0.05).fit(durations, events)
    km_low = KaplanMeierEstimator(alpha=0.05).fit(durations[groups == 0], events[groups == 0])
    km_high = KaplanMeierEstimator(alpha=0.05).fit(durations[groups == 1], events[groups == 1])

    print("Kaplan-Meier Estimation Demo (Right Censoring)")
    print(f"Total samples: {durations.size}")
    print(f"Overall censoring rate: {np.mean(events == 0):.2%}")
    print(f"Monotone survival check: {bool(np.all(np.diff(km_all.survival_) <= 1e-12))}")

    summarize_model("Overall Cohort", km_all, query_times)
    summarize_model("Low-risk Group", km_low, query_times)
    summarize_model("High-risk Group", km_high, query_times)

    print("\nEvent table head (overall):")
    print(km_all.event_table().head(10).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
