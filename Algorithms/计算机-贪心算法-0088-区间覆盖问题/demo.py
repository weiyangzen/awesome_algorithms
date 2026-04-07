"""Interval covering problem: a minimal yet honest greedy MVP.

Run:
    uv run python Algorithms/计算机-贪心算法-0088-区间覆盖问题/demo.py
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

Interval = Tuple[float, float]


def validate_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Validate intervals and normalize to float tuples."""
    cleaned: List[Interval] = []
    for idx, item in enumerate(intervals):
        if len(item) != 2:
            raise ValueError(f"interval #{idx} must contain exactly 2 values")
        start, end = float(item[0]), float(item[1])
        if start > end:
            raise ValueError(f"interval #{idx} has start > end: {item}")
        cleaned.append((start, end))
    return cleaned


def greedy_interval_cover(
    intervals: Sequence[Interval], target: Interval, eps: float = 1e-12
) -> Optional[List[int]]:
    """Return indices of a minimum-size cover for target using greedy, or None.

    The greedy rule is classic:
    - among intervals with start <= current_left, pick the one with farthest end,
    - advance current_left to that farthest end, repeat.
    """
    left, right = float(target[0]), float(target[1])
    if left > right:
        raise ValueError("target interval has left > right")
    if abs(left - right) <= eps:
        return []

    clean = validate_intervals(intervals)
    ordered = sorted(
        [(s, e, i) for i, (s, e) in enumerate(clean)],
        key=lambda x: (x[0], -x[1], x[2]),
    )

    chosen: List[int] = []
    cursor = left
    pointer = 0
    n = len(ordered)

    while cursor < right - eps:
        best_end = cursor
        best_idx: Optional[int] = None

        while pointer < n and ordered[pointer][0] <= cursor + eps:
            _, end, original_idx = ordered[pointer]
            if end > best_end + eps:
                best_end = end
                best_idx = original_idx
            pointer += 1

        if best_idx is None:
            return None

        chosen.append(best_idx)
        cursor = best_end

    return chosen


def subset_covers_target(
    intervals: Sequence[Interval], subset_indices: Sequence[int], target: Interval, eps: float = 1e-12
) -> bool:
    """Check if a chosen subset continuously covers [left, right]."""
    left, right = target
    if left > right:
        return False
    if abs(left - right) <= eps:
        return True

    sub = sorted((intervals[i] for i in subset_indices), key=lambda x: (x[0], x[1]))
    if not sub:
        return False

    reach = left
    for start, end in sub:
        if start > reach + eps:
            return False
        if end > reach:
            reach = end
        if reach >= right - eps:
            return True
    return reach >= right - eps


def brute_force_min_cover(
    intervals: Sequence[Interval], target: Interval, max_n: int = 18
) -> Optional[List[int]]:
    """Exponential baseline for small n; used only for spot-check validation."""
    n = len(intervals)
    if n > max_n:
        raise ValueError(f"brute_force_min_cover only supports n <= {max_n}, got {n}")

    left, right = target
    if left > right:
        raise ValueError("target interval has left > right")
    if abs(left - right) <= 1e-12:
        return []

    for k in range(1, n + 1):
        for comb in combinations(range(n), k):
            if subset_covers_target(intervals, comb, target):
                return list(comb)
    return None


def run_fixed_cases() -> pd.DataFrame:
    """Deterministic hand-crafted cases with expected answers."""
    target = (0.0, 10.0)
    cases = [
        {
            "name": "basic-feasible",
            "target": target,
            "intervals": [(0, 3), (2, 6), (4, 8), (7, 10), (0, 5)],
            "expected_count": 3,
        },
        {
            "name": "single-interval",
            "target": target,
            "intervals": [(-1, 11), (0, 2), (8, 10)],
            "expected_count": 1,
        },
        {
            "name": "impossible-gap",
            "target": target,
            "intervals": [(0, 2), (2, 4), (5, 10)],
            "expected_count": None,
        },
        {
            "name": "prefer-farthest",
            "target": target,
            "intervals": [(0, 2), (0, 5), (1, 4), (5, 9), (8, 10), (6, 10)],
            "expected_count": 3,
        },
    ]

    rows = []
    for case in cases:
        intervals = validate_intervals(case["intervals"])
        chosen = greedy_interval_cover(intervals, case["target"])
        got_count = None if chosen is None else len(chosen)
        expected_count = case["expected_count"]

        if got_count != expected_count:
            raise AssertionError(
                f"case={case['name']} expected_count={expected_count}, got={got_count}"
            )

        if chosen is not None and not subset_covers_target(intervals, chosen, case["target"]):
            raise AssertionError(f"case={case['name']} greedy result does not cover target")

        rows.append(
            {
                "case": case["name"],
                "interval_count": len(intervals),
                "feasible": chosen is not None,
                "greedy_count": got_count,
                "chosen_indices": chosen,
            }
        )

    return pd.DataFrame(rows)


def build_feasible_instance(
    rng: np.random.Generator,
    target: Interval,
    chain_step_range: Tuple[float, float] = (13.0, 22.0),
    chain_span_range: Tuple[float, float] = (20.0, 36.0),
    noise_count: int = 3,
) -> List[Interval]:
    """Construct a random but guaranteed-feasible instance."""
    left, right = target
    intervals: List[Interval] = []

    cursor = left
    while cursor < right - 1e-9:
        back = float(rng.uniform(0.0, 4.0))
        span = float(rng.uniform(*chain_span_range))
        start = cursor - back
        end = min(right + float(rng.uniform(0.0, 3.0)), cursor + span)
        intervals.append((start, end))

        # Force the chain to keep overlapping with the previous interval.
        # Progress is upper-bounded by span-1, so next cursor stays within
        # the current interval and cannot create a coverage gap.
        raw_step = float(rng.uniform(*chain_step_range))
        step = max(0.5, min(raw_step, span - 1.0))
        cursor += step

    for _ in range(noise_count):
        s = float(rng.uniform(left - 8.0, right - 2.0))
        e = s + float(rng.uniform(2.0, 25.0))
        intervals.append((s, e))

    return intervals


def run_optimality_spotcheck(rng: np.random.Generator, target: Interval, checks: int = 25) -> None:
    """Compare greedy count vs brute-force optimum on small random instances."""
    for _ in range(checks):
        intervals = build_feasible_instance(
            rng,
            target,
            chain_step_range=(20.0, 28.0),
            chain_span_range=(24.0, 34.0),
            noise_count=2,
        )
        intervals = validate_intervals(intervals)

        greedy = greedy_interval_cover(intervals, target)
        brute = brute_force_min_cover(intervals, target, max_n=18)

        if greedy is None or brute is None:
            raise AssertionError("spot-check generated an unexpected infeasible instance")
        if len(greedy) != len(brute):
            raise AssertionError(
                f"greedy count {len(greedy)} != brute-force optimum {len(brute)}"
            )


def collect_experiments(
    rng: np.random.Generator, target: Interval, trials: int = 80
) -> pd.DataFrame:
    """Generate benchmark table for lightweight analysis."""
    left, right = target
    target_len = right - left
    rows = []

    for trial_id in range(trials):
        intervals = build_feasible_instance(rng, target, noise_count=int(rng.integers(2, 6)))
        intervals = validate_intervals(intervals)

        chosen = greedy_interval_cover(intervals, target)
        if chosen is None:
            raise AssertionError("build_feasible_instance should always produce feasible cases")

        lengths = np.array([e - s for s, e in intervals], dtype=float)
        clipped_lengths = np.array(
            [max(0.0, min(e, right) - max(s, left)) for s, e in intervals], dtype=float
        )
        density = float(clipped_lengths.sum() / target_len)

        rows.append(
            {
                "trial": trial_id,
                "n_intervals": len(intervals),
                "avg_len": float(lengths.mean()),
                "density": density,
                "greedy_count": len(chosen),
            }
        )

    return pd.DataFrame(rows)


def run_stat_models(df: pd.DataFrame) -> None:
    """Use scipy, sklearn, torch for interpretable diagnostics."""
    rho, p_value = spearmanr(df["density"].to_numpy(), df["greedy_count"].to_numpy())

    x_cols = ["n_intervals", "avg_len", "density"]
    x = df[x_cols].to_numpy(dtype=float)
    y = df["greedy_count"].to_numpy(dtype=float)

    reg = LinearRegression()
    reg.fit(x, y)
    r2 = reg.score(x, y)

    torch.manual_seed(7)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    x_mean = x_tensor.mean(dim=0, keepdim=True)
    x_std = x_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_norm = (x_tensor - x_mean) / x_std

    model = torch.nn.Linear(x_norm.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        pred = model(x_norm)
        loss = torch.mean((pred - y_tensor) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = torch.mean((model(x_norm) - y_tensor) ** 2).item()
        w = model.weight.detach().cpu().numpy().reshape(-1)
        b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])

    print("\n[Analysis: scipy]")
    print(f"Spearman(density, greedy_count) = {rho:.4f}, p-value = {p_value:.4e}")

    print("\n[Analysis: scikit-learn LinearRegression]")
    print(f"features = {x_cols}")
    print(f"coef = {np.round(reg.coef_, 4).tolist()}, intercept = {reg.intercept_:.4f}, R^2 = {r2:.4f}")

    print("\n[Analysis: PyTorch linear regressor]")
    print(
        "weights(normalized X) = "
        f"{np.round(w, 4).tolist()}, bias = {b:.4f}, final_mse = {final_loss:.6f}"
    )


def main() -> None:
    target = (0.0, 100.0)
    np.random.seed(7)
    rng = np.random.default_rng(7)

    fixed_df = run_fixed_cases()
    run_optimality_spotcheck(rng, target, checks=25)
    exp_df = collect_experiments(rng, target, trials=80)

    print("[Fixed Cases]")
    print(fixed_df.to_string(index=False))

    print("\n[Random Experiment Preview]")
    print(exp_df.head(8).to_string(index=False))

    print("\n[Experiment Aggregates]")
    print(exp_df.describe().round(3).to_string())

    run_stat_models(exp_df)
    print("\nAll checks passed: greedy cover is validated on fixed and random spot-checks.")


if __name__ == "__main__":
    main()
