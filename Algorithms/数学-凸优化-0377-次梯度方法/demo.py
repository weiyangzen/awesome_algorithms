"""Subgradient method MVP for convex non-smooth optimization (MATH-0377).

Problem solved in this demo:
    min_x 0.5 * ||x - c||_2^2 + lam * ||x||_1

The objective is convex and non-smooth (due to L1 norm). We implement a
plain deterministic subgradient method and compare against the closed-form
soft-threshold solution for verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float]


@dataclass
class CaseConfig:
    """Configuration for one deterministic test case."""

    name: str
    c: np.ndarray
    lam: float
    step_scale: float
    max_iter: int


def check_vector(name: str, x: np.ndarray) -> None:
    """Validate 1D finite vector."""
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def objective_l1_quadratic(x: np.ndarray, c: np.ndarray, lam: float) -> float:
    """Compute f(x) = 0.5*||x-c||^2 + lam*||x||_1."""
    diff = x - c
    return 0.5 * float(diff.T @ diff) + lam * float(np.sum(np.abs(x)))


def soft_threshold(c: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form minimizer of 0.5*||x-c||^2 + lam*||x||_1."""
    return np.sign(c) * np.maximum(np.abs(c) - lam, 0.0)


def subgradient_l1(x: np.ndarray, zero_eps: float = 1e-12) -> np.ndarray:
    """Pick one valid subgradient of ||x||_1.

    For x_i != 0: sign(x_i)
    For x_i == 0: choose 0 (valid because 0 in [-1, 1])
    """
    s = np.sign(x)
    s[np.abs(x) <= zero_eps] = 0.0
    return s


def subgradient_method(
    c: np.ndarray,
    lam: float,
    x0: np.ndarray,
    step_scale: float,
    max_iter: int,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, List[HistoryItem], np.ndarray]:
    """Run deterministic subgradient descent for the non-smooth objective."""
    check_vector("c", c)
    check_vector("x0", x0)
    if c.shape != x0.shape:
        raise ValueError(f"Shape mismatch: c={c.shape}, x0={x0.shape}.")
    if lam < 0.0:
        raise ValueError("lam must be >= 0.")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    x = x0.astype(float).copy()
    history: List[HistoryItem] = []
    x_best = x.copy()
    best_obj = objective_l1_quadratic(x, c, lam)

    for t in range(1, max_iter + 1):
        s = subgradient_l1(x)
        g = (x - c) + lam * s
        g_norm = float(np.linalg.norm(g))

        eta_t = step_scale / np.sqrt(float(t))
        x_next = x - eta_t * g
        step_norm = float(np.linalg.norm(x_next - x))

        obj_next = objective_l1_quadratic(x_next, c, lam)
        history.append((t, obj_next, g_norm, step_norm))

        if obj_next < best_obj:
            best_obj = obj_next
            x_best = x_next.copy()

        if not np.isfinite(obj_next) or not np.isfinite(g_norm) or not np.isfinite(step_norm):
            raise RuntimeError("Encountered non-finite value during iteration.")

        if step_norm <= tol * (1.0 + float(np.linalg.norm(x_next))):
            return x_best, history, x_next

        x = x_next

    return x_best, history, x


def print_history(history: Sequence[HistoryItem], max_lines: int = 10) -> None:
    """Print the first few optimization records."""
    print("iter | objective        | ||subgrad||       | ||step||")
    print("-" * 64)
    for item in history[:max_lines]:
        k, obj, g_norm, step_norm = item
        print(f"{k:4d} | {obj:16.9e} | {g_norm:16.9e} | {step_norm:16.9e}")
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def build_cases() -> List[CaseConfig]:
    """Create deterministic test cases with different dimensions and sparsity."""
    rng = np.random.default_rng(377)

    c1 = np.array([2.2, -1.6, 0.5, -0.2, 1.1], dtype=float)
    c2 = np.array([1.9, -0.4, 0.15, -2.3, 0.02, 0.88, -0.76, 0.5], dtype=float)
    c3 = rng.normal(loc=0.0, scale=1.0, size=120).astype(float)

    return [
        CaseConfig(
            name="Low-dimensional mixed signs",
            c=c1,
            lam=0.6,
            step_scale=1.0,
            max_iter=9000,
        ),
        CaseConfig(
            name="Sparse-inducing medium case",
            c=c2,
            lam=0.7,
            step_scale=1.0,
            max_iter=12000,
        ),
        CaseConfig(
            name="Higher-dimensional random case",
            c=c3,
            lam=0.55,
            step_scale=0.9,
            max_iter=15000,
        ),
    ]


def run_case(case: CaseConfig) -> Dict[str, float]:
    """Run one case and return summary metrics."""
    print(f"\n=== Case: {case.name} ===")

    x0 = np.zeros_like(case.c)
    x_best, history, x_last = subgradient_method(
        c=case.c,
        lam=case.lam,
        x0=x0,
        step_scale=case.step_scale,
        max_iter=case.max_iter,
    )

    print_history(history)

    x_star = soft_threshold(case.c, case.lam)
    obj_star = objective_l1_quadratic(x_star, case.c, case.lam)
    obj_best = objective_l1_quadratic(x_best, case.c, case.lam)
    obj_last = objective_l1_quadratic(x_last, case.c, case.lam)

    gap_best = obj_best - obj_star
    gap_last = obj_last - obj_star
    rel_gap_best = gap_best / (1.0 + abs(obj_star))

    err_best = float(np.linalg.norm(x_best - x_star))
    err_last = float(np.linalg.norm(x_last - x_star))

    sparsity_best = float(np.mean(np.abs(x_best) <= 1e-8))
    sparsity_star = float(np.mean(np.abs(x_star) <= 1e-12))

    print(f"objective(best iterate): {obj_best:.9e}")
    print(f"objective(last iterate): {obj_last:.9e}")
    print(f"objective(optimal closed-form): {obj_star:.9e}")
    print(f"best absolute objective gap: {gap_best:.9e}")
    print(f"last absolute objective gap: {gap_last:.9e}")
    print(f"best relative objective gap: {rel_gap_best:.9e}")
    print(f"||x_best - x_star||_2: {err_best:.9e}")
    print(f"||x_last - x_star||_2: {err_last:.9e}")
    print(f"sparsity ratio (best iterate): {sparsity_best:.3f}")
    print(f"sparsity ratio (closed-form optimum): {sparsity_star:.3f}")
    print(f"iterations used: {len(history)}")

    if not np.all(np.isfinite(x_best)):
        raise RuntimeError("x_best has non-finite values.")

    # The threshold is intentionally modest because plain subgradient descent
    # on non-smooth objectives converges sublinearly.
    if rel_gap_best > 3e-3:
        raise RuntimeError(
            f"Relative objective gap too large: {rel_gap_best:.6e} > 3e-3"
        )

    return {
        "rel_gap_best": float(rel_gap_best),
        "abs_gap_best": float(gap_best),
        "err_best": err_best,
        "iters": float(len(history)),
    }


def main() -> None:
    print("Subgradient Method MVP (MATH-0377)")
    print("Objective: 0.5*||x-c||^2 + lam*||x||_1")
    print("=" * 72)

    cases = build_cases()
    results = [run_case(case) for case in cases]

    max_rel_gap = max(item["rel_gap_best"] for item in results)
    avg_rel_gap = float(np.mean([item["rel_gap_best"] for item in results]))
    max_err = max(item["err_best"] for item in results)
    max_iters = int(max(item["iters"] for item in results))

    print("\n=== Summary ===")
    print(f"max best relative objective gap: {max_rel_gap:.9e}")
    print(f"avg best relative objective gap: {avg_rel_gap:.9e}")
    print(f"max ||x_best - x_star||_2: {max_err:.9e}")
    print(f"max iterations used: {max_iters}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
