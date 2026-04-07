"""Nesterov accelerated gradient MVP for strongly convex quadratic problems.

Assigned item: MATH-0399 Nesterov加速梯度

We solve a diagonal strongly-convex quadratic:
    f(x) = 0.5 * sum_i curvature_i * (x_i - target_i)^2
where curvature_i > 0.

This keeps the implementation transparent while preserving the core behavior of
Nesterov Acceleration:
- Gradient descent uses x_{k+1} = x_k - (1/L) * grad(x_k)
- Nesterov (strongly convex form) uses
      y_k = x_k + beta * (x_k - x_{k-1})
      x_{k+1} = y_k - (1/L) * grad(y_k)
  with beta = (sqrt(L)-sqrt(mu)) / (sqrt(L)+sqrt(mu)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float]


@dataclass
class CaseConfig:
    """Configuration of one deterministic benchmark case."""

    name: str
    seed: int
    n: int
    mu: float
    L: float
    max_iter: int


@dataclass
class MethodResult:
    """Optimization traces for one method."""

    x_final: np.ndarray
    history: List[HistoryItem]


def validate_problem(curvature: np.ndarray, target: np.ndarray) -> None:
    """Validate shape, finiteness, and positive curvature."""
    if curvature.ndim != 1 or target.ndim != 1:
        raise ValueError("curvature and target must be 1D vectors.")
    if curvature.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: curvature={curvature.shape}, target={target.shape}."
        )
    if not np.all(np.isfinite(curvature)) or not np.all(np.isfinite(target)):
        raise ValueError("curvature/target contains non-finite values.")
    if float(np.min(curvature)) <= 0.0:
        raise ValueError("curvature must be strictly positive.")


def objective(x: np.ndarray, curvature: np.ndarray, target: np.ndarray) -> float:
    """Compute f(x) = 0.5 * sum_i curvature_i * (x_i - target_i)^2."""
    diff = x - target
    return 0.5 * float(np.sum(curvature * diff * diff))


def gradient(x: np.ndarray, curvature: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute gradient of the diagonal quadratic objective."""
    return curvature * (x - target)


def nesterov_beta_strongly_convex(mu: float, L: float) -> float:
    """Return Nesterov momentum coefficient for strongly-convex quadratics."""
    if mu <= 0.0 or L <= 0.0 or mu > L:
        raise ValueError(f"Require 0 < mu <= L, got mu={mu}, L={L}.")
    root_mu = float(np.sqrt(mu))
    root_L = float(np.sqrt(L))
    return (root_L - root_mu) / (root_L + root_mu)


def run_gradient_descent(
    curvature: np.ndarray,
    target: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float = 1e-12,
) -> MethodResult:
    """Run vanilla gradient descent with fixed step size 1/L."""
    validate_problem(curvature, target)
    if x0.shape != target.shape:
        raise ValueError(f"x0 shape mismatch: x0={x0.shape}, target={target.shape}.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    L = float(np.max(curvature))
    step = 1.0 / L

    x = x0.astype(float).copy()
    history: List[HistoryItem] = []

    for k in range(1, max_iter + 1):
        grad_x = gradient(x, curvature, target)
        x_next = x - step * grad_x

        obj = objective(x_next, curvature, target)
        grad_norm = float(np.linalg.norm(grad_x))
        step_norm = float(np.linalg.norm(x_next - x))

        if not np.isfinite(obj) or not np.isfinite(grad_norm) or not np.isfinite(step_norm):
            raise RuntimeError("Non-finite value encountered in gradient descent.")

        history.append((k, obj, grad_norm, step_norm))

        if step_norm <= tol * (1.0 + float(np.linalg.norm(x_next))):
            return MethodResult(x_final=x_next, history=history)

        x = x_next

    return MethodResult(x_final=x, history=history)


def run_nesterov_accelerated_gradient(
    curvature: np.ndarray,
    target: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float = 1e-12,
) -> MethodResult:
    """Run strongly-convex Nesterov accelerated gradient (constant momentum)."""
    validate_problem(curvature, target)
    if x0.shape != target.shape:
        raise ValueError(f"x0 shape mismatch: x0={x0.shape}, target={target.shape}.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    mu = float(np.min(curvature))
    L = float(np.max(curvature))
    step = 1.0 / L
    beta = nesterov_beta_strongly_convex(mu, L)

    x_prev = x0.astype(float).copy()
    x = x0.astype(float).copy()
    history: List[HistoryItem] = []

    for k in range(1, max_iter + 1):
        y = x + beta * (x - x_prev)
        grad_y = gradient(y, curvature, target)
        x_next = y - step * grad_y

        obj = objective(x_next, curvature, target)
        grad_norm = float(np.linalg.norm(grad_y))
        step_norm = float(np.linalg.norm(x_next - x))

        if not np.isfinite(obj) or not np.isfinite(grad_norm) or not np.isfinite(step_norm):
            raise RuntimeError("Non-finite value encountered in Nesterov AG.")

        history.append((k, obj, grad_norm, step_norm))

        if step_norm <= tol * (1.0 + float(np.linalg.norm(x_next))):
            return MethodResult(x_final=x_next, history=history)

        x_prev, x = x, x_next

    return MethodResult(x_final=x, history=history)


def build_problem(case: CaseConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Build diagonal quadratic data with controlled condition number."""
    if case.mu <= 0.0:
        raise ValueError(f"mu must be > 0, got {case.mu}.")
    if case.L <= 0.0 or case.L < case.mu:
        raise ValueError(f"Require L >= mu > 0, got mu={case.mu}, L={case.L}.")
    if case.n <= 0:
        raise ValueError(f"n must be > 0, got {case.n}.")

    rng = np.random.default_rng(case.seed)
    curvature = np.geomspace(case.mu, case.L, num=case.n).astype(float)
    rng.shuffle(curvature)
    target = rng.normal(loc=0.0, scale=1.0, size=case.n).astype(float)
    validate_problem(curvature, target)
    return curvature, target


def print_history_preview(tag: str, history: Sequence[HistoryItem], max_lines: int = 6) -> None:
    """Print first and last few history records for readability."""
    print(f"[{tag}] iter | objective        | ||grad||          | ||step||")
    print(f"[{tag}] " + "-" * 66)

    if not history:
        print(f"[{tag}] (empty history)")
        return

    head = history[:max_lines]
    tail = history[-max_lines:] if len(history) > max_lines else []

    for k, obj, grad_norm, step_norm in head:
        print(f"[{tag}] {k:4d} | {obj:16.9e} | {grad_norm:16.9e} | {step_norm:16.9e}")

    if tail:
        omitted = len(history) - (len(head) + len(tail))
        if omitted > 0:
            print(f"[{tag}] ... ({omitted} iterations omitted)")
        for k, obj, grad_norm, step_norm in tail:
            print(f"[{tag}] {k:4d} | {obj:16.9e} | {grad_norm:16.9e} | {step_norm:16.9e}")


def build_cases() -> List[CaseConfig]:
    """Create deterministic cases with increasing condition numbers."""
    return [
        CaseConfig(
            name="Moderate condition number",
            seed=3991,
            n=32,
            mu=0.05,
            L=25.0,
            max_iter=120,
        ),
        CaseConfig(
            name="High condition number",
            seed=3992,
            n=64,
            mu=0.01,
            L=40.0,
            max_iter=220,
        ),
        CaseConfig(
            name="Very high condition number",
            seed=3993,
            n=96,
            mu=0.005,
            L=60.0,
            max_iter=320,
        ),
    ]


def run_case(case: CaseConfig) -> Dict[str, float]:
    """Execute one case and perform internal correctness checks."""
    print(f"\n=== Case: {case.name} ===")
    curvature, target = build_problem(case)

    mu = float(np.min(curvature))
    L = float(np.max(curvature))
    beta = nesterov_beta_strongly_convex(mu, L)
    kappa = L / mu

    print(f"dimension: {case.n}")
    print(f"mu: {mu:.6e}, L: {L:.6e}, condition number(L/mu): {kappa:.3f}")
    print(f"Nesterov beta: {beta:.9f}")
    print(f"iterations budget: {case.max_iter}")

    x0 = np.zeros_like(target)
    f_star = 0.0

    gd_res = run_gradient_descent(curvature, target, x0, max_iter=case.max_iter)
    nag_res = run_nesterov_accelerated_gradient(
        curvature, target, x0, max_iter=case.max_iter
    )

    gd_obj = objective(gd_res.x_final, curvature, target)
    nag_obj = objective(nag_res.x_final, curvature, target)

    gd_rel_gap = gd_obj / (1.0 + abs(f_star))
    nag_rel_gap = nag_obj / (1.0 + abs(f_star))

    gd_err = float(np.linalg.norm(gd_res.x_final - target))
    nag_err = float(np.linalg.norm(nag_res.x_final - target))

    print_history_preview("GD", gd_res.history)
    print_history_preview("NAG", nag_res.history)

    print(f"final objective (GD):  {gd_obj:.9e}")
    print(f"final objective (NAG): {nag_obj:.9e}")
    print(f"relative gap (GD):      {gd_rel_gap:.9e}")
    print(f"relative gap (NAG):     {nag_rel_gap:.9e}")
    print(f"||x_gd - x*||_2:        {gd_err:.9e}")
    print(f"||x_nag - x*||_2:       {nag_err:.9e}")

    if not np.all(np.isfinite(gd_res.x_final)) or not np.all(np.isfinite(nag_res.x_final)):
        raise RuntimeError("Detected non-finite values in final solution.")

    if nag_rel_gap > 5e-3:
        raise RuntimeError(
            f"NAG relative objective gap too large: {nag_rel_gap:.6e} > 5e-3"
        )

    if not (nag_rel_gap <= 0.05 * gd_rel_gap):
        raise RuntimeError(
            "NAG did not significantly beat GD in this benchmark: "
            f"nag_rel_gap={nag_rel_gap:.6e}, gd_rel_gap={gd_rel_gap:.6e}."
        )

    return {
        "gd_rel_gap": float(gd_rel_gap),
        "nag_rel_gap": float(nag_rel_gap),
        "speedup_ratio": float(gd_rel_gap / max(nag_rel_gap, 1e-18)),
        "gd_iters": float(len(gd_res.history)),
        "nag_iters": float(len(nag_res.history)),
        "gd_err": float(gd_err),
        "nag_err": float(nag_err),
    }


def main() -> None:
    print("Nesterov Accelerated Gradient MVP (MATH-0399)")
    print("Objective: f(x)=0.5*sum_i curvature_i*(x_i-target_i)^2")
    print("Comparison: Gradient Descent vs Nesterov AG")
    print("=" * 76)

    cases = build_cases()
    metrics = [run_case(case) for case in cases]

    max_nag_gap = max(item["nag_rel_gap"] for item in metrics)
    max_gd_gap = max(item["gd_rel_gap"] for item in metrics)
    min_speedup = min(item["speedup_ratio"] for item in metrics)
    max_nag_err = max(item["nag_err"] for item in metrics)
    max_gd_err = max(item["gd_err"] for item in metrics)

    print("\n=== Summary ===")
    print(f"max relative gap (GD):  {max_gd_gap:.9e}")
    print(f"max relative gap (NAG): {max_nag_gap:.9e}")
    print(f"minimum GD/NAG gap speedup ratio: {min_speedup:.3f}x")
    print(f"max ||x_gd - x*||_2:  {max_gd_err:.9e}")
    print(f"max ||x_nag - x*||_2: {max_nag_err:.9e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
