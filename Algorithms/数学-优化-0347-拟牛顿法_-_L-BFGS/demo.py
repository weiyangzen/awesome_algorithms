"""L-BFGS: minimal runnable MVP.

This script implements a compact, auditable L-BFGS optimizer with
Armijo backtracking line search, then runs two deterministic demos:
1) Rosenbrock (non-convex, 2D)
2) SPD quadratic (convex, higher-dimensional)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float, int]


def ensure_vector(x: np.ndarray | List[float], name: str) -> np.ndarray:
    """Convert input to finite 1D float64 ndarray."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def lbfgs_two_loop(
    grad: np.ndarray,
    s_hist: List[np.ndarray],
    y_hist: List[np.ndarray],
    rho_hist: List[float],
) -> np.ndarray:
    """Compute descent direction p_k = -H_k * g_k via two-loop recursion."""
    q = grad.copy()
    alpha_vals: List[float] = []

    for s_i, y_i, rho_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        alpha_i = rho_i * float(np.dot(s_i, q))
        alpha_vals.append(alpha_i)
        q = q - alpha_i * y_i

    if s_hist:
        s_last = s_hist[-1]
        y_last = y_hist[-1]
        sy = float(np.dot(s_last, y_last))
        yy = float(np.dot(y_last, y_last))
        gamma = sy / yy if sy > 0.0 and yy > 0.0 else 1.0
    else:
        gamma = 1.0

    r = gamma * q

    for idx, (s_i, y_i, rho_i) in enumerate(zip(s_hist, y_hist, rho_hist)):
        alpha_i = alpha_vals[len(alpha_vals) - 1 - idx]
        beta_i = rho_i * float(np.dot(y_i, r))
        r = r + s_i * (alpha_i - beta_i)

    return -r


def armijo_backtracking(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    f_x: float,
    grad: np.ndarray,
    direction: np.ndarray,
    c1: float,
    shrink: float,
    min_step: float,
    max_backtracks: int,
) -> Tuple[float, np.ndarray, float, bool]:
    """Backtracking line search with Armijo condition."""
    step = 1.0
    directional_derivative = float(np.dot(grad, direction))

    for _ in range(max_backtracks):
        x_trial = x + step * direction
        f_trial = float(f(x_trial))
        if np.isfinite(f_trial) and f_trial <= f_x + c1 * step * directional_derivative:
            return step, x_trial, f_trial, True
        step *= shrink
        if step < min_step:
            break

    return 0.0, x.copy(), f_x, False


def lbfgs_optimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | List[float],
    m: int = 10,
    max_iter: int = 200,
    gtol: float = 1e-6,
    c1: float = 1e-4,
    line_search_shrink: float = 0.7,
    min_step: float = 1e-12,
    max_backtracks: int = 30,
) -> Dict[str, object]:
    """Optimize f(x) with a minimal L-BFGS implementation."""
    x = ensure_vector(x0, "x0").copy()

    if m <= 0:
        raise ValueError("m must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if gtol <= 0:
        raise ValueError("gtol must be > 0")
    if not (0.0 < c1 < 1.0):
        raise ValueError("c1 must be in (0, 1)")
    if not (0.0 < line_search_shrink < 1.0):
        raise ValueError("line_search_shrink must be in (0, 1)")
    if min_step <= 0:
        raise ValueError("min_step must be > 0")

    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []
    rho_hist: List[float] = []
    history: List[HistoryItem] = []

    message = "max_iter reached"
    converged = False

    f_x = float(f(x))
    g_x = ensure_vector(grad_f(x), "grad(x)")

    if not np.isfinite(f_x):
        raise RuntimeError("Initial f(x0) is non-finite")

    for k in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g_x, ord=2))
        if grad_norm <= gtol:
            converged = True
            message = "gradient norm below gtol"
            break

        direction = lbfgs_two_loop(g_x, s_hist, y_hist, rho_hist)
        directional_derivative = float(np.dot(g_x, direction))

        if directional_derivative >= -1e-16:
            direction = -g_x
            directional_derivative = -float(np.dot(g_x, g_x))

        step, x_next, f_next, ok = armijo_backtracking(
            f=f,
            x=x,
            f_x=f_x,
            grad=g_x,
            direction=direction,
            c1=c1,
            shrink=line_search_shrink,
            min_step=min_step,
            max_backtracks=max_backtracks,
        )

        if not ok:
            message = "line search failed"
            history.append((k, f_x, grad_norm, 0.0, len(s_hist)))
            break

        g_next = ensure_vector(grad_f(x_next), "grad(x_next)")
        if not np.all(np.isfinite(g_next)):
            message = "non-finite gradient encountered"
            history.append((k, f_next, grad_norm, step, len(s_hist)))
            break

        s_k = x_next - x
        y_k = g_next - g_x
        ys = float(np.dot(y_k, s_k))
        scale = max(1.0, float(np.linalg.norm(s_k) * np.linalg.norm(y_k)))
        curvature_eps = 1e-12 * scale

        if ys > curvature_eps:
            if len(s_hist) == m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_k)
            y_hist.append(y_k)
            rho_hist.append(1.0 / ys)

        next_grad_norm = float(np.linalg.norm(g_next, ord=2))
        history.append((k, f_next, next_grad_norm, step, len(s_hist)))

        x = x_next
        f_x = f_next
        g_x = g_next

    final_grad_norm = float(np.linalg.norm(g_x, ord=2))

    return {
        "x": x,
        "f": float(f_x),
        "grad_norm": final_grad_norm,
        "iterations": len(history),
        "converged": converged,
        "message": message,
        "history": history,
    }


def rosenbrock(x: np.ndarray) -> float:
    """2D Rosenbrock function."""
    x1, x2 = float(x[0]), float(x[1])
    return 100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of 2D Rosenbrock."""
    x1, x2 = float(x[0]), float(x[1])
    g1 = -400.0 * x1 * (x2 - x1 * x1) - 2.0 * (1.0 - x1)
    g2 = 200.0 * (x2 - x1 * x1)
    return np.asarray([g1, g2], dtype=np.float64)


def make_spd_quadratic_case(dim: int = 30, seed: int = 2026) -> Dict[str, object]:
    """Build a deterministic diagonal-SPD quadratic minimization case."""
    rng = np.random.default_rng(seed)
    diag = np.linspace(0.5, 8.0, dim, dtype=np.float64)
    b = rng.normal(size=dim)

    def f(x: np.ndarray) -> float:
        return 0.5 * float(np.dot(diag * x, x)) - float(np.dot(b, x))

    def grad_f(x: np.ndarray) -> np.ndarray:
        return diag * x - b

    x_star = b / diag
    return {
        "name": f"SPD Quadratic (dim={dim})",
        "f": f,
        "grad": grad_f,
        "x0": np.zeros(dim, dtype=np.float64),
        "x_ref": x_star,
    }


def relative_error_norm(x: np.ndarray, x_ref: np.ndarray) -> float:
    denom = max(1.0, float(np.linalg.norm(x_ref, ord=2)))
    return float(np.linalg.norm(x - x_ref, ord=2) / denom)


def print_history(history: List[HistoryItem], max_lines: int = 10) -> None:
    """Print compact optimization trajectory."""
    print("  iter | f(x)               | ||grad||           | step      | mem")
    print("  -----+--------------------+--------------------+-----------+-----")
    for k, f_val, g_norm, step, mem_size in history[:max_lines]:
        print(f"  {k:>4d} | {f_val:>18.10e} | {g_norm:>18.10e} | {step:>9.2e} | {mem_size:>3d}")
    if len(history) > max_lines:
        k, f_val, g_norm, step, mem_size = history[-1]
        print(f"  ... ({len(history) - max_lines} more iterations omitted)")
        print(f"  last | {f_val:>18.10e} | {g_norm:>18.10e} | {step:>9.2e} | {mem_size:>3d}")


def run_case(
    name: str,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    x_ref: np.ndarray,
    max_iter: int,
) -> Dict[str, object]:
    """Run one optimization case and print report."""
    print(f"\nCase: {name}")
    print(f"  dim={x0.size}, max_iter={max_iter}")

    result = lbfgs_optimize(
        f=f,
        grad_f=grad_f,
        x0=x0,
        m=10,
        max_iter=max_iter,
        gtol=1e-6,
        c1=1e-4,
        line_search_shrink=0.7,
        min_step=1e-12,
        max_backtracks=30,
    )

    history = result["history"]
    if not isinstance(history, list):
        raise RuntimeError("history format error")
    print_history(history, max_lines=10)

    x_est = ensure_vector(result["x"], "x_est")
    x_ref = ensure_vector(x_ref, "x_ref")
    abs_x_error = float(np.linalg.norm(x_est - x_ref, ord=2))
    rel_x_error = relative_error_norm(x_est, x_ref)

    print(f"  converged      = {result['converged']}")
    print(f"  message        = {result['message']}")
    print(f"  iterations     = {result['iterations']}")
    print(f"  final_f        = {result['f']:.6e}")
    print(f"  final_grad_norm= {result['grad_norm']:.6e}")
    print(f"  abs_x_error    = {abs_x_error:.6e}")
    print(f"  rel_x_error    = {rel_x_error:.6e}")

    return {
        "rel_x_error": rel_x_error,
        "final_grad_norm": float(result["grad_norm"]),
        "converged": bool(result["converged"]),
    }


def main() -> None:
    rosen_case = {
        "name": "Rosenbrock (2D)",
        "f": rosenbrock,
        "grad": rosenbrock_grad,
        "x0": np.asarray([-1.2, 1.0], dtype=np.float64),
        "x_ref": np.asarray([1.0, 1.0], dtype=np.float64),
        "max_iter": 250,
    }

    quad_case = make_spd_quadratic_case(dim=30, seed=2026)
    quad_case["max_iter"] = 200

    print("L-BFGS MVP")
    print("=" * 80)

    metrics = []
    metrics.append(
        run_case(
            name=str(rosen_case["name"]),
            f=rosen_case["f"],
            grad_f=rosen_case["grad"],
            x0=ensure_vector(rosen_case["x0"], "rosen_x0"),
            x_ref=ensure_vector(rosen_case["x_ref"], "rosen_x_ref"),
            max_iter=int(rosen_case["max_iter"]),
        )
    )

    metrics.append(
        run_case(
            name=str(quad_case["name"]),
            f=quad_case["f"],
            grad_f=quad_case["grad"],
            x0=ensure_vector(quad_case["x0"], "quad_x0"),
            x_ref=ensure_vector(quad_case["x_ref"], "quad_x_ref"),
            max_iter=int(quad_case["max_iter"]),
        )
    )

    rel_errs = np.asarray([m["rel_x_error"] for m in metrics], dtype=np.float64)
    grad_norms = np.asarray([m["final_grad_norm"] for m in metrics], dtype=np.float64)
    converged_all = bool(np.all([bool(m["converged"]) for m in metrics]))

    print("\nSummary")
    print("=" * 80)
    print(f"cases={len(metrics)}")
    print(f"max_rel_x_error={rel_errs.max():.6e}")
    print(f"mean_rel_x_error={rel_errs.mean():.6e}")
    print(f"max_final_grad_norm={grad_norms.max():.6e}")
    print(f"all_converged={converged_all}")


if __name__ == "__main__":
    main()
