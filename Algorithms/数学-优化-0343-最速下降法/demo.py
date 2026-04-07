"""Steepest descent (exact line search on SPD quadratic) MVP demo."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float, float]


def check_vector(name: str, x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_spd_matrix(a: np.ndarray) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be square 2D matrix, got shape={a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("A contains non-finite values.")
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("A must be symmetric for SPD quadratic objective.")
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A must be positive definite.") from exc


def objective(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x.T @ a @ x - b.T @ x)


def gradient(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return a @ x - b


def relative_error(error_norm: float, reference_norm: float, eps: float = 1e-15) -> float:
    return abs(error_norm) / (abs(reference_norm) + eps)


def steepest_descent_spd(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
    denominator_floor: float = 1e-18,
) -> Tuple[np.ndarray, List[HistoryItem]]:
    check_spd_matrix(a)
    check_vector("b", b)
    check_vector("x0", x0)
    if a.shape[0] != b.shape[0] or a.shape[0] != x0.shape[0]:
        raise ValueError(
            f"Shape mismatch: A={a.shape}, b={b.shape}, x0={x0.shape}."
        )
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be > 0.")

    x = x0.astype(float).copy()
    history: List[HistoryItem] = []

    b_norm = float(np.linalg.norm(b))
    grad_tol = tol * (1.0 + b_norm)

    for k in range(1, max_iter + 1):
        g = gradient(a, b, x)
        grad_norm = float(np.linalg.norm(g))
        if grad_norm <= grad_tol:
            history.append((k, objective(a, b, x), grad_norm, 0.0, 0.0))
            return x, history

        numerator = float(g.T @ g)
        denominator = float(g.T @ a @ g)
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            raise RuntimeError("Non-finite numerator/denominator encountered.")
        if denominator <= denominator_floor:
            raise RuntimeError(
                "Denominator too small in exact line search; cannot continue safely."
            )

        alpha = numerator / denominator
        x_next = x - alpha * g
        if not np.all(np.isfinite(x_next)):
            raise RuntimeError("Non-finite iterate encountered.")

        step_norm = float(np.linalg.norm(x_next - x))
        fx_next = objective(a, b, x_next)
        history.append((k, fx_next, grad_norm, alpha, step_norm))

        x = x_next
        if step_norm <= tol * (1.0 + float(np.linalg.norm(x))):
            return x, history

    raise RuntimeError(
        f"Steepest descent did not converge within max_iter={max_iter}."
    )


def print_history(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("iter | f(x_k)           | ||grad||         | alpha            | ||step||")
    print("-" * 74)
    for item in history[:max_lines]:
        k, fx, gnorm, alpha, snorm = item
        print(
            f"{k:4d} | {fx:16.9e} | {gnorm:16.9e} | {alpha:16.9e} | {snorm:16.9e}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_case(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")
    x_star, history = steepest_descent_spd(a=a, b=b, x0=x0, tol=tol, max_iter=max_iter)
    print_history(history)

    reference = np.linalg.solve(a, b)
    final_grad_norm = float(np.linalg.norm(gradient(a, b, x_star)))
    abs_error = float(np.linalg.norm(x_star - reference))
    rel_error = relative_error(abs_error, float(np.linalg.norm(reference)))
    f_star = objective(a, b, x_star)

    print(f"x* estimate: {x_star}")
    print(f"x* reference: {reference}")
    print(f"objective at estimate: {f_star:.9e}")
    print(f"absolute error (vector 2-norm): {abs_error:.9e}")
    print(f"relative error (against ||x_ref||): {rel_error:.9e}")
    print(f"final gradient norm: {final_grad_norm:.9e}")
    print(f"iterations used: {len(history)}")

    return {
        "iterations": float(len(history)),
        "abs_error": abs_error,
        "rel_error": rel_error,
        "final_grad_norm": final_grad_norm,
        "objective": f_star,
    }


def main() -> None:
    tol = 1e-10
    max_iter = 5000

    cases = [
        {
            "name": "2D SPD quadratic",
            "A": np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float),
            "b": np.array([1.0, 2.0], dtype=float),
            "x0": np.array([2.0, 2.0], dtype=float),
        },
        {
            "name": "3D SPD quadratic",
            "A": np.array(
                [[10.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]], dtype=float
            ),
            "b": np.array([3.0, -1.0, 2.0], dtype=float),
            "x0": np.array([0.0, 0.0, 0.0], dtype=float),
        },
        {
            "name": "Ill-conditioned diagonal SPD",
            "A": np.array([[1.0, 0.0], [0.0, 80.0]], dtype=float),
            "b": np.array([1.0, 1.0], dtype=float),
            "x0": np.array([-3.0, 2.0], dtype=float),
        },
    ]

    results = []
    for case in cases:
        result = run_case(
            name=case["name"],
            a=case["A"],
            b=case["b"],
            x0=case["x0"],
            tol=tol,
            max_iter=max_iter,
        )
        results.append(result)

    max_rel_error = max(item["rel_error"] for item in results)
    avg_rel_error = float(np.mean([item["rel_error"] for item in results]))
    max_grad_norm = max(item["final_grad_norm"] for item in results)
    pass_flag = max_rel_error < 1e-8 and max_grad_norm < 1e-8

    print("\n=== Summary ===")
    print(f"max relative error: {max_rel_error:.9e}")
    print(f"avg relative error: {avg_rel_error:.9e}")
    print(f"max final gradient norm: {max_grad_norm:.9e}")
    print(f"all cases pass strict check: {pass_flag}")


if __name__ == "__main__":
    main()
