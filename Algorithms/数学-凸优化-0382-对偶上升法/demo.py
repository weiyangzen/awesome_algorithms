"""Dual ascent method MVP for equality-constrained convex quadratic programs."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float, float]


def check_vector(name: str, x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_matrix(name: str, x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_spd_matrix(q: np.ndarray) -> None:
    check_matrix("Q", q)
    if q.shape[0] != q.shape[1]:
        raise ValueError(f"Q must be square, got shape={q.shape}.")
    if not np.allclose(q, q.T, atol=1e-12):
        raise ValueError("Q must be symmetric.")
    try:
        np.linalg.cholesky(q)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Q must be positive definite.") from exc


def primal_objective(q: np.ndarray, c: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x.T @ q @ x + c.T @ x)


def primal_minimizer_given_lambda(
    q_inv: np.ndarray, c: np.ndarray, a: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    return -q_inv @ (c + a.T @ lam)


def lagrangian_value(
    q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
) -> float:
    residual = a @ x - b
    return primal_objective(q, c, x) + float(lam.T @ residual)


def relative_error(error_norm: float, reference_norm: float, eps: float = 1e-15) -> float:
    return abs(error_norm) / (abs(reference_norm) + eps)


def estimate_dual_gradient_lipschitz(a: np.ndarray, q_inv: np.ndarray) -> float:
    hessian_magnitude = a @ q_inv @ a.T
    eigvals = np.linalg.eigvalsh(hessian_magnitude)
    l_value = float(np.max(eigvals))
    if l_value <= 0.0:
        raise ValueError("Dual gradient Lipschitz constant must be positive.")
    return l_value


def solve_kkt_reference(
    q: np.ndarray, c: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = q.shape[0]
    m = a.shape[0]
    kkt = np.block([[q, a.T], [a, np.zeros((m, m), dtype=float)]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n], sol[n:]


def dual_ascent_equality_qp(
    q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lam0: np.ndarray | None = None,
    step_size: float | None = None,
    step_scale: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, List[HistoryItem], float]:
    check_spd_matrix(q)
    check_vector("c", c)
    check_matrix("A", a)
    check_vector("b", b)

    n = q.shape[0]
    m = a.shape[0]
    if c.shape[0] != n:
        raise ValueError(f"Shape mismatch: Q={q.shape}, c={c.shape}.")
    if a.shape[1] != n:
        raise ValueError(f"Shape mismatch: A={a.shape}, Q={q.shape}.")
    if b.shape[0] != m:
        raise ValueError(f"Shape mismatch: A={a.shape}, b={b.shape}.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be > 0.")

    q_inv = np.linalg.inv(q)

    if lam0 is None:
        lam = np.zeros(m, dtype=float)
    else:
        check_vector("lam0", lam0)
        if lam0.shape[0] != m:
            raise ValueError(f"lam0 must have length={m}, got {lam0.shape[0]}.")
        lam = lam0.astype(float).copy()

    if step_size is None:
        l_value = estimate_dual_gradient_lipschitz(a, q_inv)
        step_size = step_scale / l_value
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0.")

    history: List[HistoryItem] = []
    residual_tol = tol * (1.0 + float(np.linalg.norm(b)))

    for k in range(1, max_iter + 1):
        x = primal_minimizer_given_lambda(q_inv, c, a, lam)
        residual = a @ x - b
        residual_norm = float(np.linalg.norm(residual))
        p_obj = primal_objective(q, c, x)
        d_obj = lagrangian_value(q, c, a, b, x, lam)

        if residual_norm <= residual_tol:
            history.append((k, p_obj, d_obj, residual_norm, 0.0))
            return x, lam, history, float(step_size)

        lam_next = lam + step_size * residual
        if not np.all(np.isfinite(lam_next)):
            raise RuntimeError("Non-finite dual iterate encountered.")
        dual_step_norm = float(np.linalg.norm(lam_next - lam))
        history.append((k, p_obj, d_obj, residual_norm, dual_step_norm))
        lam = lam_next

    raise RuntimeError(f"Dual ascent did not converge within max_iter={max_iter}.")


def print_history(history: Sequence[HistoryItem], max_lines: int = 14) -> None:
    print(
        "iter | primal_obj       | dual_obj         | ||A x - b||      | ||delta_lambda||"
    )
    print("-" * 89)
    for item in history[:max_lines]:
        k, p_obj, d_obj, residual_norm, dual_step_norm = item
        print(
            f"{k:4d} | {p_obj:16.9e} | {d_obj:16.9e} | "
            f"{residual_norm:16.9e} | {dual_step_norm:16.9e}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_case(
    name: str,
    q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    step_scale: float,
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")
    x_est, lam_est, history, alpha = dual_ascent_equality_qp(
        q=q,
        c=c,
        a=a,
        b=b,
        tol=tol,
        max_iter=max_iter,
        step_scale=step_scale,
    )
    print(f"step size alpha: {alpha:.9e}")
    print_history(history)

    x_ref, lam_ref = solve_kkt_reference(q, c, a, b)
    primal_gap = abs(primal_objective(q, c, x_est) - primal_objective(q, c, x_ref))
    residual_norm = float(np.linalg.norm(a @ x_est - b))
    x_abs_error = float(np.linalg.norm(x_est - x_ref))
    x_rel_error = relative_error(x_abs_error, float(np.linalg.norm(x_ref)))
    lam_abs_error = float(np.linalg.norm(lam_est - lam_ref))
    lam_rel_error = relative_error(lam_abs_error, float(np.linalg.norm(lam_ref)))
    duality_gap_abs = abs(
        float(primal_objective(q, c, x_est) - lagrangian_value(q, c, a, b, x_est, lam_est))
    )

    print(f"x* estimate:   {x_est}")
    print(f"x* reference:  {x_ref}")
    print(f"lambda estimate:  {lam_est}")
    print(f"lambda reference: {lam_ref}")
    print(f"primal objective (estimate): {primal_objective(q, c, x_est):.9e}")
    print(f"primal objective (reference): {primal_objective(q, c, x_ref):.9e}")
    print(f"primal objective gap: {primal_gap:.9e}")
    print(f"constraint residual ||A x - b||: {residual_norm:.9e}")
    print(f"x relative error: {x_rel_error:.9e}")
    print(f"lambda relative error: {lam_rel_error:.9e}")
    print(f"absolute duality gap surrogate: {duality_gap_abs:.9e}")
    print(f"iterations used: {len(history)}")

    return {
        "iterations": float(len(history)),
        "residual_norm": residual_norm,
        "x_rel_error": x_rel_error,
        "lam_rel_error": lam_rel_error,
        "primal_gap": primal_gap,
        "duality_gap_abs": duality_gap_abs,
    }


def main() -> None:
    tol = 1e-10
    max_iter = 10000
    step_scale = 1.0

    cases = [
        {
            "name": "3 variables / 2 equality constraints",
            "Q": np.array(
                [[4.0, 1.0, 0.0], [1.0, 3.0, 0.2], [0.0, 0.2, 2.0]],
                dtype=float,
            ),
            "c": np.array([-8.0, -3.0, 1.0], dtype=float),
            "A": np.array([[1.0, 1.0, 1.0], [2.0, -1.0, 0.5]], dtype=float),
            "b": np.array([1.0, 0.5], dtype=float),
        },
        {
            "name": "4 variables / 2 equality constraints",
            "Q": np.array(
                [
                    [6.0, 1.0, 0.0, 0.0],
                    [1.0, 5.0, 1.0, 0.0],
                    [0.0, 1.0, 4.0, 0.5],
                    [0.0, 0.0, 0.5, 3.5],
                ],
                dtype=float,
            ),
            "c": np.array([-3.0, 2.0, -1.0, 4.0], dtype=float),
            "A": np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, -1.0, 1.0]], dtype=float),
            "b": np.array([0.5, -1.5], dtype=float),
        },
    ]

    results = []
    for case in cases:
        result = run_case(
            name=case["name"],
            q=case["Q"],
            c=case["c"],
            a=case["A"],
            b=case["b"],
            tol=tol,
            max_iter=max_iter,
            step_scale=step_scale,
        )
        results.append(result)

    max_residual = max(item["residual_norm"] for item in results)
    max_x_rel_error = max(item["x_rel_error"] for item in results)
    max_lam_rel_error = max(item["lam_rel_error"] for item in results)
    max_primal_gap = max(item["primal_gap"] for item in results)
    max_duality_gap_abs = max(item["duality_gap_abs"] for item in results)
    avg_iterations = float(np.mean([item["iterations"] for item in results]))

    pass_flag = (
        max_residual < 1e-8
        and max_x_rel_error < 1e-8
        and max_lam_rel_error < 1e-8
        and max_primal_gap < 1e-8
        and max_duality_gap_abs < 1e-8
    )

    print("\n=== Summary ===")
    print(f"max residual norm: {max_residual:.9e}")
    print(f"max x relative error: {max_x_rel_error:.9e}")
    print(f"max lambda relative error: {max_lam_rel_error:.9e}")
    print(f"max primal objective gap: {max_primal_gap:.9e}")
    print(f"max absolute duality gap surrogate: {max_duality_gap_abs:.9e}")
    print(f"average iterations: {avg_iterations:.2f}")
    print(f"all cases pass strict check: {pass_flag}")


if __name__ == "__main__":
    main()
