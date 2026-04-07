"""Trust-region method MVP (dogleg subproblem solver)."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, float, float, float, int]


def check_vector(name: str, x: Array) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def predicted_reduction(g: Array, b_mat: Array, p: Array) -> float:
    return float(-(g.T @ p + 0.5 * p.T @ b_mat @ p))


def dogleg_step(g: Array, b_mat: Array, delta: float, eps: float = 1e-14) -> Array:
    g_norm = float(np.linalg.norm(g))
    if g_norm <= eps:
        return np.zeros_like(g)

    b_mat = 0.5 * (b_mat + b_mat.T)
    bg = b_mat @ g
    gbg = float(g.T @ bg)

    if gbg <= eps:
        return -(delta / g_norm) * g

    # Cauchy (steepest descent) point on quadratic model.
    alpha_u = float((g.T @ g) / gbg)
    p_u = -alpha_u * g
    p_u_norm = float(np.linalg.norm(p_u))
    if p_u_norm >= delta:
        return -(delta / g_norm) * g

    # Full Newton step (when linear solve is stable).
    p_b: Array | None = None
    try:
        p_newton = -np.linalg.solve(b_mat, g)
        if np.all(np.isfinite(p_newton)):
            p_b = p_newton
    except np.linalg.LinAlgError:
        p_b = None

    if p_b is None:
        return p_u

    p_b_norm = float(np.linalg.norm(p_b))
    if p_b_norm <= delta:
        return p_b

    # Interpolate on the dogleg segment p_u -> p_b to hit ||p|| = delta.
    d = p_b - p_u
    a = float(d.T @ d)
    b = float(2.0 * p_u.T @ d)
    c = float(p_u.T @ p_u - delta * delta)
    disc = b * b - 4.0 * a * c
    disc = max(disc, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    tau1 = (-b + sqrt_disc) / (2.0 * a)
    tau2 = (-b - sqrt_disc) / (2.0 * a)

    tau_candidates = [tau for tau in (tau1, tau2) if 0.0 <= tau <= 1.0]
    if tau_candidates:
        tau = min(tau_candidates, key=lambda t: abs(t - 1.0))
    else:
        tau = float(np.clip(tau1, 0.0, 1.0))

    return p_u + tau * d


def trust_region_dogleg(
    fun: Callable[[Array], float],
    grad: Callable[[Array], Array],
    hess: Callable[[Array], Array],
    x0: Array,
    delta0: float = 1.0,
    delta_max: float = 100.0,
    eta: float = 0.1,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Tuple[Array, List[HistoryItem]]:
    check_vector("x0", x0)
    if delta0 <= 0.0 or delta_max <= 0.0 or delta0 > delta_max:
        raise ValueError("Require 0 < delta0 <= delta_max.")
    if not (0.0 < eta < 1.0):
        raise ValueError("eta must be in (0, 1).")
    if tol <= 0.0 or max_iter <= 0:
        raise ValueError("tol/max_iter must be positive.")

    x = x0.astype(float).copy()
    delta = float(delta0)
    history: List[HistoryItem] = []

    for k in range(1, max_iter + 1):
        fx = float(fun(x))
        g = grad(x)
        check_vector("grad(x)", g)
        if g.shape != x.shape:
            raise ValueError("grad(x) shape mismatch with x.")
        g_norm = float(np.linalg.norm(g))
        if g_norm <= tol:
            history.append((k, fx, g_norm, 0.0, delta, np.inf, 1))
            return x, history

        b_mat = hess(x)
        if b_mat.ndim != 2 or b_mat.shape != (x.size, x.size):
            raise ValueError(f"hess(x) must be square matrix of shape {(x.size, x.size)}")
        if not np.all(np.isfinite(b_mat)):
            raise ValueError("hess(x) contains non-finite values.")

        p = dogleg_step(g, b_mat, delta=delta)
        p_norm = float(np.linalg.norm(p))
        if p_norm <= 1e-15:
            history.append((k, fx, g_norm, p_norm, delta, -np.inf, 0))
            return x, history

        pred = predicted_reduction(g, b_mat, p)
        if pred <= 1e-16:
            p = -(delta / max(g_norm, 1e-15)) * g
            p_norm = float(np.linalg.norm(p))
            pred = predicted_reduction(g, b_mat, p)
        if pred <= 1e-16:
            raise RuntimeError("Predicted reduction is non-positive; cannot continue safely.")

        x_trial = x + p
        f_trial = float(fun(x_trial))
        ared = fx - f_trial
        rho = ared / pred

        accept = int(rho > eta)
        if accept:
            x = x_trial
            fx = f_trial

        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(p_norm - delta) <= 1e-10 * max(1.0, delta):
            delta = min(2.0 * delta, delta_max)
        delta = float(np.clip(delta, 1e-12, delta_max))

        history.append((k, fx, g_norm, p_norm, delta, float(rho), accept))

    raise RuntimeError(f"Trust-region method did not converge within max_iter={max_iter}.")


def rosenbrock_fun(x: Array) -> float:
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_grad(x: Array) -> Array:
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def rosenbrock_hess(x: Array) -> Array:
    return np.array(
        [
            [2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0]],
            [-400.0 * x[0], 200.0],
        ],
        dtype=float,
    )


def make_quadratic_case(
    q: Array,
    b: Array,
) -> Tuple[Callable[[Array], float], Callable[[Array], Array], Callable[[Array], Array]]:
    def fun(x: Array) -> float:
        return float(0.5 * x.T @ q @ x - b.T @ x)

    def grad(x: Array) -> Array:
        return q @ x - b

    def hess(_: Array) -> Array:
        return q

    return fun, grad, hess


def print_history(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("iter | f(x_k)           | ||g||            | ||p||            | delta            | rho              | acc")
    print("-" * 108)
    for row in history[:max_lines]:
        k, fx, g_norm, p_norm, delta, rho, acc = row
        print(
            f"{k:4d} | {fx:16.9e} | {g_norm:16.9e} | {p_norm:16.9e} | "
            f"{delta:16.9e} | {rho:16.9e} | {acc:d}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def relative_error(error_norm: float, reference_norm: float, eps: float = 1e-15) -> float:
    return abs(error_norm) / (abs(reference_norm) + eps)


def run_case(
    name: str,
    fun: Callable[[Array], float],
    grad: Callable[[Array], Array],
    hess: Callable[[Array], Array],
    x0: Array,
    reference: Array,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")
    x_star, history = trust_region_dogleg(
        fun=fun,
        grad=grad,
        hess=hess,
        x0=x0,
        delta0=1.0,
        delta_max=100.0,
        eta=0.1,
        tol=tol,
        max_iter=max_iter,
    )
    print_history(history)
    gnorm = float(np.linalg.norm(grad(x_star)))
    abs_err = float(np.linalg.norm(x_star - reference))
    rel_err = relative_error(abs_err, float(np.linalg.norm(reference)))
    print(f"x* estimate:  {x_star}")
    print(f"x* reference: {reference}")
    print(f"f(x*): {fun(x_star):.9e}")
    print(f"final ||g||: {gnorm:.9e}")
    print(f"absolute error: {abs_err:.9e}")
    print(f"relative error: {rel_err:.9e}")
    print(f"iterations used: {len(history)}")
    return {
        "iterations": float(len(history)),
        "final_grad_norm": gnorm,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }


def main() -> None:
    q = np.array([[30.0, 0.0], [0.0, 1.0]], dtype=float)
    b = np.array([3.0, 2.0], dtype=float)
    quad_fun, quad_grad, quad_hess = make_quadratic_case(q=q, b=b)
    quad_ref = np.linalg.solve(q, b)

    cases = [
        {
            "name": "Rosenbrock (non-convex)",
            "fun": rosenbrock_fun,
            "grad": rosenbrock_grad,
            "hess": rosenbrock_hess,
            "x0": np.array([-1.2, 1.0], dtype=float),
            "reference": np.array([1.0, 1.0], dtype=float),
        },
        {
            "name": "Ill-conditioned quadratic",
            "fun": quad_fun,
            "grad": quad_grad,
            "hess": quad_hess,
            "x0": np.array([4.0, -3.0], dtype=float),
            "reference": quad_ref,
        },
    ]

    results = []
    for case in cases:
        result = run_case(
            name=case["name"],
            fun=case["fun"],
            grad=case["grad"],
            hess=case["hess"],
            x0=case["x0"],
            reference=case["reference"],
        )
        results.append(result)

    max_rel_error = max(item["rel_error"] for item in results)
    max_grad_norm = max(item["final_grad_norm"] for item in results)
    avg_iters = float(np.mean([item["iterations"] for item in results]))
    pass_flag = bool(max_rel_error < 1e-6 and max_grad_norm < 1e-6)

    print("\n=== Summary ===")
    print(f"max relative error: {max_rel_error:.9e}")
    print(f"max final gradient norm: {max_grad_norm:.9e}")
    print(f"average iterations: {avg_iters:.2f}")
    print(f"all cases pass strict check: {pass_flag}")


if __name__ == "__main__":
    main()
