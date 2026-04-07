"""Bundle Method MVP for nonsmooth convex optimization.

This script implements a proximal bundle method (serious/null step variant)
on a 2D convex nonsmooth objective:

    f(x) = max_i (a_i^T x + b_i) + 0.5 * mu * ||x||^2

The max-affine part creates kinks and the quadratic term makes the solution
unique at x*=0 in this setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Cut:
    """Linearization cut: f(y) + g(y)^T (x - y)."""

    y: np.ndarray
    f: float
    g: np.ndarray


def build_problem() -> Tuple[np.ndarray, np.ndarray, float]:
    """Create a max-affine + quadratic convex problem instance."""
    # First row is the zero affine term; others come in ± pairs.
    a_mat = np.array(
        [
            [0.0, 0.0],
            [1.6, 0.2],
            [-1.6, -0.2],
            [0.3, 1.5],
            [-0.3, -1.5],
            [1.0, -1.1],
            [-1.0, 1.1],
        ],
        dtype=float,
    )
    # Negative offsets keep x=0 as unique minimizer once the quadratic term is added.
    b_vec = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float)
    mu = 0.2
    return a_mat, b_vec, mu


def oracle(x: np.ndarray, a_mat: np.ndarray, b_vec: np.ndarray, mu: float) -> Tuple[float, np.ndarray]:
    """Return objective value and one valid subgradient."""
    affine_vals = a_mat @ x + b_vec
    active = int(np.argmax(affine_vals))
    max_part = float(affine_vals[active])
    f_val = max_part + 0.5 * mu * float(x @ x)
    g_val = a_mat[active] + mu * x
    return f_val, g_val


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    if v.ndim != 1:
        raise ValueError("Input for simplex projection must be a 1D array.")
    n = v.size
    if n == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u - (cssv - 1.0) / (np.arange(n) + 1.0)
    rho = np.where(rho_candidates > 0.0)[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho_idx = int(rho[-1])
        theta = (cssv[rho_idx] - 1.0) / (rho_idx + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_bundle_subproblem(
    cuts: List[Cut],
    center: np.ndarray,
    prox_t: float,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-10,
) -> Tuple[np.ndarray, float, float]:
    """Solve min_x m(x) + (1/(2*prox_t))*||x-center||^2 via dual PGD on simplex.

    Model m(x) = max_i g_i^T x + beta_i, where beta_i = f_i - g_i^T y_i.
    """
    g_mat = np.vstack([c.g for c in cuts])  # (m, d)
    y_mat = np.vstack([c.y for c in cuts])  # (m, d)
    f_vec = np.array([c.f for c in cuts], dtype=float)  # (m,)
    beta = f_vec - np.sum(g_mat * y_mat, axis=1)  # (m,)
    s = g_mat @ center + beta  # (m,)

    m_cuts = g_mat.shape[0]
    alpha = np.full(m_cuts, 1.0 / m_cuts, dtype=float)

    # Gradient-Lipschitz constant of dual objective:
    # grad = s - prox_t * G G^T alpha
    # L = prox_t * ||G G^T||_2 <= prox_t * ||G||_2^2
    spectral = float(np.linalg.norm(g_mat, ord=2) ** 2)
    lipschitz = max(prox_t * spectral, 1e-12)
    step = 1.0 / lipschitz

    for _ in range(inner_max_iter):
        g_alpha = g_mat.T @ alpha  # (d,)
        grad = s - prox_t * (g_mat @ g_alpha)
        alpha_next = project_to_simplex(alpha + step * grad)
        if np.linalg.norm(alpha_next - alpha) <= inner_tol:
            alpha = alpha_next
            break
        alpha = alpha_next

    x_trial = center - prox_t * (g_mat.T @ alpha)
    model_val = float(np.max(g_mat @ x_trial + beta))
    prox_val = 0.5 / prox_t * float(np.sum((x_trial - center) ** 2))
    subproblem_val = model_val + prox_val
    return x_trial, model_val, subproblem_val


def prune_bundle(cuts: List[Cut], center_cut: Cut, bundle_cap: int) -> List[Cut]:
    """Keep center cut and recent cuts, with bounded memory."""
    if bundle_cap < 2:
        raise ValueError("bundle_cap must be >= 2.")

    survivors: List[Cut] = []
    for c in cuts:
        if not np.allclose(c.y, center_cut.y, atol=1e-12):
            survivors.append(c)
    survivors = survivors[-(bundle_cap - 1) :]
    return [center_cut] + survivors


def bundle_method(
    x0: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    mu: float,
    max_iter: int = 100,
    prox_t: float = 1.5,
    gamma: float = 0.25,
    bundle_cap: int = 14,
    tol: float = 1e-8,
) -> dict:
    """Run proximal bundle method with serious/null step decisions."""
    center = x0.astype(float).copy()
    f_center, g_center = oracle(center, a_mat, b_vec, mu)
    center_cut = Cut(y=center.copy(), f=f_center, g=g_center.copy())
    cuts: List[Cut] = [center_cut]

    history: List[Tuple[int, float, float, float, int]] = []
    serious_steps = 0
    null_steps = 0

    for k in range(1, max_iter + 1):
        x_trial, model_trial, subproblem_val = solve_bundle_subproblem(
            cuts=cuts,
            center=center,
            prox_t=prox_t,
        )
        pred = f_center - subproblem_val
        if pred <= tol:
            history.append((k, f_center, pred, 0.0, len(cuts)))
            break

        f_trial, g_trial = oracle(x_trial, a_mat, b_vec, mu)
        actual = f_center - f_trial
        ratio = actual / max(pred, 1e-12)

        trial_cut = Cut(y=x_trial.copy(), f=f_trial, g=g_trial.copy())
        cuts.append(trial_cut)

        if ratio >= gamma:
            # Serious step: move center to trial point.
            center = x_trial
            f_center = f_trial
            g_center = g_trial
            center_cut = Cut(y=center.copy(), f=f_center, g=g_center.copy())
            serious_steps += 1
        else:
            null_steps += 1

        cuts = prune_bundle(cuts, center_cut=center_cut, bundle_cap=bundle_cap)
        history.append((k, f_center, pred, ratio, len(cuts)))

    return {
        "x": center,
        "f": f_center,
        "history": history,
        "serious_steps": serious_steps,
        "null_steps": null_steps,
    }


def subgradient_baseline(
    x0: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    mu: float,
    steps: int = 300,
    lr0: float = 0.7,
) -> dict:
    """Simple diminishing-step subgradient descent baseline."""
    x = x0.astype(float).copy()
    values: List[float] = []
    for k in range(1, steps + 1):
        f_val, g_val = oracle(x, a_mat, b_vec, mu)
        values.append(f_val)
        x = x - (lr0 / np.sqrt(k)) * g_val
    f_last, _ = oracle(x, a_mat, b_vec, mu)
    return {"x": x, "f": f_last, "history": values}


def finite_diff_check(a_mat: np.ndarray, b_vec: np.ndarray, mu: float) -> float:
    """Gradient check at a smooth point (away from max-affine kinks)."""
    x = np.array([1.35, -0.41], dtype=float)
    eps = 1e-6
    _, g = oracle(x, a_mat, b_vec, mu)

    num_grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        f_plus, _ = oracle(x + e, a_mat, b_vec, mu)
        f_minus, _ = oracle(x - e, a_mat, b_vec, mu)
        num_grad[i] = (f_plus - f_minus) / (2.0 * eps)
    return float(np.max(np.abs(num_grad - g)))


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)
    a_mat, b_vec, mu = build_problem()
    x0 = np.array([2.7, -2.2], dtype=float)
    f0, _ = oracle(x0, a_mat, b_vec, mu)

    grad_err = finite_diff_check(a_mat, b_vec, mu)
    bundle_res = bundle_method(
        x0=x0,
        a_mat=a_mat,
        b_vec=b_vec,
        mu=mu,
        max_iter=90,
        prox_t=1.5,
        gamma=0.25,
        bundle_cap=14,
        tol=1e-8,
    )
    baseline_res = subgradient_baseline(
        x0=x0,
        a_mat=a_mat,
        b_vec=b_vec,
        mu=mu,
        steps=320,
        lr0=0.7,
    )

    x_bundle = bundle_res["x"]
    f_bundle = float(bundle_res["f"])
    x_base = baseline_res["x"]
    f_base = float(baseline_res["f"])

    print("=== Bundle Method MVP (MATH-0378) ===")
    print(f"Initial x: {x0}, f(x0)={f0:.6f}")
    print(f"Finite-diff gradient error (smooth point): {grad_err:.3e}")
    print(
        f"Bundle result: x={x_bundle}, f={f_bundle:.8f}, "
        f"serious={bundle_res['serious_steps']}, null={bundle_res['null_steps']}"
    )
    print(f"Baseline subgradient: x={x_base}, f={f_base:.8f}")
    print(f"Distance to known optimum x*=0: ||x_bundle||={np.linalg.norm(x_bundle):.6e}")

    # Basic correctness gates.
    assert grad_err < 5e-5, "Gradient check failed at the chosen smooth point."
    assert f_bundle <= 5e-4, "Bundle method did not reach a near-optimal value."
    assert np.linalg.norm(x_bundle) < 6e-2, "Bundle solution is too far from x*=0."
    assert f_base <= 1e-3, "Baseline should also converge on this toy convex problem."
    assert f_bundle < f0 * 1e-2, "Bundle method should reduce the objective by at least 100x."


if __name__ == "__main__":
    main()
