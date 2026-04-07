"""Sturm-Liouville theory MVP with finite-difference discretization.

We solve the regular Sturm-Liouville eigenproblem on [a, b] with Dirichlet BC:
    -(p(x) y'(x))' + q(x) y(x) = lambda * w(x) * y(x),
    y(a) = y(b) = 0.

The continuous problem is discretized into a symmetric tridiagonal generalized
matrix eigenproblem A u = lambda W u on interior nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal

Array = np.ndarray


@dataclass(frozen=True)
class SturmLiouvilleProblem:
    name: str
    a: float
    b: float
    p_fn: Callable[[Array], Array]
    q_fn: Callable[[Array], Array]
    w_fn: Callable[[Array], Array]
    n_interior: int
    n_modes: int


@dataclass
class SturmLiouvilleResult:
    x_interior: Array
    h: float
    weight: Array
    diag_a: Array
    off_a: Array
    eigenvalues: Array
    eigenvectors: Array
    residual_rel: Array
    gram: Array


def _validate_problem(problem: SturmLiouvilleProblem) -> None:
    if not np.isfinite(problem.a) or not np.isfinite(problem.b):
        raise ValueError("区间端点必须是有限值")
    if problem.b <= problem.a:
        raise ValueError("必须满足 b > a")
    if problem.n_interior < 10:
        raise ValueError("n_interior 至少为 10")
    if problem.n_modes < 1:
        raise ValueError("n_modes 至少为 1")
    if problem.n_modes >= problem.n_interior:
        raise ValueError("n_modes 必须严格小于 n_interior")


def _tridiag_matvec(diag: Array, off: Array, vec: Array) -> Array:
    out = diag * vec
    out[:-1] += off * vec[1:]
    out[1:] += off * vec[:-1]
    return out


def _build_discretization(
    problem: SturmLiouvilleProblem,
) -> tuple[Array, float, Array, Array, Array]:
    x_full = np.linspace(problem.a, problem.b, problem.n_interior + 2)
    h = float(x_full[1] - x_full[0])

    x_interior = x_full[1:-1]
    x_mid = 0.5 * (x_full[:-1] + x_full[1:])

    p_mid = np.asarray(problem.p_fn(x_mid), dtype=float)
    q_val = np.asarray(problem.q_fn(x_interior), dtype=float)
    w_val = np.asarray(problem.w_fn(x_interior), dtype=float)

    if p_mid.shape != x_mid.shape:
        raise ValueError("p_fn 返回形状错误")
    if q_val.shape != x_interior.shape:
        raise ValueError("q_fn 返回形状错误")
    if w_val.shape != x_interior.shape:
        raise ValueError("w_fn 返回形状错误")
    if not np.all(np.isfinite(p_mid)) or not np.all(np.isfinite(q_val)):
        raise ValueError("p 或 q 含有非有限值")
    if not np.all(np.isfinite(w_val)):
        raise ValueError("w 含有非有限值")
    if np.min(p_mid) <= 0.0:
        raise ValueError("Sturm-Liouville 要求 p(x) > 0")
    if np.min(w_val) <= 0.0:
        raise ValueError("Sturm-Liouville 要求 w(x) > 0")

    p_minus = p_mid[:-1]
    p_plus = p_mid[1:]

    diag_a = (p_minus + p_plus) / (h * h) + q_val
    off_a = -p_mid[1:-1] / (h * h)
    return x_interior, h, diag_a, off_a, w_val


def solve_sturm_liouville(problem: SturmLiouvilleProblem) -> SturmLiouvilleResult:
    _validate_problem(problem)
    x, h, diag_a, off_a, w = _build_discretization(problem)

    # Convert A u = lambda W u to symmetric tridiagonal C z = lambda z,
    # where z = sqrt(W) u and C = W^{-1/2} A W^{-1/2}.
    diag_c = diag_a / w
    off_c = off_a / np.sqrt(w[:-1] * w[1:])

    eigvals, eigvecs_z = eigh_tridiagonal(
        diag_c,
        off_c,
        select="i",
        select_range=(0, problem.n_modes - 1),
    )

    eigvecs_u = eigvecs_z / np.sqrt(w)[:, None]

    # Normalize in weighted inner-product and fix deterministic sign.
    for j in range(problem.n_modes):
        norm_w = float(np.sqrt(np.dot(w, eigvecs_u[:, j] * eigvecs_u[:, j])))
        eigvecs_u[:, j] /= norm_w
        pivot = int(np.argmax(np.abs(eigvecs_u[:, j])))
        if eigvecs_u[pivot, j] < 0.0:
            eigvecs_u[:, j] *= -1.0

    gram = eigvecs_u.T @ (w[:, None] * eigvecs_u)

    residual_rel = np.empty(problem.n_modes, dtype=float)
    for j in range(problem.n_modes):
        lhs = _tridiag_matvec(diag_a, off_a, eigvecs_u[:, j])
        rhs = eigvals[j] * (w * eigvecs_u[:, j])
        residual = lhs - rhs
        residual_rel[j] = np.linalg.norm(residual) / (np.linalg.norm(lhs) + 1e-14)

    return SturmLiouvilleResult(
        x_interior=x,
        h=h,
        weight=w,
        diag_a=diag_a,
        off_a=off_a,
        eigenvalues=eigvals,
        eigenvectors=eigvecs_u,
        residual_rel=residual_rel,
        gram=gram,
    )


def run_case_with_exact(
    problem: SturmLiouvilleProblem,
    exact_eigenvalues: Callable[[Array], Array],
) -> tuple[float, float]:
    result = solve_sturm_liouville(problem)
    mode_ids = np.arange(1, problem.n_modes + 1)

    lam_exact = np.asarray(exact_eigenvalues(mode_ids), dtype=float)
    lam_num = result.eigenvalues
    abs_err = np.abs(lam_num - lam_exact)
    rel_err = abs_err / (np.abs(lam_exact) + 1e-14)

    df = pd.DataFrame(
        {
            "mode": mode_ids,
            "lambda_num": lam_num,
            "lambda_exact": lam_exact,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "rel_residual": result.residual_rel,
        }
    )

    gram_offdiag = result.gram - np.eye(problem.n_modes)
    ortho_err = float(np.max(np.abs(gram_offdiag)))
    max_rel_err = float(np.max(rel_err))

    print(f"\n=== {problem.name} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print(f"h = {result.h:.6e}")
    print(f"max relative eigenvalue error = {max_rel_err:.6e}")
    print(f"max weighted orthogonality error = {ortho_err:.6e}")

    return max_rel_err, max(ortho_err, float(np.max(result.residual_rel)))


def run_case_without_exact(problem: SturmLiouvilleProblem) -> float:
    result = solve_sturm_liouville(problem)
    mode_ids = np.arange(1, problem.n_modes + 1)

    df = pd.DataFrame(
        {
            "mode": mode_ids,
            "lambda_num": result.eigenvalues,
            "rel_residual": result.residual_rel,
        }
    )

    gram_offdiag = result.gram - np.eye(problem.n_modes)
    ortho_err = float(np.max(np.abs(gram_offdiag)))
    worst_residual = float(np.max(result.residual_rel))

    print(f"\n=== {problem.name} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print(f"h = {result.h:.6e}")
    print(f"max weighted orthogonality error = {ortho_err:.6e}")
    print(f"max relative residual = {worst_residual:.6e}")

    return max(ortho_err, worst_residual)


def main() -> None:
    # Case 1: canonical equation -y'' = lambda y on [0, pi], y(0)=y(pi)=0.
    # Exact eigenvalues are lambda_n = n^2.
    case_exact = SturmLiouvilleProblem(
        name="Canonical: -(y')' = lambda y on [0, pi]",
        a=0.0,
        b=float(np.pi),
        p_fn=lambda x: np.ones_like(x),
        q_fn=lambda x: np.zeros_like(x),
        w_fn=lambda x: np.ones_like(x),
        n_interior=320,
        n_modes=6,
    )

    # Case 2: variable coefficients without closed-form eigenvalues in this demo.
    case_variable = SturmLiouvilleProblem(
        name="Variable coefficients: -(p y')' + q y = lambda w y on [0, 1]",
        a=0.0,
        b=1.0,
        p_fn=lambda x: 1.0 + 0.5 * x,
        q_fn=lambda x: 1.0 + x,
        w_fn=lambda x: 1.0 + x,
        n_interior=280,
        n_modes=5,
    )

    max_rel_err_exact, stability_exact = run_case_with_exact(
        case_exact,
        exact_eigenvalues=lambda n: n.astype(float) ** 2,
    )
    stability_variable = run_case_without_exact(case_variable)

    ok = (
        max_rel_err_exact < 2e-3
        and stability_exact < 5e-10
        and stability_variable < 5e-10
    )

    print("\n=== Summary ===")
    print(f"max_rel_err_exact = {max_rel_err_exact:.6e}")
    print(f"stability_exact   = {stability_exact:.6e}")
    print(f"stability_var     = {stability_variable:.6e}")
    print(f"PASS: {ok}")

    if not ok:
        raise RuntimeError("Sturm-Liouville MVP checks failed")


if __name__ == "__main__":
    main()
