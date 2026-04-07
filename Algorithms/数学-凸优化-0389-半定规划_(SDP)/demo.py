"""Semidefinite Programming (SDP) MVP via log-det barrier + Newton method.

We solve the primal standard-form SDP:
    min_X <C, X>
    s.t.  <A_i, X> = b_i,  i=1..m
          X >> 0  (strictly SPD during barrier iterations)

The demo problem is the Max-Cut SDP relaxation:
    max_X 0.25 * <L, X>
    s.t.  diag(X) = 1, X >= 0

converted to minimization with C = -0.25 * L.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class SdpSolveResult:
    X: np.ndarray
    converged: bool
    outer_iterations: int
    total_newton_iterations: int
    history: List[Dict[str, float]]


def symmetrize(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + x.T)


def is_spd(x: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(symmetrize(x))
        return True
    except np.linalg.LinAlgError:
        return False


def logdet_spd(x: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(x)
    if sign <= 0:
        raise ValueError("Matrix is not SPD; logdet undefined for barrier objective.")
    return float(val)


def trace_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.trace(a @ b))


def A_map(x: np.ndarray, a_mats: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([trace_inner(ai, x) for ai in a_mats], dtype=float)


def A_adjoint(y: np.ndarray, a_mats: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(a_mats[0], dtype=float)
    for yi, ai in zip(y, a_mats):
        out += yi * ai
    return out


def barrier_objective(x: np.ndarray, c: np.ndarray, t: float) -> float:
    return t * trace_inner(c, x) - logdet_spd(x)


def solve_barrier_subproblem_newton(
    c: np.ndarray,
    a_mats: Sequence[np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    t: float,
    newton_tol: float = 1e-10,
    max_newton_iters: int = 80,
    armijo_c1: float = 1e-4,
    backtrack_beta: float = 0.5,
    max_backtracks: int = 30,
) -> tuple[np.ndarray, int]:
    """Solve min t<C,X> - logdet(X) with linear equalities using feasible Newton."""
    x = symmetrize(x0.copy())
    m = len(a_mats)

    eq_residual = np.linalg.norm(A_map(x, a_mats) - b)
    if eq_residual > 1e-6:
        raise ValueError(f"Initial point must satisfy equality constraints, got residual={eq_residual:.3e}")
    if not is_spd(x):
        raise ValueError("Initial point for barrier Newton must be SPD.")

    for it in range(1, max_newton_iters + 1):
        inv_x = np.linalg.inv(x)
        grad = t * c - inv_x

        # Build Schur complement system:
        # M w = -h, where
        # h_i = <A_i, X grad X>, M_ij = <A_i, X A_j X>.
        x_grad_x = x @ grad @ x
        h = A_map(x_grad_x, a_mats)

        m_mat = np.empty((m, m), dtype=float)
        for i, ai in enumerate(a_mats):
            for j, aj in enumerate(a_mats):
                m_mat[i, j] = trace_inner(ai @ x, aj @ x)

        try:
            w = np.linalg.solve(m_mat, -h)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(m_mat, -h, rcond=None)[0]

        grad_projected = grad + A_adjoint(w, a_mats)
        d_x = -x @ grad_projected @ x
        d_x = symmetrize(d_x)

        invx_dx = inv_x @ d_x
        decrement_sq = float(np.trace(invx_dx @ invx_dx))
        if 0.5 * decrement_sq <= newton_tol:
            return x, it

        phi = barrier_objective(x, c, t)
        directional_derivative = trace_inner(grad, d_x)

        alpha = 1.0
        accepted = False
        for _ in range(max_backtracks):
            x_candidate = symmetrize(x + alpha * d_x)
            if not is_spd(x_candidate):
                alpha *= backtrack_beta
                continue

            phi_candidate = barrier_objective(x_candidate, c, t)
            if phi_candidate <= phi + armijo_c1 * alpha * directional_derivative:
                x = x_candidate
                accepted = True
                break
            alpha *= backtrack_beta

        if not accepted:
            raise RuntimeError("Line search failed to produce an SPD descent step.")

    raise RuntimeError(f"Newton method did not converge in {max_newton_iters} iterations.")


def solve_sdp_barrier(
    c: np.ndarray,
    a_mats: Sequence[np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    gap_tol: float = 1e-5,
    mu: float = 6.0,
    max_outer_iters: int = 25,
) -> SdpSolveResult:
    """Outer barrier loop with feasible Newton centering steps."""
    n = c.shape[0]
    x = symmetrize(x0.copy())
    history: List[Dict[str, float]] = []
    t = 1.0
    total_newton_iters = 0
    converged = False

    for outer in range(1, max_outer_iters + 1):
        x, used_newton = solve_barrier_subproblem_newton(c, a_mats, b, x, t=t)
        total_newton_iters += used_newton

        primal_obj = trace_inner(c, x)
        eq_res = float(np.linalg.norm(A_map(x, a_mats) - b))
        min_eig = float(np.min(np.linalg.eigvalsh(symmetrize(x))))
        duality_gap_est = n / t
        history.append(
            {
                "outer_iter": float(outer),
                "t": float(t),
                "primal_obj": float(primal_obj),
                "eq_residual": eq_res,
                "min_eig": min_eig,
                "gap_est": float(duality_gap_est),
                "newton_iters": float(used_newton),
            }
        )

        if duality_gap_est <= gap_tol:
            converged = True
            break
        t *= mu

    return SdpSolveResult(
        X=x,
        converged=converged,
        outer_iterations=len(history),
        total_newton_iterations=total_newton_iters,
        history=history,
    )


def build_maxcut_sdp_instance() -> tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Construct a tiny weighted graph and the corresponding Max-Cut SDP."""
    w = np.array(
        [
            [0.0, 2.0, 1.0, 3.0, 0.5],
            [2.0, 0.0, 2.5, 0.7, 1.2],
            [1.0, 2.5, 0.0, 1.8, 2.2],
            [3.0, 0.7, 1.8, 0.0, 1.6],
            [0.5, 1.2, 2.2, 1.6, 0.0],
        ],
        dtype=float,
    )
    w = symmetrize(w)
    np.fill_diagonal(w, 0.0)

    n = w.shape[0]
    degree = np.diag(np.sum(w, axis=1))
    laplacian = degree - w
    c = -0.25 * laplacian  # convert max to min

    a_mats: List[np.ndarray] = []
    for i in range(n):
        ai = np.zeros((n, n), dtype=float)
        ai[i, i] = 1.0
        a_mats.append(ai)
    b = np.ones(n, dtype=float)
    x0 = np.eye(n, dtype=float)
    return c, a_mats, b, w


def sdp_objective_to_maxcut_value(c: np.ndarray, x: np.ndarray) -> float:
    # c = -0.25 * L, so maxcut_sdp_value = -<c, x>.
    return -trace_inner(c, x)


def cut_value_from_signs(w: np.ndarray, s: np.ndarray) -> float:
    # Works with symmetric W and {+1, -1} sign vector.
    return float(0.25 * np.sum(w * (1.0 - np.outer(s, s))))


def random_hyperplane_rounding(
    x: np.ndarray,
    w: np.ndarray,
    rounds: int = 2000,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(symmetrize(x))
    eigvals = np.clip(eigvals, 0.0, None)
    embedding = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    best_cut = -np.inf
    best_signs = np.ones(n, dtype=float)

    for _ in range(rounds):
        g = rng.normal(size=n)
        y = embedding @ g
        s = np.where(y >= 0.0, 1.0, -1.0)
        val = cut_value_from_signs(w, s)
        if val > best_cut:
            best_cut = val
            best_signs = s
    return float(best_cut), best_signs


def exact_maxcut_bruteforce(w: np.ndarray) -> tuple[float, np.ndarray]:
    """Bruteforce exact Max-Cut for very small graphs (n <= 20 recommended)."""
    n = w.shape[0]
    if n > 20:
        raise ValueError("Bruteforce Max-Cut is only for tiny graphs.")

    best_cut = -np.inf
    best_signs = np.ones(n, dtype=float)

    # Fix first sign to +1 to remove sign-complement duplication.
    for bits in product([-1.0, 1.0], repeat=n - 1):
        s = np.array((1.0, *bits), dtype=float)
        val = cut_value_from_signs(w, s)
        if val > best_cut:
            best_cut = val
            best_signs = s
    return float(best_cut), best_signs


def print_history(history: Sequence[Dict[str, float]]) -> None:
    print("outer | t          | <C,X> (min)    | gap_est      | eq_residual  | min_eig(X)   | newton")
    print("-" * 97)
    for row in history:
        print(
            f"{int(row['outer_iter']):5d} | "
            f"{row['t']:10.3e} | "
            f"{row['primal_obj']:13.6e} | "
            f"{row['gap_est']:11.3e} | "
            f"{row['eq_residual']:11.3e} | "
            f"{row['min_eig']:11.3e} | "
            f"{int(row['newton_iters']):6d}"
        )


def main() -> None:
    c, a_mats, b, w = build_maxcut_sdp_instance()
    n = c.shape[0]
    x0 = np.eye(n, dtype=float)

    result = solve_sdp_barrier(
        c=c,
        a_mats=a_mats,
        b=b,
        x0=x0,
        gap_tol=1e-5,
        mu=6.0,
        max_outer_iters=25,
    )
    x_star = result.X
    sdp_upper_bound = sdp_objective_to_maxcut_value(c, x_star)
    rounded_cut, rounded_signs = random_hyperplane_rounding(x_star, w, rounds=3000, seed=2026)
    exact_cut, exact_signs = exact_maxcut_bruteforce(w)

    diag_residual = np.linalg.norm(np.diag(x_star) - 1.0)
    min_eig = float(np.min(np.linalg.eigvalsh(symmetrize(x_star))))

    print("=== SDP MVP (Max-Cut Relaxation) ===")
    print(f"n={n}, outer_iters={result.outer_iterations}, total_newton={result.total_newton_iterations}")
    print(f"barrier converged by gap estimate: {result.converged}")
    print_history(result.history)
    print()
    print(f"SDP relaxation value (upper bound on Max-Cut): {sdp_upper_bound:.6f}")
    print(f"Best rounded cut from random hyperplanes:      {rounded_cut:.6f}")
    print(f"Exact Max-Cut by brute force:                  {exact_cut:.6f}")
    print(f"Bound check (rounded <= SDP):                  {rounded_cut <= sdp_upper_bound + 1e-6}")
    print(f"Bound check (exact   <= SDP):                  {exact_cut <= sdp_upper_bound + 1e-6}")
    print(f"Constraint residual ||diag(X)-1||_2:           {diag_residual:.3e}")
    print(f"Minimum eigenvalue of X:                       {min_eig:.3e}")
    print(f"Rounded sign vector: {rounded_signs.astype(int)}")
    print(f"Exact sign vector:   {exact_signs.astype(int)}")


if __name__ == "__main__":
    main()
