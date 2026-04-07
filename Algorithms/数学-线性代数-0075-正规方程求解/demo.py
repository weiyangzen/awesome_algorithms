"""Normal equation solver MVP (least squares with optional ridge regularization)."""

from __future__ import annotations

import numpy as np

# Some BLAS backends may raise spurious FP warnings on safe matmul inputs.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def normal_equation_solve(
    a: np.ndarray,
    b: np.ndarray,
    ridge_lambda: float = 0.0,
    fallback_to_pinv: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Solve least squares via normal equations.

    Solves (A^T A + lambda I)x = A^T b.
    Returns x, Gram matrix, right-hand side vector, and used solver method.
    """
    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D vector.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes.")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative.")

    gram = a.T @ a
    rhs = a.T @ b

    if ridge_lambda > 0.0:
        gram = gram + ridge_lambda * np.eye(a.shape[1], dtype=gram.dtype)

    try:
        x = np.linalg.solve(gram, rhs)
        method = "solve"
    except np.linalg.LinAlgError:
        if not fallback_to_pinv:
            raise
        x = np.linalg.pinv(gram) @ rhs
        method = "pinv"

    return x, gram, rhs, method


def make_full_rank_case(
    m: int = 200, n: int = 6, noise_std: float = 0.05, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic full-column-rank regression problem."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    noise = rng.normal(scale=noise_std, size=m)
    b = a @ x_true + noise
    return a, b, x_true


def make_rank_deficient_case(
    m: int = 80, seed: int = 17
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic rank-deficient least-squares problem."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(size=m)
    c2 = rng.normal(size=m)
    c3 = c1 + 2.0 * c2  # exact linear dependency
    c4 = rng.normal(size=m)
    a = np.column_stack([c1, c2, c3, c4])
    x_ref = np.array([1.5, -0.3, 0.2, 2.0], dtype=float)
    b = a @ x_ref + rng.normal(scale=0.02, size=m)
    return a, b


def residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean residual norm ||Ax-b||_2."""
    return float(np.linalg.norm(a @ x - b, ord=2))


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print("=== Case 1: Full-rank least squares ===")
    a1, b1, x_true = make_full_rank_case()
    x_ne, g1, _, method = normal_equation_solve(a1, b1, ridge_lambda=0.0)
    x_ls, _, _, _ = np.linalg.lstsq(a1, b1, rcond=None)

    cond_a1 = np.linalg.cond(a1)
    cond_g1 = np.linalg.cond(g1)
    res_ne = residual_norm(a1, x_ne, b1)
    res_ls = residual_norm(a1, x_ls, b1)
    param_err = float(np.linalg.norm(x_ne - x_true, ord=2))
    gap_vs_lstsq = float(np.linalg.norm(x_ne - x_ls, ord=2))

    print(f"solver method: {method}")
    print(f"cond(A):      {cond_a1:.3e}")
    print(f"cond(A^T A):  {cond_g1:.3e} (roughly cond(A)^2)")
    print(f"||Ax-b||_2 (normal eq): {res_ne:.6e}")
    print(f"||Ax-b||_2 (lstsq):     {res_ls:.6e}")
    print(f"||x_NE - x_true||_2:    {param_err:.6e}")
    print(f"||x_NE - x_lstsq||_2:   {gap_vs_lstsq:.6e}")

    print("\n=== Case 2: Rank-deficient design matrix ===")
    a2, b2 = make_rank_deficient_case()
    try:
        x_raw, g_raw, _, raw_method = normal_equation_solve(
            a2, b2, ridge_lambda=0.0, fallback_to_pinv=False
        )
        raw_res = residual_norm(a2, x_raw, b2)
        print(
            "unregularized normal equation status: "
            f"succeeded via {raw_method}, cond(A^T A)={np.linalg.cond(g_raw):.3e}, "
            f"residual={raw_res:.6e}"
        )
    except np.linalg.LinAlgError as exc:
        print("unregularized normal equation status: failed as singular")
        print(f"  {exc}")

    x_ridge, g2, _, ridge_method = normal_equation_solve(
        a2, b2, ridge_lambda=1e-2, fallback_to_pinv=False
    )
    x_ls2, _, _, _ = np.linalg.lstsq(a2, b2, rcond=None)
    res_ridge = residual_norm(a2, x_ridge, b2)
    res_ls2 = residual_norm(a2, x_ls2, b2)

    print(f"ridge solver method: {ridge_method}")
    print(f"rank(A): {np.linalg.matrix_rank(a2)} / n={a2.shape[1]}")
    print(f"cond(A^T A + λI): {np.linalg.cond(g2):.3e}")
    print(f"||Ax-b||_2 (ridge normal eq): {res_ridge:.6e}")
    print(f"||Ax-b||_2 (lstsq / SVD):     {res_ls2:.6e}")
    print(f"||x_ridge||_2: {np.linalg.norm(x_ridge):.6e}")
    print(f"||x_lstsq||_2: {np.linalg.norm(x_ls2):.6e}")


if __name__ == "__main__":
    main()
