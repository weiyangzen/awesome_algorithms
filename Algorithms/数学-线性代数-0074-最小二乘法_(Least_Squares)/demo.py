"""Least Squares MVP: normal equation, QR, and lstsq comparison."""

from __future__ import annotations

import numpy as np


def normal_equation_solve(
    a: np.ndarray,
    b: np.ndarray,
    ridge_lambda: float = 0.0,
    fallback_to_pinv: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve least squares via (A^T A + lambda I)x = A^T b."""
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

    return x, gram, method


def qr_least_squares(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, str]:
    """Solve least squares via reduced QR when rank is full, else fallback to lstsq."""
    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D vector.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes.")

    q, r = np.linalg.qr(a, mode="reduced")
    n = r.shape[1]

    if np.linalg.matrix_rank(r) < n:
        x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return x, "qr->lstsq_fallback"

    x = np.linalg.solve(r, q.T @ b)
    return x, "qr_solve"


def residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Return ||Ax-b||_2."""
    return float(np.linalg.norm(a @ x - b, ord=2))


def make_full_rank_case(
    m: int = 160, n: int = 6, noise_std: float = 0.08, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic full-rank regression data."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    b = a @ x_true + rng.normal(scale=noise_std, size=m)
    return a, b, x_true


def make_rank_deficient_case(m: int = 90, seed: int = 74) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic rank-deficient matrix (one exact dependent column)."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(size=m)
    c2 = rng.normal(size=m)
    c3 = c1 - 0.5 * c2
    c4 = rng.normal(size=m)
    c5 = 2.0 * c1 - c2  # exact dependency: c5 == 2*c3
    a = np.column_stack([c1, c2, c3, c4, c5])
    x_ref = np.array([1.2, -0.7, 0.5, 2.0, -0.3], dtype=float)
    b = a @ x_ref + rng.normal(scale=0.03, size=m)
    return a, b


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print("=== Case 1: Full-rank least squares ===")
    a1, b1, x_true = make_full_rank_case()

    x_ne, g1, ne_method = normal_equation_solve(a1, b1, ridge_lambda=0.0)
    x_qr, qr_method = qr_least_squares(a1, b1)
    x_ls, _, rank1, _ = np.linalg.lstsq(a1, b1, rcond=None)

    cond_a1 = np.linalg.cond(a1)
    cond_g1 = np.linalg.cond(g1)

    print(f"rank(A): {rank1} / n={a1.shape[1]}")
    print(f"normal-equation method: {ne_method}")
    print(f"qr method:              {qr_method}")
    print(f"cond(A):      {cond_a1:.3e}")
    print(f"cond(A^T A):  {cond_g1:.3e}")
    print(f"||Ax-b||_2 (normal): {residual_norm(a1, x_ne, b1):.6e}")
    print(f"||Ax-b||_2 (qr):     {residual_norm(a1, x_qr, b1):.6e}")
    print(f"||Ax-b||_2 (lstsq):  {residual_norm(a1, x_ls, b1):.6e}")
    print(f"||x_normal - x_lstsq||_2: {np.linalg.norm(x_ne - x_ls):.6e}")
    print(f"||x_qr - x_lstsq||_2:     {np.linalg.norm(x_qr - x_ls):.6e}")
    print(f"||x_lstsq - x_true||_2:   {np.linalg.norm(x_ls - x_true):.6e}")

    print("\n=== Case 2: Rank-deficient least squares ===")
    a2, b2 = make_rank_deficient_case()
    rank2 = np.linalg.matrix_rank(a2)
    print(f"rank(A): {rank2} / n={a2.shape[1]}")

    try:
        x_bad, g_bad, method_bad = normal_equation_solve(
            a2, b2, ridge_lambda=0.0, fallback_to_pinv=False
        )
        print(
            "unregularized normal-equation status: "
            f"succeeded via {method_bad}, cond(A^T A)={np.linalg.cond(g_bad):.3e}, "
            f"residual={residual_norm(a2, x_bad, b2):.6e}"
        )
    except np.linalg.LinAlgError as exc:
        print("unregularized normal-equation status: failed as singular")
        print(f"  {exc}")

    x_ridge, g_ridge, ridge_method = normal_equation_solve(
        a2, b2, ridge_lambda=1e-2, fallback_to_pinv=False
    )
    x_ls2, _, rank_ls2, _ = np.linalg.lstsq(a2, b2, rcond=None)

    print(f"lstsq reported rank: {rank_ls2}")
    print(f"ridge method: {ridge_method}")
    print(f"cond(A^T A + lambda I): {np.linalg.cond(g_ridge):.3e}")
    print(f"||Ax-b||_2 (ridge): {residual_norm(a2, x_ridge, b2):.6e}")
    print(f"||Ax-b||_2 (lstsq): {residual_norm(a2, x_ls2, b2):.6e}")
    print(f"||x_ridge||_2: {np.linalg.norm(x_ridge):.6e}")
    print(f"||x_lstsq||_2: {np.linalg.norm(x_ls2):.6e}")


if __name__ == "__main__":
    main()
