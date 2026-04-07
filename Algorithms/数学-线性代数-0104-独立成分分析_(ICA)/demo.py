"""Manual FastICA MVP for Independent Component Analysis (ICA)."""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np


def safe_matmul(left: np.ndarray, right: np.ndarray, context: str) -> np.ndarray:
    """Matrix multiply with warning suppression and finite-result validation."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.matmul(left, right)
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"Non-finite values produced during {context}")
    return out


def standardize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise standardized data (zero mean, unit variance)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"x must be 2D, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")

    centered = arr - arr.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    if np.any(std <= eps):
        raise ValueError("At least one row has near-zero variance")
    return centered / std


def generate_sources(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Create independent source signals with non-Gaussian distributions."""
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100 for stable ICA demo")

    t = np.linspace(0.0, 8.0, n_samples, endpoint=False)
    s1 = np.sin(2.0 * t)  # Smooth periodic signal
    s2 = np.sign(np.sin(3.0 * t))  # Square-wave-like signal
    # Heavy-tailed but bounded to avoid rare extreme outliers that can destabilize MVP demos.
    s3 = np.clip(rng.standard_t(df=3.0, size=n_samples), -8.0, 8.0)

    s = np.vstack([s1, s2, s3])
    return standardize_rows(s)


def make_mixture(s: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a well-conditioned random mixing matrix and mixed observations."""
    s_arr = np.asarray(s, dtype=float)
    if s_arr.ndim != 2:
        raise ValueError("s must be a 2D matrix with shape (n_sources, n_samples)")
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("s contains non-finite values")

    n_sources = s_arr.shape[0]
    for _ in range(1000):
        a = rng.uniform(-1.5, 1.5, size=(n_sources, n_sources))
        if abs(np.linalg.det(a)) > 0.2 and np.linalg.cond(a) < 12.0:
            x = safe_matmul(a, s_arr, context="mixing matrix application")
            return a, x

    raise RuntimeError("Failed to sample a suitable mixing matrix")


def whiten(x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and whiten observed signals.

    Returns:
        x_white: whitened signals
        whitening: whitening matrix
        dewhitening: inverse transform matrix
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"x must be 2D, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")

    x_centered = arr - arr.mean(axis=1, keepdims=True)
    n_samples = x_centered.shape[1]

    cov = safe_matmul(x_centered, x_centered.T, context="covariance build") / n_samples
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if np.any(eigvals <= eps):
        raise ValueError("Covariance matrix is singular or near-singular")

    whitening = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    dewhitening = eigvecs @ np.diag(np.sqrt(eigvals))
    x_white = safe_matmul(whitening, x_centered, context="whitening transform")

    return x_white, whitening, dewhitening


def fastica_deflation(
    x_white: np.ndarray,
    n_components: int,
    random_state: int = 0,
    max_iter: int = 2000,
    tol: float = 1e-7,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate unmixing rows via FastICA fixed-point updates (deflation mode)."""
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    xw = np.asarray(x_white, dtype=float)
    if xw.ndim != 2:
        raise ValueError(f"x_white must be 2D, got ndim={xw.ndim}")
    if not np.all(np.isfinite(xw)):
        raise ValueError("x_white contains non-finite values")

    n_features, _ = xw.shape
    if n_components > n_features:
        raise ValueError("n_components cannot exceed n_features")

    rng = np.random.default_rng(random_state)
    w_rows = np.zeros((n_components, n_features), dtype=float)
    iters = np.zeros(n_components, dtype=int)

    for p in range(n_components):
        w = rng.normal(size=n_features)
        w_norm = np.linalg.norm(w)
        if w_norm <= 1e-12:
            raise RuntimeError("Random initialization failed")
        w = w / w_norm

        converged = False
        for step in range(1, max_iter + 1):
            wx = safe_matmul(w, xw, context="projection onto candidate component")
            gwx = np.tanh(alpha * wx)
            gprime = alpha * (1.0 - gwx**2)

            w_new = (xw * gwx).mean(axis=1) - gprime.mean() * w

            if p > 0:
                # Deflation: remove projections to previously found components.
                proj = safe_matmul(w_rows[:p], w_new, context="deflation inner projection")
                correction = safe_matmul(w_rows[:p].T, proj, context="deflation back projection")
                w_new = w_new - correction

            new_norm = np.linalg.norm(w_new)
            if new_norm <= 1e-12:
                raise RuntimeError(f"Degenerate update at component {p}, iter {step}")
            w_new = w_new / new_norm

            if abs(abs(float(np.dot(w_new, w))) - 1.0) < tol:
                converged = True
                w = w_new
                iters[p] = step
                break

            w = w_new

        if not converged:
            raise RuntimeError(f"FastICA did not converge for component {p} in {max_iter} steps")

        w_rows[p] = w

    return w_rows, iters


def absolute_correlation_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute absolute row-wise correlation matrix between two signal sets."""
    a_std = standardize_rows(a)
    b_std = standardize_rows(b)
    n_samples = a_std.shape[1]
    corr = safe_matmul(a_std, b_std.T, context="correlation matrix build") / n_samples
    return np.abs(corr)


def best_permutation(abs_corr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find permutation maximizing total diagonal correlation."""
    corr = np.asarray(abs_corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("abs_corr must be square")

    n = corr.shape[0]
    if n > 8:
        raise ValueError("Brute-force permutation search is limited to n <= 8")

    best_score = -np.inf
    best_perm = None
    for perm in itertools.permutations(range(n)):
        score = float(sum(corr[perm[j], j] for j in range(n)))
        if score > best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None
    return np.array(best_perm, dtype=int), best_score


def align_estimated_sources(s_true: np.ndarray, s_est: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Align estimated sources to true order and sign for fair comparison."""
    n_components = s_true.shape[0]
    aligned = np.zeros_like(s_est)

    for est_idx in range(n_components):
        true_idx = int(perm[est_idx])
        sign = np.sign(np.dot(s_est[est_idx], s_true[true_idx]))
        if sign == 0:
            sign = 1.0
        aligned[true_idx] = sign * s_est[est_idx]

    return aligned


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    n_components = 3
    n_samples = 4000

    rng_sources = np.random.default_rng(42)
    s_true = generate_sources(n_samples=n_samples, rng=rng_sources)
    a_true, x_obs = make_mixture(s_true, rng=np.random.default_rng(123))

    x_white, whitening, _ = whiten(x_obs)
    w_rows, iters = fastica_deflation(
        x_white,
        n_components=n_components,
        random_state=7,
        max_iter=2000,
        tol=1e-7,
        alpha=1.0,
    )

    s_est = safe_matmul(w_rows, x_white, context="source recovery")
    unmixing_total = safe_matmul(w_rows, whitening, context="global unmixing build")

    abs_corr = absolute_correlation_rows(s_true, s_est)
    perm, score = best_permutation(abs_corr)
    matched_corr = np.array([abs_corr[perm[j], j] for j in range(n_components)], dtype=float)

    s_aligned = align_estimated_sources(s_true, s_est, perm)
    s_true_std = standardize_rows(s_true)
    s_aligned_std = standardize_rows(s_aligned)
    recovery_mse = float(np.mean((s_true_std - s_aligned_std) ** 2))

    whiten_cov = safe_matmul(x_white, x_white.T, context="white covariance check") / x_white.shape[1]
    whitening_error = float(np.linalg.norm(whiten_cov - np.eye(n_components), ord="fro"))

    print("=== ICA Demo (Manual FastICA) ===")
    print(f"n_components = {n_components}, n_samples = {n_samples}")

    print("\n=== True Mixing Matrix A ===")
    print(a_true)

    print("\n=== Estimated Global Unmixing Matrix W_total ===")
    print(unmixing_total)

    print("\n=== Fixed-Point Iterations per Component ===")
    print(iters)

    print("\n=== |corr(true_i, est_j)| Matrix ===")
    print(abs_corr)
    print("Best mapping (est_j -> true_i):", perm)

    print("\n=== Quality Metrics ===")
    print(f"Whitening error ||Cov(X_white)-I||_F = {whitening_error:.3e}")
    print(f"Mean matched |corr|               = {matched_corr.mean():.6f}")
    print(f"Min matched |corr|                = {matched_corr.min():.6f}")
    print(f"Permutation objective score       = {score:.6f}")
    print(f"Aligned source recovery MSE       = {recovery_mse:.6e}")

    success = bool(matched_corr.min() > 0.90 and recovery_mse < 0.20)
    print("\nRecovery success threshold met:", success)


if __name__ == "__main__":
    main()
