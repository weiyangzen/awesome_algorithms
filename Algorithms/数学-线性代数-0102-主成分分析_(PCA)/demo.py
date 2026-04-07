"""Minimal runnable MVP for Principal Component Analysis (PCA).

This implementation keeps the algorithm transparent:
1) center/standardize features
2) build covariance matrix
3) solve eigenpairs via symmetric eigendecomposition
4) sort/select principal components
5) project and reconstruct data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    """Container for PCA fit outputs."""

    mean_: np.ndarray
    scale_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray
    explained_variance_ratio_: np.ndarray
    transformed_: np.ndarray
    reconstructed_: np.ndarray
    n_components_: int


def validate_inputs(x: np.ndarray, n_components: int | None) -> None:
    """Validate input matrix and PCA hyper-parameters."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix of shape (n_samples, n_features).")
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("x must contain at least 2 samples.")
    if n_features < 1:
        raise ValueError("x must contain at least 1 feature.")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite values.")

    max_rank = min(n_samples - 1, n_features)
    if n_components is not None:
        if n_components <= 0:
            raise ValueError("n_components must be positive when provided.")
        if n_components > max_rank:
            raise ValueError(
                f"n_components={n_components} exceeds max valid rank {max_rank}."
            )


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale columns to zero mean / unit variance.

    Returns:
        x_std: standardized data
        mean_: feature means
        scale_: feature standard deviations (with zeros guarded to 1)
    """
    mean_ = np.mean(x, axis=0)
    centered = x - mean_

    scale_ = np.std(centered, axis=0, ddof=1)
    scale_safe = scale_.copy()
    scale_safe[scale_safe < 1e-15] = 1.0

    x_std = centered / scale_safe
    return x_std, mean_, scale_safe


def choose_components_by_variance(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float,
) -> int:
    """Choose smallest k such that cumulative explained ratio >= threshold."""
    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError("variance_threshold must be in (0, 1].")
    cumulative = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    return min(k, explained_variance_ratio.size)


def pca_fit_transform(
    x: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    standardize: bool = True,
) -> PCAResult:
    """Fit PCA and return transformed data using covariance eigendecomposition."""
    validate_inputs(x, n_components)

    if standardize:
        x_proc, mean_, scale_ = standardize_features(x)
    else:
        mean_ = np.mean(x, axis=0)
        x_proc = x - mean_
        scale_ = np.ones(x.shape[1], dtype=float)

    n_samples = x_proc.shape[0]

    cov = (x_proc.T @ x_proc) / (n_samples - 1)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0.0)
    total_var = float(np.sum(eigvals))
    if total_var <= 0.0:
        raise ValueError("Total variance is non-positive; PCA is ill-defined for this input.")

    explained_ratio = eigvals / total_var

    if n_components is None:
        k = choose_components_by_variance(explained_ratio, variance_threshold)
    else:
        k = n_components

    principal_axes = eigvecs[:, :k]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        transformed = x_proc @ principal_axes

    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values encountered in PCA projection.")

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        reconstructed_proc = transformed @ principal_axes.T

    if not np.isfinite(reconstructed_proc).all():
        raise ValueError("Non-finite values encountered in PCA reconstruction.")
    reconstructed = reconstructed_proc * scale_ + mean_

    components = principal_axes.T

    return PCAResult(
        mean_=mean_,
        scale_=scale_,
        components_=components,
        eigenvalues_=eigvals,
        explained_variance_ratio_=explained_ratio,
        transformed_=transformed,
        reconstructed_=reconstructed,
        n_components_=k,
    )


def build_demo_data(n_samples: int = 480) -> np.ndarray:
    """Construct deterministic correlated data from 3 latent factors to 6 features."""
    rng = np.random.default_rng(2026)

    latent = rng.normal(size=(n_samples, 3))
    mixing = np.array(
        [
            [2.0, 0.2, -0.4, 1.3, 0.0, 0.8],
            [0.0, 1.6, 0.9, -0.2, 1.4, 0.3],
            [1.0, -0.7, 1.8, 0.0, 0.5, -1.2],
        ],
        dtype=float,
    )

    noise = 0.15 * rng.normal(size=(n_samples, mixing.shape[1]))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x_linear = latent @ mixing
    if not np.isfinite(x_linear).all():
        raise RuntimeError("Non-finite values encountered in synthetic data generation.")

    x = x_linear + noise

    feature_scale = np.array([1.0, 2.2, 0.7, 1.5, 0.9, 1.8], dtype=float)
    x *= feature_scale
    x += np.array([3.0, -2.5, 1.2, 0.0, 5.0, -1.0], dtype=float)

    return x


def run_checks(x: np.ndarray, result: PCAResult) -> None:
    """Sanity checks for PCA output quality and numerical properties."""
    n_samples, _ = x.shape
    k = result.n_components_

    if k <= 0:
        raise AssertionError("n_components_ must be positive.")
    if result.transformed_.shape != (n_samples, k):
        raise AssertionError("Transformed shape mismatch.")

    gram = result.components_ @ result.components_.T
    if not np.allclose(gram, np.eye(k), atol=1e-8, rtol=0.0):
        raise AssertionError("Principal components are not orthonormal within tolerance.")

    ev = result.eigenvalues_
    if np.any(ev[:-1] < ev[1:] - 1e-12):
        raise AssertionError("Eigenvalues are not sorted in descending order.")

    evr = result.explained_variance_ratio_
    if np.any(evr < -1e-12):
        raise AssertionError("Explained variance ratio contains negative entries.")
    if abs(float(np.sum(evr)) - 1.0) > 1e-8:
        raise AssertionError("Explained variance ratio does not sum to 1.")

    recon_error = float(np.linalg.norm(result.reconstructed_ - x, ord="fro"))
    baseline_error = float(
        np.linalg.norm(np.repeat(result.mean_[None, :], n_samples, axis=0) - x, ord="fro")
    )
    if recon_error >= baseline_error:
        raise AssertionError("PCA reconstruction is not better than mean-only baseline.")

    # Cross-check with SVD on the same standardized data (independent formulation).
    x_std = (x - result.mean_) / result.scale_
    _, singular_values, _ = np.linalg.svd(x_std, full_matrices=False)
    sv_eigvals = (singular_values**2) / (n_samples - 1)
    sv_eigvals = np.maximum(sv_eigvals, 0.0)

    max_compare = min(result.eigenvalues_.size, sv_eigvals.size)
    eig_diff = float(np.max(np.abs(result.eigenvalues_[:max_compare] - sv_eigvals[:max_compare])))
    if eig_diff > 1e-7:
        raise AssertionError(f"Eigenvalue mismatch against SVD reference: {eig_diff:.3e}")


def main() -> None:
    x = build_demo_data(n_samples=480)

    result = pca_fit_transform(
        x,
        n_components=None,
        variance_threshold=0.95,
        standardize=True,
    )

    run_checks(x, result)

    cumulative = np.cumsum(result.explained_variance_ratio_)
    recon_rmse = float(np.sqrt(np.mean((result.reconstructed_ - x) ** 2)))

    print("PCA MVP report")
    print(f"data_shape={x.shape}")
    print(f"selected_n_components={result.n_components_}")
    print("top_eigenvalues=", np.round(result.eigenvalues_[:6], 6))
    print(
        "top_explained_variance_ratio=",
        np.round(result.explained_variance_ratio_[:6], 6),
    )
    print(
        "cumulative_explained_variance=",
        np.round(cumulative[: result.n_components_], 6),
    )
    print(f"reconstruction_rmse={recon_rmse:.6f}")
    print(f"transformed_shape={result.transformed_.shape}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
