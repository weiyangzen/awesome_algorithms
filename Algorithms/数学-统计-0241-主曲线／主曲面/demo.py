"""Principal curve / principal surface minimal runnable MVP.

This script implements two iterative manifold fitting demos:
1) Principal curve in 2D (Hastie-Stuetzle style loop).
2) Principal surface in 3D (2D latent coordinates + kernel smoothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge


@dataclass
class CurveResult:
    params: np.ndarray
    fitted_points: np.ndarray
    mse: float
    iterations: int
    converged: bool


@dataclass
class SurfaceResult:
    latent_uv: np.ndarray
    fitted_points: np.ndarray
    mse: float
    iterations: int
    converged: bool


def _validate_data_matrix(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] < 10:
        raise ValueError(f"{name} must have at least 10 samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _strictly_increasing(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def make_curve_data(n_samples: int = 320, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(-3.0, 3.0, size=n_samples)
    x1 = t + 0.12 * rng.normal(size=n_samples)
    x2 = np.sin(2.2 * t) + 0.22 * t + 0.18 * rng.normal(size=n_samples)
    return np.column_stack([x1, x2])


def make_surface_data(n_samples: int = 420, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-2.5, 2.5, size=n_samples)
    v = rng.uniform(-2.5, 2.5, size=n_samples)

    x = u + 0.08 * rng.normal(size=n_samples)
    y = v + 0.08 * rng.normal(size=n_samples)
    z = np.sin(u) * np.cos(v) + 0.12 * u + 0.10 * rng.normal(size=n_samples)
    return np.column_stack([x, y, z])


def fit_principal_curve(
    x: np.ndarray,
    max_iter: int = 30,
    smooth_factor: float = 0.25,
    projection_grid_size: int = 240,
    tol: float = 2e-3,
) -> CurveResult:
    data = _validate_data_matrix(x, "curve_data")
    n_samples, dim = data.shape

    if dim < 2:
        raise ValueError("principal curve demo expects at least 2D data")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if projection_grid_size < 20:
        raise ValueError("projection_grid_size must be >= 20")

    pca = PCA(n_components=1)
    params = pca.fit_transform(data).ravel()
    params = (params - np.mean(params)) / (np.std(params) + 1e-12)

    splines: list[UnivariateSpline] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        order = np.argsort(params)
        t_sorted = _strictly_increasing(params[order])

        splines = []
        for j in range(dim):
            y_sorted = data[order, j]
            # Smoothness scales with sample size and data variance.
            smoothness = smooth_factor * n_samples * float(np.var(y_sorted))
            spline = UnivariateSpline(t_sorted, y_sorted, s=smoothness, k=3)
            splines.append(spline)

        low, high = np.percentile(params, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.min(params))
            high = float(np.max(params))
        if low == high:
            high = low + 1e-3

        grid = np.linspace(low, high, projection_grid_size)
        curve_grid = np.column_stack([spline(grid) for spline in splines])

        dist2 = np.sum((data[:, None, :] - curve_grid[None, :, :]) ** 2, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        new_params = grid[nearest_idx]

        delta = float(np.mean(np.abs(new_params - params)))
        params = 0.25 * params + 0.75 * new_params

        if delta < tol:
            converged = True
            break

    fitted_points = np.column_stack([spline(params) for spline in splines])
    mse = float(np.mean(np.sum((data - fitted_points) ** 2, axis=1)))

    return CurveResult(
        params=params,
        fitted_points=fitted_points,
        mse=mse,
        iterations=iteration,
        converged=converged,
    )


def fit_principal_surface(
    x: np.ndarray,
    max_iter: int = 14,
    alpha: float = 0.08,
    gamma: float = 0.55,
    grid_size: int = 30,
    tol: float = 2e-3,
) -> SurfaceResult:
    data = _validate_data_matrix(x, "surface_data")
    n_samples, dim = data.shape

    if dim < 3:
        raise ValueError("principal surface demo expects at least 3D data")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if grid_size < 10:
        raise ValueError("grid_size must be >= 10")

    pca2 = PCA(n_components=2)
    latent_uv = pca2.fit_transform(data)
    latent_uv = (latent_uv - np.mean(latent_uv, axis=0)) / (
        np.std(latent_uv, axis=0) + 1e-12
    )

    models: list[KernelRidge] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        models = []
        for j in range(dim):
            model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            model.fit(latent_uv, data[:, j])
            models.append(model)

        u_low, u_high = np.percentile(latent_uv[:, 0], [2.0, 98.0])
        v_low, v_high = np.percentile(latent_uv[:, 1], [2.0, 98.0])
        if u_low == u_high:
            u_high = u_low + 1e-3
        if v_low == v_high:
            v_high = v_low + 1e-3

        u_grid = np.linspace(u_low, u_high, grid_size)
        v_grid = np.linspace(v_low, v_high, grid_size)
        uu, vv = np.meshgrid(u_grid, v_grid, indexing="xy")
        uv_grid = np.column_stack([uu.ravel(), vv.ravel()])

        surface_grid = np.column_stack([model.predict(uv_grid) for model in models])

        dist2 = np.sum((data[:, None, :] - surface_grid[None, :, :]) ** 2, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        new_uv = uv_grid[nearest_idx]

        delta = float(np.mean(np.linalg.norm(new_uv - latent_uv, axis=1)))
        latent_uv = 0.40 * latent_uv + 0.60 * new_uv

        if delta < tol:
            converged = True
            break

    fitted_points = np.column_stack([model.predict(latent_uv) for model in models])
    mse = float(np.mean(np.sum((data - fitted_points) ** 2, axis=1)))

    return SurfaceResult(
        latent_uv=latent_uv,
        fitted_points=fitted_points,
        mse=mse,
        iterations=iteration,
        converged=converged,
    )


def pca_reconstruction_mse(x: np.ndarray, n_components: int) -> float:
    data = _validate_data_matrix(x, "pca_data")
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(transformed)
    return float(np.mean(np.sum((data - reconstructed) ** 2, axis=1)))


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    curve_data = make_curve_data()
    curve_result = fit_principal_curve(curve_data)
    pca1_mse = pca_reconstruction_mse(curve_data, n_components=1)

    print("=== Principal Curve Demo (2D) ===")
    print(f"samples={curve_data.shape[0]}")
    print(
        f"iterations={curve_result.iterations}, converged={curve_result.converged}, "
        f"principal_curve_mse={curve_result.mse:.6f}, pca1_mse={pca1_mse:.6f}, "
        f"improvement={pca1_mse / curve_result.mse:.3f}x"
    )
    print("first_5_fitted_points:")
    print(curve_result.fitted_points[:5])
    print()

    surface_data = make_surface_data()
    surface_result = fit_principal_surface(surface_data)
    pca2_mse = pca_reconstruction_mse(surface_data, n_components=2)

    print("=== Principal Surface Demo (3D) ===")
    print(f"samples={surface_data.shape[0]}")
    print(
        f"iterations={surface_result.iterations}, converged={surface_result.converged}, "
        f"principal_surface_mse={surface_result.mse:.6f}, pca2_mse={pca2_mse:.6f}, "
        f"improvement={pca2_mse / surface_result.mse:.3f}x"
    )
    print("first_5_fitted_points:")
    print(surface_result.fitted_points[:5])


if __name__ == "__main__":
    main()
