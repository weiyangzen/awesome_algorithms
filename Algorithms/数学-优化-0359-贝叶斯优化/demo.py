"""Bayesian Optimization MVP with a hand-written Gaussian Process surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class GpPosterior:
    x_train: Array
    y_mean: float
    y_std: float
    length_scale: float
    signal_variance: float
    noise_variance: float
    chol_l: Array
    alpha: Array


def ensure_1d_vector(name: str, x: Array) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def ensure_2d_array(name: str, x: Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def forrester_function(x: float) -> float:
    """Standard 1D BO benchmark on [0, 1]."""
    t = 6.0 * x - 2.0
    return float((t * t) * np.sin(12.0 * x - 4.0))


def rbf_kernel(xa: Array, xb: Array, length_scale: float, signal_variance: float) -> Array:
    ensure_2d_array("xa", xa)
    ensure_2d_array("xb", xb)
    if xa.shape[1] != xb.shape[1]:
        raise ValueError("xa/xb feature dimension mismatch.")
    if length_scale <= 0.0 or signal_variance <= 0.0:
        raise ValueError("length_scale and signal_variance must be positive.")

    scaled_diff = (xa[:, None, :] - xb[None, :, :]) / length_scale
    sq_dist = np.sum(scaled_diff * scaled_diff, axis=2)
    return signal_variance * np.exp(-0.5 * sq_dist)


def fit_gp_posterior(
    x_train: Array,
    y_train: Array,
    length_scale: float = 0.2,
    signal_variance: float = 1.0,
    noise_variance: float = 1e-2,
    jitter: float = 1e-8,
) -> GpPosterior:
    ensure_2d_array("x_train", x_train)
    ensure_1d_vector("y_train", y_train)
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have same number of samples.")
    if x_train.shape[0] < 2:
        raise ValueError("Need at least 2 observations to fit GP posterior.")
    if noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative.")

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    y_normalized = (y_train - y_mean) / y_std

    k_mat = rbf_kernel(
        xa=x_train,
        xb=x_train,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    k_mat = k_mat + (noise_variance + jitter) * np.eye(x_train.shape[0], dtype=float)

    try:
        chol_l = np.linalg.cholesky(k_mat)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Cholesky decomposition failed; kernel matrix not SPD.") from exc

    alpha = np.linalg.solve(chol_l.T, np.linalg.solve(chol_l, y_normalized))
    if not np.all(np.isfinite(alpha)):
        raise RuntimeError("GP solve produced non-finite alpha values.")
    return GpPosterior(
        x_train=x_train,
        y_mean=y_mean,
        y_std=y_std,
        length_scale=length_scale,
        signal_variance=signal_variance,
        noise_variance=noise_variance,
        chol_l=chol_l,
        alpha=alpha,
    )


def gp_predict(posterior: GpPosterior, x_query: Array) -> Tuple[Array, Array]:
    ensure_2d_array("x_query", x_query)
    if x_query.shape[1] != posterior.x_train.shape[1]:
        raise ValueError("x_query feature dimension mismatch.")

    k_xs = rbf_kernel(
        xa=posterior.x_train,
        xb=x_query,
        length_scale=posterior.length_scale,
        signal_variance=posterior.signal_variance,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean_norm = np.sum(k_xs * posterior.alpha[:, None], axis=0)
    if not np.all(np.isfinite(mean_norm)):
        mean_norm = np.nan_to_num(mean_norm, nan=0.0, posinf=0.0, neginf=0.0)

    v = np.linalg.solve(posterior.chol_l, k_xs)
    prior_var = posterior.signal_variance * np.ones(x_query.shape[0], dtype=float)
    var_norm = np.maximum(prior_var - np.sum(v * v, axis=0), 1e-15)

    mean = posterior.y_mean + posterior.y_std * mean_norm
    std = posterior.y_std * np.sqrt(var_norm)
    return mean.astype(float), std.astype(float)


def expected_improvement(
    mu: Array,
    sigma: Array,
    best_y: float,
    xi: float = 0.01,
) -> Array:
    ensure_1d_vector("mu", mu)
    ensure_1d_vector("sigma", sigma)
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma shape mismatch.")
    if xi < 0.0:
        raise ValueError("xi must be non-negative.")

    improve = best_y - mu - xi
    safe_sigma = np.maximum(sigma, 1e-15)
    z = improve / safe_sigma
    cdf_z = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = improve * cdf_z + safe_sigma * pdf_z
    ei = np.where(sigma <= 1e-14, 0.0, ei)
    return np.maximum(ei, 0.0)


def is_duplicate(x_value: float, x_seen: Array, tol: float = 5e-3) -> bool:
    return bool(np.any(np.abs(x_seen[:, 0] - x_value) <= tol))


def golden_section_maximize(
    func,
    lower: float,
    upper: float,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> Tuple[float, float]:
    if not (lower < upper):
        raise ValueError("golden section requires lower < upper.")

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    a = float(lower)
    b = float(upper)
    c = b - (b - a) * inv_phi
    d = a + (b - a) * inv_phi
    fc = float(func(c))
    fd = float(func(d))

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * inv_phi
            fc = float(func(c))
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * inv_phi
            fd = float(func(d))

    x_best = c if fc >= fd else d
    y_best = max(fc, fd)
    return float(x_best), float(y_best)


def propose_next_point(
    posterior: GpPosterior,
    bounds: Tuple[float, float],
    x_seen: Array,
    y_seen: Array,
    rng: np.random.Generator,
    n_candidates: int = 2000,
    n_restarts: int = 6,
) -> Tuple[float, float]:
    lower, upper = bounds
    if not (lower < upper):
        raise ValueError("bounds must satisfy lower < upper.")

    # Step 1: global random scan over acquisition.
    x_rand = rng.uniform(lower, upper, size=(n_candidates, 1))
    mu_rand, sigma_rand = gp_predict(posterior, x_rand)
    ei_rand = expected_improvement(mu_rand, sigma_rand, best_y=float(np.min(y_seen)))
    order = np.argsort(-ei_rand)

    best_x = float(x_rand[order[0], 0])
    best_ei = float(ei_rand[order[0]])

    for idx in order:
        candidate = float(x_rand[idx, 0])
        if not is_duplicate(candidate, x_seen):
            best_x = candidate
            best_ei = float(ei_rand[idx])
            break

    # Step 2: local refinement by golden-section maximize on EI.
    def ei_at_scalar(x_scalar: float) -> float:
        x_clipped = float(np.clip(x_scalar, lower, upper))
        xq = np.array([[x_clipped]], dtype=float)
        mu, sigma = gp_predict(posterior, xq)
        ei_val = expected_improvement(mu, sigma, best_y=float(np.min(y_seen)))[0]
        return float(ei_val)

    seeds = x_rand[order[:n_restarts], 0]
    local_radius = 0.2 * (upper - lower)
    for seed in seeds:
        left = float(max(lower, seed - local_radius))
        right = float(min(upper, seed + local_radius))
        if right - left < 1e-12:
            continue
        candidate, ei_loc = golden_section_maximize(
            func=ei_at_scalar,
            lower=left,
            upper=right,
        )
        if is_duplicate(candidate, x_seen):
            continue

        if ei_loc > best_ei:
            best_x = candidate
            best_ei = ei_loc

    # Step 3: if still duplicate (rare), force a farthest random point.
    if is_duplicate(best_x, x_seen):
        distances = np.min(np.abs(x_rand - x_seen.T), axis=1)
        best_x = float(x_rand[int(np.argmax(distances)), 0])
        mu_far, sigma_far = gp_predict(posterior, np.array([[best_x]], dtype=float))
        best_ei = float(expected_improvement(mu_far, sigma_far, best_y=float(np.min(y_seen)))[0])

    return best_x, best_ei


def bayesian_optimization(
    objective,
    bounds: Tuple[float, float],
    initial_points: Sequence[float],
    n_iter: int,
    rng_seed: int = 7,
) -> Tuple[float, float, Array, Array, List[Dict[str, float]]]:
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    lower, upper = bounds
    x_init = np.array(initial_points, dtype=float)
    ensure_1d_vector("initial_points", x_init)
    if np.any((x_init < lower) | (x_init > upper)):
        raise ValueError("initial_points must stay within bounds.")

    x_obs = x_init.reshape(-1, 1)
    y_obs = np.array([objective(float(x)) for x in x_init], dtype=float)

    rng = np.random.default_rng(rng_seed)
    history: List[Dict[str, float]] = []

    for i in range(1, n_iter + 1):
        posterior = fit_gp_posterior(x_obs, y_obs)
        x_next, ei_value = propose_next_point(
            posterior=posterior,
            bounds=bounds,
            x_seen=x_obs,
            y_seen=y_obs,
            rng=rng,
        )
        y_next = float(objective(x_next))

        x_obs = np.vstack([x_obs, np.array([[x_next]], dtype=float)])
        y_obs = np.concatenate([y_obs, np.array([y_next], dtype=float)])

        best_idx = int(np.argmin(y_obs))
        history.append(
            {
                "iter": float(i),
                "x_next": x_next,
                "y_next": y_next,
                "best_x": float(x_obs[best_idx, 0]),
                "best_y": float(y_obs[best_idx]),
                "ei": float(ei_value),
            }
        )

    best_idx = int(np.argmin(y_obs))
    best_x = float(x_obs[best_idx, 0])
    best_y = float(y_obs[best_idx])
    return best_x, best_y, x_obs, y_obs, history


def approximate_true_minimum(objective, bounds: Tuple[float, float], grid_size: int = 20000) -> Tuple[float, float]:
    lower, upper = bounds
    grid = np.linspace(lower, upper, num=grid_size, dtype=float)
    values = np.array([objective(float(x)) for x in grid], dtype=float)
    idx = int(np.argmin(values))
    return float(grid[idx]), float(values[idx])


def print_history(history: Sequence[Dict[str, float]], max_lines: int = 15) -> None:
    print("iter | x_next       | y_next        | best_x       | best_y        | EI")
    print("-" * 82)
    for row in history[:max_lines]:
        print(
            f"{int(row['iter']):4d} | {row['x_next']:12.8f} | {row['y_next']:12.8f} | "
            f"{row['best_x']:12.8f} | {row['best_y']:12.8f} | {row['ei']:.6e}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def main() -> None:
    bounds = (0.0, 1.0)
    initial_points = [0.05, 0.35, 0.65, 0.95]
    bo_iterations = 18

    best_x, best_y, x_obs, y_obs, history = bayesian_optimization(
        objective=forrester_function,
        bounds=bounds,
        initial_points=initial_points,
        n_iter=bo_iterations,
        rng_seed=7,
    )

    ref_x, ref_y = approximate_true_minimum(forrester_function, bounds)
    abs_x_err = abs(best_x - ref_x)
    abs_y_err = abs(best_y - ref_y)

    print("=== Bayesian Optimization MVP (Forrester, minimize) ===")
    print(f"search bounds: {bounds}")
    print(f"initial points: {initial_points}")
    print(f"iterations: {bo_iterations}")
    print(f"total evaluations: {x_obs.shape[0]}")
    print_history(history)

    print("\n=== Final Result ===")
    print(f"BO best x: {best_x:.10f}")
    print(f"BO best y: {best_y:.10f}")
    print(f"Reference x (dense grid): {ref_x:.10f}")
    print(f"Reference y (dense grid): {ref_y:.10f}")
    print(f"Absolute x error: {abs_x_err:.6e}")
    print(f"Absolute y error: {abs_y_err:.6e}")

    pass_flag = bool(abs_x_err < 5e-2 and abs_y_err < 5e-1)
    print(f"Pass loose accuracy check: {pass_flag}")


if __name__ == "__main__":
    main()
