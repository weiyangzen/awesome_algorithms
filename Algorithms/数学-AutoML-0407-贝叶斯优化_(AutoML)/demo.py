"""Bayesian Optimization MVP for AutoML hyper-parameter tuning.

This script tunes two log-scale hyper-parameters of Kernel Ridge Regression:
- log10(alpha): regularization strength
- log10(gamma): RBF kernel width

The optimizer is implemented explicitly with:
1) Gaussian Process surrogate modeling
2) Expected Improvement (EI) acquisition
3) Candidate-based acquisition maximization

The script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HistoryItem = Tuple[int, float, float, float, float]
# (eval_id, current_rmse, best_rmse, log10_alpha, log10_gamma)


@dataclass
class BOResult:
    best_x: np.ndarray
    best_y: float
    x_obs: np.ndarray
    y_obs: np.ndarray
    history: List[HistoryItem]


def validate_bounds(bounds: np.ndarray) -> None:
    if bounds.ndim != 2:
        raise ValueError(f"bounds must be 2D, got shape={bounds.shape}.")
    if bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("bounds must be finite.")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each bound must satisfy lower < upper.")


def sample_uniform(bounds: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return rng.uniform(lower, upper, size=(n_points, bounds.shape[0]))


def to_hyperparams(log_params: np.ndarray) -> Dict[str, float]:
    return {
        "alpha": float(10.0 ** log_params[0]),
        "gamma": float(10.0 ** log_params[1]),
    }


def make_automl_dataset(seed: int = 2026, n_samples: int = 280) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic non-linear regression dataset for AutoML tuning."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n_samples, 6))

    y_clean = (
        np.sin(1.7 * x[:, 0])
        + 0.65 * (x[:, 1] ** 2)
        - 1.15 * x[:, 2] * x[:, 3]
        + 0.30 * x[:, 4]
        - 0.25 * np.cos(2.0 * x[:, 5])
    )
    noise = 0.20 * rng.normal(size=n_samples)
    y = y_clean + noise

    return x.astype(float), y.astype(float)


def evaluate_cv_rmse(
    log_params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    penalty: float = 1e6,
) -> float:
    """Expensive black-box objective: mean CV RMSE (to minimize)."""
    params = to_hyperparams(log_params)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "krr",
                KernelRidge(
                    kernel="rbf",
                    alpha=params["alpha"],
                    gamma=params["gamma"],
                ),
            ),
        ]
    )

    try:
        neg_mse_scores = cross_val_score(
            model,
            x,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=None,
            error_score="raise",
        )
        rmse_scores = np.sqrt(np.maximum(-neg_mse_scores, 0.0))
        mean_rmse = float(np.mean(rmse_scores))
        if not np.isfinite(mean_rmse):
            return float(penalty)
        return mean_rmse
    except Exception:
        return float(penalty)


def build_gp_model(dim: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )


def expected_improvement(
    candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """EI for minimization objective."""
    mu, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-12)

    improvement = best_y - mu - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    near_zero = std <= 1e-12
    ei[near_zero] = 0.0
    return ei


def choose_next_point(
    bounds: np.ndarray,
    gp: GaussianProcessRegressor,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    candidates = sample_uniform(bounds, n_candidates, rng)
    ei_values = expected_improvement(candidates, gp, best_y=float(np.min(y_obs)), xi=0.01)

    ranked = np.argsort(-ei_values)
    for idx in ranked:
        point = candidates[idx]
        # Skip near-duplicate points to avoid redundant expensive evaluations.
        min_dist = float(np.min(np.linalg.norm(x_obs - point, axis=1)))
        if min_dist > 1e-9:
            return point, float(ei_values[idx])

    # Very unlikely fallback.
    return candidates[ranked[0]], float(ei_values[ranked[0]])


def bayesian_optimization(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_init: int = 6,
    n_iter: int = 18,
    n_candidates: int = 3000,
    seed: int = 2026,
) -> BOResult:
    validate_bounds(bounds)
    if n_init <= 0 or n_iter <= 0 or n_candidates <= 0:
        raise ValueError("n_init, n_iter, n_candidates must all be positive.")

    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]

    x_obs = sample_uniform(bounds, n_init, rng)
    y_obs = np.array([objective(xi) for xi in x_obs], dtype=float)

    history: List[HistoryItem] = []
    running_best = float(np.min(y_obs))
    for i in range(n_init):
        running_best = min(running_best, float(y_obs[i]))
        history.append((i + 1, float(y_obs[i]), running_best, float(x_obs[i, 0]), float(x_obs[i, 1])))

    for step in range(1, n_iter + 1):
        gp = build_gp_model(dim=dim)
        gp.fit(x_obs, y_obs)

        x_next, best_ei = choose_next_point(
            bounds=bounds,
            gp=gp,
            x_obs=x_obs,
            y_obs=y_obs,
            n_candidates=n_candidates,
            rng=rng,
        )
        y_next = float(objective(x_next))

        x_obs = np.vstack([x_obs, x_next])
        y_obs = np.append(y_obs, y_next)

        running_best = min(running_best, y_next)
        eval_id = n_init + step
        history.append((eval_id, y_next, running_best, float(x_next[0]), float(x_next[1])))

        print(
            f"[BO] eval={eval_id:02d}, rmse={y_next:.6f}, best_rmse={running_best:.6f}, "
            f"EI={best_ei:.6e}, log_alpha={x_next[0]:.3f}, log_gamma={x_next[1]:.3f}"
        )

    best_idx = int(np.argmin(y_obs))
    return BOResult(
        best_x=x_obs[best_idx].copy(),
        best_y=float(y_obs[best_idx]),
        x_obs=x_obs,
        y_obs=y_obs,
        history=history,
    )


def random_search_baseline(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    total_evals: int,
    seed: int = 999,
) -> Tuple[np.ndarray, float]:
    if total_evals <= 0:
        raise ValueError("total_evals must be positive.")

    rng = np.random.default_rng(seed)
    points = sample_uniform(bounds, total_evals, rng)
    scores = np.array([objective(xi) for xi in points], dtype=float)

    best_idx = int(np.argmin(scores))
    return points[best_idx], float(scores[best_idx])


def print_history_snippet(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("eval | rmse(current) | rmse(best)   | log10(alpha) | log10(gamma)")
    print("---------------------------------------------------------------")

    show = min(len(history), max_lines)
    for i in range(show):
        eval_id, y_cur, y_best, la, lg = history[i]
        print(f"{eval_id:4d} | {y_cur:12.6f} | {y_best:11.6f} | {la:12.5f} | {lg:12.5f}")

    if len(history) > max_lines:
        omitted = len(history) - max_lines
        eval_id, y_cur, y_best, la, lg = history[-1]
        print(f"... ({omitted} evaluations omitted)")
        print(f"{eval_id:4d} | {y_cur:12.6f} | {y_best:11.6f} | {la:12.5f} | {lg:12.5f}  (last)")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    x, y = make_automl_dataset(seed=2026, n_samples=280)
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)

    bounds = np.array(
        [
            [-6.0, 1.0],  # log10(alpha)
            [-6.0, 1.0],  # log10(gamma)
        ],
        dtype=float,
    )

    n_init = 6
    n_iter = 18
    n_candidates = 3000
    total_budget = n_init + n_iter

    print("Bayesian Optimization for AutoML hyper-parameter tuning")
    print(f"dataset: X.shape={x.shape}, y.shape={y.shape}")
    print(f"search space (log-scale): {bounds.tolist()}")
    print(f"budget: n_init={n_init}, n_iter={n_iter}, total={total_budget}")

    objective = lambda log_params: evaluate_cv_rmse(log_params, x=x, y=y, cv=cv)

    bo_result = bayesian_optimization(
        objective=objective,
        bounds=bounds,
        n_init=n_init,
        n_iter=n_iter,
        n_candidates=n_candidates,
        seed=2026,
    )

    baseline_x, baseline_y = random_search_baseline(
        objective=objective,
        bounds=bounds,
        total_evals=total_budget,
        seed=17,
    )

    best_params = to_hyperparams(bo_result.best_x)
    baseline_params = to_hyperparams(baseline_x)

    print("\n=== BO History Snippet ===")
    print_history_snippet(bo_result.history, max_lines=12)

    print("\n=== Best by Bayesian Optimization ===")
    print(
        f"best_rmse={bo_result.best_y:.6f}, "
        f"log_alpha={bo_result.best_x[0]:.5f}, log_gamma={bo_result.best_x[1]:.5f}, "
        f"alpha={best_params['alpha']:.6e}, gamma={best_params['gamma']:.6e}"
    )

    print("\n=== Best by Random Search (same budget) ===")
    print(
        f"best_rmse={baseline_y:.6f}, "
        f"log_alpha={baseline_x[0]:.5f}, log_gamma={baseline_x[1]:.5f}, "
        f"alpha={baseline_params['alpha']:.6e}, gamma={baseline_params['gamma']:.6e}"
    )

    improvement = baseline_y - bo_result.best_y
    print("\n=== Summary Checks ===")
    print(f"improvement_vs_random={improvement:.6f} (positive means BO is better)")
    print(f"all_scores_finite={bool(np.all(np.isfinite(bo_result.y_obs)))}")
    print(f"history_length_ok={len(bo_result.history) == total_budget}")


if __name__ == "__main__":
    main()
