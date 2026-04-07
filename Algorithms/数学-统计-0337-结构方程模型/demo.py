"""Minimal runnable MVP for Structural Equation Modeling (MATH-0337).

Model in this demo:
- Two latent variables: exogenous xi and endogenous eta.
- Structural equation: eta = beta * xi + zeta.
- Measurement equations:
  x1,x2,x3 load on xi; y1,y2,y3 load on eta.

The script:
1) generates synthetic data from known parameters,
2) fits SEM parameters by minimizing Gaussian covariance discrepancy F_ML,
3) reports parameter recovery and fit diagnostics.

No black-box SEM package is used. Covariance mapping and objective are implemented
explicitly in source for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


EPS = 1e-8


@dataclass(frozen=True)
class SEMConfig:
    """Reproducible settings for one SEM experiment run."""

    n_samples: int = 1800
    seed: int = 337
    maxiter: int = 500


@dataclass(frozen=True)
class SEMTrueParams:
    """Ground-truth parameters used to synthesize data."""

    loadings_x: Tuple[float, float, float] = (1.0, 0.90, 0.75)
    loadings_y: Tuple[float, float, float] = (1.0, 0.85, 0.95)
    beta: float = 0.72
    var_xi: float = 1.15
    var_zeta: float = 0.55
    theta_diag: Tuple[float, float, float, float, float, float] = (
        0.30,
        0.36,
        0.42,
        0.28,
        0.33,
        0.40,
    )


@dataclass(frozen=True)
class OptimizationResult:
    """Small optimizer status object with scipy-like fields."""

    x: np.ndarray
    success: bool
    message: str
    nit: int
    fun: float


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus to keep variance parameters positive."""
    x_arr = np.asarray(x)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
    if np.isscalar(x):
        return float(out)
    return out


def build_lambda(loadings_x: np.ndarray, loadings_y: np.ndarray) -> np.ndarray:
    """Construct 6x2 loading matrix for (xi, eta)."""
    lam = np.array(
        [
            [loadings_x[0], 0.0],
            [loadings_x[1], 0.0],
            [loadings_x[2], 0.0],
            [0.0, loadings_y[0]],
            [0.0, loadings_y[1]],
            [0.0, loadings_y[2]],
        ],
        dtype=np.float64,
    )
    return lam


def latent_covariance(beta: float, var_xi: float, var_zeta: float) -> np.ndarray:
    """Latent covariance implied by eta = beta*xi + zeta."""
    cov_xi_eta = beta * var_xi
    var_eta = beta * beta * var_xi + var_zeta
    return np.array(
        [[var_xi, cov_xi_eta], [cov_xi_eta, var_eta]],
        dtype=np.float64,
    )


def simulate_sem_data(params: SEMTrueParams, config: SEMConfig) -> pd.DataFrame:
    """Generate observed SEM data from the configured true parameters."""
    rng = np.random.default_rng(config.seed)

    var_xi = params.var_xi
    var_zeta = params.var_zeta
    beta = params.beta

    xi = rng.normal(loc=0.0, scale=np.sqrt(var_xi), size=config.n_samples)
    zeta = rng.normal(loc=0.0, scale=np.sqrt(var_zeta), size=config.n_samples)
    eta = beta * xi + zeta

    theta = np.array(params.theta_diag, dtype=np.float64)
    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(theta),
        size=(config.n_samples, 6),
    )

    lx = np.array(params.loadings_x, dtype=np.float64)
    ly = np.array(params.loadings_y, dtype=np.float64)

    x_block = np.column_stack(
        [
            lx[0] * xi + noise[:, 0],
            lx[1] * xi + noise[:, 1],
            lx[2] * xi + noise[:, 2],
        ]
    )
    y_block = np.column_stack(
        [
            ly[0] * eta + noise[:, 3],
            ly[1] * eta + noise[:, 4],
            ly[2] * eta + noise[:, 5],
        ]
    )

    data = np.column_stack([x_block, y_block])
    columns = ["x1", "x2", "x3", "y1", "y2", "y3"]
    return pd.DataFrame(data, columns=columns)


def unpack_raw_params(raw: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Map unconstrained optimization vector to SEM parameters."""
    if raw.shape[0] != 13:
        raise ValueError(f"Expected 13 raw params, got {raw.shape[0]}.")

    l_x2, l_x3, l_y2, l_y3, beta = raw[:5]
    var_xi = float(softplus(raw[5]) + EPS)
    var_zeta = float(softplus(raw[6]) + EPS)
    theta_diag = np.asarray(softplus(raw[7:13]) + EPS, dtype=np.float64)

    loadings_x = np.array([1.0, l_x2, l_x3], dtype=np.float64)
    loadings_y = np.array([1.0, l_y2, l_y3], dtype=np.float64)

    return {
        "loadings_x": loadings_x,
        "loadings_y": loadings_y,
        "beta": float(beta),
        "var_xi": var_xi,
        "var_zeta": var_zeta,
        "theta_diag": theta_diag,
    }


def implied_covariance_from_raw(raw: np.ndarray) -> np.ndarray:
    """Compute model-implied covariance Sigma(theta)."""
    p = unpack_raw_params(raw)
    lam = build_lambda(
        loadings_x=np.asarray(p["loadings_x"]),
        loadings_y=np.asarray(p["loadings_y"]),
    )
    phi = latent_covariance(
        beta=float(p["beta"]),
        var_xi=float(p["var_xi"]),
        var_zeta=float(p["var_zeta"]),
    )
    theta = np.diag(np.asarray(p["theta_diag"], dtype=np.float64))
    sigma = lam @ phi @ lam.T + theta
    return sigma


def f_ml_objective(raw: np.ndarray, sample_cov: np.ndarray, logdet_sample_cov: float) -> float:
    """Gaussian SEM discrepancy F_ML = log|Sigma| + tr(S Sigma^-1) - log|S| - p."""
    p_dim = sample_cov.shape[0]

    sigma = implied_covariance_from_raw(raw)
    sign_sigma, logdet_sigma = np.linalg.slogdet(sigma)
    if sign_sigma <= 0 or not np.isfinite(logdet_sigma):
        return 1e12

    try:
        solve_sigma_s = np.linalg.solve(sigma, sample_cov)
    except np.linalg.LinAlgError:
        return 1e12

    trace_term = float(np.trace(solve_sigma_s))
    f_ml = float(logdet_sigma + trace_term - logdet_sample_cov - p_dim)
    if not np.isfinite(f_ml):
        return 1e12
    return f_ml


def finite_diff_grad(
    objective: callable,
    x: np.ndarray,
    step: float = 1e-4,
) -> np.ndarray:
    """Central-difference gradient for small-dimensional unconstrained params."""
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        fp = objective(xp)
        fm = objective(xm)
        if np.isfinite(fp) and np.isfinite(fm):
            grad[i] = (fp - fm) / (2.0 * step)
        else:
            grad[i] = 0.0
    return grad


def optimize_without_scipy(
    objective: callable,
    x0: np.ndarray,
    maxiter: int,
    seed: int = 2026,
) -> OptimizationResult:
    """Fallback optimizer: finite-difference Adam + backtracking."""
    rng = np.random.default_rng(seed)
    x = x0.copy()
    fx = float(objective(x))
    best_x = x.copy()
    best_fx = fx

    m = np.zeros_like(x, dtype=np.float64)
    v = np.zeros_like(x, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    base_lr = 0.08

    success = np.isfinite(best_fx)
    message = "max iterations reached"
    stale_rounds = 0
    it = 0

    for it in range(1, maxiter + 1):
        prev_best = best_fx
        grad = finite_diff_grad(objective=objective, x=x, step=1e-4)
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm):
            message = "non-finite gradient encountered"
            break

        if grad_norm < 1e-6:
            success = True
            message = "gradient norm small enough"
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**it)
        v_hat = v / (1.0 - beta2**it)
        direction = m_hat / (np.sqrt(v_hat) + 1e-8)

        step_size = base_lr
        accepted = False
        for _ in range(10):
            cand = x - step_size * direction
            f_cand = float(objective(cand))
            if np.isfinite(f_cand) and f_cand <= fx:
                x = cand
                fx = f_cand
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            # Mild random perturbation to escape flat/poor local area.
            cand = x + rng.normal(loc=0.0, scale=0.02, size=x.shape[0])
            f_cand = float(objective(cand))
            if np.isfinite(f_cand) and f_cand < fx:
                x = cand
                fx = f_cand

        if fx < best_fx:
            best_fx = fx
            best_x = x.copy()

        if prev_best - best_fx < 1e-9:
            stale_rounds += 1
        else:
            stale_rounds = 0

        if stale_rounds >= 25:
            message = "objective improvement below tolerance"
            break

    return OptimizationResult(
        x=best_x,
        success=success and np.isfinite(best_fx),
        message=message,
        nit=it,
        fun=best_fx,
    )


def fit_sem_covariance_mle(data: pd.DataFrame, maxiter: int) -> Tuple[np.ndarray, np.ndarray, float, OptimizationResult]:
    """Estimate SEM parameters by minimizing covariance discrepancy."""
    x = data.to_numpy(dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    n = x_centered.shape[0]
    sample_cov = (x_centered.T @ x_centered) / n

    sign_s, logdet_s = np.linalg.slogdet(sample_cov)
    if sign_s <= 0:
        raise RuntimeError("Sample covariance is not positive definite.")

    # Unconstrained initial point; variances become positive via softplus.
    raw0 = np.zeros(13, dtype=np.float64)
    raw0[:5] = np.array([0.7, 0.7, 0.7, 0.7, 0.5], dtype=np.float64)

    objective = lambda raw: f_ml_objective(raw, sample_cov, float(logdet_s))
    if HAS_SCIPY:
        raw_result = minimize(
            f_ml_objective,
            x0=raw0,
            args=(sample_cov, float(logdet_s)),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "disp": False},
        )
        result = OptimizationResult(
            x=np.asarray(raw_result.x, dtype=np.float64),
            success=bool(raw_result.success),
            message=str(raw_result.message),
            nit=int(raw_result.nit),
            fun=float(raw_result.fun),
        )
    else:
        result = optimize_without_scipy(
            objective=objective,
            x0=raw0,
            maxiter=maxiter,
            seed=337,
        )

    est_raw = np.asarray(result.x, dtype=np.float64)
    est_sigma = implied_covariance_from_raw(est_raw)
    est_fml = f_ml_objective(est_raw, sample_cov, float(logdet_s))
    return est_raw, est_sigma, est_fml, result


def format_recovery_table(true_params: SEMTrueParams, est_raw: np.ndarray) -> pd.DataFrame:
    """Build a compact table comparing true and estimated core parameters."""
    est = unpack_raw_params(est_raw)

    true_names: List[str] = [
        "lambda_x2",
        "lambda_x3",
        "lambda_y2",
        "lambda_y3",
        "beta",
        "var_xi",
        "var_zeta",
        "theta_x1",
        "theta_x2",
        "theta_x3",
        "theta_y1",
        "theta_y2",
        "theta_y3",
    ]
    true_values = [
        true_params.loadings_x[1],
        true_params.loadings_x[2],
        true_params.loadings_y[1],
        true_params.loadings_y[2],
        true_params.beta,
        true_params.var_xi,
        true_params.var_zeta,
        *list(true_params.theta_diag),
    ]

    est_values = [
        float(np.asarray(est["loadings_x"])[1]),
        float(np.asarray(est["loadings_x"])[2]),
        float(np.asarray(est["loadings_y"])[1]),
        float(np.asarray(est["loadings_y"])[2]),
        float(est["beta"]),
        float(est["var_xi"]),
        float(est["var_zeta"]),
        *[float(v) for v in np.asarray(est["theta_diag"])],
    ]

    df = pd.DataFrame(
        {
            "parameter": true_names,
            "true": true_values,
            "estimated": est_values,
        }
    )
    df["abs_error"] = (df["estimated"] - df["true"]).abs()
    return df


def main() -> None:
    cfg = SEMConfig()
    true_params = SEMTrueParams()

    data = simulate_sem_data(params=true_params, config=cfg)
    est_raw, est_sigma, est_fml, opt_result = fit_sem_covariance_mle(data=data, maxiter=cfg.maxiter)

    x = data.to_numpy(dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    n, p = x_centered.shape
    sample_cov = (x_centered.T @ x_centered) / n

    resid = sample_cov - est_sigma
    rms_cov_resid = float(np.sqrt(np.mean(resid * resid)))

    # Approximate chi-square style statistics.
    n_stats = n - 1
    chi2 = n_stats * est_fml
    free_params = 13
    unique_cov_elems = p * (p + 1) // 2
    df = unique_cov_elems - free_params
    rmsea = float(np.sqrt(max((chi2 - df) / (df * n_stats), 0.0))) if df > 0 else 0.0

    recovery = format_recovery_table(true_params=true_params, est_raw=est_raw)

    print("Structural Equation Model MVP (MATH-0337)")
    print("=" * 78)
    optimizer_name = "L-BFGS-B(scipy)" if HAS_SCIPY else "gradient-descent(fallback)"
    print(f"n_samples={cfg.n_samples}, seed={cfg.seed}, optimizer={optimizer_name}")
    print(
        f"optimizer_success={opt_result.success}, iterations={opt_result.nit}, "
        f"message={opt_result.message}"
    )
    print("-" * 78)
    print(f"F_ML minimum: {est_fml:.6f}")
    print(f"chi-square approx: {chi2:.4f} with df={df}")
    print(f"RMSEA (approx): {rmsea:.6f}")
    print(f"RMS covariance residual: {rms_cov_resid:.6f}")
    print("-" * 78)
    print("Parameter recovery (true vs estimated):")
    print(recovery.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    # Sanity checks for this reproducible synthetic setup.
    assert np.isfinite(est_fml), "F_ML should be finite."
    assert np.all(np.isfinite(est_sigma)), "Implied covariance should be finite."

    sign_hat, _ = np.linalg.slogdet(est_sigma)
    assert sign_hat > 0, "Estimated implied covariance must be positive definite."

    beta_hat = float(unpack_raw_params(est_raw)["beta"])
    assert abs(beta_hat - true_params.beta) < 0.25, "Estimated structural coefficient deviates too much."
    assert rms_cov_resid < 0.20, "Covariance fit is unexpectedly poor for synthetic data."

    print("All checks passed.")


if __name__ == "__main__":
    main()
