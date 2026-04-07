"""Factor Analysis (EM) minimal runnable MVP.

This script implements Gaussian factor analysis with diagonal uniqueness via a
transparent EM routine (no black-box estimator call), then evaluates on a
fixed synthetic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FAHistory:
    iter: int
    avg_loglike: float
    delta: float


@dataclass
class FAResult:
    mean: np.ndarray
    loadings: np.ndarray
    psi: np.ndarray
    sigma: np.ndarray
    history: List[FAHistory]
    iterations: int
    converged: bool
    message: str


@dataclass
class SyntheticCase:
    x_train: np.ndarray
    x_test: np.ndarray
    mean_true: np.ndarray
    loadings_true: np.ndarray
    psi_true: np.ndarray


def ensure_finite_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 rows and 2 columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def average_log_likelihood(
    x_centered: np.ndarray,
    loadings: np.ndarray,
    psi: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return average log-likelihood under N(0, Sigma) and Sigma itself."""
    n, d = x_centered.shape
    sigma = loadings @ loadings.T + np.diag(psi)

    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Sigma is not SPD during log-likelihood computation") from exc

    logdet_sigma = 2.0 * float(np.sum(np.log(np.diag(chol))))
    solved = np.linalg.solve(chol, x_centered.T)
    quad = np.sum(solved * solved, axis=0)

    const = d * np.log(2.0 * np.pi) + logdet_sigma
    avg_ll = float(np.mean(-0.5 * (const + quad)))
    if not np.isfinite(avg_ll):
        raise RuntimeError("Non-finite average log-likelihood encountered")

    return avg_ll, sigma


def initialize_parameters(
    x_centered: np.ndarray,
    n_factors: int,
    min_psi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize loadings and uniqueness from sample covariance eigensystem."""
    n, d = x_centered.shape
    sxx = (x_centered.T @ x_centered) / float(n)

    eigvals, eigvecs = np.linalg.eigh(sxx)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_factors < d:
        tail = eigvals[n_factors:]
        noise_floor = float(np.mean(tail)) if tail.size > 0 else float(np.min(eigvals))
    else:
        noise_floor = float(np.min(eigvals))

    noise_floor = max(noise_floor, min_psi)
    signal = np.maximum(eigvals[:n_factors] - noise_floor, min_psi)

    loadings = eigvecs[:, :n_factors] * np.sqrt(signal)[None, :]
    psi = np.clip(np.diag(sxx) - noise_floor, min_psi, None)

    if not np.all(np.isfinite(loadings)) or not np.all(np.isfinite(psi)):
        raise RuntimeError("Initialization produced non-finite parameters")

    return loadings, psi


def e_step_stats(
    x_centered: np.ndarray,
    loadings: np.ndarray,
    psi: np.ndarray,
    ridge: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ez, Sxz, Szz for Gaussian FA E-step."""
    n, _ = x_centered.shape
    _, k = loadings.shape

    psi_inv = 1.0 / psi
    weighted_loadings = psi_inv[:, None] * loadings
    m_mat = np.eye(k, dtype=np.float64) + loadings.T @ weighted_loadings
    m_mat = m_mat + ridge * np.eye(k, dtype=np.float64)
    m_inv = np.linalg.inv(m_mat)

    beta = m_inv @ (loadings.T * psi_inv[None, :])
    ez = x_centered @ beta.T

    sxz = (x_centered.T @ ez) / float(n)
    szz = m_inv + (ez.T @ ez) / float(n)
    return ez, sxz, szz


def posterior_mean_factors(
    x: np.ndarray,
    mean: np.ndarray,
    loadings: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """Compute E[z|x] for each sample under fitted parameters."""
    x = ensure_finite_2d(x, "x_for_scores")
    x_centered = x - mean[None, :]

    psi_inv = 1.0 / psi
    k = loadings.shape[1]
    m_mat = np.eye(k, dtype=np.float64) + loadings.T @ (psi_inv[:, None] * loadings)
    m_inv = np.linalg.inv(m_mat + 1e-9 * np.eye(k, dtype=np.float64))
    beta = m_inv @ (loadings.T * psi_inv[None, :])
    return x_centered @ beta.T


def fit_factor_analysis_em(
    x: np.ndarray,
    n_factors: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_psi: float = 1e-6,
) -> FAResult:
    """Fit Gaussian factor analysis by EM."""
    x = ensure_finite_2d(x, "x")
    n, d = x.shape

    if not (1 <= n_factors < d):
        raise ValueError(f"n_factors must satisfy 1 <= n_factors < d, got {n_factors} with d={d}")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")
    if min_psi <= 0.0:
        raise ValueError("min_psi must be > 0")

    mean = np.mean(x, axis=0)
    x_centered = x - mean[None, :]
    sxx = (x_centered.T @ x_centered) / float(n)

    loadings, psi = initialize_parameters(x_centered, n_factors=n_factors, min_psi=min_psi)

    history: List[FAHistory] = []
    converged = False
    message = "max_iter reached"

    prev_ll: float | None = None
    sigma = loadings @ loadings.T + np.diag(psi)

    for it in range(1, max_iter + 1):
        _, sxz, szz = e_step_stats(x_centered=x_centered, loadings=loadings, psi=psi)

        szz_inv = np.linalg.inv(szz + 1e-9 * np.eye(n_factors, dtype=np.float64))
        loadings_new = sxz @ szz_inv

        residual_diag = np.diag(sxx - loadings_new @ sxz.T)
        psi_new = np.clip(residual_diag, min_psi, None)

        avg_ll, sigma_new = average_log_likelihood(
            x_centered=x_centered,
            loadings=loadings_new,
            psi=psi_new,
        )

        delta = np.nan if prev_ll is None else float(avg_ll - prev_ll)
        history.append(FAHistory(iter=it, avg_loglike=avg_ll, delta=delta))

        loadings = loadings_new
        psi = psi_new
        sigma = sigma_new

        if prev_ll is not None:
            threshold = tol * max(1.0, abs(prev_ll))
            if abs(delta) <= threshold:
                converged = True
                message = "log-likelihood increment below tolerance"
                break
        prev_ll = avg_ll

    return FAResult(
        mean=mean,
        loadings=loadings,
        psi=psi,
        sigma=sigma,
        history=history,
        iterations=len(history),
        converged=converged,
        message=message,
    )


def mean_principal_cosine(loadings_true: np.ndarray, loadings_est: np.ndarray) -> float:
    """Subspace agreement metric in [0, 1], higher is better."""
    q_true, _ = np.linalg.qr(loadings_true)
    q_est, _ = np.linalg.qr(loadings_est)
    sv = np.linalg.svd(q_true.T @ q_est, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return float(np.mean(sv))


def build_synthetic_case(
    n_train: int = 1200,
    n_test: int = 400,
    seed: int = 20260407,
) -> SyntheticCase:
    """Deterministic synthetic data from a known FA model."""
    rng = np.random.default_rng(seed)

    loadings_true = np.asarray(
        [
            [0.90, 0.05],
            [0.75, -0.10],
            [0.82, 0.18],
            [0.68, -0.22],
            [0.12, 0.88],
            [0.08, 0.78],
            [-0.05, 0.92],
            [0.20, 0.65],
        ],
        dtype=np.float64,
    )
    psi_true = np.asarray([0.18, 0.22, 0.20, 0.25, 0.16, 0.21, 0.19, 0.24], dtype=np.float64)
    mean_true = np.asarray([1.5, 0.8, 1.1, 1.2, -0.5, -0.8, -0.3, -0.6], dtype=np.float64)

    k = loadings_true.shape[1]
    d = loadings_true.shape[0]

    z_train = rng.normal(size=(n_train, k))
    eps_train = rng.normal(scale=np.sqrt(psi_true), size=(n_train, d))
    x_train = mean_true[None, :] + z_train @ loadings_true.T + eps_train

    z_test = rng.normal(size=(n_test, k))
    eps_test = rng.normal(scale=np.sqrt(psi_true), size=(n_test, d))
    x_test = mean_true[None, :] + z_test @ loadings_true.T + eps_test

    return SyntheticCase(
        x_train=x_train,
        x_test=x_test,
        mean_true=mean_true,
        loadings_true=loadings_true,
        psi_true=psi_true,
    )


def evaluate_result(case: SyntheticCase, result: FAResult) -> None:
    sigma_true = case.loadings_true @ case.loadings_true.T + np.diag(case.psi_true)

    rel_cov_error = float(
        np.linalg.norm(result.sigma - sigma_true, ord="fro")
        / np.linalg.norm(sigma_true, ord="fro")
    )

    train_centered = case.x_train - result.mean[None, :]
    test_centered = case.x_test - result.mean[None, :]
    train_ll, _ = average_log_likelihood(train_centered, result.loadings, result.psi)
    test_ll, _ = average_log_likelihood(test_centered, result.loadings, result.psi)

    mean_cos = mean_principal_cosine(case.loadings_true, result.loadings)
    latent_scores = posterior_mean_factors(case.x_train[:5], result.mean, result.loadings, result.psi)

    print("Factor Analysis EM MVP")
    print("=" * 80)
    print(f"Train shape: {case.x_train.shape}")
    print(f"Test shape : {case.x_test.shape}")
    print(f"Converged  : {result.converged}")
    print(f"Iterations : {result.iterations}")
    print(f"Message    : {result.message}")
    print(f"Final avg loglike (train): {result.history[-1].avg_loglike:.6f}")
    print(f"Train avg loglike        : {train_ll:.6f}")
    print(f"Test avg loglike         : {test_ll:.6f}")
    print(f"Relative covariance error: {rel_cov_error:.6f}")
    print(f"Mean principal cosine    : {mean_cos:.6f}")
    print(f"First 5 uniqueness (psi) : {np.round(result.psi[:5], 6)}")

    print("\nHistory preview (first 8 iterations)")
    print(" iter | avg_loglike  | delta")
    print("------+--------------+--------------")
    for item in result.history[:8]:
        delta_text = "nan" if np.isnan(item.delta) else f"{item.delta: .6e}"
        print(f" {item.iter:>4d} | {item.avg_loglike:>12.6f} | {delta_text:>12}")

    if len(result.history) > 8:
        last = result.history[-1]
        delta_text = "nan" if np.isnan(last.delta) else f"{last.delta: .6e}"
        print(" ...")
        print(f" {last.iter:>4d} | {last.avg_loglike:>12.6f} | {delta_text:>12}")

    print("\nFirst 5 latent posterior means E[z|x]")
    print(np.round(latent_scores, 4))


def main() -> None:
    case = build_synthetic_case(n_train=1200, n_test=400, seed=20260407)

    result = fit_factor_analysis_em(
        x=case.x_train,
        n_factors=2,
        max_iter=200,
        tol=1e-6,
        min_psi=1e-6,
    )

    evaluate_result(case, result)


if __name__ == "__main__":
    main()
