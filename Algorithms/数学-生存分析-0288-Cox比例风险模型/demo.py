"""Cox proportional hazards model (minimal runnable MVP).

This script implements Cox PH fitting via Newton-Raphson on the
negative partial log-likelihood with optional L2 regularization.
It is intentionally self-contained and avoids black-box survival packages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CoxFitResult:
    beta: np.ndarray
    history: pd.DataFrame
    converged: bool


def validate_inputs(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if time.ndim != 1 or event.ndim != 1:
        raise ValueError("time and event must be 1D arrays")
    n_samples = X.shape[0]
    if time.shape[0] != n_samples or event.shape[0] != n_samples:
        raise ValueError("X, time, event must have the same number of rows")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(time)):
        raise ValueError("X and time must be finite")
    if np.any(time <= 0):
        raise ValueError("time must be strictly positive")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must only contain 0/1")
    if int(np.sum(event)) == 0:
        raise ValueError("at least one observed event is required")


def make_synthetic_data(
    n_samples: int = 320,
    n_features: int = 4,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    base = rng.normal(size=(n_samples, n_features))
    # Inject mild correlation to make optimization less trivial.
    X = base.copy()
    if n_features >= 2:
        X[:, 1] = 0.65 * base[:, 0] + np.sqrt(1.0 - 0.65**2) * base[:, 1]

    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    true_beta = np.array([0.9, -0.7, 0.55, -0.25], dtype=float)[:n_features]
    if true_beta.shape[0] < n_features:
        extra = rng.normal(loc=0.0, scale=0.35, size=n_features - true_beta.shape[0])
        true_beta = np.concatenate([true_beta, extra])

    linear_predictor = X @ true_beta

    baseline_rate = 0.08
    event_rate = baseline_rate * np.exp(linear_predictor)

    event_time = rng.exponential(scale=1.0 / event_rate)

    censor_scale = np.quantile(event_time, 0.58)
    censor_time = rng.exponential(scale=censor_scale, size=n_samples)

    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    # Guarantee at least one censored and one event sample.
    if event.sum() == n_samples:
        event[rng.integers(0, n_samples)] = 0
    if event.sum() == 0:
        event[rng.integers(0, n_samples)] = 1

    return X, observed_time, event, true_beta


def cox_nll_grad_hess(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l2: float = 1e-3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Negative partial log-likelihood + gradient + Hessian (Breslow ties)."""
    validate_inputs(X, time, event)
    if beta.ndim != 1 or beta.shape[0] != X.shape[1]:
        raise ValueError("beta shape mismatch")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    n_features = X.shape[1]
    eta = X @ beta

    nll = 0.0
    grad = np.zeros(n_features, dtype=float)
    hess = np.zeros((n_features, n_features), dtype=float)

    for i in range(X.shape[0]):
        if event[i] != 1:
            continue

        risk_mask = time >= time[i]
        X_risk = X[risk_mask]
        eta_risk = eta[risk_mask]

        max_eta = np.max(eta_risk)
        w = np.exp(eta_risk - max_eta)

        s0 = np.sum(w)
        s1 = X_risk.T @ w
        s2 = (X_risk.T * w) @ X_risk

        log_denom = np.log(s0) + max_eta

        nll += -(eta[i] - log_denom)

        mean_x = s1 / s0
        grad += -(X[i] - mean_x)

        second_moment = s2 / s0
        hess += second_moment - np.outer(mean_x, mean_x)

    if l2 > 0:
        nll += 0.5 * l2 * float(beta @ beta)
        grad += l2 * beta
        hess += l2 * np.eye(n_features)

    return float(nll), grad, hess


def cox_neg_log_partial_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l2: float = 1e-3,
) -> float:
    nll, _, _ = cox_nll_grad_hess(beta=beta, X=X, time=time, event=event, l2=l2)
    return nll


def fit_cox_newton(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l2: float = 1e-3,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> CoxFitResult:
    validate_inputs(X, time, event)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    beta = np.zeros(X.shape[1], dtype=float)
    history_rows: list[dict[str, float | int]] = []
    converged = False

    for it in range(1, max_iter + 1):
        nll, grad, hess = cox_nll_grad_hess(beta=beta, X=X, time=time, event=event, l2=l2)
        grad_norm = float(np.linalg.norm(grad))

        if not np.isfinite(nll) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            raise RuntimeError("non-finite quantity encountered during optimization")

        if grad_norm < tol:
            history_rows.append(
                {
                    "iter": it,
                    "nll": nll,
                    "grad_norm": grad_norm,
                    "step_norm": 0.0,
                    "line_search_alpha": 0.0,
                }
            )
            converged = True
            break

        damping = 1e-8
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(hess + damping * np.eye(hess.shape[0]), grad)

        # Backtracking Armijo line search for robust monotonic decrease of NLL.
        alpha = 1.0
        directional_derivative = float(grad @ step)
        current_nll = nll

        while alpha >= 1e-8:
            candidate = beta - alpha * step
            candidate_nll = cox_neg_log_partial_likelihood(
                beta=candidate,
                X=X,
                time=time,
                event=event,
                l2=l2,
            )
            if candidate_nll <= current_nll - 1e-4 * alpha * directional_derivative:
                break
            alpha *= 0.5

        beta_next = beta - alpha * step
        step_norm = float(np.linalg.norm(beta_next - beta))

        history_rows.append(
            {
                "iter": it,
                "nll": nll,
                "grad_norm": grad_norm,
                "step_norm": step_norm,
                "line_search_alpha": alpha,
            }
        )

        beta = beta_next

        if step_norm < tol:
            converged = True
            break

    history = pd.DataFrame(history_rows)
    return CoxFitResult(beta=beta, history=history, converged=converged)


def breslow_baseline_hazard(
    time: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
) -> pd.DataFrame:
    """Estimate baseline hazard increments and cumulative hazard via Breslow."""
    validate_inputs(np.column_stack([eta]), time, event)

    event_times = np.sort(np.unique(time[event == 1]))
    rows: list[dict[str, float | int]] = []
    cumulative = 0.0

    for t in event_times:
        d_t = int(np.sum((time == t) & (event == 1)))
        eta_risk = eta[time >= t]
        max_eta = float(np.max(eta_risk))
        denom = float(np.exp(max_eta) * np.sum(np.exp(eta_risk - max_eta)))

        h0_increment = d_t / denom
        cumulative += h0_increment

        rows.append(
            {
                "time": float(t),
                "events_at_time": d_t,
                "baseline_hazard_increment": h0_increment,
                "baseline_cumulative_hazard": cumulative,
            }
        )

    return pd.DataFrame(rows)


def concordance_index(time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
    """Harrell's C-index for right-censored data (naive O(n^2) implementation)."""
    n = time.shape[0]
    concordant = 0.0
    ties = 0.0
    permissible = 0.0

    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[i] < time[j]:
                permissible += 1.0
                if risk_score[i] > risk_score[j]:
                    concordant += 1.0
                elif risk_score[i] == risk_score[j]:
                    ties += 1.0

    if permissible == 0.0:
        return float("nan")
    return float((concordant + 0.5 * ties) / permissible)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    X, time, event, beta_true = make_synthetic_data(n_samples=320, n_features=4, seed=7)

    result = fit_cox_newton(
        X=X,
        time=time,
        event=event,
        l2=1e-3,
        max_iter=80,
        tol=1e-7,
    )

    beta_hat = result.beta
    eta_hat = X @ beta_hat

    baseline = breslow_baseline_hazard(time=time, event=event, eta=eta_hat)

    c_index = concordance_index(time=time, event=event, risk_score=eta_hat)
    c_index_true = concordance_index(time=time, event=event, risk_score=X @ beta_true)

    coef_table = pd.DataFrame(
        {
            "coef_true": beta_true,
            "coef_hat": beta_hat,
            "abs_error": np.abs(beta_hat - beta_true),
        },
        index=[f"x{k}" for k in range(X.shape[1])],
    )

    print("=== Cox PH Newton-Raphson MVP ===")
    print(f"n_samples={X.shape[0]}, n_features={X.shape[1]}")
    print(f"events={int(event.sum())}, censoring_rate={1.0 - event.mean():.3f}")
    print(f"converged={result.converged}, iters={len(result.history)}")

    print("\n[Coefficient recovery]")
    print(coef_table.to_string(float_format=lambda x: f"{x: .4f}"))

    print("\n[Optimization trace: head]")
    print(result.history.head(8).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\n[Optimization trace: tail]")
    print(result.history.tail(5).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\n[Concordance index]")
    print(f"C-index (estimated risk score): {c_index:.4f}")
    print(f"C-index (oracle true score):    {c_index_true:.4f}")

    print("\n[Breslow baseline cumulative hazard: first 8 rows]")
    print(baseline.head(8).to_string(index=False, float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()
