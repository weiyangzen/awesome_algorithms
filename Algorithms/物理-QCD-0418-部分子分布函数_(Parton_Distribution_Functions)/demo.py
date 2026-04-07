"""Minimal runnable MVP for Parton Distribution Functions (PDF) fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import beta as beta_fn
from sklearn.metrics import mean_squared_error, r2_score


Q2_0 = 2.0
LAMBDA2 = 0.04
# Keep gluon shape fixed in the MVP; its normalization comes from momentum sum rule.
A_G = -0.05
B_G = 5.0
K_G = 0.25

PARAM_ORDER = [
    "a_u",
    "b_u",
    "a_d",
    "b_d",
    "a_s",
    "b_s",
    "sea_momentum",
    "k_valence",
    "k_sea",
]


@dataclass(frozen=True)
class PDFParams:
    """Compact parameter container for the simplified PDF model."""

    a_u: float
    b_u: float
    a_d: float
    b_d: float
    a_s: float
    b_s: float
    sea_momentum: float
    k_valence: float
    k_sea: float

    def to_theta(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in PARAM_ORDER], dtype=float)

    @classmethod
    def from_theta(cls, theta: np.ndarray) -> "PDFParams":
        if theta.shape != (len(PARAM_ORDER),):
            raise ValueError("theta has unexpected shape")
        return cls(**{name: float(theta[i]) for i, name in enumerate(PARAM_ORDER)})


@dataclass(frozen=True)
class FitSummary:
    """Numerical summary of the nonlinear least-squares fit."""

    params: PDFParams
    chi2: float
    dof: int
    rmse: float
    r2: float
    success: bool
    message: str
    nfev: int


def compute_normalization_constants(params: PDFParams) -> dict[str, float]:
    """Compute analytic normalization constants constrained by sum rules."""
    n_u = 2.0 / beta_fn(params.a_u + 1.0, params.b_u + 1.0)
    n_d = 1.0 / beta_fn(params.a_d + 1.0, params.b_d + 1.0)
    n_s = params.sea_momentum / beta_fn(params.a_s + 2.0, params.b_s + 1.0)

    momentum_u = n_u * beta_fn(params.a_u + 2.0, params.b_u + 1.0)
    momentum_d = n_d * beta_fn(params.a_d + 2.0, params.b_d + 1.0)
    momentum_s = params.sea_momentum

    momentum_g = 1.0 - momentum_u - momentum_d - momentum_s
    if momentum_g <= 1e-8:
        raise ValueError("non-positive gluon momentum fraction")

    n_g = momentum_g / beta_fn(A_G + 2.0, B_G + 1.0)
    return {
        "n_u": n_u,
        "n_d": n_d,
        "n_s": n_s,
        "n_g": n_g,
        "momentum_u": momentum_u,
        "momentum_d": momentum_d,
        "momentum_s": momentum_s,
        "momentum_g": momentum_g,
    }


def pdf_components_at_q0(
    x: np.ndarray,
    params: PDFParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u_valence, d_valence, sea, gluon) at Q0^2."""
    x = np.asarray(x, dtype=float)
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("x must stay in (0, 1)")

    norms = compute_normalization_constants(params)
    u_valence = norms["n_u"] * np.power(x, params.a_u) * np.power(1.0 - x, params.b_u)
    d_valence = norms["n_d"] * np.power(x, params.a_d) * np.power(1.0 - x, params.b_d)
    sea = norms["n_s"] * np.power(x, params.a_s) * np.power(1.0 - x, params.b_s)
    gluon = norms["n_g"] * np.power(x, A_G) * np.power(1.0 - x, B_G)
    return u_valence, d_valence, sea, gluon


def evolve_component(
    base_component: np.ndarray,
    x: np.ndarray,
    q2: np.ndarray,
    k: float,
) -> np.ndarray:
    """Apply a simple phenomenological Q^2 evolution factor."""
    log_ratio = np.log((q2 + LAMBDA2) / (Q2_0 + LAMBDA2))
    factor = np.exp(k * log_ratio * (1.0 - x))
    return base_component * factor


def f2_model(x: np.ndarray, q2: np.ndarray, params: PDFParams) -> np.ndarray:
    """Compute a simplified LO DIS structure function F2(x, Q^2)."""
    x_arr = np.asarray(x, dtype=float)
    q2_arr = np.asarray(q2, dtype=float)
    x_b, q2_b = np.broadcast_arrays(x_arr, q2_arr)

    if np.any((x_b <= 0.0) | (x_b >= 1.0)):
        raise ValueError("x must be in (0, 1)")
    if np.any(q2_b <= 0.0):
        raise ValueError("q2 must be positive")

    u0, d0, sea0, _ = pdf_components_at_q0(x_b, params)
    u = evolve_component(u0, x_b, q2_b, params.k_valence)
    d = evolve_component(d0, x_b, q2_b, params.k_valence)
    sea = evolve_component(sea0, x_b, q2_b, params.k_sea)

    # Flavor decomposition for q + qbar combinations entering F2.
    u_plus = u + 0.5 * sea
    d_plus = d + 0.3 * sea
    s_plus = 0.2 * sea

    return x_b * ((4.0 / 9.0) * u_plus + (1.0 / 9.0) * d_plus + (1.0 / 9.0) * s_plus)


def make_synthetic_dataset(
    rng: np.random.Generator,
    true_params: PDFParams,
    x_points: np.ndarray,
    q2_points: np.ndarray,
) -> pd.DataFrame:
    """Generate pseudo-DIS observations with heteroscedastic Gaussian noise."""
    x_grid, q2_grid = np.meshgrid(x_points, q2_points, indexing="ij")
    x_flat = x_grid.ravel()
    q2_flat = q2_grid.ravel()

    f2_true = f2_model(x_flat, q2_flat, true_params)
    sigma = 0.03 * np.abs(f2_true) + 0.002
    noise = rng.normal(loc=0.0, scale=sigma)
    f2_obs = f2_true + noise

    return pd.DataFrame(
        {
            "x": x_flat,
            "q2": q2_flat,
            "f2_true": f2_true,
            "f2_obs": f2_obs,
            "sigma": sigma,
        }
    )


def residual_vector(
    theta: np.ndarray,
    x: np.ndarray,
    q2: np.ndarray,
    y_obs: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Weighted residual vector for least squares optimization."""
    try:
        params = PDFParams.from_theta(theta)
        y_pred = f2_model(x, q2, params)
    except Exception:
        return np.full_like(y_obs, 1e6, dtype=float)

    denom = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    if not np.all(np.isfinite(y_pred)):
        return np.full_like(y_obs, 1e6, dtype=float)
    return (y_pred - y_obs) / denom


def fit_pdf_parameters(
    df: pd.DataFrame,
    initial_theta: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> FitSummary:
    """Fit model parameters to pseudo data with bounded least squares."""
    x = df["x"].to_numpy(dtype=float)
    q2 = df["q2"].to_numpy(dtype=float)
    y_obs = df["f2_obs"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)

    fit = least_squares(
        residual_vector,
        x0=initial_theta,
        bounds=bounds,
        args=(x, q2, y_obs, sigma),
        max_nfev=800,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    fitted_params = PDFParams.from_theta(fit.x)
    y_fit = f2_model(x, q2, fitted_params)
    pull = (y_fit - y_obs) / np.maximum(sigma, 1e-6)

    chi2 = float(np.sum(pull**2))
    dof = int(y_obs.size - initial_theta.size)
    rmse = float(np.sqrt(mean_squared_error(y_obs, y_fit)))
    r2 = float(r2_score(y_obs, y_fit))

    return FitSummary(
        params=fitted_params,
        chi2=chi2,
        dof=dof,
        rmse=rmse,
        r2=r2,
        success=bool(fit.success),
        message=str(fit.message),
        nfev=int(fit.nfev),
    )


def sum_rule_diagnostics(params: PDFParams) -> dict[str, float]:
    """Numerically verify valence-number and momentum sum rules at Q0^2."""
    x_dense = np.linspace(1e-5, 1.0 - 1e-5, 12000)
    u_valence, d_valence, sea, gluon = pdf_components_at_q0(x_dense, params)

    valence_u = float(np.trapezoid(u_valence, x_dense))
    valence_d = float(np.trapezoid(d_valence, x_dense))

    momentum_u = float(np.trapezoid(x_dense * u_valence, x_dense))
    momentum_d = float(np.trapezoid(x_dense * d_valence, x_dense))
    momentum_s = float(np.trapezoid(x_dense * sea, x_dense))
    momentum_g = float(np.trapezoid(x_dense * gluon, x_dense))
    momentum_total = momentum_u + momentum_d + momentum_s + momentum_g

    return {
        "valence_u": valence_u,
        "valence_d": valence_d,
        "momentum_u": momentum_u,
        "momentum_d": momentum_d,
        "momentum_s": momentum_s,
        "momentum_g": momentum_g,
        "momentum_total": momentum_total,
    }


def build_parameter_report(true_params: PDFParams, fitted_params: PDFParams) -> pd.DataFrame:
    """Create a compact comparison table of true vs fitted parameters."""
    rows = []
    for name in PARAM_ORDER:
        truth = float(getattr(true_params, name))
        fit = float(getattr(fitted_params, name))
        rel = abs(fit - truth) / max(abs(truth), 1e-9)
        rows.append(
            {
                "parameter": name,
                "truth": truth,
                "fit": fit,
                "abs_error": fit - truth,
                "rel_error": rel,
            }
        )
    return pd.DataFrame(rows)


def run_optional_torch_check(df: pd.DataFrame, params: PDFParams) -> Optional[float]:
    """If torch is available, compare torch and numpy forward predictions."""
    try:
        import torch
    except Exception:
        return None

    torch.set_default_dtype(torch.float64)
    x_t = torch.tensor(df["x"].to_numpy(dtype=float), dtype=torch.float64)
    q2_t = torch.tensor(df["q2"].to_numpy(dtype=float), dtype=torch.float64)

    norms = compute_normalization_constants(params)
    u0 = norms["n_u"] * torch.pow(x_t, params.a_u) * torch.pow(1.0 - x_t, params.b_u)
    d0 = norms["n_d"] * torch.pow(x_t, params.a_d) * torch.pow(1.0 - x_t, params.b_d)
    s0 = norms["n_s"] * torch.pow(x_t, params.a_s) * torch.pow(1.0 - x_t, params.b_s)

    log_ratio = torch.log((q2_t + LAMBDA2) / (Q2_0 + LAMBDA2))
    u = u0 * torch.exp(params.k_valence * log_ratio * (1.0 - x_t))
    d = d0 * torch.exp(params.k_valence * log_ratio * (1.0 - x_t))
    sea = s0 * torch.exp(params.k_sea * log_ratio * (1.0 - x_t))

    u_plus = u + 0.5 * sea
    d_plus = d + 0.3 * sea
    s_plus = 0.2 * sea
    f2_torch = x_t * ((4.0 / 9.0) * u_plus + (1.0 / 9.0) * d_plus + (1.0 / 9.0) * s_plus)

    f2_numpy = f2_model(df["x"].to_numpy(dtype=float), df["q2"].to_numpy(dtype=float), params)
    diff = torch.max(torch.abs(f2_torch - torch.tensor(f2_numpy, dtype=torch.float64))).item()
    return float(diff)


def run_assertions(summary: FitSummary, sums: dict[str, float], torch_diff: Optional[float]) -> None:
    """Minimal automatic acceptance checks for the MVP."""
    if not summary.success:
        raise AssertionError(f"fit did not converge: {summary.message}")

    chi2_dof = summary.chi2 / max(summary.dof, 1)
    if chi2_dof > 2.5:
        raise AssertionError(f"chi2/dof too large: {chi2_dof:.3f}")
    if summary.r2 < 0.92:
        raise AssertionError(f"R^2 too low: {summary.r2:.4f}")

    if abs(sums["valence_u"] - 2.0) > 5e-3:
        raise AssertionError("u valence sum rule violated")
    if abs(sums["valence_d"] - 1.0) > 5e-3:
        raise AssertionError("d valence sum rule violated")
    if abs(sums["momentum_total"] - 1.0) > 6e-3:
        raise AssertionError("momentum sum rule violated")

    if min(sums["momentum_u"], sums["momentum_d"], sums["momentum_s"], sums["momentum_g"]) <= 0.0:
        raise AssertionError("all momentum fractions must be positive")

    if torch_diff is not None and torch_diff > 1e-10:
        raise AssertionError(f"torch/numpy forward mismatch too large: {torch_diff:.3e}")


def main() -> None:
    rng = np.random.default_rng(20260407)

    true_params = PDFParams(
        a_u=0.55,
        b_u=3.30,
        a_d=0.85,
        b_d=4.20,
        a_s=0.10,
        b_s=7.20,
        sea_momentum=0.16,
        k_valence=0.08,
        k_sea=0.26,
    )

    x_points = np.geomspace(1e-3, 0.7, 45)
    q2_points = np.array([2.0, 5.0, 10.0, 20.0, 50.0], dtype=float)

    df = make_synthetic_dataset(rng=rng, true_params=true_params, x_points=x_points, q2_points=q2_points)

    initial = np.array([0.75, 3.9, 1.10, 5.3, 0.35, 8.5, 0.20, 0.02, 0.12], dtype=float)
    lower = np.array([0.05, 1.0, 0.05, 1.0, -0.25, 2.0, 0.04, -0.25, -0.25], dtype=float)
    upper = np.array([1.60, 8.5, 2.20, 10.0, 1.20, 13.0, 0.35, 0.65, 1.10], dtype=float)

    fit_summary = fit_pdf_parameters(df=df, initial_theta=initial, bounds=(lower, upper))

    f2_fit = f2_model(df["x"].to_numpy(), df["q2"].to_numpy(), fit_summary.params)
    df_eval = df.copy()
    df_eval["f2_fit"] = f2_fit
    df_eval["pull"] = (df_eval["f2_fit"] - df_eval["f2_obs"]) / np.maximum(df_eval["sigma"], 1e-6)

    sum_rules = sum_rule_diagnostics(fit_summary.params)
    param_report = build_parameter_report(true_params, fit_summary.params)
    torch_diff = run_optional_torch_check(df_eval, fit_summary.params)

    print("Parton Distribution Functions MVP")
    print("Model: Beta-shape PDFs + simplified Q^2 evolution + weighted least squares")
    print()
    print("Fit diagnostics")
    print(f"  success      : {fit_summary.success}")
    print(f"  message      : {fit_summary.message}")
    print(f"  nfev         : {fit_summary.nfev}")
    print(f"  chi2 / dof   : {fit_summary.chi2:.3f} / {fit_summary.dof} = {fit_summary.chi2 / max(fit_summary.dof, 1):.4f}")
    print(f"  RMSE         : {fit_summary.rmse:.6f}")
    print(f"  R^2          : {fit_summary.r2:.6f}")
    print()

    print("Parameter recovery (truth vs fit)")
    print(param_report.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    print("Sum-rule diagnostics at Q0^2")
    for key in [
        "valence_u",
        "valence_d",
        "momentum_u",
        "momentum_d",
        "momentum_s",
        "momentum_g",
        "momentum_total",
    ]:
        print(f"  {key:14s}: {sum_rules[key]:.6f}")
    print()

    if torch_diff is None:
        print("PyTorch check: skipped (torch not available)")
    else:
        print(f"PyTorch check: max |F2_torch - F2_numpy| = {torch_diff:.3e}")
    print()

    print("Prediction sample (first 12 rows)")
    cols = ["x", "q2", "f2_obs", "f2_fit", "sigma", "pull"]
    print(df_eval.loc[:11, cols].to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    run_assertions(fit_summary, sum_rules, torch_diff)
    print("All checks passed.")


if __name__ == "__main__":
    main()
