"""Minimal runnable MVP for QCD factorization theorem (toy hadronic observable).

The script demonstrates a transparent factorized pipeline:
1) build toy PDFs and a parton-luminosity convolution,
2) multiply by a perturbative hard kernel and a power-suppressed term,
3) fit selected coefficients from synthetic data,
4) check factorization-scale stability and physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from scipy.optimize import least_squares
from scipy.special import beta as beta_fn
from sklearn.metrics import mean_squared_error, r2_score


@dataclass(frozen=True)
class FactorizationConfig:
    """Configuration for the toy factorization workflow."""

    mu0_gev: float = 2.0
    alpha_s: float = 0.22
    # q PDF: N x^a (1-x)^b with fixed momentum fraction.
    q_a: float = -0.25
    q_b: float = 3.8
    q_momentum: float = 0.36
    q_lambda: float = -0.18
    # qbar PDF: sea-like distribution.
    qb_a: float = -0.10
    qb_b: float = 6.2
    qb_momentum: float = 0.09
    qb_lambda: float = 0.30
    # Ground-truth coefficients for pseudo data.
    c0_true: float = 1.35
    c_np_true: float = 18.0
    # Dataset and integration settings.
    tau_values: tuple[float, ...] = (
        0.015,
        0.022,
        0.032,
        0.045,
        0.064,
        0.090,
        0.125,
        0.170,
        0.225,
        0.290,
    )
    q_values_gev: tuple[float, ...] = (20.0, 30.0, 50.0, 80.0, 120.0)
    integration_points: int = 400
    # Central and variation scales: mu_F = ratio * Q.
    scale_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    # Pseudo-experimental uncertainty model.
    noise_rel: float = 0.018
    noise_abs: float = 8e-5
    random_seed: int = 20260407


@dataclass(frozen=True)
class FitSummary:
    """Numerical summary of least-squares fit."""

    c0_fit: float
    c_np_fit: float
    chi2: float
    dof: int
    rmse: float
    r2: float
    success: bool
    message: str
    nfev: int


def validate_config(cfg: FactorizationConfig) -> None:
    """Validate numerical/physics guards for the toy model."""
    if cfg.mu0_gev <= 0.0:
        raise ValueError("mu0_gev must be positive.")
    if cfg.alpha_s <= 0.0:
        raise ValueError("alpha_s must be positive.")
    if cfg.integration_points < 120:
        raise ValueError("integration_points must be >= 120.")
    if len(cfg.tau_values) < 6:
        raise ValueError("Need at least 6 tau points.")
    if len(cfg.q_values_gev) < 3:
        raise ValueError("Need at least 3 Q values.")
    if sorted(cfg.scale_ratios) != list(cfg.scale_ratios):
        raise ValueError("scale_ratios must be sorted ascending.")
    if 1.0 not in cfg.scale_ratios:
        raise ValueError("scale_ratios must include 1.0 for central scale.")
    if min(cfg.tau_values) <= 0.0 or max(cfg.tau_values) >= 0.7:
        raise ValueError("tau values must lie in (0, 0.7).")
    if min(cfg.q_values_gev) <= 5.0:
        raise ValueError("Q values must stay safely in the perturbative regime.")
    if cfg.noise_rel <= 0.0 or cfg.noise_abs <= 0.0:
        raise ValueError("noise parameters must be positive.")


def _momentum_normalized_pdf(x: np.ndarray, a: float, b: float, momentum: float) -> np.ndarray:
    """Return N * x^a * (1-x)^b with N fixed by momentum fraction integral."""
    n = momentum / beta_fn(a + 2.0, b + 1.0)
    return n * np.power(x, a) * np.power(1.0 - x, b)


def toy_pdf_q(x: np.ndarray, mu_gev: float, cfg: FactorizationConfig) -> np.ndarray:
    """Toy quark PDF with phenomenological scale evolution."""
    x_arr = np.asarray(x, dtype=float)
    base = _momentum_normalized_pdf(x_arr, cfg.q_a, cfg.q_b, cfg.q_momentum)
    evo = np.exp(cfg.q_lambda * np.log(mu_gev / cfg.mu0_gev) * (1.0 - x_arr))
    return base * evo


def toy_pdf_qbar(x: np.ndarray, mu_gev: float, cfg: FactorizationConfig) -> np.ndarray:
    """Toy antiquark PDF with phenomenological scale evolution."""
    x_arr = np.asarray(x, dtype=float)
    base = _momentum_normalized_pdf(x_arr, cfg.qb_a, cfg.qb_b, cfg.qb_momentum)
    evo = np.exp(cfg.qb_lambda * np.log(mu_gev / cfg.mu0_gev) * (1.0 - x_arr))
    return base * evo


def build_luminosity_getter(cfg: FactorizationConfig) -> Callable[[float, float], float]:
    """Create cached luminosity evaluator L(tau, muF)."""
    cache: dict[tuple[float, float], float] = {}

    def get_luminosity(tau: float, mu_f: float) -> float:
        key = (round(float(tau), 10), round(float(mu_f), 10))
        if key in cache:
            return cache[key]

        x_min = tau + 1e-6
        x_max = 1.0 - 1e-6
        x = np.linspace(x_min, x_max, cfg.integration_points, dtype=float)
        x2 = tau / x

        fq = toy_pdf_q(x, mu_f, cfg)
        fqb = toy_pdf_qbar(x2, mu_f, cfg)
        integrand = fq * fqb / x
        lum = float(simpson(integrand, x=x))
        cache[key] = lum
        return lum

    return get_luminosity


def hard_kernel(q_gev: np.ndarray, mu_f_gev: np.ndarray, c0: float, c1: float, cfg: FactorizationConfig) -> np.ndarray:
    """Perturbative hard coefficient with explicit factorization-scale logs."""
    q_arr = np.asarray(q_gev, dtype=float)
    mu_arr = np.asarray(mu_f_gev, dtype=float)
    log_term = np.log((q_arr * q_arr) / (mu_arr * mu_arr))
    return 1.0 + cfg.alpha_s * (c0 + c1 * log_term)


def sigma_factorized_from_luminosity(
    luminosity: np.ndarray,
    q_gev: np.ndarray,
    mu_f_gev: np.ndarray,
    c0: float,
    c_np: float,
    c1: float,
    cfg: FactorizationConfig,
) -> np.ndarray:
    """Compute sigma = H * L * (1 + c_np / Q^2)."""
    q_arr = np.asarray(q_gev, dtype=float)
    lum_arr = np.asarray(luminosity, dtype=float)
    hard = hard_kernel(q_arr, np.asarray(mu_f_gev, dtype=float), c0=c0, c1=c1, cfg=cfg)
    power = 1.0 + c_np / (q_arr * q_arr)
    return lum_arr * hard * power


def estimate_compensating_c1(cfg: FactorizationConfig, get_luminosity: Callable[[float, float], float], c0_ref: float) -> tuple[float, float]:
    """Estimate c1 so d ln(H)/d ln(muF) roughly cancels d ln(L)/d ln(muF)."""
    eps = 0.10
    slopes: list[float] = []
    for q in cfg.q_values_gev:
        for tau in cfg.tau_values:
            mu_plus = q * np.exp(eps)
            mu_minus = q * np.exp(-eps)
            l_plus = get_luminosity(tau, mu_plus)
            l_minus = get_luminosity(tau, mu_minus)
            slope = (np.log(l_plus) - np.log(l_minus)) / (2.0 * eps)
            slopes.append(float(slope))

    mean_slope = float(np.mean(slopes))
    h_center = 1.0 + cfg.alpha_s * c0_ref
    c1 = mean_slope * h_center / (2.0 * cfg.alpha_s)
    return float(c1), mean_slope


def build_synthetic_dataset(
    cfg: FactorizationConfig,
    get_luminosity: Callable[[float, float], float],
    c1: float,
) -> pd.DataFrame:
    """Generate pseudo measurements from the toy factorized model."""
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, float]] = []

    for q in cfg.q_values_gev:
        for tau in cfg.tau_values:
            mu_f = q
            lum = get_luminosity(tau, mu_f)
            sigma_true = sigma_factorized_from_luminosity(
                luminosity=np.array([lum]),
                q_gev=np.array([q]),
                mu_f_gev=np.array([mu_f]),
                c0=cfg.c0_true,
                c_np=cfg.c_np_true,
                c1=c1,
                cfg=cfg,
            )[0]
            sigma_err = cfg.noise_rel * abs(sigma_true) + cfg.noise_abs
            sigma_obs = float(sigma_true + rng.normal(loc=0.0, scale=sigma_err))

            rows.append(
                {
                    "tau": float(tau),
                    "Q": float(q),
                    "Q2": float(q * q),
                    "mu_f": float(mu_f),
                    "luminosity": float(lum),
                    "sigma_true": float(sigma_true),
                    "sigma_obs": sigma_obs,
                    "sigma_err": float(sigma_err),
                }
            )

    return pd.DataFrame(rows)


def residual_vector(theta: np.ndarray, df: pd.DataFrame, c1: float, cfg: FactorizationConfig) -> np.ndarray:
    """Weighted residuals for least-squares fit of (c0, c_np)."""
    c0, c_np = float(theta[0]), float(theta[1])
    q2 = df["Q2"].to_numpy(dtype=float)
    if c_np <= -0.95 * float(np.min(q2)):
        return np.full(df.shape[0], 1e6, dtype=float)

    pred = sigma_factorized_from_luminosity(
        luminosity=df["luminosity"].to_numpy(dtype=float),
        q_gev=df["Q"].to_numpy(dtype=float),
        mu_f_gev=df["mu_f"].to_numpy(dtype=float),
        c0=c0,
        c_np=c_np,
        c1=c1,
        cfg=cfg,
    )
    if not np.all(np.isfinite(pred)):
        return np.full(df.shape[0], 1e6, dtype=float)

    err = np.maximum(df["sigma_err"].to_numpy(dtype=float), 1e-12)
    obs = df["sigma_obs"].to_numpy(dtype=float)
    return (pred - obs) / err


def fit_factorized_coefficients(df: pd.DataFrame, c1: float, cfg: FactorizationConfig) -> FitSummary:
    """Fit c0 and c_np from pseudo data."""
    x0 = np.array([0.8, 8.0], dtype=float)
    bounds = (np.array([-5.0, 0.0], dtype=float), np.array([5.0, 50.0], dtype=float))

    fit = least_squares(
        residual_vector,
        x0=x0,
        bounds=bounds,
        args=(df, c1, cfg),
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    c0_fit, c_np_fit = float(fit.x[0]), float(fit.x[1])

    pred = sigma_factorized_from_luminosity(
        luminosity=df["luminosity"].to_numpy(dtype=float),
        q_gev=df["Q"].to_numpy(dtype=float),
        mu_f_gev=df["mu_f"].to_numpy(dtype=float),
        c0=c0_fit,
        c_np=c_np_fit,
        c1=c1,
        cfg=cfg,
    )
    obs = df["sigma_obs"].to_numpy(dtype=float)
    err = np.maximum(df["sigma_err"].to_numpy(dtype=float), 1e-12)
    pull = (pred - obs) / err

    chi2 = float(np.sum(pull**2))
    dof = int(obs.size - 2)
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    r2 = float(r2_score(obs, pred))

    return FitSummary(
        c0_fit=c0_fit,
        c_np_fit=c_np_fit,
        chi2=chi2,
        dof=dof,
        rmse=rmse,
        r2=r2,
        success=bool(fit.success),
        message=str(fit.message),
        nfev=int(fit.nfev),
    )


def momentum_sum_rule_diagnostics(cfg: FactorizationConfig) -> dict[str, float]:
    """Verify momentum closure at the reference scale mu0."""
    x = np.linspace(1e-6, 1.0 - 1e-6, 12000)
    q = toy_pdf_q(x, cfg.mu0_gev, cfg)
    qb = toy_pdf_qbar(x, cfg.mu0_gev, cfg)
    m_q = float(np.trapezoid(x * q, x))
    m_qb = float(np.trapezoid(x * qb, x))
    m_g = 1.0 - m_q - m_qb
    return {"m_q": m_q, "m_qbar": m_qb, "m_gluon_inferred": m_g, "momentum_total": m_q + m_qb + m_g}


def scale_variation_table(
    cfg: FactorizationConfig,
    get_luminosity: Callable[[float, float], float],
    c0: float,
    c_np: float,
    c1: float,
) -> pd.DataFrame:
    """Evaluate residual muF dependence over (tau, Q) points."""
    rows: list[dict[str, float]] = []
    for q in cfg.q_values_gev:
        for tau in cfg.tau_values:
            sigmas: list[float] = []
            for ratio in cfg.scale_ratios:
                mu_f = ratio * q
                lum = get_luminosity(tau, mu_f)
                sigma = sigma_factorized_from_luminosity(
                    luminosity=np.array([lum]),
                    q_gev=np.array([q]),
                    mu_f_gev=np.array([mu_f]),
                    c0=c0,
                    c_np=c_np,
                    c1=c1,
                    cfg=cfg,
                )[0]
                sigmas.append(float(sigma))

            central_idx = cfg.scale_ratios.index(1.0)
            central = sigmas[central_idx]
            rel_var = (max(sigmas) - min(sigmas)) / max(abs(central), 1e-12)
            rows.append({"tau": tau, "Q": q, "sigma_central": central, "rel_scale_var": rel_var})
    return pd.DataFrame(rows)


def torch_consistency_and_gradients(df: pd.DataFrame, c0: float, c_np: float, c1: float, cfg: FactorizationConfig) -> dict[str, float]:
    """Use PyTorch to verify forward consistency and differentiability."""
    q = torch.tensor(df["Q"].to_numpy(dtype=float), dtype=torch.float64)
    mu_f = torch.tensor(df["mu_f"].to_numpy(dtype=float), dtype=torch.float64)
    lum = torch.tensor(df["luminosity"].to_numpy(dtype=float), dtype=torch.float64)
    obs = torch.tensor(df["sigma_obs"].to_numpy(dtype=float), dtype=torch.float64)
    err = torch.tensor(df["sigma_err"].to_numpy(dtype=float), dtype=torch.float64)

    c0_t = torch.tensor(float(c0), dtype=torch.float64, requires_grad=True)
    c_np_t = torch.tensor(float(c_np), dtype=torch.float64, requires_grad=True)

    hard_t = 1.0 + cfg.alpha_s * (c0_t + c1 * torch.log((q * q) / (mu_f * mu_f)))
    power_t = 1.0 + c_np_t / (q * q)
    pred_t = lum * hard_t * power_t
    loss_t = torch.mean(((pred_t - obs) / err) ** 2)
    loss_t.backward()

    pred_np = sigma_factorized_from_luminosity(
        luminosity=df["luminosity"].to_numpy(dtype=float),
        q_gev=df["Q"].to_numpy(dtype=float),
        mu_f_gev=df["mu_f"].to_numpy(dtype=float),
        c0=c0,
        c_np=c_np,
        c1=c1,
        cfg=cfg,
    )
    max_forward_diff = float(np.max(np.abs(pred_t.detach().cpu().numpy() - pred_np)))
    grad_norm = float(torch.sqrt(c0_t.grad * c0_t.grad + c_np_t.grad * c_np_t.grad).item())
    return {"max_forward_diff": max_forward_diff, "autodiff_grad_norm": grad_norm}


def run_quality_checks(
    cfg: FactorizationConfig,
    fit: FitSummary,
    momentum_diag: dict[str, float],
    scale_with_log: pd.DataFrame,
    scale_no_log: pd.DataFrame,
    torch_diag: dict[str, float],
) -> None:
    """Assert core numerical and physics-facing quality gates."""
    if not fit.success:
        raise AssertionError(f"Least-squares did not converge: {fit.message}")

    chi2_dof = fit.chi2 / max(fit.dof, 1)
    if chi2_dof > 2.4:
        raise AssertionError(f"chi2/dof too large: {chi2_dof:.3f}")
    if fit.r2 < 0.97:
        raise AssertionError(f"R^2 too small: {fit.r2:.4f}")

    if abs(fit.c0_fit - cfg.c0_true) > 0.30 * abs(cfg.c0_true):
        raise AssertionError("Recovered c0 deviates too much from truth.")
    if abs(fit.c_np_fit - cfg.c_np_true) > 0.40 * abs(cfg.c_np_true):
        raise AssertionError("Recovered c_np deviates too much from truth.")

    if abs(momentum_diag["momentum_total"] - 1.0) > 3e-4:
        raise AssertionError("Momentum closure check failed.")
    if momentum_diag["m_gluon_inferred"] <= 0.0:
        raise AssertionError("Inferred gluon momentum must be positive.")

    mean_var_log = float(scale_with_log["rel_scale_var"].mean())
    mean_var_nolog = float(scale_no_log["rel_scale_var"].mean())
    if mean_var_log >= mean_var_nolog:
        raise AssertionError("Factorized hard-log compensation did not improve scale stability.")
    if mean_var_log > 0.18:
        raise AssertionError("Residual scale variation remains too large.")

    if torch_diag["max_forward_diff"] > 5e-11:
        raise AssertionError("Torch/NumPy forward mismatch too large.")
    if torch_diag["autodiff_grad_norm"] <= 0.0:
        raise AssertionError("Torch gradient norm should be positive.")


def main() -> None:
    cfg = FactorizationConfig()
    validate_config(cfg)

    get_luminosity = build_luminosity_getter(cfg)
    c1_comp, mean_lumi_slope = estimate_compensating_c1(cfg, get_luminosity, c0_ref=cfg.c0_true)

    df = build_synthetic_dataset(cfg, get_luminosity, c1=c1_comp)
    fit = fit_factorized_coefficients(df, c1=c1_comp, cfg=cfg)

    momentum_diag = momentum_sum_rule_diagnostics(cfg)
    scale_with_log = scale_variation_table(
        cfg,
        get_luminosity,
        c0=fit.c0_fit,
        c_np=fit.c_np_fit,
        c1=c1_comp,
    )
    scale_no_log = scale_variation_table(
        cfg,
        get_luminosity,
        c0=fit.c0_fit,
        c_np=fit.c_np_fit,
        c1=0.0,
    )
    torch_diag = torch_consistency_and_gradients(df, c0=fit.c0_fit, c_np=fit.c_np_fit, c1=c1_comp, cfg=cfg)

    run_quality_checks(
        cfg=cfg,
        fit=fit,
        momentum_diag=momentum_diag,
        scale_with_log=scale_with_log,
        scale_no_log=scale_no_log,
        torch_diag=torch_diag,
    )

    df_report = df.copy()
    pred_fit = sigma_factorized_from_luminosity(
        luminosity=df["luminosity"].to_numpy(dtype=float),
        q_gev=df["Q"].to_numpy(dtype=float),
        mu_f_gev=df["mu_f"].to_numpy(dtype=float),
        c0=fit.c0_fit,
        c_np=fit.c_np_fit,
        c1=c1_comp,
        cfg=cfg,
    )
    df_report["sigma_fit"] = pred_fit
    df_report["pull"] = (df_report["sigma_fit"] - df_report["sigma_obs"]) / df_report["sigma_err"]

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 20)

    print("=== QCD Factorization Theorem MVP ===")
    print(
        {
            "alpha_s": cfg.alpha_s,
            "mu0_gev": cfg.mu0_gev,
            "points": int(df.shape[0]),
            "tau_count": len(cfg.tau_values),
            "Q_count": len(cfg.q_values_gev),
            "estimated_mean_dlnL_dlnmu": round(mean_lumi_slope, 6),
            "compensating_c1": round(c1_comp, 6),
        }
    )
    print("\nFit summary:")
    print(
        {
            "success": fit.success,
            "message": fit.message,
            "nfev": fit.nfev,
            "c0_true": cfg.c0_true,
            "c0_fit": round(fit.c0_fit, 6),
            "c_np_true": cfg.c_np_true,
            "c_np_fit": round(fit.c_np_fit, 6),
            "chi2_dof": round(fit.chi2 / max(fit.dof, 1), 6),
            "rmse": round(fit.rmse, 8),
            "r2": round(fit.r2, 6),
        }
    )
    print("\nMomentum diagnostics (mu0):")
    print({k: round(v, 6) for k, v in momentum_diag.items()})

    print("\nScale-variation diagnostics:")
    print(
        {
            "mean_rel_var_with_log": round(float(scale_with_log["rel_scale_var"].mean()), 6),
            "max_rel_var_with_log": round(float(scale_with_log["rel_scale_var"].max()), 6),
            "mean_rel_var_no_log": round(float(scale_no_log["rel_scale_var"].mean()), 6),
            "max_rel_var_no_log": round(float(scale_no_log["rel_scale_var"].max()), 6),
        }
    )
    print("\nTorch diagnostics:")
    print({k: round(v, 12) for k, v in torch_diag.items()})

    print("\nSample rows (first 12):")
    print(
        df_report.loc[:, ["tau", "Q", "sigma_obs", "sigma_fit", "sigma_err", "pull"]]
        .head(12)
        .to_string(index=False, float_format=lambda x: f"{x: .6e}")
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
