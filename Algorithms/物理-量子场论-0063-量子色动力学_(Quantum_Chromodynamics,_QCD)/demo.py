"""Minimal runnable MVP for Quantum Chromodynamics (QCD).

Pipeline:
1) Build two-loop running coupling alpha_s(mu) for SU(3) QCD,
2) Generate synthetic e+e- hadronic R-ratio measurements,
3) Invert measurements to alpha_s estimates,
4) Recover QCD scale information with linear regression and nonlinear fitting,
5) Validate physics-consistent checks (asymptotic freedom, monotonic running).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class QCDConfig:
    n_f: int = 5
    mu_ref_gev: float = 10.0
    alpha_ref_true: float = 0.18
    energy_max_gev: float = 400.0
    num_points: int = 18
    noise_sigma: float = 0.004
    loops: int = 2
    seed: int = 20260407


def validate_config(config: QCDConfig) -> None:
    if not (1 <= config.n_f <= 6):
        raise ValueError("This MVP supports n_f in [1, 6].")
    if config.mu_ref_gev <= 0.0:
        raise ValueError("mu_ref_gev must be positive.")
    if config.energy_max_gev <= config.mu_ref_gev:
        raise ValueError("energy_max_gev must be larger than mu_ref_gev.")
    if config.num_points < 6:
        raise ValueError("num_points must be >= 6.")
    if config.alpha_ref_true <= 0.0:
        raise ValueError("alpha_ref_true must be positive.")
    if config.noise_sigma < 0.0:
        raise ValueError("noise_sigma must be non-negative.")
    if config.loops not in (1, 2):
        raise ValueError("loops must be 1 or 2.")


def qcd_beta_coefficients(n_f: int) -> tuple[float, float]:
    """Return (beta0, beta1) for SU(3) QCD."""
    beta0 = 11.0 - (2.0 / 3.0) * n_f
    beta1 = 102.0 - (38.0 / 3.0) * n_f
    return beta0, beta1


def reduced_beta_coefficients(n_f: int) -> tuple[float, float]:
    """Return (b0, b1) for d alpha / d ln(mu) = -b0*alpha^2 - b1*alpha^3."""
    beta0, beta1 = qcd_beta_coefficients(n_f)
    b0 = beta0 / (2.0 * np.pi)
    b1 = beta1 / (4.0 * np.pi**2)
    return b0, b1


def beta_alpha(alpha: float, n_f: int, loops: int = 2) -> float:
    """QCD beta value in ln(mu) variable."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if loops not in (1, 2):
        raise ValueError("loops must be 1 or 2.")

    b0, b1 = reduced_beta_coefficients(n_f)
    value = -b0 * alpha**2
    if loops == 2:
        value -= b1 * alpha**3
    return value


def integrate_running_alpha(
    mu_grid: np.ndarray,
    n_f: int,
    mu_ref: float,
    alpha_ref: float,
    loops: int = 2,
) -> np.ndarray:
    """Integrate alpha_s(mu) on an ascending positive mu grid."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if mu_grid.ndim != 1 or mu_grid.size < 2:
        raise ValueError("mu_grid must be a 1D array with at least 2 points.")
    if np.any(mu_grid <= 0.0):
        raise ValueError("All mu_grid values must be positive.")
    if not np.all(np.diff(mu_grid) > 0.0):
        raise ValueError("mu_grid must be strictly ascending.")
    if not np.isclose(mu_grid[0], mu_ref):
        raise ValueError("mu_grid[0] must match mu_ref.")

    t_eval = np.log(mu_grid)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        alpha = max(float(y[0]), 1e-14)
        return np.array([beta_alpha(alpha=alpha, n_f=n_f, loops=loops)], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=np.array([alpha_ref], dtype=float),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    alpha_vals = sol.y[0]
    if np.any(~np.isfinite(alpha_vals)) or np.any(alpha_vals <= 0.0):
        raise RuntimeError("Invalid alpha values encountered during integration.")
    return alpha_vals


def active_charge_squared_sum(n_f: int) -> float:
    """Sum of quark electric charges squared for first n_f quarks."""
    charges = np.array([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0])
    return float(np.sum(charges[:n_f] ** 2))


def r_ratio_coefficient_c2(n_f: int) -> float:
    """Approximate NNLO coefficient for massless e+e- -> hadrons R-ratio."""
    return 1.9857 - 0.1153 * n_f


def qcd_r_ratio(alpha_s: np.ndarray | float, n_f: int) -> np.ndarray:
    """Perturbative R-ratio model: R = 3*sum(q_f^2)*(1 + a + c2*a^2), a=alpha_s/pi."""
    alpha_arr = np.asarray(alpha_s, dtype=float)
    if np.any(alpha_arr <= 0.0):
        raise ValueError("alpha_s must be positive.")

    base = 3.0 * active_charge_squared_sum(n_f)
    a = alpha_arr / np.pi
    c2 = r_ratio_coefficient_c2(n_f)
    return base * (1.0 + a + c2 * a**2)


def invert_r_ratio_to_alpha(r_values: np.ndarray, n_f: int) -> np.ndarray:
    """Invert quadratic R-ratio model to alpha_s estimates."""
    r_values = np.asarray(r_values, dtype=float)
    base = 3.0 * active_charge_squared_sum(n_f)
    c2 = r_ratio_coefficient_c2(n_f)

    q = np.maximum(r_values / base - 1.0, 1e-10)
    # Solve c2*x^2 + x - q = 0 for x>0, then alpha = pi*x.
    disc = 1.0 + 4.0 * c2 * q
    x = (-1.0 + np.sqrt(disc)) / (2.0 * c2)
    alpha = np.pi * np.maximum(x, 1e-12)
    return alpha


def one_loop_lambda_from_reference(mu_ref: float, alpha_ref: float, n_f: int) -> float:
    """Infer Lambda_QCD from one-loop relation at reference point."""
    if mu_ref <= 0.0 or alpha_ref <= 0.0:
        raise ValueError("mu_ref and alpha_ref must be positive.")
    b0, _ = reduced_beta_coefficients(n_f)
    if b0 <= 0.0:
        raise ValueError("One-loop Lambda extraction requires b0 > 0.")
    return mu_ref * np.exp(-1.0 / (b0 * alpha_ref))


def fit_alpha_ref_from_r_data(
    energies: np.ndarray,
    r_observed: np.ndarray,
    config: QCDConfig,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Fit alpha_ref by least squares against R-ratio observations."""

    def residuals(params: np.ndarray) -> np.ndarray:
        alpha0 = float(params[0])
        if alpha0 <= 0.0:
            return np.full_like(r_observed, 1e6, dtype=float)
        try:
            alpha_curve = integrate_running_alpha(
                mu_grid=energies,
                n_f=config.n_f,
                mu_ref=config.mu_ref_gev,
                alpha_ref=alpha0,
                loops=config.loops,
            )
        except Exception:
            return np.full_like(r_observed, 1e6, dtype=float)

        r_model = qcd_r_ratio(alpha_curve, n_f=config.n_f)
        return r_model - r_observed

    result = least_squares(
        residuals,
        x0=np.array([0.15], dtype=float),
        bounds=(0.05, 0.6),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=300,
    )
    if not result.success:
        raise RuntimeError(f"Nonlinear fit failed: {result.message}")

    alpha_ref_fit = float(result.x[0])
    alpha_fit_curve = integrate_running_alpha(
        mu_grid=energies,
        n_f=config.n_f,
        mu_ref=config.mu_ref_gev,
        alpha_ref=alpha_ref_fit,
        loops=config.loops,
    )
    r_fit = qcd_r_ratio(alpha_fit_curve, n_f=config.n_f)
    rmse = float(np.sqrt(np.mean((r_fit - r_observed) ** 2)))
    return alpha_ref_fit, alpha_fit_curve, r_fit, rmse


def fit_one_loop_lambda_with_sklearn(
    energies: np.ndarray,
    alpha_estimates: np.ndarray,
    n_f: int,
) -> dict[str, float]:
    """Fit 1/alpha = b0*ln(mu) - b0*ln(Lambda) using linear regression."""
    x = np.log(np.asarray(energies, dtype=float)).reshape(-1, 1)
    y = 1.0 / np.asarray(alpha_estimates, dtype=float)

    model = LinearRegression().fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    if slope <= 0.0:
        raise RuntimeError("Unexpected non-positive slope from one-loop regression.")

    lambda_est = float(np.exp(-intercept / slope))
    b0_expected, _ = reduced_beta_coefficients(n_f)
    r2 = float(model.score(x, y))

    return {
        "slope_fit": slope,
        "intercept_fit": intercept,
        "lambda_est_gev": lambda_est,
        "b0_expected": b0_expected,
        "r2": r2,
    }


def torch_beta_derivative(alpha_value: float, n_f: int, loops: int = 2) -> tuple[float, float, float]:
    """Compare autograd derivative d(beta)/d(alpha) with analytic derivative."""
    a = torch.tensor(alpha_value, dtype=torch.float64, requires_grad=True)
    b0, b1 = reduced_beta_coefficients(n_f)

    beta_val = -b0 * a**2
    if loops == 2:
        beta_val = beta_val - b1 * a**3

    beta_val.backward()
    autograd_derivative = float(a.grad.item())

    analytic_derivative = -2.0 * b0 * alpha_value
    if loops == 2:
        analytic_derivative -= 3.0 * b1 * alpha_value**2

    abs_error = abs(autograd_derivative - analytic_derivative)
    return autograd_derivative, float(analytic_derivative), float(abs_error)


def main() -> None:
    config = QCDConfig()
    validate_config(config)

    rng = np.random.default_rng(config.seed)
    energies = np.geomspace(config.mu_ref_gev, config.energy_max_gev, config.num_points)

    alpha_true = integrate_running_alpha(
        mu_grid=energies,
        n_f=config.n_f,
        mu_ref=config.mu_ref_gev,
        alpha_ref=config.alpha_ref_true,
        loops=config.loops,
    )

    r_true = qcd_r_ratio(alpha_true, n_f=config.n_f)
    r_observed = r_true + rng.normal(loc=0.0, scale=config.noise_sigma, size=r_true.size)
    alpha_from_r = invert_r_ratio_to_alpha(r_observed, n_f=config.n_f)

    alpha_ref_fit, alpha_fit_curve, r_fit, rmse = fit_alpha_ref_from_r_data(
        energies=energies,
        r_observed=r_observed,
        config=config,
    )

    reg_stats = fit_one_loop_lambda_with_sklearn(
        energies=energies,
        alpha_estimates=alpha_from_r,
        n_f=config.n_f,
    )

    lambda_true_1loop = one_loop_lambda_from_reference(
        mu_ref=config.mu_ref_gev,
        alpha_ref=config.alpha_ref_true,
        n_f=config.n_f,
    )

    deriv_auto, deriv_analytic, deriv_abs_err = torch_beta_derivative(
        alpha_value=config.alpha_ref_true,
        n_f=config.n_f,
        loops=config.loops,
    )

    coeff_rows = []
    for n_f in [3, 5, 6, 17]:
        beta0, beta1 = qcd_beta_coefficients(n_f)
        coeff_rows.append(
            {
                "N_f": n_f,
                "beta0": beta0,
                "beta1": beta1,
                "asymptotically_free(beta0>0)": beta0 > 0.0,
            }
        )
    coeff_df = pd.DataFrame(coeff_rows)

    report_df = pd.DataFrame(
        {
            "mu_GeV": energies,
            "alpha_true": alpha_true,
            "R_true": r_true,
            "R_observed": r_observed,
            "alpha_from_R": alpha_from_r,
            "alpha_fit_curve": alpha_fit_curve,
            "R_fit": r_fit,
        }
    )
    report_df["abs_alpha_err_from_inversion"] = np.abs(report_df["alpha_from_R"] - report_df["alpha_true"])

    print("=== QCD Beta Coefficients (SU(3)) ===")
    print(coeff_df.to_string(index=False))
    print()
    print("=== Configuration ===")
    print(
        " ".join(
            [
                f"N_f={config.n_f}",
                f"mu_ref={config.mu_ref_gev:.3f} GeV",
                f"alpha_ref_true={config.alpha_ref_true:.4f}",
                f"energy_max={config.energy_max_gev:.1f} GeV",
                f"points={config.num_points}",
                f"noise_sigma={config.noise_sigma:.4f}",
                f"loops={config.loops}",
            ]
        )
    )
    print()
    print("=== Running + R-ratio Table ===")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()
    print("=== Inference Summary ===")
    print(f"True alpha_ref at mu_ref:        {config.alpha_ref_true:.8f}")
    print(f"Fitted alpha_ref (nonlinear):    {alpha_ref_fit:.8f}")
    print(f"Nonlinear fit RMSE in R:         {rmse:.8e}")
    print(f"One-loop lambda from true ref:   {lambda_true_1loop:.8f} GeV")
    print(f"One-loop lambda via regression:  {reg_stats['lambda_est_gev']:.8f} GeV")
    print(f"One-loop slope fit b0:           {reg_stats['slope_fit']:.8f}")
    print(f"Expected one-loop b0:            {reg_stats['b0_expected']:.8f}")
    print(f"Regression R^2:                  {reg_stats['r2']:.8f}")
    print()
    print("=== Torch Autograd Check for d(beta)/d(alpha) ===")
    print(f"autograd derivative:             {deriv_auto:.12f}")
    print(f"analytic derivative:             {deriv_analytic:.12f}")
    print(f"absolute error:                  {deriv_abs_err:.3e}")

    beta0_nf5, _ = qcd_beta_coefficients(5)
    beta0_nf17, _ = qcd_beta_coefficients(17)

    # Deterministic validation checks.
    assert beta0_nf5 > 0.0, "N_f=5 should be asymptotically free."
    assert beta0_nf17 < 0.0, "N_f=17 should lose asymptotic freedom."
    assert np.all(np.diff(alpha_true) < 0.0), "True running alpha should decrease with mu."
    assert np.all(np.diff(alpha_fit_curve) < 0.0), "Fitted running alpha should decrease with mu."
    assert abs(alpha_ref_fit - config.alpha_ref_true) < 0.03, "Recovered alpha_ref is too far from truth."
    assert rmse < 0.02, "R-ratio fit RMSE is unexpectedly high."
    assert 0.6 * reg_stats["b0_expected"] < reg_stats["slope_fit"] < 1.4 * reg_stats["b0_expected"], "Linearized slope is inconsistent with one-loop expectation."
    assert reg_stats["r2"] > 0.97, "Linearized one-loop fit quality is too low."
    assert deriv_abs_err < 1e-12, "Torch derivative check failed."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
