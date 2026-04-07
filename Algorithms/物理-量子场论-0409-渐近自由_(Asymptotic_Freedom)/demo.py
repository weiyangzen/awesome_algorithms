"""Asymptotic freedom MVP for QCD running coupling.

This script demonstrates:
1) one-loop and two-loop beta-function driven running of alpha_s(mu),
2) asymptotic freedom for N_f < 16.5 in SU(3),
3) sign flip when N_f is too large (non-asymptotically free case).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class QCDScenario:
    n_f: int
    mu_ref_gev: float
    alpha_ref: float


def qcd_beta_coefficients(n_f: int) -> tuple[float, float]:
    """Return (beta0, beta1) in the common QCD convention.

    beta(alpha_s) = mu d(alpha_s)/dmu
                  = -beta0 * alpha_s^2 / (2*pi) - beta1 * alpha_s^3 / (4*pi^2) + ...
    """
    if n_f < 0:
        raise ValueError("n_f must be non-negative.")
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
    """Beta function value d alpha / d ln(mu)."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if loops not in (1, 2):
        raise ValueError("loops must be 1 or 2.")

    b0, b1 = reduced_beta_coefficients(n_f)
    value = -b0 * alpha**2
    if loops == 2:
        value -= b1 * alpha**3
    return value


def one_loop_lambda_qcd(mu_ref: float, alpha_ref: float, n_f: int) -> float:
    """Infer Lambda_QCD at one-loop from a reference point (mu_ref, alpha_ref)."""
    if mu_ref <= 0.0:
        raise ValueError("mu_ref must be positive.")
    if alpha_ref <= 0.0:
        raise ValueError("alpha_ref must be positive.")
    b0, _ = reduced_beta_coefficients(n_f)
    if b0 <= 0.0:
        raise ValueError("Asymptotic-freedom one-loop Lambda needs b0 > 0.")
    return mu_ref * np.exp(-1.0 / (b0 * alpha_ref))


def alpha_one_loop_closed_form(mu: np.ndarray, n_f: int, lambda_qcd: float) -> np.ndarray:
    """One-loop closed-form alpha_s(mu) = 1 / (b0 * ln(mu / Lambda))."""
    mu = np.asarray(mu, dtype=float)
    if np.any(mu <= 0.0):
        raise ValueError("All mu must be positive.")
    if lambda_qcd <= 0.0:
        raise ValueError("lambda_qcd must be positive.")
    b0, _ = reduced_beta_coefficients(n_f)
    if b0 <= 0.0:
        raise ValueError("Closed form requires b0 > 0.")
    log_term = np.log(mu / lambda_qcd)
    if np.any(log_term <= 0.0):
        raise ValueError("mu must be larger than lambda_qcd for perturbative formula.")
    return 1.0 / (b0 * log_term)


def integrate_running_alpha(
    mu_grid: np.ndarray,
    n_f: int,
    mu_ref: float,
    alpha_ref: float,
    loops: int = 2,
) -> np.ndarray:
    """Integrate d alpha / d ln(mu) on an ascending mu grid."""
    mu_grid = np.asarray(mu_grid, dtype=float)
    if mu_grid.ndim != 1 or mu_grid.size < 2:
        raise ValueError("mu_grid must be 1D with at least two points.")
    if np.any(mu_grid <= 0.0):
        raise ValueError("mu_grid values must be positive.")
    if not np.all(np.diff(mu_grid) > 0.0):
        raise ValueError("mu_grid must be strictly ascending.")
    if not np.isclose(mu_grid[0], mu_ref):
        raise ValueError("mu_grid[0] must match mu_ref.")

    t_eval = np.log(mu_grid)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        alpha = max(float(y[0]), 1e-14)
        return np.array([beta_alpha(alpha=alpha, n_f=n_f, loops=loops)], dtype=float)

    solution = solve_ivp(
        rhs,
        t_span=t_span,
        y0=np.array([alpha_ref], dtype=float),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    alpha_values = solution.y[0]
    if np.any(~np.isfinite(alpha_values)) or np.any(alpha_values <= 0.0):
        raise RuntimeError("Non-finite or non-positive alpha encountered in integration.")
    return alpha_values


def summarize_coefficients(nf_list: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for n_f in nf_list:
        beta0, beta1 = qcd_beta_coefficients(n_f)
        rows.append(
            {
                "N_f": n_f,
                "beta0": beta0,
                "beta1": beta1,
                "asymptotically_free(beta0>0)": beta0 > 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    scenario = QCDScenario(n_f=5, mu_ref_gev=2.0, alpha_ref=0.30)
    mu_grid = scenario.mu_ref_gev * np.geomspace(1.0, 500.0, 14)

    lambda_qcd = one_loop_lambda_qcd(
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        n_f=scenario.n_f,
    )

    alpha_1loop_numeric = integrate_running_alpha(
        mu_grid=mu_grid,
        n_f=scenario.n_f,
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        loops=1,
    )
    alpha_2loop_numeric = integrate_running_alpha(
        mu_grid=mu_grid,
        n_f=scenario.n_f,
        mu_ref=scenario.mu_ref_gev,
        alpha_ref=scenario.alpha_ref,
        loops=2,
    )
    alpha_1loop_analytic = alpha_one_loop_closed_form(
        mu=mu_grid,
        n_f=scenario.n_f,
        lambda_qcd=lambda_qcd,
    )

    report_df = pd.DataFrame(
        {
            "mu_GeV": mu_grid,
            "alpha_1loop_analytic": alpha_1loop_analytic,
            "alpha_1loop_numeric": alpha_1loop_numeric,
            "alpha_2loop_numeric": alpha_2loop_numeric,
        }
    )
    report_df["abs_err_1loop"] = np.abs(
        report_df["alpha_1loop_numeric"] - report_df["alpha_1loop_analytic"]
    )

    coeff_df = summarize_coefficients([3, 5, 6, 16, 17])

    beta_nf5 = beta_alpha(alpha=0.10, n_f=5, loops=1)
    beta_nf17 = beta_alpha(alpha=0.10, n_f=17, loops=1)

    print("=== QCD Beta Coefficients (SU(3)) ===")
    print(coeff_df.to_string(index=False))
    print()
    print(f"Reference scenario: N_f={scenario.n_f}, mu_ref={scenario.mu_ref_gev:.3f} GeV, alpha_ref={scenario.alpha_ref:.3f}")
    print(f"Inferred one-loop Lambda_QCD = {lambda_qcd:.6f} GeV")
    print()
    print("=== Running Coupling Table ===")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()
    print("=== Beta Function Sign Check at alpha=0.10 (one-loop) ===")
    print(f"beta(alpha=0.10, N_f=5)  = {beta_nf5:.8f}  (should be negative)")
    print(f"beta(alpha=0.10, N_f=17) = {beta_nf17:.8f}  (should be positive)")

    max_rel_err_1loop = float(
        np.max(
            np.abs(
                (alpha_1loop_numeric - alpha_1loop_analytic)
                / np.maximum(alpha_1loop_analytic, 1e-14)
            )
        )
    )

    # Assertions for asymptotic freedom and numerical consistency.
    assert max_rel_err_1loop < 5e-6, f"1-loop numeric/analytic mismatch too large: {max_rel_err_1loop:.3e}"
    assert np.all(np.diff(alpha_1loop_numeric) < 0.0), "1-loop alpha should decrease with increasing mu."
    assert np.all(np.diff(alpha_2loop_numeric) < 0.0), "2-loop alpha should decrease with increasing mu."
    assert beta_nf5 < 0.0, "QCD with N_f=5 should be asymptotically free (negative beta at small alpha)."
    assert beta_nf17 > 0.0, "N_f=17 should lose asymptotic freedom (positive beta at small alpha)."
    assert alpha_2loop_numeric[-1] < scenario.alpha_ref, "High-energy alpha should be lower than reference alpha."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
