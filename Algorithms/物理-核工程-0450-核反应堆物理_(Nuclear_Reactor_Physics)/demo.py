"""Minimal runnable MVP for Nuclear Reactor Physics.

This script implements a transparent one-group neutron diffusion criticality model
in a 1D bare slab reactor with vacuum boundaries:

    -D d2(phi)/dx2 + Sigma_a * phi = (1/k_eff) * nuSigma_f * phi

The algorithm is shown explicitly (matrix assembly + power iteration), while
SciPy/Torch are used as independent numerical cross-checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LinearRegression
import torch


@dataclass(frozen=True)
class Reactor1DConfig:
    """Configuration for one-group 1D slab diffusion criticality."""

    length_cm: float = 60.0
    n_cells: int = 120
    diffusion_coeff_cm: float = 1.3
    sigma_a_cm_inv: float = 0.09
    nu_sigma_f_cm_inv: float = 0.11


def validate_config(cfg: Reactor1DConfig) -> tuple[bool, str]:
    """Sanity checks for physically meaningful and numerically safe inputs."""

    if cfg.length_cm <= 0.0:
        return False, "length_cm must be > 0"
    if cfg.n_cells < 10:
        return False, "n_cells must be >= 10"
    if cfg.diffusion_coeff_cm <= 0.0:
        return False, "diffusion_coeff_cm must be > 0"
    if cfg.sigma_a_cm_inv <= 0.0:
        return False, "sigma_a_cm_inv must be > 0"
    if cfg.nu_sigma_f_cm_inv <= 0.0:
        return False, "nu_sigma_f_cm_inv must be > 0"
    return True, "ok"


def build_diffusion_system(cfg: Reactor1DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build finite-difference operators A and F for A*phi=(1/k)*F*phi.

    Returns:
      x: interior cell-center coordinates
      A: loss operator (leakage + absorption)
      F_diag: fission production diagonal entries
      dx: mesh spacing
    """

    n = cfg.n_cells
    dx = cfg.length_cm / (n + 1)
    x = dx * np.arange(1, n + 1)

    leak_main = 2.0 * cfg.diffusion_coeff_cm / (dx * dx)
    leak_off = -cfg.diffusion_coeff_cm / (dx * dx)

    main_diag = np.full(n, leak_main + cfg.sigma_a_cm_inv, dtype=np.float64)
    off_diag = np.full(n - 1, leak_off, dtype=np.float64)

    A = np.diag(main_diag)
    A += np.diag(off_diag, k=1)
    A += np.diag(off_diag, k=-1)

    F_diag = np.full(n, cfg.nu_sigma_f_cm_inv, dtype=np.float64)
    return x, A, F_diag, dx


def power_iteration_keff(
    A: np.ndarray,
    F_diag: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-11,
) -> tuple[float, np.ndarray, int, bool]:
    """Dominant eigenpair of M=A^{-1}F using explicit source iteration."""

    n = A.shape[0]
    phi = np.ones(n, dtype=np.float64)
    phi /= np.linalg.norm(phi)

    k_old = 1.0
    converged = False

    for it in range(1, max_iter + 1):
        fission_source = F_diag * phi
        phi_new = np.linalg.solve(A, fission_source)

        norm = np.linalg.norm(phi_new)
        if norm == 0.0:
            break
        phi_new /= norm

        numerator = float(phi_new @ (F_diag * phi_new))
        denominator = float(phi_new @ (A @ phi_new))
        if denominator <= 0.0:
            break

        k_new = numerator / denominator

        if abs(k_new - k_old) < tol:
            phi = phi_new
            k_old = k_new
            converged = True
            return k_old, phi, it, converged

        phi = phi_new
        k_old = k_new

    return k_old, phi, it, converged


def scipy_generalized_keff(A: np.ndarray, F_diag: np.ndarray) -> float:
    """Solve generalized eigenproblem A v = lambda F v (lambda=1/k)."""

    F = np.diag(F_diag)
    lambdas = linalg.eigvalsh(A, F)
    lambda_min = float(np.min(lambdas))
    return 1.0 / lambda_min


def torch_power_iteration_keff(
    A: np.ndarray,
    F_diag: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-11,
) -> tuple[float, np.ndarray, int, bool]:
    """Independent PyTorch implementation for cross-checking."""

    A_t = torch.tensor(A, dtype=torch.float64)
    F_t = torch.tensor(F_diag, dtype=torch.float64)

    n = A_t.shape[0]
    phi = torch.ones(n, dtype=torch.float64)
    phi = phi / torch.linalg.vector_norm(phi)

    k_old = 1.0
    converged = False

    for it in range(1, max_iter + 1):
        source = F_t * phi
        phi_new = torch.linalg.solve(A_t, source)

        norm = torch.linalg.vector_norm(phi_new)
        if float(norm) == 0.0:
            break
        phi_new = phi_new / norm

        numerator = torch.dot(phi_new, F_t * phi_new)
        denominator = torch.dot(phi_new, A_t @ phi_new)
        if float(denominator) <= 0.0:
            break

        k_new = float(numerator / denominator)

        if abs(k_new - k_old) < tol:
            phi = phi_new
            k_old = k_new
            converged = True
            return k_old, phi.cpu().numpy(), it, converged

        phi = phi_new
        k_old = k_new

    return k_old, phi.cpu().numpy(), it, converged


def analytic_keff_uniform(cfg: Reactor1DConfig) -> float:
    """Continuous bare slab formula for uniform one-group coefficients."""

    buckling = (math.pi / cfg.length_cm) ** 2
    denom = cfg.sigma_a_cm_inv + cfg.diffusion_coeff_cm * buckling
    return cfg.nu_sigma_f_cm_inv / denom


def normalize_flux(phi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize flux to unit integral over the slab."""

    area = float(np.sum(phi) * dx)
    if area <= 0.0:
        return phi
    return phi / area


def run_length_sweep(base_cfg: Reactor1DConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sweep reactor size and fit 1/k = a + b*(1/L^2) via scikit-learn."""

    lengths = np.array([18.0, 22.0, 26.0, 30.0, 40.0, 55.0, 75.0], dtype=np.float64)
    rows: list[dict[str, float]] = []

    for L in lengths:
        cfg = Reactor1DConfig(
            length_cm=float(L),
            n_cells=base_cfg.n_cells,
            diffusion_coeff_cm=base_cfg.diffusion_coeff_cm,
            sigma_a_cm_inv=base_cfg.sigma_a_cm_inv,
            nu_sigma_f_cm_inv=base_cfg.nu_sigma_f_cm_inv,
        )
        _, A, F_diag, _dx = build_diffusion_system(cfg)
        k_num, _phi, _it, _ok = power_iteration_keff(A, F_diag)
        k_an = analytic_keff_uniform(cfg)
        rows.append(
            {
                "length_cm": L,
                "inv_length2_cm-2": 1.0 / (L * L),
                "k_numeric": k_num,
                "k_analytic": k_an,
                "rel_err_percent": 100.0 * abs(k_num - k_an) / k_an,
            }
        )

    df = pd.DataFrame(rows)
    X = df[["inv_length2_cm-2"]].to_numpy()
    y = (1.0 / df["k_numeric"].to_numpy()).reshape(-1)

    model = LinearRegression()
    model.fit(X, y)

    intercept_fit = float(model.intercept_)
    slope_fit = float(model.coef_[0])

    intercept_theory = base_cfg.sigma_a_cm_inv / base_cfg.nu_sigma_f_cm_inv
    slope_theory = (base_cfg.diffusion_coeff_cm * math.pi * math.pi) / base_cfg.nu_sigma_f_cm_inv

    metrics = {
        "fit_intercept": intercept_fit,
        "fit_slope": slope_fit,
        "theory_intercept": intercept_theory,
        "theory_slope": slope_theory,
        "intercept_rel_err_percent": 100.0 * abs(intercept_fit - intercept_theory) / intercept_theory,
        "slope_rel_err_percent": 100.0 * abs(slope_fit - slope_theory) / slope_theory,
        "fit_r2": float(model.score(X, y)),
    }

    return df, metrics


def main() -> None:
    cfg = Reactor1DConfig()
    ok, reason = validate_config(cfg)
    if not ok:
        raise ValueError(f"Invalid Reactor1DConfig: {reason}")

    x, A, F_diag, dx = build_diffusion_system(cfg)

    k_power, phi_power, it_power, conv_power = power_iteration_keff(A, F_diag)
    k_scipy = scipy_generalized_keff(A, F_diag)
    k_torch, phi_torch, it_torch, conv_torch = torch_power_iteration_keff(A, F_diag)
    k_analytic = analytic_keff_uniform(cfg)

    phi_norm = normalize_flux(phi_power, dx)
    peak_idx = int(np.argmax(phi_norm))

    flux_df = pd.DataFrame(
        {
            "x_cm": x,
            "flux": phi_norm,
            "torch_flux": normalize_flux(phi_torch, dx),
        }
    )
    flux_df["abs_flux_diff"] = np.abs(flux_df["flux"] - flux_df["torch_flux"])

    summary = {
        "k_power": k_power,
        "k_scipy": k_scipy,
        "k_torch": k_torch,
        "k_analytic": k_analytic,
        "abs_diff_power_vs_scipy": abs(k_power - k_scipy),
        "abs_diff_power_vs_torch": abs(k_power - k_torch),
        "abs_diff_power_vs_analytic": abs(k_power - k_analytic),
        "power_iterations": float(it_power),
        "torch_iterations": float(it_torch),
        "power_converged": float(conv_power),
        "torch_converged": float(conv_torch),
        "peak_x_cm": float(x[peak_idx]),
        "max_flux_abs_diff": float(flux_df["abs_flux_diff"].max()),
    }

    sweep_df, fit_metrics = run_length_sweep(cfg)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v: .6e}")

    print("Nuclear Reactor Physics MVP: 1D One-Group Diffusion Criticality")
    print("\n[Config]")
    print(cfg)

    print("\n[Core k-effective Results]")
    for key, value in summary.items():
        print(f"{key:>28}: {value}")

    print("\n[Flux Sample: first 8 rows]")
    print(flux_df.head(8))

    print("\n[Length Sweep for 1/k vs 1/L^2]")
    print(sweep_df)

    print("\n[Linear Fit Metrics (scikit-learn)]")
    for key, value in fit_metrics.items():
        print(f"{key:>28}: {value: .6e}")

    # Automatic consistency gates for validation pipelines.
    assert conv_power, "Power iteration did not converge"
    assert conv_torch, "Torch power iteration did not converge"
    assert abs(k_power - k_scipy) < 1e-9, "Power vs SciPy mismatch too large"
    assert abs(k_power - k_torch) < 1e-9, "Power vs Torch mismatch too large"
    assert fit_metrics["fit_r2"] > 0.999, "1/k vs 1/L^2 linear law fit is unexpectedly weak"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
