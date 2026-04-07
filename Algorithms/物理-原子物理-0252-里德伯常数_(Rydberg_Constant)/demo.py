"""Minimal runnable MVP for estimating the Rydberg constant from Balmer lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None


@dataclass
class BalmerDataset:
    n2: np.ndarray
    x: np.ndarray
    wavelength_m: np.ndarray
    inv_wavelength_m_inv: np.ndarray


def rydberg_infinite_from_constants() -> float:
    """Compute R_infinity from fundamental constants."""
    alpha = constants.alpha
    m_e = constants.m_e
    c = constants.c
    h = constants.h
    return (alpha * alpha * m_e * c) / (2.0 * h)


def rydberg_with_reduced_mass(rydberg_inf: float, nuclear_mass_kg: float) -> float:
    """Apply reduced-mass correction R_M = R_inf / (1 + m_e/M)."""
    return rydberg_inf / (1.0 + constants.m_e / nuclear_mass_kg)


def balmer_factor(n2: np.ndarray, n1: int = 2) -> np.ndarray:
    n2 = np.asarray(n2, dtype=np.float64)
    if np.any(n2 <= n1):
        raise ValueError("All n2 must be > n1 for emission lines")
    return (1.0 / float(n1 * n1)) - (1.0 / np.square(n2))


def generate_balmer_dataset(
    rydberg_true: float,
    n2_values: np.ndarray,
    noise_ppm: float = 80.0,
    seed: int = 42,
) -> BalmerDataset:
    """Create deterministic synthetic Balmer observations with ppm-level multiplicative noise."""
    n2 = np.asarray(n2_values, dtype=np.float64)
    x = balmer_factor(n2=n2, n1=2)

    inv_lambda_true = rydberg_true * x

    rng = np.random.default_rng(seed)
    rel_noise = rng.normal(loc=0.0, scale=noise_ppm * 1e-6, size=n2.shape)
    inv_lambda_obs = inv_lambda_true * (1.0 + rel_noise)
    wavelength_obs = 1.0 / inv_lambda_obs

    return BalmerDataset(
        n2=n2,
        x=x,
        wavelength_m=wavelength_obs,
        inv_wavelength_m_inv=inv_lambda_obs,
    )


def estimate_rydberg_sklearn(dataset: BalmerDataset) -> float:
    """Estimate R from y=R*x with zero intercept (physics-constrained model)."""
    x = dataset.x.reshape(-1, 1)
    y = dataset.inv_wavelength_m_inv
    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    return float(model.coef_[0])


def estimate_rydberg_sklearn_with_intercept(dataset: BalmerDataset) -> tuple[float, float]:
    """Diagnostic fit with an intercept to inspect systematic bias."""
    x = dataset.x.reshape(-1, 1)
    y = dataset.inv_wavelength_m_inv
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    return float(model.coef_[0]), float(model.intercept_)


def estimate_rydberg_curve_fit(dataset: BalmerDataset) -> float:
    """Estimate R by fitting wavelength-domain nonlinear model."""

    def wavelength_model(n2: np.ndarray, rydberg: float) -> np.ndarray:
        return 1.0 / (rydberg * balmer_factor(n2=n2, n1=2))

    popt, _ = curve_fit(
        f=wavelength_model,
        xdata=dataset.n2,
        ydata=dataset.wavelength_m,
        p0=np.array([constants.Rydberg], dtype=np.float64),
        bounds=(1.0e7, 1.2e7),
        maxfev=20000,
    )
    return float(popt[0])


def estimate_rydberg_torch_optional(
    dataset: BalmerDataset,
    lr: float = 0.05,
    steps: int = 800,
) -> Optional[float]:
    """Optional single-parameter gradient fit in PyTorch."""
    if torch is None:
        return None

    dtype = torch.float64
    device = torch.device("cpu")

    x = torch.tensor(dataset.x, dtype=dtype, device=device)
    y = torch.tensor(dataset.inv_wavelength_m_inv / 1.0e6, dtype=dtype, device=device)

    scale = 1.0e6
    r_scaled = torch.nn.Parameter(torch.tensor(10.96, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([r_scaled], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = r_scaled * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    return float(r_scaled.detach().cpu().item() * scale)


def build_observation_table(dataset: BalmerDataset) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "n2": dataset.n2.astype(int),
            "balmer_factor": dataset.x,
            "inv_lambda_obs_m^-1": dataset.inv_wavelength_m_inv,
            "lambda_obs_nm": dataset.wavelength_m * 1e9,
        }
    )
    return table


def build_estimate_summary(reference_rh: float, estimates: dict[str, Optional[float]]) -> pd.DataFrame:
    rows = []
    for name, value in estimates.items():
        if value is None:
            rows.append({"method": name, "R_est_m^-1": np.nan, "rel_err_vs_R_H": np.nan})
            continue
        rel_err = abs(value - reference_rh) / reference_rh
        rows.append({"method": name, "R_est_m^-1": value, "rel_err_vs_R_H": rel_err})
    return pd.DataFrame(rows)


def main() -> None:
    rydberg_inf_formula = rydberg_infinite_from_constants()
    rydberg_inf_codata = constants.Rydberg
    rel_err_formula = abs(rydberg_inf_formula - rydberg_inf_codata) / rydberg_inf_codata

    rydberg_h = rydberg_with_reduced_mass(rydberg_inf=rydberg_inf_formula, nuclear_mass_kg=constants.m_p)

    n2_values = np.arange(3, 13, dtype=np.float64)
    dataset = generate_balmer_dataset(
        rydberg_true=rydberg_h,
        n2_values=n2_values,
        noise_ppm=80.0,
        seed=42,
    )

    obs_table = build_observation_table(dataset)

    r_sklearn = estimate_rydberg_sklearn(dataset)
    r_sklearn_int, intercept = estimate_rydberg_sklearn_with_intercept(dataset)
    r_curve = estimate_rydberg_curve_fit(dataset)
    r_torch = estimate_rydberg_torch_optional(dataset)

    estimates = {
        "sklearn_linear_no_intercept": r_sklearn,
        "sklearn_linear_with_intercept": r_sklearn_int,
        "scipy_curve_fit": r_curve,
        "torch_gradient_descent": r_torch,
    }
    summary = build_estimate_summary(reference_rh=rydberg_h, estimates=estimates)

    print("Rydberg constant MVP")
    print(f"R_inf_formula      = {rydberg_inf_formula:.9f} m^-1")
    print(f"R_inf_scipy        = {rydberg_inf_codata:.9f} m^-1")
    print(f"relative_error     = {rel_err_formula:.3e}")
    print(f"R_H_reference      = {rydberg_h:.9f} m^-1")
    print()

    print("Balmer synthetic observations (first 5 rows):")
    print(obs_table.head(5).to_string(index=False))
    print()

    print("Estimator summary:")
    print(summary.to_string(index=False, justify="left"))
    print(f"diagnostic_intercept_m^-1 = {intercept:.3f}")

    sklearn_rel_err = abs(r_sklearn - rydberg_h) / rydberg_h
    curve_rel_err = abs(r_curve - rydberg_h) / rydberg_h

    assert rel_err_formula < 1.0e-10, f"R_inf formula mismatch too large: {rel_err_formula}"
    assert sklearn_rel_err < 3.0e-4, f"sklearn estimate too far: {sklearn_rel_err}"
    assert curve_rel_err < 8.0e-4, f"curve_fit estimate too far: {curve_rel_err}"
    assert abs(intercept) < 4.0e3, f"unexpectedly large intercept: {intercept}"

    if r_torch is not None:
        torch_rel_err = abs(r_torch - rydberg_h) / rydberg_h
        assert torch_rel_err < 8.0e-4, f"torch estimate too far: {torch_rel_err}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
