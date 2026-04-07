"""Tensor-to-scalar ratio (r) MVP with a toy CMB B-mode likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class TensorScalarParams:
    """Parameters for a minimal tensor-to-scalar ratio inference demo."""

    ell_min: int = 20
    ell_max: int = 200
    n_bins: int = 12

    # Synthetic truth used to generate mock data.
    r_true: float = 0.035
    lensing_amp_true: float = 1.0
    noise_floor_uk2: float = 0.0012
    frac_noise: float = 0.18
    random_seed: int = 7

    # Cosmological constants / pivot parameters.
    pivot_k_mpc: float = 0.05
    a_s: float = 2.1e-9
    m_pl_reduced_gev: float = 2.435e18

    # Inference bounds for r (physical domain r >= 0).
    r_min: float = 0.0
    r_max: float = 0.3


def make_ell_grid(params: TensorScalarParams) -> np.ndarray:
    """Construct effective multipole-bin centers."""
    return np.linspace(params.ell_min, params.ell_max, params.n_bins)


def primordial_bb_template_r1(ell: np.ndarray) -> np.ndarray:
    """Toy primordial B-mode template for r=1 in uK^2.

    The shape combines a reionization-like low-ell bump and a recombination bump.
    This is an educational proxy, not a Boltzmann-solver output.
    """
    x = np.asarray(ell, dtype=float)
    reion_bump = 0.0026 * np.exp(-0.5 * ((x - 8.0) / 6.0) ** 2)
    recomb_bump = 0.012 * (x / 80.0) ** 2 * np.exp(-(x / 90.0) ** 1.3)
    return reion_bump + recomb_bump


def lensing_bb_template(ell: np.ndarray) -> np.ndarray:
    """Toy lensing B-mode template in uK^2."""
    x = np.asarray(ell, dtype=float)
    return 0.001 + 0.0045 * (x / 120.0) ** 0.9 * np.exp(-(x / 420.0) ** 1.15)


def build_mock_dataset(params: TensorScalarParams) -> pd.DataFrame:
    """Generate synthetic observed bandpowers with heteroscedastic Gaussian noise."""
    ell = make_ell_grid(params)
    t_unit = primordial_bb_template_r1(ell)
    l_temp = lensing_bb_template(ell)

    signal = params.r_true * t_unit + params.lensing_amp_true * l_temp
    sigma = params.noise_floor_uk2 + params.frac_noise * signal

    rng = np.random.default_rng(params.random_seed)
    obs = signal + rng.normal(0.0, sigma)

    return pd.DataFrame(
        {
            "ell": ell,
            "tensor_template_r1_uk2": t_unit,
            "lensing_template_uk2": l_temp,
            "bb_signal_true_uk2": signal,
            "sigma_uk2": sigma,
            "bb_obs_uk2": obs,
        }
    )


def chi2_of_r(r_value: float, data: pd.DataFrame) -> float:
    """Chi-square under fixed-lensing-amplitude model: D = r*T + L."""
    model = r_value * data["tensor_template_r1_uk2"].to_numpy() + data["lensing_template_uk2"].to_numpy()
    resid = (data["bb_obs_uk2"].to_numpy() - model) / data["sigma_uk2"].to_numpy()
    return float(np.sum(resid**2))


def fit_r_bounded(data: pd.DataFrame, r_min: float, r_max: float) -> tuple[float, float]:
    """Estimate r by minimizing chi^2 on a bounded interval."""
    result = optimize.minimize_scalar(
        lambda r: chi2_of_r(r, data),
        method="bounded",
        bounds=(r_min, r_max),
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"r fit failed: {result.message}")
    return float(result.x), float(result.fun)


def estimate_sigma_from_curvature(r_hat: float, data: pd.DataFrame, step: float = 1e-4) -> float:
    """Approximate 1-sigma uncertainty using chi^2 local curvature.

    Around the minimum: chi^2 ~ chi^2_min + (r-r_hat)^2 / sigma_r^2.
    Therefore d2(chi^2)/dr2 = 2 / sigma_r^2.
    """
    f_minus = chi2_of_r(r_hat - step, data)
    f_0 = chi2_of_r(r_hat, data)
    f_plus = chi2_of_r(r_hat + step, data)
    second_derivative = (f_plus - 2.0 * f_0 + f_minus) / (step**2)
    if second_derivative <= 0.0:
        return float("nan")
    return float(np.sqrt(2.0 / second_derivative))


def upper_limit_r_95(r_hat: float, chi2_min: float, data: pd.DataFrame, r_max: float) -> float:
    """One-sided 95% CL upper limit via Delta chi^2 = 2.71."""
    delta_target = 2.71

    def root_fn(r_value: float) -> float:
        return chi2_of_r(r_value, data) - chi2_min - delta_target

    if root_fn(r_max) <= 0.0:
        return r_max

    r_lo = max(r_hat, 0.0)
    return float(optimize.brentq(root_fn, r_lo, r_max, xtol=1e-10, rtol=1e-10, maxiter=300))


def slow_roll_from_r(r_value: float) -> tuple[float, float]:
    """Return (epsilon, n_t) from lowest-order slow-roll consistency."""
    epsilon = r_value / 16.0
    n_t = -r_value / 8.0
    return float(epsilon), float(n_t)


def inflation_energy_scale_gev(r_value: float, a_s: float, m_pl_reduced_gev: float) -> float:
    """Estimate inflationary energy scale V^(1/4) in GeV.

    Uses V = (3*pi^2/2) * A_s * r * M_pl^4 in slow-roll approximation.
    """
    if r_value <= 0.0:
        return 0.0
    prefactor = (3.0 * np.pi**2 / 2.0) * a_s * r_value
    return float((prefactor ** 0.25) * m_pl_reduced_gev)


def run_demo() -> None:
    params = TensorScalarParams()
    data = build_mock_dataset(params)

    r_hat, chi2_min = fit_r_bounded(data, params.r_min, params.r_max)
    sigma_r = estimate_sigma_from_curvature(r_hat, data)
    r_95 = upper_limit_r_95(r_hat, chi2_min, data, params.r_max)

    eps_hat, nt_hat = slow_roll_from_r(r_hat)
    e_inf_hat = inflation_energy_scale_gev(r_hat, params.a_s, params.m_pl_reduced_gev)
    e_inf_true = inflation_energy_scale_gev(params.r_true, params.a_s, params.m_pl_reduced_gev)

    print("=== Tensor-to-Scalar Ratio MVP ===")
    print(f"Pivot scale k* = {params.pivot_k_mpc:.3f} Mpc^-1")
    print(f"Synthetic truth: r_true = {params.r_true:.4f}")
    print(f"Best-fit r_hat = {r_hat:.6e}")
    print(f"Approx 1-sigma uncertainty = {sigma_r:.6e}")
    print(f"One-sided 95% upper limit r_95 = {r_95:.6e}")
    print(f"chi2_min = {chi2_min:.4f}, dof = {params.n_bins - 1}")

    print("\nDerived slow-roll quantities from r_hat:")
    print(f"epsilon = r/16 = {eps_hat:.5f}")
    print(f"n_t = -r/8 = {nt_hat:.5f}")

    print("\nInflation energy scale estimate:")
    print(f"V^(1/4) from r_hat  = {e_inf_hat:.3e} GeV")
    print(f"V^(1/4) from r_true = {e_inf_true:.3e} GeV")

    # Provide a compact likelihood profile for auditing.
    grid = np.linspace(params.r_min, params.r_max, 16)
    profile_rows: list[dict[str, float]] = []
    for r_val in grid:
        chi2 = chi2_of_r(float(r_val), data)
        profile_rows.append(
            {
                "r": float(r_val),
                "chi2": chi2,
                "delta_chi2": chi2 - chi2_min,
            }
        )
    profile = pd.DataFrame(profile_rows)

    print("\nChi-square profile sample:")
    print(profile.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nSynthetic bandpower table:")
    print(data.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
