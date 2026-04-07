"""Resonant scattering MVP using a transparent Breit-Wigner fit.

This script generates synthetic resonance-scattering data and fits the
Breit-Wigner line shape with a hand-written Levenberg-Marquardt loop
(no black-box curve fitting call).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t


@dataclass
class ResonanceParams:
    """Breit-Wigner parameters.

    Attributes:
        e0: Resonance energy (center).
        gamma: Full width at half maximum (FWHM).
        amplitude: Peak contribution above background.
        background: Non-resonant baseline.
    """

    e0: float
    gamma: float
    amplitude: float
    background: float

    def to_array(self) -> np.ndarray:
        return np.array([self.e0, self.gamma, self.amplitude, self.background], dtype=float)

    @staticmethod
    def from_array(theta: np.ndarray) -> "ResonanceParams":
        return ResonanceParams(
            e0=float(theta[0]),
            gamma=float(theta[1]),
            amplitude=float(theta[2]),
            background=float(theta[3]),
        )


@dataclass
class FitConfig:
    max_iter: int = 160
    damping_init: float = 1e-2
    damping_up: float = 2.5
    damping_down: float = 0.45
    tol_step: float = 1e-9
    min_gamma: float = 1e-8


@dataclass
class FitResult:
    params: ResonanceParams
    stderr: np.ndarray
    ci95_low: np.ndarray
    ci95_high: np.ndarray
    rmse: float
    r2: float
    sse: float
    iterations: int
    accepted_steps: int
    converged: bool


def validate_inputs(energy: np.ndarray, sigma_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize input arrays."""
    e = np.asarray(energy, dtype=float)
    y = np.asarray(sigma_obs, dtype=float)

    if e.ndim != 1 or y.ndim != 1:
        raise ValueError("energy and sigma_obs must be 1D arrays")
    if len(e) != len(y):
        raise ValueError("energy and sigma_obs must have the same length")
    if len(e) < 8:
        raise ValueError("at least 8 points are required for a stable fit")
    if not np.isfinite(e).all() or not np.isfinite(y).all():
        raise ValueError("inputs contain NaN or Inf")

    order = np.argsort(e)
    e = e[order]
    y = y[order]

    if np.unique(e).size < 8:
        raise ValueError("energy grid must contain enough distinct points")
    if np.ptp(y) <= 0.0:
        raise ValueError("sigma_obs must have non-zero dynamic range")

    return e, y


def breit_wigner_cross_section(energy: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Breit-Wigner profile with constant background.

    f(E) = background + amplitude * (w^2) / ((E - E0)^2 + w^2),
    where w = gamma / 2.
    """
    e0, gamma, amplitude, background = theta
    gamma = max(gamma, 1e-12)
    w = 0.5 * gamma
    d = energy - e0
    q = d * d + w * w
    lorentz = (w * w) / q
    return background + amplitude * lorentz


def breit_wigner_jacobian(energy: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Jacobian of model output wrt [E0, gamma, amplitude, background]."""
    e0, gamma, amplitude, _ = theta
    gamma = max(gamma, 1e-12)
    w = 0.5 * gamma
    d = energy - e0
    q = d * d + w * w
    q2 = q * q

    lorentz = (w * w) / q

    dlorentz_de0 = 2.0 * d * (w * w) / q2
    dlorentz_dgamma = w * (d * d) / q2

    jac = np.empty((energy.size, 4), dtype=float)
    jac[:, 0] = amplitude * dlorentz_de0
    jac[:, 1] = amplitude * dlorentz_dgamma
    jac[:, 2] = lorentz
    jac[:, 3] = 1.0
    return jac


def estimate_initial_guess(energy: np.ndarray, sigma_obs: np.ndarray) -> np.ndarray:
    """Simple, robust initializer from peak and half-maximum width."""
    p10 = np.percentile(sigma_obs, 10)
    peak_idx = int(np.argmax(sigma_obs))
    peak_sigma = float(sigma_obs[peak_idx])
    e0_0 = float(energy[peak_idx])

    amp_0 = max(peak_sigma - p10, 1e-6)
    half_level = p10 + 0.5 * amp_0
    mask = sigma_obs >= half_level
    if mask.sum() >= 2:
        gamma_0 = float(energy[mask][-1] - energy[mask][0])
    else:
        gamma_0 = float((energy[-1] - energy[0]) / 8.0)

    gamma_0 = max(gamma_0, (energy[-1] - energy[0]) / 200.0)
    return np.array([e0_0, gamma_0, amp_0, p10], dtype=float)


def fit_resonance_lm(
    energy: np.ndarray,
    sigma_obs: np.ndarray,
    theta0: np.ndarray,
    config: FitConfig,
) -> FitResult:
    """Fit resonance with a hand-written LM loop."""
    n = energy.size
    p = 4

    theta = theta0.copy()
    lam = float(config.damping_init)

    accepted = 0
    converged = False
    iterations = 0

    for it in range(1, config.max_iter + 1):
        iterations = it
        model = breit_wigner_cross_section(energy, theta)
        residual = sigma_obs - model
        sse = float(residual @ residual)

        jac = breit_wigner_jacobian(energy, theta)
        h = jac.T @ jac
        g = jac.T @ residual

        # LM damping on diagonal terms for numerical stability.
        h_lm = h + lam * np.diag(np.diag(h) + 1e-12)
        try:
            delta = np.linalg.solve(h_lm, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(h_lm, g, rcond=None)[0]

        trial = theta + delta
        trial[1] = max(trial[1], config.min_gamma)
        trial[2] = max(trial[2], 0.0)

        trial_model = breit_wigner_cross_section(energy, trial)
        trial_residual = sigma_obs - trial_model
        trial_sse = float(trial_residual @ trial_residual)

        if trial_sse < sse:
            theta = trial
            accepted += 1
            lam = max(lam * config.damping_down, 1e-12)

            step_norm = float(np.linalg.norm(delta))
            theta_norm = float(np.linalg.norm(theta))
            if step_norm <= config.tol_step * (theta_norm + config.tol_step):
                converged = True
                break
        else:
            lam = min(lam * config.damping_up, 1e12)

    final_model = breit_wigner_cross_section(energy, theta)
    final_residual = sigma_obs - final_model
    sse = float(final_residual @ final_residual)
    mse = sse / n
    rmse = float(np.sqrt(mse))

    ss_tot = float(((sigma_obs - sigma_obs.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0.0 else 0.0

    jac_final = breit_wigner_jacobian(energy, theta)
    jtj = jac_final.T @ jac_final
    dof = max(n - p, 1)
    sigma2 = sse / dof
    cov = sigma2 * np.linalg.pinv(jtj)
    stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))

    tcrit = float(t.ppf(0.975, dof))
    ci_low = theta - tcrit * stderr
    ci_high = theta + tcrit * stderr

    return FitResult(
        params=ResonanceParams.from_array(theta),
        stderr=stderr,
        ci95_low=ci_low,
        ci95_high=ci_high,
        rmse=rmse,
        r2=r2,
        sse=sse,
        iterations=iterations,
        accepted_steps=accepted,
        converged=converged,
    )


def make_synthetic_dataset(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, ResonanceParams]:
    """Generate a synthetic resonant-scattering dataset."""
    rng = np.random.default_rng(seed)

    truth = ResonanceParams(e0=5.0, gamma=0.46, amplitude=12.5, background=0.9)
    energy = np.linspace(3.7, 6.3, 140)
    sigma_clean = breit_wigner_cross_section(energy, truth.to_array())

    noise_std = 0.22
    sigma_obs = sigma_clean + rng.normal(loc=0.0, scale=noise_std, size=energy.size)
    sigma_obs = np.clip(sigma_obs, 1e-8, None)

    return energy, sigma_obs, sigma_clean, truth


def main() -> None:
    energy, sigma_obs, _, truth = make_synthetic_dataset(seed=42)
    energy, sigma_obs = validate_inputs(energy, sigma_obs)

    theta0 = estimate_initial_guess(energy, sigma_obs)

    config = FitConfig(max_iter=180, damping_init=1e-2)
    fit = fit_resonance_lm(energy, sigma_obs, theta0, config)

    fitted = fit.params
    sigma_fit = breit_wigner_cross_section(energy, fitted.to_array())

    q_factor = fitted.e0 / fitted.gamma
    peak_sigma = fitted.background + fitted.amplitude

    summary = pd.DataFrame(
        {
            "parameter": ["E0", "Gamma", "Amplitude", "Background"],
            "true": truth.to_array(),
            "fitted": fitted.to_array(),
            "std_err": fit.stderr,
            "ci95_low": fit.ci95_low,
            "ci95_high": fit.ci95_high,
        }
    )
    summary["abs_error"] = (summary["fitted"] - summary["true"]).abs()

    preview = pd.DataFrame(
        {
            "E": energy,
            "sigma_obs": sigma_obs,
            "sigma_fit": sigma_fit,
            "residual": sigma_obs - sigma_fit,
        }
    )

    print("=== Resonant Scattering MVP (Breit-Wigner + Manual LM) ===")
    print(f"converged={fit.converged}, iterations={fit.iterations}, accepted_steps={fit.accepted_steps}")
    print(f"RMSE={fit.rmse:.6f}, R2={fit.r2:.6f}, SSE={fit.sse:.6f}")
    print(f"Resonance E0={fitted.e0:.6f}, Gamma(FWHM)={fitted.gamma:.6f}, Q=E0/Gamma={q_factor:.6f}")
    print(f"Estimated peak cross-section ~ background + amplitude = {peak_sigma:.6f}")

    print("\n--- Parameter Summary ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n--- Data Preview (first 10 rows) ---")
    print(preview.head(10).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
