"""Transition radiation MVP (normal incidence, ideal conductor limit).

Model summary:
- A charged particle with Lorentz factor gamma crosses a planar interface at normal incidence.
- In the vacuum -> perfect-conductor limit, backward transition radiation follows
  the Ginzburg-Frank spectral-angular density:

    d^2W/(dω dΩ) = (z^2 α ħ / π^2) * [β^2 sin^2θ / (1 - β^2 cos^2θ)^2]

- Photon yield density is obtained by dividing by ħω:

    d^2N/(dω dΩ) = (z^2 α / (π^2 ω)) * [β^2 sin^2θ / (1 - β^2 cos^2θ)^2]

This script integrates the distributions over angle and frequency band, then checks
consistency against separated analytic structure (Δω and ln(ωmax/ωmin)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical constants (SI)
ALPHA = 7.297_352_569_3e-3  # fine-structure constant
HBAR = 1.054_571_817e-34  # reduced Planck constant (J*s)


@dataclass(frozen=True)
class TRConfig:
    """Configuration for transition-radiation MVP."""

    gamma: float = 30.0  # Lorentz factor γ
    charge_state_z: int = 1  # particle charge in units of +e
    omega_min_rad_s: float = 1.0e14  # integration lower bound for angular frequency
    omega_max_rad_s: float = 2.0e16  # integration upper bound for angular frequency
    theta_max_rad: float = np.pi / 2.0  # backward hemisphere limit (normal incidence)
    n_theta: int = 4000  # angular discretization
    n_omega: int = 2000  # frequency discretization


def beta_from_gamma(gamma: float) -> float:
    """Compute beta = v/c from gamma."""
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for a moving relativistic particle.")
    return float(np.sqrt(1.0 - 1.0 / (gamma * gamma)))


def angular_kernel(theta: np.ndarray, beta: float) -> np.ndarray:
    """Return K(theta)=β²sin²θ/(1-β²cos²θ)²."""
    numerator = (beta * beta) * np.sin(theta) ** 2
    denominator = (1.0 - (beta * beta) * np.cos(theta) ** 2) ** 2
    return numerator / denominator


def d2w_domega_domega(theta: np.ndarray, beta: float, z: int) -> np.ndarray:
    """Spectral-angular energy density d²W/(dω dΩ) in J*s/sr."""
    prefactor = (z * z) * ALPHA * HBAR / (np.pi * np.pi)
    return prefactor * angular_kernel(theta, beta)


def d2n_domega_domega(theta: np.ndarray, omega: np.ndarray, beta: float, z: int) -> np.ndarray:
    """Spectral-angular photon yield density d²N/(dω dΩ)."""
    prefactor = (z * z) * ALPHA / (np.pi * np.pi)
    k_theta = angular_kernel(theta, beta)
    return prefactor * k_theta[None, :] / omega[:, None]


def integrate_over_theta(f_theta: np.ndarray, theta: np.ndarray) -> float:
    """Integrate azimuth-symmetric quantity over solid angle up to theta_max.

    Input f_theta should represent density per steradian as function of theta.
    Integration uses dΩ = 2π sinθ dθ.
    """
    integrand = f_theta * (2.0 * np.pi * np.sin(theta))
    return float(np.trapezoid(integrand, theta))


def run_mvp(cfg: TRConfig) -> dict[str, float | np.ndarray]:
    """Run deterministic transition-radiation calculations."""
    if cfg.omega_min_rad_s <= 0.0 or cfg.omega_max_rad_s <= cfg.omega_min_rad_s:
        raise ValueError("Require 0 < omega_min < omega_max.")
    if cfg.n_theta < 100 or cfg.n_omega < 100:
        raise ValueError("Use at least 100 points for theta and omega grids.")

    beta = beta_from_gamma(cfg.gamma)
    theta = np.linspace(0.0, cfg.theta_max_rad, cfg.n_theta)
    # Geometric frequency grid improves accuracy for 1/omega photon-yield integration.
    omega = np.geomspace(cfg.omega_min_rad_s, cfg.omega_max_rad_s, cfg.n_omega)

    # 1) Energy integral W over (ω, Ω)
    d2w_theta = d2w_domega_domega(theta, beta=beta, z=cfg.charge_state_z)
    dW_domega_scalar = integrate_over_theta(d2w_theta, theta)

    dW_domega_grid = np.full_like(omega, dW_domega_scalar)
    W_band_2d = float(np.trapezoid(dW_domega_grid, omega))
    W_band_sep = dW_domega_scalar * (cfg.omega_max_rad_s - cfg.omega_min_rad_s)

    # 2) Photon-number integral N over (ω, Ω)
    d2n_grid = d2n_domega_domega(theta, omega, beta=beta, z=cfg.charge_state_z)
    dN_domega_grid = np.trapezoid(d2n_grid * (2.0 * np.pi * np.sin(theta))[None, :], theta, axis=1)
    N_band_2d = float(np.trapezoid(dN_domega_grid, omega))

    angle_prefactor_n = integrate_over_theta(
        ((cfg.charge_state_z * cfg.charge_state_z) * ALPHA / (np.pi * np.pi))
        * angular_kernel(theta, beta),
        theta,
    )
    N_band_sep = angle_prefactor_n * np.log(cfg.omega_max_rad_s / cfg.omega_min_rad_s)

    # 3) Characteristic angle check: θ_peak ~ 1/γ
    theta_peak = float(theta[np.argmax(angular_kernel(theta, beta))])
    theta_char = 1.0 / cfg.gamma

    np.testing.assert_allclose(W_band_2d, W_band_sep, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(N_band_2d, N_band_sep, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(theta_peak, theta_char, rtol=3e-2, atol=0.0)

    sample_deg = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    sample_theta = np.deg2rad(sample_deg)
    sample_d2w = d2w_domega_domega(sample_theta, beta=beta, z=cfg.charge_state_z)

    return {
        "beta": beta,
        "theta_grid": theta,
        "omega_grid": omega,
        "dW_domega": dW_domega_scalar,
        "W_band_2d": W_band_2d,
        "W_band_sep": W_band_sep,
        "N_band_2d": N_band_2d,
        "N_band_sep": N_band_sep,
        "theta_peak_rad": theta_peak,
        "theta_char_rad": theta_char,
        "sample_deg": sample_deg,
        "sample_d2w": sample_d2w,
    }


def main() -> None:
    cfg = TRConfig()
    out = run_mvp(cfg)

    print("Transition Radiation MVP")
    print(
        f"Config: gamma={cfg.gamma:.2f}, z={cfg.charge_state_z}, "
        f"omega=[{cfg.omega_min_rad_s:.3e}, {cfg.omega_max_rad_s:.3e}] rad/s"
    )
    print(f"Grid: n_theta={cfg.n_theta}, n_omega={cfg.n_omega}")
    print("--- Integrated observables ---")
    print(f"dW/domega (angle-integrated): {out['dW_domega']:.6e} J")
    print(f"W_band_2d                  : {out['W_band_2d']:.6e} J")
    print(f"W_band_sep                 : {out['W_band_sep']:.6e} J")
    print(
        f"RelErr(W)                  : "
        f"{abs(out['W_band_2d'] - out['W_band_sep']) / out['W_band_sep']:.3e}"
    )
    print(f"N_band_2d                  : {out['N_band_2d']:.6e}")
    print(f"N_band_sep                 : {out['N_band_sep']:.6e}")
    print(
        f"RelErr(N)                  : "
        f"{abs(out['N_band_2d'] - out['N_band_sep']) / out['N_band_sep']:.3e}"
    )
    print("--- Angular feature ---")
    print(f"theta_peak                 : {out['theta_peak_rad']:.6e} rad")
    print(f"1/gamma                    : {out['theta_char_rad']:.6e} rad")
    print(
        f"peak ratio theta_peak/(1/gamma): "
        f"{out['theta_peak_rad'] / out['theta_char_rad']:.4f}"
    )
    print("--- Samples: d^2W/(dω dΩ) [J*s/sr] ---")
    for deg, value in zip(out["sample_deg"], out["sample_d2w"]):
        print(f"theta={deg:5.1f} deg -> {value:.6e}")


if __name__ == "__main__":
    main()
