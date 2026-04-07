"""Dimensional Regularization MVP.

This demo focuses on the standard one-loop Euclidean scalar integral:

    I_d(Delta) = mu^(2*epsilon) * ∫ d^d k / (2π)^d * 1/(k^2 + Delta)^2
    with d = 4 - 2*epsilon

Closed form (for Re(epsilon) > 0):
    I_d(Delta) = mu^(2*epsilon) / (4π)^(d/2) * Γ(epsilon) * Delta^(-epsilon)

The script demonstrates:
1) exact formula evaluation;
2) Laurent expansion around epsilon -> 0;
3) MS and MS-bar subtraction to isolate finite parts;
4) a numerical radial-integral cross-check for d < 4.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special


def exact_one_loop_integral(epsilon: np.ndarray | float, delta: float, mu: float) -> np.ndarray:
    """Closed-form dimensional-regularized integral for alpha=2."""
    eps = np.asarray(epsilon, dtype=float)
    d = 4.0 - 2.0 * eps
    prefactor = np.power(mu, 2.0 * eps) / np.power(4.0 * np.pi, d / 2.0)
    return prefactor * special.gamma(eps) * np.power(delta, -eps)


def laurent_up_to_constant(epsilon: np.ndarray | float, delta: float, mu: float) -> np.ndarray:
    """Laurent expansion up to O(epsilon^0) around epsilon=0."""
    eps = np.asarray(epsilon, dtype=float)
    c = math.log((mu * mu) / delta) - np.euler_gamma + math.log(4.0 * np.pi)
    return (1.0 / (16.0 * np.pi * np.pi)) * ((1.0 / eps) + c)


def ms_counterterm(epsilon: np.ndarray | float) -> np.ndarray:
    """Minimal subtraction counterterm: (1/16π²) * (1/epsilon)."""
    eps = np.asarray(epsilon, dtype=float)
    return (1.0 / (16.0 * np.pi * np.pi)) * (1.0 / eps)


def msbar_counterterm(epsilon: np.ndarray | float) -> np.ndarray:
    """MS-bar counterterm: (1/16π²) * (1/epsilon - gamma_E + ln 4π)."""
    eps = np.asarray(epsilon, dtype=float)
    return (1.0 / (16.0 * np.pi * np.pi)) * (
        (1.0 / eps) - np.euler_gamma + math.log(4.0 * np.pi)
    )


def sphere_surface_area(d: float) -> float:
    """Surface area of the unit (d-1)-sphere."""
    return 2.0 * np.pi ** (d / 2.0) / special.gamma(d / 2.0)


def numeric_radial_integral(d: float, delta: float) -> float:
    """Numerically compute ∫ d^d k/(2π)^d * 1/(k²+Delta)^2 for d<4."""
    if d >= 4.0:
        raise ValueError("numeric_radial_integral requires d < 4 for convergence.")

    sd_minus_1 = sphere_surface_area(d)
    angular_prefactor = sd_minus_1 / ((2.0 * np.pi) ** d)

    def integrand(k: float) -> float:
        return (k ** (d - 1.0)) / ((k * k + delta) ** 2)

    value, abs_err = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=400)
    if not np.isfinite(value) or abs_err > 1e-7:
        raise RuntimeError(f"Unexpected integration issue: value={value}, abs_err={abs_err}")
    return angular_prefactor * value


def main() -> None:
    delta = 2.3
    mu = 1.7
    eps_grid = np.array([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3], dtype=float)

    exact_vals = exact_one_loop_integral(eps_grid, delta, mu)
    laurent_vals = laurent_up_to_constant(eps_grid, delta, mu)
    approx_err = np.abs(exact_vals - laurent_vals)

    print("=== Dimensional Regularization: one-loop scalar integral ===")
    print(f"delta = {delta:.6f}, mu = {mu:.6f}")
    print("epsilon     exact               laurent(O(eps^0))     abs_error")
    for eps, exact_v, laurent_v, err_v in zip(eps_grid, exact_vals, laurent_vals, approx_err):
        print(f"{eps:8.4g}  {exact_v: .12e}  {laurent_v: .12e}  {err_v: .3e}")

    # Since truncation error is O(epsilon), |exact - laurent| / epsilon should stay bounded.
    scaled_err = approx_err / eps_grid
    print(f"\nmax(|exact-laurent|/epsilon) = {np.max(scaled_err):.6e}")
    assert np.max(scaled_err) < 3.0e-2, "Laurent expansion error is larger than expected O(epsilon)."

    # MS and MS-bar finite-part checks at tiny epsilon.
    eps_tiny = 1.0e-6
    exact_tiny = float(exact_one_loop_integral(eps_tiny, delta, mu))
    i_ms = exact_tiny - float(ms_counterterm(eps_tiny))
    i_msbar = exact_tiny - float(msbar_counterterm(eps_tiny))

    expected_ms_finite = (1.0 / (16.0 * np.pi * np.pi)) * (
        math.log((mu * mu) / delta) - np.euler_gamma + math.log(4.0 * np.pi)
    )
    expected_msbar_finite = (1.0 / (16.0 * np.pi * np.pi)) * math.log((mu * mu) / delta)

    print("\nMS finite part    :", f"{i_ms:.12e}")
    print("MS expected finite:", f"{expected_ms_finite:.12e}")
    print("MS-bar finite part    :", f"{i_msbar:.12e}")
    print("MS-bar expected finite:", f"{expected_msbar_finite:.12e}")

    assert abs(i_ms - expected_ms_finite) < 5.0e-8, "MS finite part check failed."
    assert abs(i_msbar - expected_msbar_finite) < 5.0e-8, "MS-bar finite part check failed."

    # Numerical cross-check for d<4 using radial integration.
    d_test = 3.2
    eps_test = (4.0 - d_test) / 2.0
    exact_d_test = float(exact_one_loop_integral(eps_test, delta, mu=1.0))  # set mu=1 to match integral dimension
    numeric_d_test = numeric_radial_integral(d_test, delta)
    rel_err = abs(numeric_d_test - exact_d_test) / abs(exact_d_test)

    print(f"\nCross-check at d={d_test:.1f} (epsilon={eps_test:.3f}, mu=1):")
    print("exact closed-form :", f"{exact_d_test:.12e}")
    print("numeric radial int:", f"{numeric_d_test:.12e}")
    print("relative error     :", f"{rel_err:.3e}")
    assert rel_err < 2.0e-6, "Numerical radial integral check failed."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
