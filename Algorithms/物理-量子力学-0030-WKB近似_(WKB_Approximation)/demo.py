"""WKB approximation MVP for a 1D quartic oscillator.

This script compares semiclassical (WKB / Bohr-Sommerfeld) energy levels
against numerical eigenvalues from a finite-difference Schr\u00f6dinger solver.

Units: hbar = 1, m = 1, lambda = 1.
Potential: V(x) = (lambda/4) * x^4.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, linalg, optimize


def quartic_potential(x: np.ndarray | float, lam: float = 1.0) -> np.ndarray | float:
    """Return V(x) = (lam/4) * x^4."""
    return 0.25 * lam * np.asarray(x) ** 4


def wkb_action(E: float, lam: float = 1.0, mass: float = 1.0) -> float:
    """Compute I(E) = integral_{x1}^{x2} p(x) dx for the quartic potential.

    Here p(x) = sqrt(2m(E - V(x))). Turning points satisfy E = V(x).
    """
    if E <= 0.0:
        return 0.0

    x_turn = (4.0 * E / lam) ** 0.25
    prefactor = math.sqrt(2.0 * mass)

    def integrand(x: float) -> float:
        remaining = E - 0.25 * lam * x**4
        if remaining <= 0.0:
            return 0.0
        return prefactor * math.sqrt(remaining)

    integral_value, _ = integrate.quad(
        integrand,
        -x_turn,
        x_turn,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )
    return float(integral_value)


def wkb_energy_level(
    n: int,
    lam: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> float:
    """Solve Bohr-Sommerfeld condition: I(E) = pi*hbar*(n + 1/2)."""
    if n < 0:
        raise ValueError("n must be non-negative.")

    target = math.pi * hbar * (n + 0.5)

    def residual(E: float) -> float:
        return wkb_action(E, lam=lam, mass=mass) - target

    lo = 1e-12
    hi = 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket WKB root.")

    root = optimize.brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=300)
    return float(root)


def numerical_levels(
    n_levels: int,
    x_max: float = 8.0,
    n_grid: int = 2200,
    lam: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Compute low-lying energies by finite-difference discretization.

    Dirichlet boundary condition: psi(-x_max)=psi(x_max)=0.
    """
    if n_levels <= 0:
        raise ValueError("n_levels must be positive.")
    if n_grid < n_levels + 10:
        raise ValueError("n_grid is too small for stable low-energy extraction.")

    x_full = np.linspace(-x_max, x_max, n_grid + 2)
    x_inner = x_full[1:-1]
    dx = x_full[1] - x_full[0]

    v = quartic_potential(x_inner, lam=lam)

    # H = -(hbar^2/2m) d2/dx2 + V(x), with central finite difference d2.
    kinetic_diag = hbar**2 / (mass * dx**2)
    kinetic_off = -hbar**2 / (2.0 * mass * dx**2)

    diag = kinetic_diag + v
    offdiag = np.full(n_grid - 1, kinetic_off)

    evals = linalg.eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(0, n_levels - 1),
        check_finite=False,
    )[0]
    return np.asarray(evals, dtype=float)


def main() -> None:
    n_levels = 6
    lam = 1.0
    mass = 1.0
    hbar = 1.0

    numeric = numerical_levels(n_levels=n_levels, lam=lam, mass=mass, hbar=hbar)
    wkb = np.array([wkb_energy_level(n, lam=lam, mass=mass, hbar=hbar) for n in range(n_levels)])

    abs_err = np.abs(wkb - numeric)
    rel_err = abs_err / numeric

    if not (np.all(np.diff(numeric) > 0.0) and np.all(np.diff(wkb) > 0.0)):
        raise RuntimeError("Energy levels are not strictly increasing; check solver settings.")

    print("WKB Approximation Demo: 1D Quartic Oscillator")
    print("Potential V(x) = x^4 / 4, units: hbar=1, m=1")
    print("-" * 68)
    print(f"{'n':>2} {'E_numeric':>14} {'E_WKB':>14} {'abs_err':>12} {'rel_err(%)':>12}")
    print("-" * 68)
    for n in range(n_levels):
        print(
            f"{n:2d} {numeric[n]:14.8f} {wkb[n]:14.8f} {abs_err[n]:12.8f} {100.0 * rel_err[n]:12.4f}"
        )
    print("-" * 68)
    print(f"Mean relative error (%): {100.0 * rel_err.mean():.4f}")
    print(f"Max  relative error (%): {100.0 * rel_err.max():.4f}")


if __name__ == "__main__":
    main()
