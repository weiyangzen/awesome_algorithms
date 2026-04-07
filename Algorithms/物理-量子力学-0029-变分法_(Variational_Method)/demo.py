"""Minimal runnable MVP for the Variational Method (PHYS-0029)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar


EPS = 1e-12


@dataclass
class VariationalRecord:
    g: float
    alpha_opt: float
    e_var: float
    e_ref: float
    gap: float
    grad_at_opt: float
    stationarity_residual: float
    norm_error: float
    bound_ok: bool


def gaussian_energy(alpha: float, g: float) -> float:
    """E(alpha) for Gaussian trial wavefunction in anharmonic oscillator."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return 0.25 * alpha + 0.25 / alpha + 0.75 * g / (alpha**2)


def gaussian_energy_grad(alpha: float, g: float) -> float:
    """dE/dalpha for optimization diagnostics."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return 0.25 - 0.25 / (alpha**2) - 1.5 * g / (alpha**3)


def stationarity_polynomial_residual(alpha: float, g: float) -> float:
    """Residual of alpha^3 - alpha - 6g = 0 derived from dE/dalpha = 0."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return alpha**3 - alpha - 6.0 * g


def optimize_variational_parameter(g: float, alpha_bounds: tuple[float, float] = (1e-4, 10.0)) -> tuple[float, float]:
    """Find alpha* that minimizes E(alpha)."""
    if g < 0.0:
        raise ValueError("this MVP assumes g >= 0")

    result = minimize_scalar(
        lambda a: gaussian_energy(float(a), g),
        method="bounded",
        bounds=alpha_bounds,
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"variational optimization failed for g={g}: {result.message}")

    alpha_opt = float(result.x)
    e_var = float(result.fun)
    return alpha_opt, e_var


def gaussian_wavefunction(x: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized Gaussian trial wavefunction on grid."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    prefactor = (alpha / np.pi) ** 0.25
    return prefactor * np.exp(-0.5 * alpha * x**2)


def normalization_error(alpha: float, xmax: float = 8.0, num_points: int = 20001) -> float:
    """Numerical check of ∫|psi|^2 dx = 1."""
    x = np.linspace(-xmax, xmax, num_points)
    psi = gaussian_wavefunction(x, alpha)
    norm = np.trapezoid(np.abs(psi) ** 2, x)
    return float(abs(norm - 1.0))


def finite_difference_ground_energy(g: float, xmax: float = 8.0, num_points: int = 2401) -> float:
    """Reference ground energy from finite-difference Schrödinger discretization."""
    if num_points < 5:
        raise ValueError("num_points must be >= 5")
    if g < 0.0:
        raise ValueError("this MVP assumes g >= 0")

    x = np.linspace(-xmax, xmax, num_points)
    dx = float(x[1] - x[0])

    x_inner = x[1:-1]
    potential = 0.5 * x_inner**2 + g * x_inner**4

    # H = -0.5*d2/dx2 + V(x), central difference for d2/dx2.
    main_diag = (1.0 / dx**2) + potential
    off_diag = np.full(main_diag.size - 1, -0.5 / dx**2, dtype=float)

    e0 = eigh_tridiagonal(
        main_diag,
        off_diag,
        eigvals_only=True,
        select="i",
        select_range=(0, 0),
        check_finite=False,
    )[0]
    return float(e0)


def analyze_single_coupling(g: float, bound_tol: float = 2e-3) -> VariationalRecord:
    """Run full variational workflow for one coupling constant g."""
    alpha_opt, e_var = optimize_variational_parameter(g)
    grad = gaussian_energy_grad(alpha_opt, g)
    residual = stationarity_polynomial_residual(alpha_opt, g)
    norm_err = normalization_error(alpha_opt)

    e_ref = finite_difference_ground_energy(g)
    gap = e_var - e_ref
    bound_ok = bool(gap >= -bound_tol)

    return VariationalRecord(
        g=g,
        alpha_opt=alpha_opt,
        e_var=e_var,
        e_ref=e_ref,
        gap=gap,
        grad_at_opt=grad,
        stationarity_residual=residual,
        norm_error=norm_err,
        bound_ok=bound_ok,
    )


def main() -> None:
    g_list = np.array([0.0, 0.1, 0.5, 1.0, 2.0], dtype=float)

    records = [analyze_single_coupling(float(g)) for g in g_list]

    table = pd.DataFrame(
        {
            "g": [r.g for r in records],
            "alpha_opt": [r.alpha_opt for r in records],
            "E_var": [r.e_var for r in records],
            "E_ref(fd)": [r.e_ref for r in records],
            "gap(E_var-E_ref)": [r.gap for r in records],
            "dE/dalpha@opt": [r.grad_at_opt for r in records],
            "stationarity_residual": [r.stationarity_residual for r in records],
            "norm_error": [r.norm_error for r in records],
            "bound_ok": [r.bound_ok for r in records],
        }
    )

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    max_abs_grad = float(max(abs(r.grad_at_opt) for r in records))
    max_abs_residual = float(max(abs(r.stationarity_residual) for r in records))
    max_norm_error = float(max(r.norm_error for r in records))
    all_bounds_ok = bool(all(r.bound_ok for r in records))

    checks = {
        "all alpha_opt > 0": bool(all(r.alpha_opt > 0.0 for r in records)),
        "max |dE/dalpha| < 1e-8": max_abs_grad < 1e-8,
        "max |alpha^3-alpha-6g| < 1e-7": max_abs_residual < 1e-7,
        "max normalization error < 1e-10": max_norm_error < 1e-10,
        "all variational bounds hold": all_bounds_ok,
    }

    print("=== Variational Method MVP (PHYS-0029) ===")
    print("Model: H = -1/2 d^2/dx^2 + x^2/2 + g x^4")
    print("Trial: psi_alpha(x) = (alpha/pi)^(1/4) exp(-alpha x^2 / 2)")
    print("\nResult table:")
    print(table.to_string(index=False))

    print("\nDiagnostics summary:")
    print(f"max |dE/dalpha| = {max_abs_grad:.3e}")
    print(f"max |alpha^3-alpha-6g| = {max_abs_residual:.3e}")
    print(f"max normalization error = {max_norm_error:.3e}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
