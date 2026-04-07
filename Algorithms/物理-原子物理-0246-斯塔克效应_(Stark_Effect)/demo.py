"""Minimal runnable MVP for Stark effect (hydrogen atom).

This script demonstrates:
1) dipole matrix element <2s|z|2p,m=0> from explicit wavefunctions,
2) linear Stark splitting in the n=2 degenerate manifold,
3) quadratic Stark shift for 1s state,
4) cross-check of fitted linear slope using sklearn/scipy/(optional torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import eval_genlaguerre, factorial, lpmv
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


BOHR_RADIUS_AU = 1.0
N2_ENERGY_AU = -1.0 / 8.0
GROUND_ENERGY_AU = -0.5
GROUND_POLARIZABILITY_AU = 4.5  # alpha_1s = 9/2 in atomic units


@dataclass
class IntegrationGrid:
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray


def validate_quantum_numbers(n: int, l: int, m: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    if abs(m) > l:
        raise ValueError("m must satisfy |m| <= l")


def radial_wavefunction(n: int, l: int, r: np.ndarray, a0: float = BOHR_RADIUS_AU) -> np.ndarray:
    """Hydrogen normalized radial function R_{n,l}(r) in atomic units."""
    validate_quantum_numbers(n=n, l=l, m=0)

    rho = 2.0 * r / (float(n) * float(a0))
    k = n - l - 1
    alpha = 2 * l + 1

    prefactor = np.sqrt(
        (2.0 / (n * a0)) ** 3
        * float(factorial(k, exact=False))
        / (2.0 * n * float(factorial(n + l, exact=False)))
    )
    laguerre = eval_genlaguerre(k, alpha, rho)
    return prefactor * np.exp(-rho / 2.0) * np.power(rho, l) * laguerre


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Normalized spherical harmonic Y_l^m(theta, phi)."""
    if l < 0:
        raise ValueError("l must be >= 0")
    if abs(m) > l:
        raise ValueError("m must satisfy |m| <= l")

    if m < 0:
        mp = -m
        y_pos = spherical_harmonic(l=l, m=mp, theta=theta, phi=phi)
        return ((-1) ** mp) * np.conjugate(y_pos)

    x = np.cos(theta)
    plm = lpmv(m, l, x)
    norm = np.sqrt(
        (2.0 * l + 1.0)
        / (4.0 * np.pi)
        * float(factorial(l - m, exact=False))
        / float(factorial(l + m, exact=False))
    )
    return norm * plm * np.exp(1j * m * phi)


def build_integration_grid(
    r_max: float = 80.0,
    nr: int = 1200,
    ntheta: int = 200,
    nphi: int = 220,
) -> IntegrationGrid:
    """Build spherical grid used for dipole matrix integration."""
    r = np.linspace(1e-6, r_max, nr, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, ntheta, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, dtype=np.float64)
    return IntegrationGrid(r=r, theta=theta, phi=phi)


def dipole_matrix_element_2s_2p0(grid: IntegrationGrid) -> float:
    """Compute <2s,m=0| z |2p,m=0> in atomic units.

    The integral is factorized into radial and angular parts for efficiency.
    """
    r = grid.r
    theta = grid.theta
    phi = grid.phi

    r20 = radial_wavefunction(2, 0, r)
    r21 = radial_wavefunction(2, 1, r)

    radial_integrand = r20 * r21 * np.power(r, 3)
    radial_term = np.trapezoid(radial_integrand, r)

    theta2d = theta[:, None]
    phi2d = phi[None, :]
    y00 = spherical_harmonic(0, 0, theta2d, phi2d)
    y10 = spherical_harmonic(1, 0, theta2d, phi2d)

    angular_integrand = np.conjugate(y00) * np.cos(theta2d) * y10 * np.sin(theta2d)
    angular_phi = np.trapezoid(angular_integrand, phi, axis=1)
    angular_term = np.trapezoid(angular_phi, theta, axis=0)

    value = radial_term * angular_term
    return float(np.real(value))


def stark_hamiltonian_n2(field_au: float, dipole_au: float, e0_au: float = N2_ENERGY_AU) -> np.ndarray:
    """Hamiltonian in basis [|2s,0>, |2p,0>, |2p,1>, |2p,-1>]."""
    h = np.eye(4, dtype=np.float64) * e0_au
    coupling = dipole_au * field_au
    h[0, 1] = coupling
    h[1, 0] = coupling
    return h


def first_order_ground_shift(field_au: np.ndarray, alpha_au: float = GROUND_POLARIZABILITY_AU) -> np.ndarray:
    """Quadratic Stark shift for 1s state: DeltaE = -0.5 * alpha * F^2."""
    return -0.5 * alpha_au * np.square(field_au)


def fit_linear_abs_shift_sklearn(fields: np.ndarray, shifts_abs: np.ndarray) -> float:
    x = np.abs(fields).reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, shifts_abs)
    return float(model.coef_[0])


def fit_linear_abs_shift_curve_fit(fields: np.ndarray, shifts_abs: np.ndarray) -> float:
    def model(x: np.ndarray, beta: float) -> np.ndarray:
        return beta * x

    beta, _ = curve_fit(model, np.abs(fields), shifts_abs, p0=np.array([2.0], dtype=np.float64))
    return float(beta[0])


def fit_linear_abs_shift_torch_optional(
    fields: np.ndarray,
    shifts_abs: np.ndarray,
    steps: int = 2500,
    lr: float = 0.05,
) -> Optional[float]:
    if torch is None:
        return None

    # Scale features for stable gradients while keeping beta in physical units.
    x = torch.tensor(np.abs(fields) * 1.0e5, dtype=torch.float64)
    y = torch.tensor(shifts_abs * 1.0e5, dtype=torch.float64)
    beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = beta * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    return float(beta.detach().cpu().item())


def build_scan_table(fields_au: np.ndarray, dipole_au: float) -> pd.DataFrame:
    rows = []
    for f in fields_au:
        eig = np.linalg.eigvalsh(stark_hamiltonian_n2(field_au=float(f), dipole_au=dipole_au))
        shifts = eig - N2_ENERGY_AU
        rows.append(
            {
                "field_au": float(f),
                "e_n2_1_au": float(eig[0]),
                "e_n2_2_au": float(eig[1]),
                "e_n2_3_au": float(eig[2]),
                "e_n2_4_au": float(eig[3]),
                "shift_max_abs_au": float(np.max(np.abs(shifts))),
                "shift_mid1_au": float(eig[1] - N2_ENERGY_AU),
                "shift_mid2_au": float(eig[2] - N2_ENERGY_AU),
                "delta_e1s_au": float(first_order_ground_shift(np.array([f]))[0]),
                "e1s_total_au": float(GROUND_ENERGY_AU + first_order_ground_shift(np.array([f]))[0]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    grid = build_integration_grid()
    dipole = dipole_matrix_element_2s_2p0(grid)
    dipole_abs = abs(dipole)

    fields = np.linspace(-5.0e-5, 5.0e-5, 51, dtype=np.float64)
    table = build_scan_table(fields_au=fields, dipole_au=dipole)

    slope_sk = fit_linear_abs_shift_sklearn(fields=fields, shifts_abs=table["shift_max_abs_au"].to_numpy())
    slope_cf = fit_linear_abs_shift_curve_fit(fields=fields, shifts_abs=table["shift_max_abs_au"].to_numpy())
    slope_torch = fit_linear_abs_shift_torch_optional(fields=fields, shifts_abs=table["shift_max_abs_au"].to_numpy())

    x_quad = np.square(fields).reshape(-1, 1)
    y_quad = -table["delta_e1s_au"].to_numpy()
    quad_model = LinearRegression(fit_intercept=False)
    quad_model.fit(x_quad, y_quad)
    quad_slope = float(quad_model.coef_[0])
    quad_expected = 0.5 * GROUND_POLARIZABILITY_AU

    max_mid_shift = float(np.max(np.abs(table[["shift_mid1_au", "shift_mid2_au"]].to_numpy())))

    print("Stark effect MVP (hydrogen atom, atomic units)")
    print(
        "grid="
        f"(nr={grid.r.size}, ntheta={grid.theta.size}, nphi={grid.phi.size})"
    )
    print(f"dipole_<2s|z|2p0>={dipole:.8f} a0")
    print(f"|dipole|={dipole_abs:.8f} a0")
    print(f"slope_sklearn={slope_sk:.8f}")
    print(f"slope_curve_fit={slope_cf:.8f}")
    if slope_torch is None:
        print("slope_torch=unavailable")
    else:
        print(f"slope_torch={slope_torch:.8f}")
    print(f"max_mid_state_shift={max_mid_shift:.3e} Ha")
    print(f"quadratic_slope_fit={quad_slope:.8f}")
    print(f"quadratic_slope_expected={quad_expected:.8f}")
    print("scan_table_preview=")
    print(table.head(8).to_string(index=False))

    # Main physical checks.
    assert abs(dipole_abs - 3.0) < 5.0e-2, f"Dipole matrix element mismatch: |d|={dipole_abs}"
    assert abs(slope_sk - dipole_abs) < 1.0e-8, "sklearn slope mismatch"
    assert abs(slope_cf - dipole_abs) < 1.0e-8, "curve_fit slope mismatch"
    if slope_torch is not None:
        assert abs(slope_torch - dipole_abs) < 1.0e-3, "torch slope mismatch"

    # m=+/-1 first-order shifts should stay zero in this MVP subspace.
    assert max_mid_shift < 1.0e-12, f"middle states should be unshifted at first order: {max_mid_shift}"

    # Ground state should follow quadratic shift with slope alpha/2.
    assert abs(quad_slope - quad_expected) < 1.0e-12, "quadratic Stark slope mismatch"

    print("All checks passed.")


if __name__ == "__main__":
    main()
