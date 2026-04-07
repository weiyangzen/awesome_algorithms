"""Minimal runnable MVP for Method of Images in electrostatics.

Scenario:
- A point charge q is located at (0, a) above a grounded infinite conducting plane y=0.
- In region y>0, the potential can be constructed by replacing the conductor with
  an image charge -q at (0, -a).

Potential (for y >= 0):
    V(x, y) = (1 / (4*pi*epsilon_0)) * [ q/r_real - q/r_image ]
where
    r_real  = sqrt(x^2 + (y-a)^2)
    r_image = sqrt(x^2 + (y+a)^2)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import constants, integrate


@dataclass
class ImagesResult:
    """Container for MVP outputs and diagnostics."""

    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    q: float
    a: float
    boundary_max_abs: float
    laplace_residual_raw_inf: float
    laplace_residual_scaled_inf: float
    induced_charge_estimate: float
    induced_charge_rel_error: float
    force_exact_y: float
    force_fd_y: float
    force_rel_error: float


def validate_inputs(
    q: float,
    a: float,
    nx: int,
    ny: int,
    x_limit: float,
    y_max: float,
) -> None:
    """Validate main physical and numerical parameters."""
    if q == 0.0:
        raise ValueError("q must be non-zero for this demo.")
    if a <= 0.0:
        raise ValueError("a must be > 0.")
    if nx < 41 or ny < 41:
        raise ValueError("nx and ny must be >= 41 for stable diagnostics.")
    if x_limit <= 0.0 or y_max <= 0.0:
        raise ValueError("x_limit and y_max must be > 0.")
    if y_max <= a:
        raise ValueError("y_max must be greater than a to include useful field region.")


def method_of_images_potential(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
    a: float,
    epsilon_0: float,
) -> np.ndarray:
    """Compute potential field V(x,y) in y>=0 from real + image charges."""
    k = 1.0 / (4.0 * math.pi * epsilon_0)
    X, Y = np.meshgrid(x, y, indexing="xy")

    with np.errstate(divide="ignore", invalid="ignore"):
        r_real = np.sqrt(X * X + (Y - a) * (Y - a))
        r_image = np.sqrt(X * X + (Y + a) * (Y + a))
        V = k * q * (1.0 / r_real - 1.0 / r_image)

    return V


def discrete_laplacian_residual(
    potential: np.ndarray,
    hx: float,
    hy: float,
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    exclusion_radius: float,
) -> tuple[float, float]:
    """Compute raw and scaled discrete Laplacian residual away from singularity."""
    hxx = hx * hx
    hyy = hy * hy

    lap = (
        (potential[1:-1, 2:] - 2.0 * potential[1:-1, 1:-1] + potential[1:-1, :-2]) / hxx
        + (potential[2:, 1:-1] - 2.0 * potential[1:-1, 1:-1] + potential[:-2, 1:-1]) / hyy
    )

    X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing="xy")
    dist_to_real = np.sqrt(X * X + (Y - a) * (Y - a))
    mask = dist_to_real > exclusion_radius

    if not np.any(mask):
        raise ValueError("No valid points remain after singularity exclusion.")

    raw_inf = float(np.max(np.abs(lap[mask])))

    v_scale = float(np.max(np.abs(potential[np.isfinite(potential)])))
    h_char = max(hx, hy)
    scaled_inf = raw_inf * (h_char * h_char) / max(v_scale, 1.0e-12)
    return raw_inf, scaled_inf


def force_y_exact(q: float, a: float, epsilon_0: float) -> float:
    """Closed-form y-force on real charge from conductor via image equivalence."""
    return -(q * q) / (16.0 * math.pi * epsilon_0 * a * a)


def force_y_finite_difference(q: float, a: float, epsilon_0: float, delta: float) -> float:
    """Estimate y-force by finite-difference of image potential at charge location."""
    if delta <= 0.0 or delta >= 0.25 * a:
        raise ValueError("delta must satisfy 0 < delta < 0.25*a.")

    k = 1.0 / (4.0 * math.pi * epsilon_0)

    def phi_image(y_pos: float) -> float:
        # Potential from image charge -q at (0, -a), evaluated at x=0.
        return k * (-q) / (y_pos + a)

    phi_plus = phi_image(a + delta)
    phi_minus = phi_image(a - delta)
    ey_fd = -(phi_plus - phi_minus) / (2.0 * delta)
    return q * ey_fd


def induced_sigma(rho: np.ndarray, q: float, a: float) -> np.ndarray:
    """Surface charge density on grounded plane from image method (axisymmetric form)."""
    return -(q * a) / (2.0 * math.pi * np.power(rho * rho + a * a, 1.5))


def estimate_total_induced_charge(
    q: float,
    a: float,
    rho_max_factor: float = 800.0,
    n_rho: int = 50_001,
) -> tuple[float, float]:
    """Integrate sigma over plane to estimate total induced charge."""
    if rho_max_factor <= 1.0:
        raise ValueError("rho_max_factor must be > 1.")
    if n_rho < 1001:
        raise ValueError("n_rho must be >= 1001.")

    rho_max = rho_max_factor * a
    rho = np.linspace(0.0, rho_max, n_rho, dtype=float)
    sigma = induced_sigma(rho=rho, q=q, a=a)

    # dQ = sigma(rho) * 2*pi*rho dr
    integrand = sigma * (2.0 * math.pi * rho)
    q_ind = float(integrate.simpson(integrand, x=rho))
    rel_err = abs(q_ind + q) / abs(q)
    return q_ind, rel_err


def build_profile_table(result: ImagesResult) -> pd.DataFrame:
    """Create a compact profile table for terminal inspection."""
    y_targets = np.array([0.0, 0.5 * result.a, 0.9 * result.a, 2.0 * result.a], dtype=float)
    y_idx = [int(np.argmin(np.abs(result.y - yt))) for yt in y_targets]

    x_idx = np.linspace(0, len(result.x) - 1, 9, dtype=int)
    rows: list[dict[str, float]] = []
    for j, yt in zip(y_idx, y_targets):
        for i in x_idx:
            rows.append(
                {
                    "x": float(result.x[i]),
                    "y_target": float(yt),
                    "y_used": float(result.y[j]),
                    "V": float(result.potential[j, i]),
                }
            )
    return pd.DataFrame(rows)


def solve_method_of_images_mvp(
    q: float = 1.0e-9,
    a: float = 0.05,
    nx: int = 401,
    ny: int = 321,
    x_limit: float = 0.20,
    y_max: float = 0.25,
) -> ImagesResult:
    """Run full MVP pipeline for the grounded-plane image-charge problem."""
    validate_inputs(q=q, a=a, nx=nx, ny=ny, x_limit=x_limit, y_max=y_max)

    epsilon_0 = constants.epsilon_0
    x = np.linspace(-x_limit, x_limit, nx, dtype=float)
    y = np.linspace(0.0, y_max, ny, dtype=float)
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])

    potential = method_of_images_potential(x=x, y=y, q=q, a=a, epsilon_0=epsilon_0)

    boundary_max_abs = float(np.max(np.abs(potential[0, :])))

    raw_residual, scaled_residual = discrete_laplacian_residual(
        potential=potential,
        hx=hx,
        hy=hy,
        x=x,
        y=y,
        a=a,
        exclusion_radius=0.25 * a,
    )

    f_exact = force_y_exact(q=q, a=a, epsilon_0=epsilon_0)
    f_fd = force_y_finite_difference(q=q, a=a, epsilon_0=epsilon_0, delta=1.0e-4 * a)
    force_rel_error = abs(f_fd - f_exact) / abs(f_exact)

    q_ind, q_rel_error = estimate_total_induced_charge(q=q, a=a)

    return ImagesResult(
        x=x,
        y=y,
        potential=potential,
        q=q,
        a=a,
        boundary_max_abs=boundary_max_abs,
        laplace_residual_raw_inf=raw_residual,
        laplace_residual_scaled_inf=scaled_residual,
        induced_charge_estimate=q_ind,
        induced_charge_rel_error=q_rel_error,
        force_exact_y=f_exact,
        force_fd_y=f_fd,
        force_rel_error=force_rel_error,
    )


def run_checks(result: ImagesResult) -> None:
    """Fail fast when physical or numerical checks are not satisfied."""
    if result.boundary_max_abs > 1.0e-9:
        raise AssertionError(
            f"Boundary condition violation too high: {result.boundary_max_abs:.3e} V"
        )
    if result.laplace_residual_scaled_inf > 1.0e-3:
        raise AssertionError(
            "Scaled Laplace residual too high: "
            f"{result.laplace_residual_scaled_inf:.3e}"
        )
    if result.force_rel_error > 1.0e-8:
        raise AssertionError(f"Force relative error too high: {result.force_rel_error:.3e}")
    if result.induced_charge_rel_error > 1.5e-3:
        raise AssertionError(
            f"Induced-charge relative error too high: {result.induced_charge_rel_error:.3e}"
        )


def main() -> None:
    result = solve_method_of_images_mvp()
    run_checks(result)

    print("Method of Images MVP report")
    print(f"q (C)                             : {result.q:.6e}")
    print(f"a (m)                             : {result.a:.6e}")
    print(f"Grid size                          : {len(result.x)} x {len(result.y)}")
    print(f"Boundary max |V(x,0)| (V)          : {result.boundary_max_abs:.3e}")
    print(f"Laplace residual inf (raw)         : {result.laplace_residual_raw_inf:.3e}")
    print(f"Laplace residual inf (scaled)      : {result.laplace_residual_scaled_inf:.3e}")
    print(f"Force Fy exact (N)                 : {result.force_exact_y:.6e}")
    print(f"Force Fy finite-difference (N)     : {result.force_fd_y:.6e}")
    print(f"Force relative error               : {result.force_rel_error:.3e}")
    print(f"Induced total charge estimate (C)  : {result.induced_charge_estimate:.6e}")
    print(f"Induced charge relative error      : {result.induced_charge_rel_error:.3e}")

    profile = build_profile_table(result)
    print("\nPotential profile sample:")
    print(profile.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
