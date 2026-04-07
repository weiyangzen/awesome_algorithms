"""Minimal runnable MVP for Electric Dipole Field (exact vs far-field approximation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON_0 = 8.854_187_8128e-12  # Vacuum permittivity (F/m).
COULOMB_K = 1.0 / (4.0 * np.pi * EPSILON_0)


@dataclass
class DipoleFieldResult:
    """Container for shell-wise approximation diagnostics."""

    charge_c: float
    separation_m: float
    dipole_moment_c_m: np.ndarray
    radii_m: np.ndarray
    r_over_d: np.ndarray
    mean_relative_error: np.ndarray
    p95_relative_error: np.ndarray
    max_relative_error: np.ndarray
    loglog_slope: float
    directions_per_shell: int


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate quasi-uniform unit vectors on a sphere (deterministic)."""
    if n_points < 8:
        raise ValueError("n_points must be >= 8.")

    i = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    dirs = np.stack([x, y, z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def exact_dipole_field(points: np.ndarray, charge_c: float, separation_m: float) -> np.ndarray:
    """Exact electric field from +q and -q separated along x-axis."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if charge_c <= 0.0:
        raise ValueError("charge_c must be positive.")
    if separation_m <= 0.0:
        raise ValueError("separation_m must be positive.")

    plus_pos = np.array([0.5 * separation_m, 0.0, 0.0], dtype=float)
    minus_pos = -plus_pos

    r_plus = points - plus_pos
    r_minus = points - minus_pos

    n_plus = np.linalg.norm(r_plus, axis=1, keepdims=True)
    n_minus = np.linalg.norm(r_minus, axis=1, keepdims=True)

    if np.any(n_plus <= 1.0e-12) or np.any(n_minus <= 1.0e-12):
        raise ValueError("Sampling point is too close to one of the charges.")

    field_plus = COULOMB_K * charge_c * r_plus / (n_plus**3)
    field_minus = COULOMB_K * (-charge_c) * r_minus / (n_minus**3)
    return field_plus + field_minus


def dipole_far_field(points: np.ndarray, dipole_moment: np.ndarray) -> np.ndarray:
    """Far-field dipole approximation: E = k*(3(p·r_hat)r_hat - p)/r^3."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if dipole_moment.shape != (3,):
        raise ValueError("dipole_moment must have shape (3,).")

    r = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(r <= 0.0):
        raise ValueError("points cannot include the origin.")

    r_hat = points / r
    p_dot = np.sum(r_hat * dipole_moment.reshape(1, 3), axis=1, keepdims=True)
    numer = 3.0 * p_dot * r_hat - dipole_moment.reshape(1, 3)
    return COULOMB_K * numer / (r**3)


def evaluate_shell_errors(
    charge_c: float,
    separation_m: float,
    radii_m: np.ndarray,
    directions_per_shell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute shell-wise relative errors between exact and dipole-approx fields."""
    if radii_m.ndim != 1 or radii_m.size < 4:
        raise ValueError("radii_m must be 1D with at least 4 radii.")
    if not np.all(np.isfinite(radii_m)):
        raise ValueError("radii_m must be finite.")
    if np.any(radii_m <= 0.0):
        raise ValueError("radii_m must be positive.")
    if not np.all(np.diff(radii_m) > 0.0):
        raise ValueError("radii_m must be strictly increasing.")

    directions = fibonacci_sphere(directions_per_shell)
    dipole_moment = np.array([charge_c * separation_m, 0.0, 0.0], dtype=float)

    mean_rel = np.empty_like(radii_m)
    p95_rel = np.empty_like(radii_m)
    max_rel = np.empty_like(radii_m)

    for i, radius in enumerate(radii_m):
        points = radius * directions
        e_exact = exact_dipole_field(points=points, charge_c=charge_c, separation_m=separation_m)
        e_approx = dipole_far_field(points=points, dipole_moment=dipole_moment)

        exact_norm = np.linalg.norm(e_exact, axis=1)
        rel_err = np.linalg.norm(e_approx - e_exact, axis=1) / (exact_norm + 1.0e-30)

        mean_rel[i] = float(np.mean(rel_err))
        p95_rel[i] = float(np.quantile(rel_err, 0.95))
        max_rel[i] = float(np.max(rel_err))

    r_over_d = radii_m / separation_m
    loglog_slope = float(np.polyfit(np.log(r_over_d), np.log(mean_rel), deg=1)[0])
    return r_over_d, mean_rel, p95_rel, max_rel, loglog_slope


def run_dipole_field_mvp(
    charge_c: float = 2.0e-9,
    separation_m: float = 0.04,
    radii_m: np.ndarray | None = None,
    directions_per_shell: int = 1200,
) -> DipoleFieldResult:
    """Run the MVP: exact field vs dipole far-field approximation."""
    if charge_c <= 0.0:
        raise ValueError("charge_c must be positive.")
    if separation_m <= 0.0:
        raise ValueError("separation_m must be positive.")

    if radii_m is None:
        radii_m = np.array([0.08, 0.10, 0.12, 0.16, 0.20, 0.30, 0.40, 0.60, 0.80], dtype=float)
    else:
        radii_m = np.asarray(radii_m, dtype=float)

    r_over_d, mean_rel, p95_rel, max_rel, loglog_slope = evaluate_shell_errors(
        charge_c=charge_c,
        separation_m=separation_m,
        radii_m=radii_m,
        directions_per_shell=directions_per_shell,
    )

    dipole_moment = np.array([charge_c * separation_m, 0.0, 0.0], dtype=float)
    return DipoleFieldResult(
        charge_c=float(charge_c),
        separation_m=float(separation_m),
        dipole_moment_c_m=dipole_moment,
        radii_m=radii_m,
        r_over_d=r_over_d,
        mean_relative_error=mean_rel,
        p95_relative_error=p95_rel,
        max_relative_error=max_rel,
        loglog_slope=loglog_slope,
        directions_per_shell=int(directions_per_shell),
    )


def run_checks(result: DipoleFieldResult) -> None:
    """Fail fast if approximation quality or trend is implausible."""
    if not np.all(np.isfinite(result.mean_relative_error)):
        raise AssertionError("mean_relative_error contains non-finite values.")
    if not np.all(np.diff(result.mean_relative_error) < 0.0):
        raise AssertionError("mean_relative_error should decrease as r/d increases.")

    first_err = float(result.mean_relative_error[0])
    last_err = float(result.mean_relative_error[-1])
    reduction_ratio = first_err / last_err

    if last_err >= 0.002:
        raise AssertionError(f"Far-field mean relative error too large: {last_err:.4e}")
    if reduction_ratio <= 20.0:
        raise AssertionError(f"Error reduction ratio unexpectedly small: {reduction_ratio:.3f}")
    if not (-2.4 <= result.loglog_slope <= -1.6):
        raise AssertionError(f"Unexpected log-log slope: {result.loglog_slope:.4f}")


def preview_table(result: DipoleFieldResult) -> pd.DataFrame:
    """Create a compact table for shell-wise diagnostics."""
    return pd.DataFrame(
        {
            "radius_m": result.radii_m,
            "r_over_d": result.r_over_d,
            "mean_rel_error": result.mean_relative_error,
            "p95_rel_error": result.p95_relative_error,
            "max_rel_error": result.max_relative_error,
        }
    )


def main() -> None:
    result = run_dipole_field_mvp()
    run_checks(result)

    table = preview_table(result)
    first_err = float(result.mean_relative_error[0])
    last_err = float(result.mean_relative_error[-1])
    reduction_ratio = first_err / last_err

    print("Electric Dipole Field MVP report")
    print(f"epsilon_0 (F/m)                 : {EPSILON_0:.10e}")
    print(f"Coulomb k (N m^2 / C^2)         : {COULOMB_K:.10e}")
    print(f"charge q (C)                    : {result.charge_c:.3e}")
    print(f"separation d (m)                : {result.separation_m:.3e}")
    print(f"dipole moment |p| (C m)         : {np.linalg.norm(result.dipole_moment_c_m):.3e}")
    print(f"directions per shell            : {result.directions_per_shell}")
    print(f"mean rel error @ min r/d        : {first_err:.4e}")
    print(f"mean rel error @ max r/d        : {last_err:.4e}")
    print(f"error reduction ratio           : {reduction_ratio:.3f}")
    print(f"log-log slope (mean_err vs r/d) : {result.loglog_slope:.4f}")

    print("\nShell-wise diagnostics:")
    print(table.to_string(index=False, float_format=lambda x: f"{x: .4e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
