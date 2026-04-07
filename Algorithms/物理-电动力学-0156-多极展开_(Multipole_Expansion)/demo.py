"""Minimal runnable MVP for Multipole Expansion.

The script compares:
1) Exact Coulomb potential from discrete charges
2) Multipole approximation truncated at monopole / dipole / quadrupole

Goal: verify that, in the far field, adding higher multipole orders
systematically improves approximation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


EPS0 = 8.854_187_812_8e-12
K_COULOMB = 1.0 / (4.0 * np.pi * EPS0)


@dataclass(frozen=True)
class MultipoleMoments:
    """Moments of a localized charge distribution around a chosen origin."""

    total_charge: float
    dipole: np.ndarray
    quadrupole: np.ndarray
    origin: np.ndarray
    source_radius: float


def _validate_charges_positions(charges: np.ndarray, positions: np.ndarray) -> None:
    """Basic shape and finiteness checks."""
    if charges.ndim != 1:
        raise ValueError("charges must be a 1D array")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be of shape (N, 3)")
    if charges.shape[0] != positions.shape[0]:
        raise ValueError("charges and positions must have the same length")
    if charges.shape[0] < 1:
        raise ValueError("at least one charge is required")
    if not np.isfinite(charges).all() or not np.isfinite(positions).all():
        raise ValueError("charges and positions must be finite")


def compute_multipole_moments(
    charges: np.ndarray,
    positions: np.ndarray,
    origin: np.ndarray | None = None,
) -> MultipoleMoments:
    """Compute monopole, dipole and traceless quadrupole moments.

    Potential convention:
    phi(r) = k * [ Q/r + (p·n)/r^2 + (1/2) * (n^T Q2 n)/r^3 + ... ]
    where Q2_ij = sum_a q_a (3 x_i x_j - |x|^2 delta_ij), x = r_a - origin.
    """
    charges = np.asarray(charges, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    _validate_charges_positions(charges, positions)

    if origin is None:
        origin_vec = np.zeros(3, dtype=np.float64)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
        if origin_vec.shape != (3,) or not np.isfinite(origin_vec).all():
            raise ValueError("origin must be a finite vector with shape (3,)")

    rel = positions - origin_vec[None, :]

    total_charge = float(np.sum(charges))
    dipole = np.sum(charges[:, None] * rel, axis=0)

    second_raw = np.einsum("a,ai,aj->ij", charges, rel, rel)
    trace_raw = float(np.sum(charges * np.sum(rel * rel, axis=1)))
    quadrupole = 3.0 * second_raw - trace_raw * np.eye(3, dtype=np.float64)

    source_radius = float(np.max(np.linalg.norm(rel, axis=1)))
    return MultipoleMoments(
        total_charge=total_charge,
        dipole=dipole,
        quadrupole=quadrupole,
        origin=origin_vec,
        source_radius=source_radius,
    )


def exact_potential(
    observation_points: np.ndarray,
    charges: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute exact Coulomb potential from point charges."""
    observation_points = np.asarray(observation_points, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if observation_points.ndim != 2 or observation_points.shape[1] != 3:
        raise ValueError("observation_points must be of shape (M, 3)")
    _validate_charges_positions(charges, positions)

    displacement = observation_points[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance < 1e-12):
        raise ValueError("observation point too close to a source point")

    return K_COULOMB * np.sum(charges[None, :] / distance, axis=1)


def multipole_potential(
    observation_points: np.ndarray,
    moments: MultipoleMoments,
    max_order: int,
) -> np.ndarray:
    """Evaluate multipole potential truncated at a given order.

    max_order = 0 -> monopole
    max_order = 1 -> monopole + dipole
    max_order = 2 -> monopole + dipole + quadrupole
    """
    if max_order not in (0, 1, 2):
        raise ValueError("max_order must be one of 0, 1, 2")

    observation_points = np.asarray(observation_points, dtype=np.float64)
    if observation_points.ndim != 2 or observation_points.shape[1] != 3:
        raise ValueError("observation_points must be of shape (M, 3)")

    rel_obs = observation_points - moments.origin[None, :]
    radius = np.linalg.norm(rel_obs, axis=1)
    if np.any(radius < 1e-12):
        raise ValueError("observation point too close to multipole origin")
    direction = rel_obs / radius[:, None]

    series = moments.total_charge / radius
    if max_order >= 1:
        dipole_projection = np.einsum("i,ni->n", moments.dipole, direction)
        series = series + dipole_projection / (radius**2)
    if max_order >= 2:
        quad_projection = np.einsum("ni,ij,nj->n", direction, moments.quadrupole, direction)
        series = series + 0.5 * quad_projection / (radius**3)

    return K_COULOMB * series


def sample_observation_points(
    radii: np.ndarray,
    points_per_shell: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points on multiple spherical shells."""
    if points_per_shell < 8:
        raise ValueError("points_per_shell must be >= 8")
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or np.any(radii <= 0.0):
        raise ValueError("radii must be a positive 1D array")

    rng = np.random.default_rng(seed)
    points_list = []
    shell_index = []
    for idx, radius in enumerate(radii):
        vec = rng.normal(size=(points_per_shell, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        points_list.append(float(radius) * vec)
        shell_index.append(np.full(points_per_shell, idx, dtype=np.int64))

    return np.vstack(points_list), np.concatenate(shell_index)


def error_metrics(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    """Relative L2 and max-abs relative metrics."""
    diff = estimate - reference
    l2_rel = float(np.linalg.norm(diff) / max(1e-12, np.linalg.norm(reference)))
    max_rel = float(np.max(np.abs(diff)) / max(1e-12, np.max(np.abs(reference))))
    return {"rel_l2": l2_rel, "rel_max": max_rel}


def run_demo() -> Dict[str, float]:
    """Run the multipole expansion demo and return scalar diagnostics."""
    charges = np.array([1.2e-9, -2.0e-9, 1.5e-9, -0.7e-9, 0.9e-9], dtype=np.float64)
    positions = np.array(
        [
            [0.05, 0.01, -0.03],
            [-0.04, 0.02, 0.06],
            [0.01, -0.07, -0.02],
            [-0.06, -0.03, 0.04],
            [0.03, 0.06, 0.01],
        ],
        dtype=np.float64,
    )

    moments = compute_multipole_moments(charges, positions, origin=np.zeros(3, dtype=np.float64))

    shell_radii = np.array([0.8, 1.1, 1.6], dtype=np.float64)
    obs_points, shell_ids = sample_observation_points(shell_radii, points_per_shell=260, seed=42)

    phi_exact = exact_potential(obs_points, charges, positions)
    phi_mono = multipole_potential(obs_points, moments, max_order=0)
    phi_dip = multipole_potential(obs_points, moments, max_order=1)
    phi_quad = multipole_potential(obs_points, moments, max_order=2)

    err_mono = error_metrics(phi_exact, phi_mono)
    err_dip = error_metrics(phi_exact, phi_dip)
    err_quad = error_metrics(phi_exact, phi_quad)

    report: Dict[str, float] = {
        "source_radius_m": moments.source_radius,
        "min_shell_radius_m": float(np.min(shell_radii)),
        "min_far_field_ratio_r_over_a": float(np.min(shell_radii) / max(1e-12, moments.source_radius)),
        "total_charge_C": moments.total_charge,
        "dipole_norm_Cm": float(np.linalg.norm(moments.dipole)),
        "quadrupole_fro_norm_Cm2": float(np.linalg.norm(moments.quadrupole)),
        "rel_l2_monopole": err_mono["rel_l2"],
        "rel_l2_dipole": err_dip["rel_l2"],
        "rel_l2_quadrupole": err_quad["rel_l2"],
        "rel_max_monopole": err_mono["rel_max"],
        "rel_max_dipole": err_dip["rel_max"],
        "rel_max_quadrupole": err_quad["rel_max"],
    }

    # Shell-wise diagnostics to show far-field convergence trend.
    for idx, radius in enumerate(shell_radii):
        mask = shell_ids == idx
        shell_metric = error_metrics(phi_exact[mask], phi_quad[mask])
        report[f"rel_l2_quadrupole_shell_{idx}"] = shell_metric["rel_l2"]
        report[f"shell_radius_{idx}_m"] = float(radius)

    return report


def main() -> None:
    report = run_demo()

    print("=== Multipole Expansion MVP: point-charge potential ===")
    for key, value in report.items():
        print(f"{key:>34s}: {value:.6e}")

    # Core quality gates:
    # 1) Higher-order expansion should be strictly better in this far-field setup.
    # 2) Quadrupole truncation should be reasonably accurate.
    assert report["rel_l2_monopole"] > report["rel_l2_dipole"] > report["rel_l2_quadrupole"], (
        "Expected monotonic improvement from monopole -> dipole -> quadrupole"
    )
    assert report["rel_l2_quadrupole"] < 1.0e-3, "Quadrupole relative L2 error is too large"
    assert report["rel_max_quadrupole"] < 3.0e-3, "Quadrupole max relative error is too large"

    # On larger radius shells, quadrupole approximation should improve.
    assert report["rel_l2_quadrupole_shell_0"] > report["rel_l2_quadrupole_shell_1"] > report[
        "rel_l2_quadrupole_shell_2"
    ], "Expected smaller quadrupole error on farther shells"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
