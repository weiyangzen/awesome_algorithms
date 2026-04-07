"""Robertson-Walker (FLRW) metric MVP demo.

This script computes:
1) Metric tensor components in spherical comoving coordinates.
2) Basic background-expansion quantities E(z), H(z).
3) Cosmological distances on a small redshift grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad

# Speed of light in km/s, convenient for Mpc distances with H0 in km/s/Mpc.
C_KM_S = 299792.458


def scale_factor_from_redshift(z: float) -> float:
    """Return a(z)=1/(1+z) for z >= 0."""
    if z < 0:
        raise ValueError(f"Redshift must satisfy z >= 0, got z={z}.")
    return 1.0 / (1.0 + z)


def robertson_walker_metric(a_t: float, r: float, theta: float, k: int, c: float = 1.0) -> np.ndarray:
    """Build the diagonal RW metric tensor g_{mu nu} in coordinates (t, r, theta, phi).

    ds^2 = -c^2 dt^2 + a(t)^2 [dr^2/(1-k r^2) + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2]
    with k in {-1, 0, +1}.
    """
    if k not in (-1, 0, 1):
        raise ValueError(f"Curvature index k must be -1, 0, or 1, got {k}.")
    if a_t <= 0:
        raise ValueError(f"Scale factor must be positive, got a_t={a_t}.")
    if r < 0:
        raise ValueError(f"Comoving radius must satisfy r >= 0, got r={r}.")
    if not (0.0 <= theta <= math.pi):
        raise ValueError(f"Theta must satisfy 0 <= theta <= pi, got theta={theta}.")

    denom = 1.0 - k * r * r
    if denom <= 0.0:
        raise ValueError(
            "Coordinate singularity reached: 1 - k r^2 <= 0. "
            f"(k={k}, r={r}, 1-k r^2={denom})."
        )

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -(c**2)
    g[1, 1] = (a_t * a_t) / denom
    g[2, 2] = a_t * a_t * r * r
    g[3, 3] = a_t * a_t * r * r * (math.sin(theta) ** 2)
    return g


@dataclass(frozen=True)
class FLRWCosmology:
    """Minimal FLRW background model with matter + curvature + cosmological constant."""

    h0: float = 70.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7

    @property
    def omega_k(self) -> float:
        return 1.0 - self.omega_m - self.omega_lambda

    def e_z(self, z: float) -> float:
        """Dimensionless expansion function E(z)=H(z)/H0."""
        if z < 0:
            raise ValueError(f"Redshift must satisfy z >= 0, got z={z}.")
        term = (
            self.omega_m * (1.0 + z) ** 3
            + self.omega_k * (1.0 + z) ** 2
            + self.omega_lambda
        )
        if term <= 0.0:
            raise ValueError(f"E(z)^2 became non-positive at z={z}: {term}.")
        return math.sqrt(term)

    def hubble(self, z: float) -> float:
        """Hubble parameter in km/s/Mpc."""
        return self.h0 * self.e_z(z)

    def comoving_distance(self, z: float) -> float:
        """Line-of-sight comoving distance D_C(z) in Mpc."""
        if z < 0:
            raise ValueError(f"Redshift must satisfy z >= 0, got z={z}.")
        integral, _ = quad(
            lambda zp: 1.0 / self.e_z(zp),
            0.0,
            z,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=200,
        )
        d_h = C_KM_S / self.h0
        return d_h * integral

    def transverse_comoving_distance(self, z: float) -> float:
        """Transverse comoving distance D_M(z) in Mpc."""
        d_c = self.comoving_distance(z)
        d_h = C_KM_S / self.h0
        omega_k = self.omega_k

        if abs(omega_k) < 1e-12:
            return d_c

        sqrt_ok = math.sqrt(abs(omega_k))
        arg = sqrt_ok * d_c / d_h
        if omega_k > 0:
            return (d_h / sqrt_ok) * math.sinh(arg)
        return (d_h / sqrt_ok) * math.sin(arg)

    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance D_L(z) in Mpc."""
        return (1.0 + z) * self.transverse_comoving_distance(z)

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance D_A(z) in Mpc."""
        return self.transverse_comoving_distance(z) / (1.0 + z)


def build_distance_table(cosmology: FLRWCosmology, z_values: np.ndarray) -> pd.DataFrame:
    """Assemble a compact result table for a redshift grid."""
    rows = []
    for z in z_values:
        zf = float(z)
        rows.append(
            {
                "z": zf,
                "a(z)": scale_factor_from_redshift(zf),
                "E(z)": cosmology.e_z(zf),
                "H(z) [km/s/Mpc]": cosmology.hubble(zf),
                "D_C [Mpc]": cosmology.comoving_distance(zf),
                "D_M [Mpc]": cosmology.transverse_comoving_distance(zf),
                "D_L [Mpc]": cosmology.luminosity_distance(zf),
                "D_A [Mpc]": cosmology.angular_diameter_distance(zf),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cosmology = FLRWCosmology(h0=70.0, omega_m=0.3, omega_lambda=0.7)

    z_grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=float)
    table = build_distance_table(cosmology, z_grid)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

    print("=== FLRW Background Quantities ===")
    print(f"Parameters: H0={cosmology.h0}, Ωm={cosmology.omega_m}, ΩΛ={cosmology.omega_lambda}, Ωk={cosmology.omega_k}")
    print(table.to_string(index=False))
    print()

    z_metric = 1.0
    a_t = scale_factor_from_redshift(z_metric)
    r = 0.3
    theta = math.pi / 3.0

    print("=== Robertson-Walker Metric Tensors at z=1 (a=0.5), r=0.3, theta=pi/3 ===")
    for k in (-1, 0, 1):
        g = robertson_walker_metric(a_t=a_t, r=r, theta=theta, k=k, c=1.0)
        print(f"k={k}:")
        print(g)


if __name__ == "__main__":
    main()
