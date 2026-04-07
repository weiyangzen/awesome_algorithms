"""Minimal runnable MVP for Coulomb's Law (PHYS-0012)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import constants
from sklearn.linear_model import LinearRegression

# Coulomb constant in vacuum: k = 1 / (4 * pi * epsilon_0)
K_VACUUM = 1.0 / (4.0 * np.pi * constants.epsilon_0)


def coulomb_force_magnitude(
    q1: np.ndarray | float,
    q2: np.ndarray | float,
    r: np.ndarray | float,
    relative_permittivity: float = 1.0,
) -> np.ndarray:
    """Compute |F| = k * |q1*q2| / (eps_r * r^2)."""
    if relative_permittivity <= 0:
        raise ValueError("relative_permittivity must be positive.")

    q1_arr = np.asarray(q1, dtype=float)
    q2_arr = np.asarray(q2, dtype=float)
    r_arr = np.asarray(r, dtype=float)

    if np.any(r_arr <= 0):
        raise ValueError("All distances r must be strictly positive.")

    return (K_VACUUM / relative_permittivity) * np.abs(q1_arr * q2_arr) / np.square(r_arr)


def coulomb_force_vector(
    q1: float,
    q2: float,
    r1: np.ndarray,
    r2: np.ndarray,
    relative_permittivity: float = 1.0,
) -> np.ndarray:
    """Compute force vector on charge q1 from q2 in 3D space."""
    if relative_permittivity <= 0:
        raise ValueError("relative_permittivity must be positive.")

    r1_arr = np.asarray(r1, dtype=float)
    r2_arr = np.asarray(r2, dtype=float)
    diff = r1_arr - r2_arr
    distance = np.linalg.norm(diff)

    if distance <= 0:
        raise ValueError("Charges cannot occupy the same position.")

    scale = (K_VACUUM / relative_permittivity) * (q1 * q2) / (distance**3)
    return scale * diff


def torch_force_magnitude(
    q1: np.ndarray,
    q2: np.ndarray,
    r: np.ndarray,
    relative_permittivity: float = 1.0,
) -> np.ndarray:
    """PyTorch implementation used only for cross-checking NumPy results."""
    q1_t = torch.tensor(np.asarray(q1, dtype=np.float64), dtype=torch.float64)
    q2_t = torch.tensor(np.asarray(q2, dtype=np.float64), dtype=torch.float64)
    r_t = torch.tensor(np.asarray(r, dtype=np.float64), dtype=torch.float64)

    if torch.any(r_t <= 0):
        raise ValueError("All distances r must be strictly positive.")

    force_t = (K_VACUUM / relative_permittivity) * torch.abs(q1_t * q2_t) / torch.square(r_t)
    return force_t.detach().cpu().numpy()


def inverse_square_fit(distances: np.ndarray, forces: np.ndarray) -> tuple[float, float, float]:
    """Fit log(F) = a * log(r) + b and return (a, b, R^2)."""
    X = np.log(distances).reshape(-1, 1)
    y = np.log(forces)

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, y))
    return slope, intercept, r2


def build_demo_table() -> pd.DataFrame:
    """Create a small deterministic batch demo table."""
    q1 = np.array([1.0e-6, 1.0e-6, 2.0e-6, -3.0e-6, -2.5e-6], dtype=float)
    q2 = np.array([2.0e-6, -2.0e-6, 1.0e-6, -1.0e-6, 4.0e-6], dtype=float)
    r = np.array([0.02, 0.03, 0.05, 0.04, 0.06], dtype=float)

    force_mag = coulomb_force_magnitude(q1, q2, r)
    interaction = np.where(q1 * q2 > 0, "repulsive", "attractive")

    return pd.DataFrame(
        {
            "q1_C": q1,
            "q2_C": q2,
            "r_m": r,
            "interaction": interaction,
            "|F|_N": force_mag,
        }
    )


def main() -> None:
    print("=== PHYS-0012 Coulomb's Law MVP ===")
    print(f"k (vacuum) = {K_VACUUM:.6e} N*m^2/C^2")

    demo_df = build_demo_table()
    print("\n[Batch force magnitudes]")
    print(demo_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # Inverse-square verification on synthetic noiseless data.
    distances = np.linspace(0.02, 0.20, 15)
    fixed_q1, fixed_q2 = 2.0e-6, 3.0e-6
    forces = coulomb_force_magnitude(fixed_q1, fixed_q2, distances)
    slope, intercept, r2 = inverse_square_fit(distances, forces)

    print("\n[Inverse-square law regression: log(F)=a*log(r)+b]")
    print(f"slope a = {slope:.6f} (ideal: -2)")
    print(f"intercept b = {intercept:.6f}")
    print(f"R^2 = {r2:.6f}")

    # Cross-check NumPy vs PyTorch outputs.
    np_force = coulomb_force_magnitude(
        demo_df["q1_C"].to_numpy(),
        demo_df["q2_C"].to_numpy(),
        demo_df["r_m"].to_numpy(),
    )
    torch_force = torch_force_magnitude(
        demo_df["q1_C"].to_numpy(),
        demo_df["q2_C"].to_numpy(),
        demo_df["r_m"].to_numpy(),
    )
    max_abs_diff = float(np.max(np.abs(np_force - torch_force)))
    print("\n[NumPy vs PyTorch consistency]")
    print(f"max |difference| = {max_abs_diff:.3e} N")

    # 3D vector force example.
    q1, q2 = 3.0e-6, -5.0e-6
    r1 = np.array([0.00, 0.00, 0.00], dtype=float)
    r2 = np.array([0.03, 0.04, 0.00], dtype=float)
    f_vec = coulomb_force_vector(q1, q2, r1, r2)
    f_mag = np.linalg.norm(f_vec)

    print("\n[3D vector force example: force on q1 due to q2]")
    print(f"r1 = {r1}, r2 = {r2}")
    print(f"F_vector (N) = {f_vec}")
    print(f"|F_vector| (N) = {f_mag:.6e}")


if __name__ == "__main__":
    main()
