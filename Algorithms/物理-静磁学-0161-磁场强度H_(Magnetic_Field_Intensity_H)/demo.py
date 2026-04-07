"""MVP: compute magnetic field intensity H for infinite straight wires."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class InfiniteWire:
    """Infinite straight wire parallel to z-axis."""

    x: float
    y: float
    current: float  # ampere, positive along +z


def magnetic_field_intensity_xy(
    points_xy: np.ndarray,
    wires: list[InfiniteWire],
    core_radius: float = 1e-4,
) -> np.ndarray:
    """
    Compute H=(Hx, Hy) for infinite wires in the xy plane.

    Formula for a wire at (x0, y0):
      Hx = -I/(2*pi*rho^2) * (y - y0)
      Hy =  I/(2*pi*rho^2) * (x - x0)
    where rho^2=(x-x0)^2+(y-y0)^2.
    """
    h = np.zeros((points_xy.shape[0], 2), dtype=float)
    min_r2 = core_radius * core_radius

    for wire in wires:
        dx = points_xy[:, 0] - wire.x
        dy = points_xy[:, 1] - wire.y
        rho2 = np.maximum(dx * dx + dy * dy, min_r2)
        coef = wire.current / (2.0 * np.pi * rho2)
        h[:, 0] += -coef * dy
        h[:, 1] += coef * dx

    return h


def enclosed_current(
    wires: list[InfiniteWire],
    center_xy: tuple[float, float],
    radius: float,
) -> float:
    """Return algebraic current enclosed by a circular Amperian loop."""
    cx, cy = center_xy
    return float(
        sum(
            wire.current
            for wire in wires
            if (wire.x - cx) ** 2 + (wire.y - cy) ** 2 < radius**2
        )
    )


def ampere_loop_integral(
    wires: list[InfiniteWire],
    center_xy: tuple[float, float],
    radius: float,
    samples: int = 4096,
) -> float:
    """Numerically evaluate integral H·dl on a circular loop."""
    cx, cy = center_xy
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    points = np.column_stack(
        [cx + radius * np.cos(theta), cy + radius * np.sin(theta)]
    )
    h = magnetic_field_intensity_xy(points, wires)
    tangent = np.column_stack([-np.sin(theta), np.cos(theta)])  # CCW direction
    dtheta = 2.0 * np.pi / samples
    return float(np.sum(np.einsum("ij,ij->i", h, tangent)) * radius * dtheta)


def run_demo() -> None:
    wires = [
        InfiniteWire(x=-0.03, y=0.0, current=5.0),
        InfiniteWire(x=0.03, y=0.0, current=-3.0),
    ]

    x = np.linspace(-0.12, 0.12, 13)
    points = np.column_stack([x, np.zeros_like(x)])
    h = magnetic_field_intensity_xy(points, wires)
    b = MU0 * h

    table = pd.DataFrame(
        {
            "x_m": points[:, 0],
            "y_m": points[:, 1],
            "Hx_A_per_m": h[:, 0],
            "Hy_A_per_m": h[:, 1],
            "|H|_A_per_m": np.linalg.norm(h, axis=1),
            "By_T": b[:, 1],
        }
    )
    print("=== Magnetic Field Intensity H on y=0 line ===")
    print(table.round(6).to_string(index=False))

    print("\n=== Ampere Law Numerical Check ===")
    center = (0.0, 0.0)
    for radius in (0.02, 0.06):
        integral = ampere_loop_integral(wires, center_xy=center, radius=radius)
        i_enc = enclosed_current(wires, center_xy=center, radius=radius)
        denom = max(abs(i_enc), 1e-12)
        rel_err = abs(integral - i_enc) / denom
        print(
            f"radius={radius:.3f} m, "
            f"integral(H·dl)={integral:.6f} A, "
            f"I_enclosed={i_enc:.6f} A, "
            f"relative_error={rel_err:.3e}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
