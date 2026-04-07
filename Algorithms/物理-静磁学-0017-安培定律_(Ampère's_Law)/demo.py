"""Minimal runnable MVP for Ampere's Law (magnetostatics).

This script numerically verifies
    ∮ B · dl = μ0 * I_enclosed
for 2D cross-sections of infinitely long straight currents (along +z / -z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class CurrentWire:
    """Infinite straight wire parallel to z-axis, represented in xy-plane."""

    x: float
    y: float
    current: float  # +z direction positive, -z direction negative


Vector2 = np.ndarray


def magnetic_field_from_wires(point_xy: Vector2, wires: Sequence[CurrentWire]) -> Vector2:
    """Return Bx, By at one xy point from superposition of infinite-wire fields.

    For one wire at (x0, y0) carrying current I along +z:
        B = μ0 I / (2π r^2) * (-dy, dx)
    where dx = x - x0, dy = y - y0.
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    b = np.zeros(2, dtype=np.float64)

    for wire in wires:
        dx = x - wire.x
        dy = y - wire.y
        r2 = dx * dx + dy * dy
        if r2 < 1e-20:
            raise ValueError("Integration path hits a wire singularity.")

        coeff = MU0 * wire.current / (2.0 * np.pi * r2)
        b[0] += -coeff * dy
        b[1] += coeff * dx

    return b


def enclosed_current_circle(center_xy: Vector2, radius: float, wires: Sequence[CurrentWire]) -> float:
    """Return algebraic enclosed current for a circular loop."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r2 = float(radius) * float(radius)
    total = 0.0
    for wire in wires:
        d2 = (wire.x - cx) ** 2 + (wire.y - cy) ** 2
        if d2 < r2:
            total += wire.current
    return total


def ampere_circulation_circle(
    center_xy: Vector2,
    radius: float,
    wires: Sequence[CurrentWire],
    n_segments: int = 2400,
) -> float:
    """Numerically approximate ∮ B·dl on a circle (counter-clockwise)."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if n_segments < 16:
        raise ValueError("n_segments too small")

    cx, cy = float(center_xy[0]), float(center_xy[1])
    dtheta = 2.0 * np.pi / float(n_segments)

    circulation = 0.0
    for k in range(n_segments):
        theta = (k + 0.5) * dtheta
        ct, st = np.cos(theta), np.sin(theta)

        point = np.array([cx + radius * ct, cy + radius * st], dtype=np.float64)
        dl = np.array([-radius * st * dtheta, radius * ct * dtheta], dtype=np.float64)

        b = magnetic_field_from_wires(point, wires)
        circulation += float(np.dot(b, dl))

    return circulation


def ampere_circulation_square(
    center_xy: Vector2,
    half_side: float,
    wires: Sequence[CurrentWire],
    n_per_side: int = 1200,
) -> float:
    """Numerically approximate ∮ B·dl on an axis-aligned square (counter-clockwise)."""
    if half_side <= 0.0:
        raise ValueError("half_side must be positive")
    if n_per_side < 8:
        raise ValueError("n_per_side too small")

    cx, cy = float(center_xy[0]), float(center_xy[1])
    corners = [
        np.array([cx - half_side, cy - half_side], dtype=np.float64),
        np.array([cx + half_side, cy - half_side], dtype=np.float64),
        np.array([cx + half_side, cy + half_side], dtype=np.float64),
        np.array([cx - half_side, cy + half_side], dtype=np.float64),
    ]

    circulation = 0.0
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        dvec = (p1 - p0) / float(n_per_side)

        for j in range(n_per_side):
            point = p0 + (j + 0.5) * dvec
            b = magnetic_field_from_wires(point, wires)
            circulation += float(np.dot(b, dvec))

    return circulation


def run_single_wire_radius_invariance_demo() -> List[Dict[str, float]]:
    """Same enclosed current, different circular radii -> same circulation."""
    wires = [CurrentWire(0.0, 0.0, 8.0)]
    center = np.array([0.0, 0.0], dtype=np.float64)
    expected = MU0 * wires[0].current

    rows: List[Dict[str, float]] = []
    for radius in [0.03, 0.07, 0.16, 0.30]:
        circulation = ampere_circulation_circle(center_xy=center, radius=radius, wires=wires, n_segments=2600)
        rel_err = abs(circulation - expected) / abs(expected)
        rows.append(
            {
                "radius_m": radius,
                "i_enclosed_a": wires[0].current,
                "circulation": circulation,
                "expected": expected,
                "rel_err": rel_err,
            }
        )

    max_rel_err = max(r["rel_err"] for r in rows)
    assert max_rel_err < 8e-4, f"single-wire circle integral error too large: {max_rel_err:.3e}"
    return rows


def run_multiwire_enclosure_demo() -> List[Dict[str, float]]:
    """Different radii enclose different currents; circulation tracks algebraic sum."""
    wires = [
        CurrentWire(0.00, 0.00, +6.0),
        CurrentWire(0.08, 0.00, -2.0),
        CurrentWire(-0.16, 0.02, +3.0),
        CurrentWire(0.00, 0.18, -1.5),
    ]
    center = np.array([0.0, 0.0], dtype=np.float64)

    rows: List[Dict[str, float]] = []
    for radius in [0.05, 0.12, 0.24]:
        i_enclosed = enclosed_current_circle(center_xy=center, radius=radius, wires=wires)
        expected = MU0 * i_enclosed
        circulation = ampere_circulation_circle(center_xy=center, radius=radius, wires=wires, n_segments=3200)

        denom = max(abs(expected), 1e-15)
        rel_err = abs(circulation - expected) / denom

        rows.append(
            {
                "radius_m": radius,
                "i_enclosed_a": i_enclosed,
                "circulation": circulation,
                "expected": expected,
                "rel_err": rel_err,
            }
        )

    # Also test a loop enclosing no wire => near-zero circulation.
    far_center = np.array([0.40, 0.35], dtype=np.float64)
    far_radius = 0.08
    far_i = enclosed_current_circle(center_xy=far_center, radius=far_radius, wires=wires)
    far_circ = ampere_circulation_circle(center_xy=far_center, radius=far_radius, wires=wires, n_segments=2600)

    assert abs(far_i) < 1e-12, "far loop should enclose zero current"
    assert abs(far_circ) < 2e-9, f"zero-enclosure loop circulation too large: {far_circ:.3e}"

    max_rel_err = max(r["rel_err"] for r in rows)
    assert max_rel_err < 2.5e-3, f"multiwire integral mismatch too large: {max_rel_err:.3e}"
    return rows


def run_path_independence_demo() -> Dict[str, float]:
    """Circle and square enclosing the same currents should yield same ∮B·dl."""
    wires = [
        CurrentWire(0.00, 0.00, +6.0),
        CurrentWire(0.08, 0.00, -2.0),
        CurrentWire(-0.16, 0.02, +3.0),
        CurrentWire(0.00, 0.18, -1.5),
    ]
    center = np.array([0.0, 0.0], dtype=np.float64)

    # These two loops enclose +6 A and -2 A, but exclude the other two wires.
    circle_radius = 0.13
    square_half_side = 0.13

    i_enclosed = 4.0
    expected = MU0 * i_enclosed

    circ_circle = ampere_circulation_circle(center_xy=center, radius=circle_radius, wires=wires, n_segments=3600)
    circ_square = ampere_circulation_square(center_xy=center, half_side=square_half_side, wires=wires, n_per_side=1300)

    rel_err_circle = abs(circ_circle - expected) / abs(expected)
    rel_err_square = abs(circ_square - expected) / abs(expected)
    shape_diff_rel = abs(circ_circle - circ_square) / abs(expected)

    assert rel_err_circle < 2.5e-3, "circle path mismatch too large"
    assert rel_err_square < 4.0e-3, "square path mismatch too large"
    assert shape_diff_rel < 5.0e-3, "path independence check failed"

    return {
        "i_enclosed_a": i_enclosed,
        "expected": expected,
        "circle_circulation": circ_circle,
        "square_circulation": circ_square,
        "circle_rel_err": rel_err_circle,
        "square_rel_err": rel_err_square,
        "shape_diff_rel": shape_diff_rel,
    }


def main() -> None:
    print("=== Demo A: Single wire, radius invariance on circular Ampere loops ===")
    rows_a = run_single_wire_radius_invariance_demo()
    for row in rows_a:
        print(
            f"r={row['radius_m']:.3f} m | I_enc={row['i_enclosed_a']:.3f} A "
            f"| integral={row['circulation']:.9e} | mu0I={row['expected']:.9e} "
            f"| rel_err={row['rel_err']:.3e}"
        )

    print("\n=== Demo B: Multi-wire enclosed-current law on circular loops ===")
    rows_b = run_multiwire_enclosure_demo()
    for row in rows_b:
        print(
            f"r={row['radius_m']:.3f} m | I_enc={row['i_enclosed_a']:.3f} A "
            f"| integral={row['circulation']:.9e} | mu0I={row['expected']:.9e} "
            f"| rel_err={row['rel_err']:.3e}"
        )

    print("\n=== Demo C: Path independence (circle vs square, same I_enclosed) ===")
    report_c = run_path_independence_demo()
    for key, value in report_c.items():
        print(f"{key:>18s}: {value:.9e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
