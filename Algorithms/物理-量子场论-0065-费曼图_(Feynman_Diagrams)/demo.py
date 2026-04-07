"""Minimal runnable MVP for Feynman Diagrams (phi^3, tree-level 2->2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Diagram:
    """A single tree-level phi^3 diagram for 2->2 scattering."""

    channel: str
    left_pair: tuple[str, str]
    right_pair: tuple[str, str]


def minkowski_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Minkowski dot product with metric signature (+, -, -, -)."""
    return float(a[0] * b[0] - np.dot(a[1:], b[1:]))


def build_com_momenta(energy: float, mass: float, theta_rad: float) -> dict[str, np.ndarray]:
    """Construct COM-frame external momenta p1, p2 -> p3, p4."""
    if energy <= mass:
        raise ValueError("energy must be larger than mass for scattering kinematics")

    p_abs = float(np.sqrt(energy * energy - mass * mass))
    sin_t = float(np.sin(theta_rad))
    cos_t = float(np.cos(theta_rad))

    p1 = np.array([energy, 0.0, 0.0, p_abs], dtype=np.float64)
    p2 = np.array([energy, 0.0, 0.0, -p_abs], dtype=np.float64)
    p3 = np.array([energy, p_abs * sin_t, 0.0, p_abs * cos_t], dtype=np.float64)
    p4 = np.array([energy, -p_abs * sin_t, 0.0, -p_abs * cos_t], dtype=np.float64)
    return {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


def mandelstam_variables(momenta: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute Mandelstam invariants s, t, u for 2->2 scattering."""
    p1 = momenta["p1"]
    p2 = momenta["p2"]
    p3 = momenta["p3"]
    p4 = momenta["p4"]

    s = minkowski_dot(p1 + p2, p1 + p2)
    t = minkowski_dot(p1 - p3, p1 - p3)
    u = minkowski_dot(p1 - p4, p1 - p4)
    return {"s": s, "t": t, "u": u}


def enumerate_phi3_2to2_diagrams() -> list[Diagram]:
    """Enumerate the 3 tree-level channels for phi^3 2->2 scattering."""
    return [
        Diagram(channel="s", left_pair=("p1", "p2"), right_pair=("p3", "p4")),
        Diagram(channel="t", left_pair=("p1", "p3"), right_pair=("p2", "p4")),
        Diagram(channel="u", left_pair=("p1", "p4"), right_pair=("p2", "p3")),
    ]


def channel_q2(channel: str, invariants: dict[str, float]) -> float:
    """Map channel label to internal momentum square q^2."""
    if channel not in invariants:
        raise KeyError(f"unknown channel: {channel}")
    return float(invariants[channel])


def scalar_propagator(q2: float, mass: float, eps: float) -> complex:
    """Scalar Feynman propagator denominator part: 1/(q^2 - m^2 + i eps)."""
    return 1.0 / complex(q2 - mass * mass, eps)


def reduced_diagram_amplitude(g: float, q2: float, mass: float, eps: float) -> complex:
    """Reduced amplitude without the global -i factor: g^2/(q^2 - m^2 + i eps)."""
    return (g * g) * scalar_propagator(q2=q2, mass=mass, eps=eps)


def evaluate_diagrams(
    diagrams: list[Diagram], invariants: dict[str, float], g: float, mass: float, eps: float
) -> tuple[pd.DataFrame, complex]:
    """Evaluate each channel contribution and return a detailed table."""
    rows: list[dict[str, float | str]] = []
    total_reduced = 0.0 + 0.0j

    for diagram in diagrams:
        q2 = channel_q2(diagram.channel, invariants)
        prop = scalar_propagator(q2=q2, mass=mass, eps=eps)
        amp = reduced_diagram_amplitude(g=g, q2=q2, mass=mass, eps=eps)
        total_reduced += amp

        rows.append(
            {
                "channel": diagram.channel,
                "left_pair": f"{diagram.left_pair[0]}+{diagram.left_pair[1]}",
                "right_pair": f"{diagram.right_pair[0]}+{diagram.right_pair[1]}",
                "q2": q2,
                "propagator_real": float(np.real(prop)),
                "propagator_imag": float(np.imag(prop)),
                "reduced_amp_real": float(np.real(amp)),
                "reduced_amp_imag": float(np.imag(amp)),
                "reduced_amp_abs": float(np.abs(amp)),
            }
        )

    table = pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)
    return table, total_reduced


def main() -> None:
    mass = 1.0
    g = 0.8
    energy = 2.0
    theta_deg = 60.0
    eps = 1e-9

    theta_rad = np.deg2rad(theta_deg)
    momenta = build_com_momenta(energy=energy, mass=mass, theta_rad=theta_rad)
    invariants = mandelstam_variables(momenta)

    diagrams = enumerate_phi3_2to2_diagrams()
    table, total_reduced = evaluate_diagrams(
        diagrams=diagrams,
        invariants=invariants,
        g=g,
        mass=mass,
        eps=eps,
    )

    total_full = -1j * total_reduced

    s = invariants["s"]
    t = invariants["t"]
    u = invariants["u"]
    identity_error = abs((s + t + u) - 4.0 * mass * mass)

    kinematics_df = pd.DataFrame(
        [
            {
                "s": s,
                "t": t,
                "u": u,
                "s+t+u": s + t + u,
                "4m^2": 4.0 * mass * mass,
                "identity_error": identity_error,
            }
        ]
    )

    summary_df = pd.DataFrame(
        [
            {
                "quantity": "reduced_total_M_tilde",
                "real": float(np.real(total_reduced)),
                "imag": float(np.imag(total_reduced)),
                "abs": float(np.abs(total_reduced)),
            },
            {
                "quantity": "full_total_M",
                "real": float(np.real(total_full)),
                "imag": float(np.imag(total_full)),
                "abs": float(np.abs(total_full)),
            },
        ]
    )

    # Consistency checks
    assert len(diagrams) == 3, "phi^3 2->2 tree-level must have exactly 3 channels"
    assert set(table["channel"].tolist()) == {"s", "t", "u"}, "missing channel(s)"
    assert identity_error < 1e-10, "kinematic identity s+t+u=4m^2 is violated"

    summed = complex(table["reduced_amp_real"].sum(), table["reduced_amp_imag"].sum())
    assert abs(summed - total_reduced) < 1e-12, "channel sum mismatch"

    print("=== Kinematics (COM frame) ===")
    print(kinematics_df.to_string(index=False))
    print()

    print("=== Tree-level phi^3 diagrams (2->2) ===")
    print(table.to_string(index=False))
    print()

    print("=== Amplitude summary ===")
    print(summary_df.to_string(index=False))
    print()

    print("All checks passed.")


if __name__ == "__main__":
    main()
