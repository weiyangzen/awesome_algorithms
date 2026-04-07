"""Momentum conservation MVP: two-body 1D collision with validations.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CollisionCase:
    """One deterministic two-body collision configuration."""

    name: str
    m1: float
    m2: float
    u1: float
    u2: float
    restitution: float
    external_impulse: float = 0.0


def solve_two_body_collision_1d(
    m1: float,
    m2: float,
    u1: float,
    u2: float,
    restitution: float,
) -> tuple[float, float]:
    """Closed-system 1D collision solver.

    Uses momentum conservation and restitution:
        m1*u1 + m2*u2 = m1*v1 + m2*v2
        v2 - v1 = e*(u1 - u2)
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Mass must be positive.")
    if not (0.0 <= restitution <= 1.0):
        raise ValueError("Restitution must be in [0, 1].")

    denom = m1 + m2
    v1 = ((m1 - restitution * m2) * u1 + (1.0 + restitution) * m2 * u2) / denom
    v2 = ((m2 - restitution * m1) * u2 + (1.0 + restitution) * m1 * u1) / denom
    return float(v1), float(v2)


def kinetic_energy(m1: float, v1: float, m2: float, v2: float) -> float:
    return 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2


def run_case(case: CollisionCase) -> dict[str, float]:
    # Model an external impulse as a pre-collision velocity kick on body 1.
    # Then the collision itself is solved as a closed-system interaction.
    u1_eff = case.u1 + case.external_impulse / case.m1
    u2_eff = case.u2

    v1, v2 = solve_two_body_collision_1d(
        case.m1,
        case.m2,
        u1_eff,
        u2_eff,
        case.restitution,
    )

    p_before = case.m1 * case.u1 + case.m2 * case.u2
    p_after = case.m1 * v1 + case.m2 * v2
    expected_p_after = p_before + case.external_impulse
    momentum_residual = p_after - expected_p_after

    ke_before = kinetic_energy(case.m1, u1_eff, case.m2, u2_eff)
    ke_after = kinetic_energy(case.m1, v1, case.m2, v2)

    return {
        "v1": v1,
        "v2": v2,
        "p_before": p_before,
        "p_after": p_after,
        "expected_p_after": expected_p_after,
        "momentum_residual": momentum_residual,
        "ke_before_effective": ke_before,
        "ke_after": ke_after,
    }


def run_random_stress_test(num_cases: int = 200, seed: int = 7) -> float:
    """Randomized consistency check for momentum accounting."""
    rng = np.random.default_rng(seed)
    max_abs_residual = 0.0
    for _ in range(num_cases):
        m1 = rng.uniform(0.1, 10.0)
        m2 = rng.uniform(0.1, 10.0)
        u1 = rng.uniform(-8.0, 8.0)
        u2 = rng.uniform(-8.0, 8.0)
        restitution = rng.uniform(0.0, 1.0)
        external_impulse = rng.uniform(-3.0, 3.0) if rng.random() < 0.5 else 0.0

        case = CollisionCase(
            name="random",
            m1=float(m1),
            m2=float(m2),
            u1=float(u1),
            u2=float(u2),
            restitution=float(restitution),
            external_impulse=float(external_impulse),
        )
        result = run_case(case)
        max_abs_residual = max(max_abs_residual, abs(result["momentum_residual"]))
    return max_abs_residual


def print_case_result(case: CollisionCase, result: dict[str, float]) -> None:
    print(f"[{case.name}]")
    print(
        "  masses=(%.3f, %.3f), velocities_before=(%.3f, %.3f), e=%.3f, impulse=%.3f"
        % (
            case.m1,
            case.m2,
            case.u1,
            case.u2,
            case.restitution,
            case.external_impulse,
        )
    )
    print("  velocities_after=(%.6f, %.6f)" % (result["v1"], result["v2"]))
    print(
        "  p_before=%.6f, p_after=%.6f, expected_p_after=%.6f, residual=%+.3e"
        % (
            result["p_before"],
            result["p_after"],
            result["expected_p_after"],
            result["momentum_residual"],
        )
    )
    print(
        "  ke_effective_before=%.6f, ke_after=%.6f"
        % (result["ke_before_effective"], result["ke_after"])
    )


def main() -> None:
    cases: List[CollisionCase] = [
        CollisionCase(
            name="Elastic, closed system",
            m1=2.0,
            m2=3.0,
            u1=5.0,
            u2=-1.0,
            restitution=1.0,
        ),
        CollisionCase(
            name="Inelastic, closed system",
            m1=1.5,
            m2=4.0,
            u1=3.5,
            u2=0.2,
            restitution=0.45,
        ),
        CollisionCase(
            name="Inelastic + external impulse",
            m1=1.2,
            m2=2.3,
            u1=-1.0,
            u2=2.5,
            restitution=0.7,
            external_impulse=1.8,
        ),
    ]

    tolerance = 1e-10
    for case in cases:
        result = run_case(case)
        print_case_result(case, result)
        assert abs(result["momentum_residual"]) < tolerance, (
            f"Momentum balance failed for case {case.name}: "
            f"{result['momentum_residual']}"
        )
        if case.restitution == 1.0 and abs(case.external_impulse) < 1e-14:
            assert abs(result["ke_after"] - result["ke_before_effective"]) < 1e-10, (
                "Elastic collision should conserve kinetic energy in closed system."
            )
        if case.restitution < 1.0:
            assert result["ke_after"] <= result["ke_before_effective"] + 1e-12, (
                "Inelastic collision should not gain kinetic energy."
            )
        print()

    max_residual = run_random_stress_test(num_cases=400, seed=42)
    print("[Random stress test]")
    print("  max momentum residual over 400 cases: %.3e" % max_residual)
    assert max_residual < 1e-10, "Random stress test residual is too large."
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
