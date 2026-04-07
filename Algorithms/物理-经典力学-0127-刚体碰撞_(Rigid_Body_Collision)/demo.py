"""Rigid Body Collision MVP (PHYS-0127).

This script demonstrates a minimal, auditable rigid-body collision pipeline in 2D:
1) Time-of-impact (TOI) solve for two moving disks.
2) Normal + tangential impulse computation at the contact point.
3) Linear/angular velocity update with restitution and Coulomb friction.
4) Conservation checks (linear momentum, angular momentum) and energy dissipation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RigidDiskState:
    """Planar rigid-disk state used by the MVP."""

    mass: float
    radius: float
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)
    omega: float  # scalar z-angular velocity

    @property
    def inertia(self) -> float:
        """Moment of inertia for a solid disk around center."""

        return 0.5 * self.mass * self.radius * self.radius

    def clone(self) -> "RigidDiskState":
        """Deep-copy state arrays to avoid accidental aliasing."""

        return RigidDiskState(
            mass=float(self.mass),
            radius=float(self.radius),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            omega=float(self.omega),
        )


@dataclass
class CollisionScenario:
    """Configuration for a single collision event simulation."""

    restitution: float = 0.82
    friction: float = 0.25
    t_end: float = 2.0
    num_samples: int = 121


def perp(v: np.ndarray) -> np.ndarray:
    """2D +90 degree rotation: [x, y] -> [-y, x]."""

    return np.array([-v[1], v[0]], dtype=float)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D scalar cross product a x b."""

    return float(a[0] * b[1] - a[1] * b[0])


def point_velocity(body: RigidDiskState, r_contact: np.ndarray) -> np.ndarray:
    """Velocity of a body point: v + omega x r."""

    return body.velocity + body.omega * perp(r_contact)


def kinetic_energy(body: RigidDiskState) -> float:
    """Total kinetic energy (translational + rotational)."""

    trans = 0.5 * body.mass * float(np.dot(body.velocity, body.velocity))
    rot = 0.5 * body.inertia * body.omega * body.omega
    return trans + rot


def total_linear_momentum(a: RigidDiskState, b: RigidDiskState) -> np.ndarray:
    """System linear momentum in world frame."""

    return a.mass * a.velocity + b.mass * b.velocity


def total_angular_momentum_about_origin(a: RigidDiskState, b: RigidDiskState) -> float:
    """System scalar angular momentum Lz about world origin."""

    l_a = cross2(a.position, a.mass * a.velocity) + a.inertia * a.omega
    l_b = cross2(b.position, b.mass * b.velocity) + b.inertia * b.omega
    return l_a + l_b


def advance_ballistic(body: RigidDiskState, dt: float) -> None:
    """Advance center position under zero external force."""

    body.position = body.position + body.velocity * dt


def time_of_impact(a: RigidDiskState, b: RigidDiskState, t_max: float) -> float | None:
    """Solve first nonnegative disk-disk TOI under constant linear velocities."""

    rel_p = b.position - a.position
    rel_v = b.velocity - a.velocity
    radius_sum = a.radius + b.radius

    c_term = float(np.dot(rel_p, rel_p) - radius_sum * radius_sum)
    if c_term <= 0.0:
        return 0.0

    a_term = float(np.dot(rel_v, rel_v))
    if a_term <= 1e-14:
        return None

    b_term = 2.0 * float(np.dot(rel_p, rel_v))
    disc = b_term * b_term - 4.0 * a_term * c_term
    if disc < 0.0:
        return None

    sqrt_disc = float(np.sqrt(disc))
    roots = sorted(
        [(-b_term - sqrt_disc) / (2.0 * a_term), (-b_term + sqrt_disc) / (2.0 * a_term)]
    )

    for root in roots:
        if root < -1e-12:
            continue
        toi = max(0.0, float(root))
        if toi > t_max + 1e-12:
            continue
        sep_vec = rel_p + rel_v * toi
        # Require approaching motion to avoid picking a separating root.
        if float(np.dot(sep_vec, rel_v)) < 0.0:
            return toi

    return None


def resolve_collision(
    a: RigidDiskState,
    b: RigidDiskState,
    restitution: float,
    friction: float,
) -> dict[str, float | np.ndarray]:
    """Resolve a single contact impulse between two rigid disks."""

    center_delta = b.position - a.position
    dist = float(np.linalg.norm(center_delta))
    if dist <= 1e-12:
        raise RuntimeError("Degenerate collision: coincident centers")

    n_hat = center_delta / dist
    contact = a.position + a.radius * n_hat
    r_a = contact - a.position
    r_b = contact - b.position

    v_rel_before = point_velocity(b, r_b) - point_velocity(a, r_a)
    v_rel_n_before = float(np.dot(v_rel_before, n_hat))
    if v_rel_n_before >= 0.0:
        raise RuntimeError("Bodies are not approaching along contact normal")

    inv_mass = (1.0 / a.mass) + (1.0 / b.mass)
    k_n = inv_mass + (cross2(r_a, n_hat) ** 2) / a.inertia + (cross2(r_b, n_hat) ** 2) / b.inertia
    j_n = -(1.0 + restitution) * v_rel_n_before / k_n

    tangential = v_rel_before - v_rel_n_before * n_hat
    tangential_norm = float(np.linalg.norm(tangential))
    if tangential_norm > 1e-12:
        t_hat = tangential / tangential_norm
    else:
        t_hat = perp(n_hat)

    k_t = inv_mass + (cross2(r_a, t_hat) ** 2) / a.inertia + (cross2(r_b, t_hat) ** 2) / b.inertia
    j_t_unc = -float(np.dot(v_rel_before, t_hat)) / k_t
    j_t_limit = friction * j_n
    j_t = float(np.clip(j_t_unc, -j_t_limit, j_t_limit))

    impulse = j_n * n_hat + j_t * t_hat

    a.velocity = a.velocity - impulse / a.mass
    b.velocity = b.velocity + impulse / b.mass
    a.omega = a.omega - cross2(r_a, impulse) / a.inertia
    b.omega = b.omega + cross2(r_b, impulse) / b.inertia

    v_rel_after = point_velocity(b, r_b) - point_velocity(a, r_a)
    v_rel_n_after = float(np.dot(v_rel_after, n_hat))
    v_rel_t_before = float(np.dot(v_rel_before, t_hat))
    v_rel_t_after = float(np.dot(v_rel_after, t_hat))

    return {
        "j_n": float(j_n),
        "j_t": float(j_t),
        "j_t_unc": float(j_t_unc),
        "contact_x": float(contact[0]),
        "contact_y": float(contact[1]),
        "n_x": float(n_hat[0]),
        "n_y": float(n_hat[1]),
        "v_rel_n_before": v_rel_n_before,
        "v_rel_n_after": v_rel_n_after,
        "v_rel_t_before": v_rel_t_before,
        "v_rel_t_after": v_rel_t_after,
    }


def sample_piecewise_trajectory(
    body_init: RigidDiskState,
    pos_collision: np.ndarray,
    vel_after: np.ndarray,
    t_collision: float,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Sample center trajectory before/after impact with piecewise-constant velocity."""

    traj = np.empty((len(t_grid), 2), dtype=float)
    for i, t in enumerate(t_grid):
        if t <= t_collision:
            traj[i] = body_init.position + body_init.velocity * t
        else:
            traj[i] = pos_collision + vel_after * (t - t_collision)
    return traj


def run_simulation(scenario: CollisionScenario) -> dict[str, float]:
    """Execute one collision scenario and return diagnostics."""

    body_a_init = RigidDiskState(
        mass=2.0,
        radius=0.45,
        position=np.array([-1.8, 0.0], dtype=float),
        velocity=np.array([2.4, 0.45], dtype=float),
        omega=3.2,
    )
    body_b_init = RigidDiskState(
        mass=1.4,
        radius=0.55,
        position=np.array([1.4, 0.3], dtype=float),
        velocity=np.array([-1.6, -0.3], dtype=float),
        omega=-2.1,
    )

    body_a = body_a_init.clone()
    body_b = body_b_init.clone()

    toi = time_of_impact(body_a, body_b, scenario.t_end)
    if toi is None:
        raise RuntimeError("No collision within time horizon")

    advance_ballistic(body_a, toi)
    advance_ballistic(body_b, toi)

    momentum_before = total_linear_momentum(body_a, body_b)
    angular_before = total_angular_momentum_about_origin(body_a, body_b)
    energy_before = kinetic_energy(body_a) + kinetic_energy(body_b)

    collision = resolve_collision(
        a=body_a,
        b=body_b,
        restitution=scenario.restitution,
        friction=scenario.friction,
    )

    momentum_after = total_linear_momentum(body_a, body_b)
    angular_after = total_angular_momentum_about_origin(body_a, body_b)
    energy_after = kinetic_energy(body_a) + kinetic_energy(body_b)

    separation_at_collision = float(np.linalg.norm(body_b.position - body_a.position))

    t_grid = np.linspace(0.0, scenario.t_end, scenario.num_samples)
    traj_a = sample_piecewise_trajectory(
        body_init=body_a_init,
        pos_collision=body_a.position.copy(),
        vel_after=body_a.velocity.copy(),
        t_collision=toi,
        t_grid=t_grid,
    )
    traj_b = sample_piecewise_trajectory(
        body_init=body_b_init,
        pos_collision=body_b.position.copy(),
        vel_after=body_b.velocity.copy(),
        t_collision=toi,
        t_grid=t_grid,
    )
    center_dist = np.linalg.norm(traj_b - traj_a, axis=1)

    momentum_error = float(np.linalg.norm(momentum_after - momentum_before))
    angular_error = float(abs(angular_after - angular_before))
    restitution_residual = float(
        abs(collision["v_rel_n_after"] + scenario.restitution * collision["v_rel_n_before"])
    )
    tangential_ratio = float(
        abs(collision["v_rel_t_after"]) / (abs(collision["v_rel_t_before"]) + 1e-12)
    )

    return {
        "toi": float(toi),
        "j_n": float(collision["j_n"]),
        "j_t": float(collision["j_t"]),
        "j_t_unc": float(collision["j_t_unc"]),
        "v_rel_n_before": float(collision["v_rel_n_before"]),
        "v_rel_n_after": float(collision["v_rel_n_after"]),
        "v_rel_t_before": float(collision["v_rel_t_before"]),
        "v_rel_t_after": float(collision["v_rel_t_after"]),
        "momentum_error": momentum_error,
        "angular_error": angular_error,
        "energy_before": float(energy_before),
        "energy_after": float(energy_after),
        "energy_ratio": float(energy_after / energy_before),
        "restitution_residual": restitution_residual,
        "tangential_ratio": tangential_ratio,
        "center_distance_min_sampled": float(np.min(center_dist)),
        "separation_at_collision": separation_at_collision,
        "radius_sum": float(body_a.radius + body_b.radius),
        "final_a_x": float(traj_a[-1, 0]),
        "final_a_y": float(traj_a[-1, 1]),
        "final_b_x": float(traj_b[-1, 0]),
        "final_b_y": float(traj_b[-1, 1]),
    }


def print_report(metrics: dict[str, float], scenario: CollisionScenario) -> None:
    """Print compact simulation report."""

    scenario_df = pd.DataFrame(
        [
            {"field": "restitution", "value": f"{scenario.restitution:.3f}"},
            {"field": "friction", "value": f"{scenario.friction:.3f}"},
            {"field": "t_end", "value": f"{scenario.t_end:.3f}"},
            {"field": "num_samples", "value": f"{scenario.num_samples:d}"},
        ]
    )

    diagnostics_df = pd.DataFrame(
        [
            {"metric": "toi", "value": f"{metrics['toi']:.6f}"},
            {"metric": "j_n", "value": f"{metrics['j_n']:.6f}"},
            {"metric": "j_t", "value": f"{metrics['j_t']:.6f}"},
            {"metric": "j_t_unc", "value": f"{metrics['j_t_unc']:.6f}"},
            {"metric": "v_rel_n_before", "value": f"{metrics['v_rel_n_before']:.6f}"},
            {"metric": "v_rel_n_after", "value": f"{metrics['v_rel_n_after']:.6f}"},
            {"metric": "v_rel_t_before", "value": f"{metrics['v_rel_t_before']:.6f}"},
            {"metric": "v_rel_t_after", "value": f"{metrics['v_rel_t_after']:.6f}"},
            {"metric": "momentum_error", "value": f"{metrics['momentum_error']:.3e}"},
            {"metric": "angular_error", "value": f"{metrics['angular_error']:.3e}"},
            {"metric": "energy_before", "value": f"{metrics['energy_before']:.6f}"},
            {"metric": "energy_after", "value": f"{metrics['energy_after']:.6f}"},
            {"metric": "energy_ratio", "value": f"{metrics['energy_ratio']:.6f}"},
            {
                "metric": "restitution_residual",
                "value": f"{metrics['restitution_residual']:.3e}",
            },
            {"metric": "tangential_ratio", "value": f"{metrics['tangential_ratio']:.6f}"},
            {
                "metric": "center_distance_min_sampled",
                "value": f"{metrics['center_distance_min_sampled']:.6f}",
            },
            {
                "metric": "separation_at_collision",
                "value": f"{metrics['separation_at_collision']:.6f}",
            },
            {"metric": "radius_sum", "value": f"{metrics['radius_sum']:.6f}"},
            {"metric": "final_a_x", "value": f"{metrics['final_a_x']:.6f}"},
            {"metric": "final_a_y", "value": f"{metrics['final_a_y']:.6f}"},
            {"metric": "final_b_x", "value": f"{metrics['final_b_x']:.6f}"},
            {"metric": "final_b_y", "value": f"{metrics['final_b_y']:.6f}"},
        ]
    )

    print("=== Rigid Body Collision MVP (PHYS-0127) ===")
    print("Scenario")
    print(scenario_df.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics_df.to_string(index=False))


def main() -> None:
    """Entry point with deterministic checks for CI-friendly execution."""

    scenario = CollisionScenario()
    metrics = run_simulation(scenario)
    print_report(metrics, scenario)

    assert metrics["toi"] >= 0.0
    assert metrics["j_n"] > 0.0
    assert metrics["momentum_error"] < 1e-10
    assert metrics["angular_error"] < 1e-10
    assert metrics["energy_after"] <= metrics["energy_before"] + 1e-10
    assert metrics["restitution_residual"] < 1e-10
    assert abs(metrics["separation_at_collision"] - metrics["radius_sum"]) < 1e-9
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
