"""Minimal runnable MVP for Twin Paradox (special relativity).

This script builds a piecewise-constant velocity worldline for the traveling twin,
computes proper time numerically, compares against the closed-form solution,
and demonstrates the relativity-of-simultaneity jump at turnaround.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One inertial leg in Earth frame."""

    duration: float  # Earth-frame coordinate time
    velocity: float  # Earth-frame velocity in units of c


def lorentz_gamma(beta: float) -> float:
    """Return gamma = 1/sqrt(1-beta^2)."""
    if abs(beta) >= 1.0:
        raise ValueError("|beta| must be < 1.")
    return 1.0 / np.sqrt(1.0 - beta * beta)


def build_piecewise_velocity(time: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Create piecewise-constant velocity profile over a uniform time grid."""
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with at least 2 points.")
    dt = np.diff(time)
    if not np.allclose(dt, dt[0]):
        raise ValueError("time grid must be uniform.")

    total_duration = sum(seg.duration for seg in segments)
    if not np.isclose(total_duration, time[-1] - time[0]):
        raise ValueError("sum(segment.duration) must equal total simulation time.")

    v = np.empty_like(time)
    for i, t in enumerate(time):
        elapsed = t - time[0]
        cumulative = 0.0
        chosen = segments[-1].velocity
        for seg in segments:
            cumulative += seg.duration
            if elapsed <= cumulative + 1e-12:
                chosen = seg.velocity
                break
        v[i] = chosen
    return v


def integrate_position(time: np.ndarray, velocity: np.ndarray, x0: float = 0.0) -> np.ndarray:
    """Integrate x(t) via left Riemann sum on a uniform grid."""
    if time.shape != velocity.shape:
        raise ValueError("time and velocity must have the same shape.")
    dt = np.diff(time)
    x = np.empty_like(time)
    x[0] = x0
    x[1:] = x0 + np.cumsum(velocity[:-1] * dt)
    return x


def integrate_proper_time(time: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Integrate d\tau = dt * sqrt(1 - beta^2) on the same grid."""
    if time.shape != velocity.shape:
        raise ValueError("time and velocity must have the same shape.")
    if np.any(np.abs(velocity) >= 1.0):
        raise ValueError("All |beta| values must be < 1.")

    dt = np.diff(time)
    d_tau = dt * np.sqrt(1.0 - velocity[:-1] ** 2)
    tau = np.empty_like(time)
    tau[0] = 0.0
    tau[1:] = np.cumsum(d_tau)
    return tau


def analytic_traveler_proper_time(duration_each_leg: float, beta: float) -> float:
    """Closed-form proper time for symmetric out-and-back trip."""
    return 2.0 * duration_each_leg * np.sqrt(1.0 - beta * beta)


def simultaneity_jump_at_turnaround(t_turn: float, x_turn: float, beta: float) -> tuple[float, float, float]:
    """Earth times simultaneous with turnaround in outbound vs inbound inertial frames.

    Outbound frame (velocity +beta):  t_sim_out = t_turn - beta*x_turn
    Inbound frame  (velocity -beta):  t_sim_in  = t_turn + beta*x_turn

    (Natural units with c=1.)
    """
    t_sim_out = t_turn - beta * x_turn
    t_sim_in = t_turn + beta * x_turn
    return t_sim_out, t_sim_in, t_sim_in - t_sim_out


def main() -> None:
    # Natural units: c = 1 light-year/year.
    duration_each_leg = 10.0  # years in Earth frame
    beta = 0.8  # traveler speed as fraction of c
    total_time = 2.0 * duration_each_leg

    n_steps = 20_001
    time = np.linspace(0.0, total_time, n_steps)
    segments = [
        Segment(duration=duration_each_leg, velocity=beta),
        Segment(duration=duration_each_leg, velocity=-beta),
    ]

    v = build_piecewise_velocity(time, segments)
    x = integrate_position(time, v, x0=0.0)
    tau_traveler = integrate_proper_time(time, v)

    earth_proper_time = total_time
    traveler_tau_num = float(tau_traveler[-1])
    traveler_tau_ref = analytic_traveler_proper_time(duration_each_leg, beta)
    gamma = lorentz_gamma(beta)

    age_gap = earth_proper_time - traveler_tau_num

    t_turn = duration_each_leg
    x_turn = beta * duration_each_leg
    t_sim_out, t_sim_in, jump = simultaneity_jump_at_turnaround(t_turn, x_turn, beta)

    print("Twin Paradox MVP (natural units: c=1 ly/year)")
    print(f"beta = {beta:.3f}, gamma = {gamma:.6f}")
    print(f"Earth proper time at reunion      : {earth_proper_time:.6f} years")
    print(f"Traveler proper time (numerical) : {traveler_tau_num:.6f} years")
    print(f"Traveler proper time (analytic)  : {traveler_tau_ref:.6f} years")
    print(f"Aging gap (Earth - Traveler)     : {age_gap:.6f} years")
    print()
    print("Relativity of simultaneity at turnaround:")
    print(f"Earth time simultaneous before turn (outbound frame): {t_sim_out:.6f} years")
    print(f"Earth time simultaneous after turn  (inbound frame): {t_sim_in:.6f} years")
    print(f"Simultaneity jump                                    : {jump:.6f} years")

    # Non-interactive validation checks.
    reunion_position_error = abs(float(x[-1]))
    tau_error = abs(traveler_tau_num - traveler_tau_ref)

    assert reunion_position_error < 2e-3, "Traveler should return near Earth (x=0)."
    assert tau_error < 2e-3, "Numerical proper time should match analytic reference."
    assert earth_proper_time > traveler_tau_num, "Earth twin must age more in this setup."
    assert jump > 0.0, "Simultaneity jump must be positive for this turnaround geometry."


if __name__ == "__main__":
    main()
