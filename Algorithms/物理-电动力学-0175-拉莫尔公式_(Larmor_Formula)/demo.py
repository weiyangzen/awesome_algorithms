"""Minimal runnable MVP for Larmor Formula (non-relativistic radiation power)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback path
    np = None

EPSILON_0 = 8.8541878128e-12  # vacuum permittivity (F/m)
C_LIGHT = 2.99792458e8  # speed of light (m/s)


@dataclass
class SimulationResult:
    period: float
    numeric_avg_power: float
    analytic_avg_power: float
    relative_error: float
    times: Sequence[float]
    accelerations: Sequence[float]
    powers: Sequence[float]


def larmor_power_scalar(charge_c: float, acceleration_m_s2: float) -> float:
    """Instantaneous radiated power from Larmor formula (SI units)."""
    return (charge_c**2 * acceleration_m_s2**2) / (6.0 * math.pi * EPSILON_0 * C_LIGHT**3)


def harmonic_acceleration(a0: float, omega: float, t: float) -> float:
    """a(t) = a0 * cos(omega * t)."""
    return a0 * math.cos(omega * t)


def _trapz_manual(y: Sequence[float], x: Sequence[float]) -> float:
    total = 0.0
    for i in range(len(y) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def simulate_one_period(
    charge_c: float,
    a0: float,
    omega: float,
    samples: int = 5000,
) -> SimulationResult:
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if samples < 3:
        raise ValueError("samples must be >= 3")

    period = 2.0 * math.pi / omega

    if np is not None:
        times = np.linspace(0.0, period, num=samples, endpoint=True, dtype=float)
        accelerations = a0 * np.cos(omega * times)
        powers = (charge_c**2 * accelerations**2) / (6.0 * math.pi * EPSILON_0 * C_LIGHT**3)
        radiated_energy = float(np.trapezoid(powers, times))
        times_out = times.tolist()
        accelerations_out = accelerations.tolist()
        powers_out = powers.tolist()
    else:
        dt = period / (samples - 1)
        times_out = [i * dt for i in range(samples)]
        accelerations_out = [harmonic_acceleration(a0, omega, t) for t in times_out]
        powers_out = [larmor_power_scalar(charge_c, a) for a in accelerations_out]
        radiated_energy = _trapz_manual(powers_out, times_out)

    numeric_avg_power = radiated_energy / period
    analytic_avg_power = (charge_c**2 * a0**2) / (12.0 * math.pi * EPSILON_0 * C_LIGHT**3)

    denom = max(abs(analytic_avg_power), 1e-30)
    relative_error = abs(numeric_avg_power - analytic_avg_power) / denom

    return SimulationResult(
        period=period,
        numeric_avg_power=numeric_avg_power,
        analytic_avg_power=analytic_avg_power,
        relative_error=relative_error,
        times=times_out,
        accelerations=accelerations_out,
        powers=powers_out,
    )


def main() -> None:
    # A deterministic, non-interactive benchmark setup.
    charge_c = 1.0e-9
    freq_hz = 1.0e9
    omega = 2.0 * math.pi * freq_hz
    a0 = 5.0e13
    samples = 5000

    result = simulate_one_period(
        charge_c=charge_c,
        a0=a0,
        omega=omega,
        samples=samples,
    )

    print("=== Larmor Formula MVP (One-Period Harmonic Test) ===")
    print(f"charge q            = {charge_c:.6e} C")
    print(f"frequency f         = {freq_hz:.6e} Hz")
    print(f"omega               = {omega:.6e} rad/s")
    print(f"acceleration a0     = {a0:.6e} m/s^2")
    print(f"period T            = {result.period:.6e} s")
    print(f"samples             = {samples}")
    print(f"numeric <P>         = {result.numeric_avg_power:.6e} W")
    print(f"analytic <P>        = {result.analytic_avg_power:.6e} W")
    print(f"relative error      = {result.relative_error:.6e}")
    print("check (<1e-3)       =", "PASS" if result.relative_error < 1e-3 else "WARN")

    preview = min(6, len(result.times))
    print("\nSample points [index, t(s), a(m/s^2), P(W)]:")
    for i in range(preview):
        print(
            f"{i:>2d}  {result.times[i]:.6e}  "
            f"{result.accelerations[i]:.6e}  {result.powers[i]:.6e}"
        )


if __name__ == "__main__":
    main()
