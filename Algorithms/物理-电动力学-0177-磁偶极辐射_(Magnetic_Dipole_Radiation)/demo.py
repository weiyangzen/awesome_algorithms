"""Minimal runnable MVP for Magnetic Dipole Radiation.

This script demonstrates three core checks:
1) Time-domain average radiated power from m''(t) matches harmonic closed form.
2) Angular power density dP/dOmega ~ sin^2(theta) integrates to total power.
3) Small-loop radiation resistance from dipole-power formula matches textbook form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


MU0 = 4.0e-7 * np.pi
C0 = 299_792_458.0
EPS = 1e-15


@dataclass(frozen=True)
class SmallLoop:
    """Small loop antenna surrogate that behaves as a magnetic dipole."""

    turns: int
    area_m2: float

    def magnetic_moment_amplitude(self, current_amp_a: float) -> float:
        """m0 = N * I0 * A (SI)."""
        return float(self.turns) * float(self.area_m2) * float(current_amp_a)


def central_second_derivative(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return d2(values)/dt2 on uniform grid via second-order central differences."""
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and values must be 1-D arrays")
    if time.shape[0] != values.shape[0] or time.shape[0] < 5:
        raise ValueError("time and values must have same length >= 5")

    dt = np.diff(time)
    dt0 = float(dt[0])
    if not np.allclose(dt, dt0, rtol=1e-10, atol=1e-14):
        raise ValueError("uniform time grid is required for this MVP")

    out = np.zeros_like(values, dtype=np.float64)

    # 4th-order central stencil on the interior to reduce phase/amplitude error.
    out[2:-2] = (
        -values[4:]
        + 16.0 * values[3:-1]
        - 30.0 * values[2:-2]
        + 16.0 * values[1:-3]
        - values[:-4]
    ) / (12.0 * dt0 * dt0)

    # 2nd-order central stencil near boundaries where 4th-order is unavailable.
    out[1] = (values[2] - 2.0 * values[1] + values[0]) / (dt0 * dt0)
    out[-2] = (values[-1] - 2.0 * values[-2] + values[-3]) / (dt0 * dt0)

    # 2nd-order one-sided boundaries.
    out[0] = (2.0 * values[0] - 5.0 * values[1] + 4.0 * values[2] - values[3]) / (dt0 * dt0)
    out[-1] = (2.0 * values[-1] - 5.0 * values[-2] + 4.0 * values[-3] - values[-4]) / (dt0 * dt0)
    return out


def instantaneous_total_power_from_mddot(m_ddot: np.ndarray) -> np.ndarray:
    """P(t) = mu0/(6*pi*c^3) * |m''(t)|^2 for magnetic dipole radiation."""
    coeff = MU0 / (6.0 * np.pi * C0**3)
    return coeff * (m_ddot * m_ddot)


def average_power_harmonic_closed_form(m0: float, omega: float) -> float:
    """<P> for m(t)=m0*cos(omega t): mu0*omega^4*m0^2/(12*pi*c^3)."""
    return MU0 * (omega**4) * (m0**2) / (12.0 * np.pi * C0**3)


def angular_density_harmonic(theta: np.ndarray, m0: float, omega: float) -> np.ndarray:
    """Time-averaged angular power density dP/dOmega for axis-aligned harmonic dipole."""
    prefac = MU0 * (omega**4) * (m0**2) / (32.0 * np.pi**2 * C0**3)
    return prefac * (np.sin(theta) ** 2)


def integrate_over_solid_angle(theta: np.ndarray, density_theta: np.ndarray) -> float:
    """Integrate axisymmetric dP/dOmega(theta) over 4pi: \\int dphi \\int dtheta ..."""
    if theta.ndim != 1 or density_theta.ndim != 1 or theta.shape != density_theta.shape:
        raise ValueError("theta and density arrays must be same-shape 1-D arrays")
    integrand = density_theta * np.sin(theta)
    return float(2.0 * np.pi * np.trapezoid(integrand, theta))


def radiation_resistance_from_power_formula(turns: int, area_m2: float, omega: float) -> float:
    """Rrad from dipole power with I_rms convention.

    m0 = N A I0, I_rms = I0/sqrt(2), Rrad = <P>/I_rms^2
    -> Rrad = mu0 * omega^4 * (N*A)^2 / (6*pi*c^3)
    """
    na = float(turns) * float(area_m2)
    return MU0 * (omega**4) * (na**2) / (6.0 * np.pi * C0**3)


def radiation_resistance_small_loop_textbook(turns: int, area_m2: float, wavelength_m: float) -> float:
    """Textbook small-loop form: Rrad = K * (N*A/lambda^2)^2, with SI-exact K."""
    ratio = (float(turns) * float(area_m2)) / (float(wavelength_m) ** 2)
    coeff = MU0 * (2.0 * np.pi) ** 4 * C0 / (6.0 * np.pi)
    return coeff * (ratio**2)


def run_power_consistency_demo() -> Dict[str, float]:
    """Numerical time-average from m''(t) vs closed-form harmonic average power."""
    loop = SmallLoop(turns=18, area_m2=0.018)
    current_amp = 1.5
    freq_hz = 8.0e6
    omega = 2.0 * np.pi * freq_hz

    m0 = loop.magnetic_moment_amplitude(current_amp)

    periods = 24
    points_per_period = 2400
    t_end = periods / freq_hz
    npts = periods * points_per_period + 1
    time = np.linspace(0.0, t_end, npts, dtype=np.float64)

    m_t = m0 * np.cos(omega * time)
    m_ddot_num = central_second_derivative(time, m_t)
    p_inst = instantaneous_total_power_from_mddot(m_ddot_num)

    # Ignore tiny boundary region affected by one-sided differences.
    interior = slice(20, -20)
    p_avg_num = float(np.mean(p_inst[interior]))
    p_avg_ref = average_power_harmonic_closed_form(m0=m0, omega=omega)

    rel_err = abs(p_avg_num - p_avg_ref) / max(EPS, abs(p_avg_ref))

    assert rel_err < 1.0e-3, f"Average power mismatch too large: rel_err={rel_err:.3e}"

    return {
        "freq_hz": freq_hz,
        "m0_a_m2": m0,
        "p_avg_num_w": p_avg_num,
        "p_avg_ref_w": p_avg_ref,
        "p_avg_rel_err": rel_err,
    }


def run_angular_integration_demo() -> Dict[str, float]:
    """Integrate dP/dOmega(theta) and compare with closed-form total power."""
    m0 = 0.42
    freq_hz = 12.0e6
    omega = 2.0 * np.pi * freq_hz

    theta = np.linspace(0.0, np.pi, 20_001, dtype=np.float64)
    dpdomega = angular_density_harmonic(theta=theta, m0=m0, omega=omega)

    p_int = integrate_over_solid_angle(theta=theta, density_theta=dpdomega)
    p_ref = average_power_harmonic_closed_form(m0=m0, omega=omega)

    rel_err = abs(p_int - p_ref) / max(EPS, abs(p_ref))
    ratio_equator_to_pole = float(dpdomega[theta.shape[0] // 2] / max(EPS, dpdomega[0] + EPS))

    assert rel_err < 1e-7, f"Solid-angle integration mismatch: rel_err={rel_err:.3e}"
    assert dpdomega[0] < 1e-20 and dpdomega[-1] < 1e-20, "Pattern must vanish at poles (sin^2 theta)."

    return {
        "freq_hz": freq_hz,
        "p_integrated_w": p_int,
        "p_closed_form_w": p_ref,
        "integration_rel_err": rel_err,
        "equator_to_pole_ratio": ratio_equator_to_pole,
    }


def run_small_loop_resistance_demo() -> Dict[str, float]:
    """Cross-check radiation resistance with textbook small-loop formula."""
    turns = 22
    area_m2 = 0.0095
    freq_hz = 5.0e6
    omega = 2.0 * np.pi * freq_hz
    wavelength_m = C0 / freq_hz

    r_power = radiation_resistance_from_power_formula(turns=turns, area_m2=area_m2, omega=omega)
    r_text = radiation_resistance_small_loop_textbook(
        turns=turns,
        area_m2=area_m2,
        wavelength_m=wavelength_m,
    )

    rel_err = abs(r_power - r_text) / max(EPS, abs(r_text))

    # Frequency scaling sanity: Rrad ~ f^4.
    f2 = 2.0 * freq_hz
    r2 = radiation_resistance_from_power_formula(
        turns=turns,
        area_m2=area_m2,
        omega=2.0 * np.pi * f2,
    )
    scaling_ratio = r2 / max(EPS, r_power)

    assert rel_err < 1e-12, f"Radiation resistance formula mismatch: rel_err={rel_err:.3e}"
    assert abs(scaling_ratio - 16.0) < 5e-12, "Rrad should scale as f^4."

    return {
        "freq_hz": freq_hz,
        "r_rad_power_ohm": r_power,
        "r_rad_textbook_ohm": r_text,
        "resistance_rel_err": rel_err,
        "f2_over_f1_resistance_ratio": scaling_ratio,
    }


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print("=== Demo A: Time-domain power consistency ===")
    rep_a = run_power_consistency_demo()
    for key, value in rep_a.items():
        print(f"{key:>28s}: {value:.12e}")

    print("\n=== Demo B: Angular pattern integration ===")
    rep_b = run_angular_integration_demo()
    for key, value in rep_b.items():
        print(f"{key:>28s}: {value:.12e}")

    print("\n=== Demo C: Small-loop radiation resistance ===")
    rep_c = run_small_loop_resistance_demo()
    for key, value in rep_c.items():
        print(f"{key:>28s}: {value:.12e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
