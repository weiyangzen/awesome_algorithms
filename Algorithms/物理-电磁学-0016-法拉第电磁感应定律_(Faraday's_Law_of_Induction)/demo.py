"""Minimal runnable MVP for Faraday's Law of Induction.

This script demonstrates:
1) Flux-linkage -> induced EMF using finite differences.
2) RL circuit response to the induced EMF and comparison with harmonic steady-state theory.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


MU0 = 4.0e-7 * np.pi


def magnetic_flux_linkage(num_turns: int, area: float, b_normal: np.ndarray) -> np.ndarray:
    """Return flux linkage lambda = N * Phi = N * A * B_n."""
    return float(num_turns) * float(area) * b_normal


def finite_difference_derivative(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return d(values)/dt on a uniform time grid using centered differences."""
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and values must be 1-D arrays")
    if time.shape[0] != values.shape[0] or time.shape[0] < 3:
        raise ValueError("time and values must have the same length >= 3")

    dt = np.diff(time)
    dt0 = float(dt[0])
    if not np.allclose(dt, dt0, rtol=1e-10, atol=1e-12):
        raise ValueError("time grid must be uniform for this MVP")

    deriv = np.zeros_like(values, dtype=np.float64)
    deriv[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt0)
    deriv[0] = (values[1] - values[0]) / dt0
    deriv[-1] = (values[-1] - values[-2]) / dt0
    return deriv


def induced_emf_from_flux_linkage(time: np.ndarray, flux_linkage: np.ndarray) -> np.ndarray:
    """Return induced EMF using Faraday's law: epsilon = - d(lambda)/dt."""
    return -finite_difference_derivative(time, flux_linkage)


def sinusoidal_b_field(time: np.ndarray, b0: float, omega: float, phase: float) -> np.ndarray:
    """Return B(t) = B0 * sin(omega * t + phase)."""
    return float(b0) * np.sin(float(omega) * time + float(phase))


def analytical_emf_for_sinusoidal_b(
    time: np.ndarray,
    num_turns: int,
    area: float,
    b0: float,
    omega: float,
    phase: float,
) -> np.ndarray:
    """Return analytical epsilon(t) for sinusoidal B(t)."""
    amplitude = float(num_turns) * float(area) * float(b0) * float(omega)
    return -amplitude * np.cos(float(omega) * time + float(phase))


def fit_first_harmonic(signal: np.ndarray, time: np.ndarray, omega: float) -> np.ndarray:
    """Fit signal ≈ c0 + c1*cos(omega t) + c2*sin(omega t) via least squares."""
    design = np.column_stack(
        [
            np.ones_like(time, dtype=np.float64),
            np.cos(float(omega) * time),
            np.sin(float(omega) * time),
        ]
    )
    coeff, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return coeff


def simulate_rl_response(
    emf: np.ndarray,
    time: np.ndarray,
    resistance: float,
    inductance: float,
    i0: float = 0.0,
) -> np.ndarray:
    """Simulate L di/dt + R i = emf using explicit Euler."""
    if inductance <= 0.0 or resistance <= 0.0:
        raise ValueError("resistance and inductance must be positive")

    dt = np.diff(time)
    dt0 = float(dt[0])
    if not np.allclose(dt, dt0, rtol=1e-10, atol=1e-12):
        raise ValueError("time grid must be uniform")

    current = np.zeros_like(emf, dtype=np.float64)
    current[0] = float(i0)
    for k in range(time.shape[0] - 1):
        di_dt = (emf[k] - resistance * current[k]) / inductance
        current[k + 1] = current[k] + dt0 * di_dt

    return current


def run_flux_emf_demo() -> Dict[str, float]:
    """Check numerical EMF against analytical EMF for sinusoidal magnetic field."""
    num_turns = 240
    area = 0.012
    b0 = 0.65
    frequency = 50.0
    omega = 2.0 * np.pi * frequency
    phase = 0.35

    time = np.linspace(0.0, 0.12, 6001, dtype=np.float64)
    b_t = sinusoidal_b_field(time, b0=b0, omega=omega, phase=phase)
    flux_linkage = magnetic_flux_linkage(num_turns=num_turns, area=area, b_normal=b_t)

    emf_num = induced_emf_from_flux_linkage(time, flux_linkage)
    emf_ref = analytical_emf_for_sinusoidal_b(
        time,
        num_turns=num_turns,
        area=area,
        b0=b0,
        omega=omega,
        phase=phase,
    )

    interior = slice(5, -5)
    abs_err = np.abs(emf_num[interior] - emf_ref[interior])
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))

    db_dt = finite_difference_derivative(time, b_t)
    lenz_indicator = float(np.mean(emf_num[interior] * db_dt[interior]))

    assert max_abs_err < 0.8, f"EMF finite-difference error too large: {max_abs_err:.6f}"
    assert lenz_indicator < 0.0, "Lenz's law sign check failed: epsilon * dB/dt should be negative on average"

    return {
        "num_turns": float(num_turns),
        "area_m2": float(area),
        "b0_t": float(b0),
        "omega_rad_s": float(omega),
        "max_abs_err_v": max_abs_err,
        "mean_abs_err_v": mean_abs_err,
        "lenz_indicator": lenz_indicator,
    }


def run_rl_demo() -> Dict[str, float]:
    """Drive an RL circuit with induced EMF and compare to harmonic steady-state theory."""
    num_turns = 180
    area = 0.010
    b0 = 0.80
    frequency = 40.0
    omega = 2.0 * np.pi * frequency
    phase = 0.15

    resistance = 9.0
    inductance = 0.045

    time = np.linspace(0.0, 0.40, 16001, dtype=np.float64)
    b_t = sinusoidal_b_field(time, b0=b0, omega=omega, phase=phase)
    flux_linkage = magnetic_flux_linkage(num_turns=num_turns, area=area, b_normal=b_t)
    emf = induced_emf_from_flux_linkage(time, flux_linkage)

    current = simulate_rl_response(
        emf=emf,
        time=time,
        resistance=resistance,
        inductance=inductance,
        i0=0.0,
    )

    keep = int(0.60 * time.shape[0])
    t_tail = time[keep:]
    e_tail = emf[keep:]
    i_tail = current[keep:]

    e_coeff = fit_first_harmonic(e_tail, t_tail, omega)
    i_coeff = fit_first_harmonic(i_tail, t_tail, omega)

    e_vec = np.array([e_coeff[1], e_coeff[2]], dtype=np.float64)
    i_vec_num = np.array([i_coeff[1], i_coeff[2]], dtype=np.float64)

    a = np.array(
        [[resistance, inductance * omega], [-inductance * omega, resistance]],
        dtype=np.float64,
    )
    i_vec_theory = np.linalg.solve(a, e_vec)

    amp_e = float(np.linalg.norm(e_vec))
    amp_i_num = float(np.linalg.norm(i_vec_num))
    amp_i_theory = float(np.linalg.norm(i_vec_theory))

    amplitude_rel_err = abs(amp_i_num - amp_i_theory) / max(1e-12, amp_i_theory)
    coeff_rel_err = float(np.linalg.norm(i_vec_num - i_vec_theory) / max(1e-12, np.linalg.norm(i_vec_theory)))

    energy_rate = emf * current - resistance * current * current
    stored_energy = 0.5 * inductance * current * current
    d_stored_dt = finite_difference_derivative(time, stored_energy)
    balance_tail = float(np.mean(np.abs(energy_rate[keep:] - d_stored_dt[keep:])))
    balance_scale = float(np.mean(np.abs(energy_rate[keep:])))
    balance_rel = balance_tail / max(1e-12, balance_scale)

    assert amplitude_rel_err < 0.05, f"RL steady-state amplitude mismatch: {amplitude_rel_err:.4f}"
    assert coeff_rel_err < 0.07, f"RL harmonic coefficient mismatch: {coeff_rel_err:.4f}"
    assert balance_rel < 0.02, f"Energy balance relative residual too large: {balance_rel:.6f}"

    return {
        "resistance_ohm": float(resistance),
        "inductance_h": float(inductance),
        "emf_amp_v": amp_e,
        "i_amp_num_a": amp_i_num,
        "i_amp_theory_a": amp_i_theory,
        "amplitude_rel_err": float(amplitude_rel_err),
        "coeff_rel_err": coeff_rel_err,
        "energy_balance_residual": balance_tail,
        "energy_balance_rel_residual": balance_rel,
    }


def estimate_uniform_field_energy_density(b_values: np.ndarray) -> Tuple[float, float]:
    """Return mean and max magnetic energy density u = B^2 / (2*mu0)."""
    u = (b_values * b_values) / (2.0 * MU0)
    return float(np.mean(u)), float(np.max(u))


def main() -> None:
    print("=== Demo A: Faraday law (flux linkage -> induced EMF) ===")
    flux_report = run_flux_emf_demo()
    for key, value in flux_report.items():
        print(f"{key:>20s}: {value:.6f}")

    print("\n=== Demo B: RL response driven by induced EMF ===")
    rl_report = run_rl_demo()
    for key, value in rl_report.items():
        print(f"{key:>20s}: {value:.6f}")

    probe_b = np.array([0.10, 0.25, 0.40, 0.60], dtype=np.float64)
    u_mean, u_max = estimate_uniform_field_energy_density(probe_b)
    print("\n=== Extra: Magnetic energy density from B samples ===")
    print(f"{'u_mean_j_m3':>20s}: {u_mean:.6f}")
    print(f"{'u_max_j_m3':>20s}: {u_max:.6f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
