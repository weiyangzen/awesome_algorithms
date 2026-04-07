"""Minimal runnable MVP for electroluminescence (EL).

This script builds a compact LED EL pipeline:
1) Electrical injection from a Shockley diode with series resistance.
2) Carrier-density solve from ABC recombination.
3) Internal/External quantum efficiency and optical power estimation.
4) EL spectrum synthesis with Varshni bandgap shift + Gaussian broadening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Q_E = 1.602176634e-19  # C


@dataclass(frozen=True)
class LEDParams:
    area_m2: float
    active_thickness_m: float
    i0_a: float
    ideality: float
    series_resistance_ohm: float
    shunt_resistance_ohm: float
    a_srh: float
    b_rad: float
    c_auger: float
    extraction_efficiency: float
    injection_efficiency: float
    eg0_ev: float
    varshni_alpha_ev_per_k: float
    varshni_beta_k: float
    ambient_temp_k: float
    thermal_resistance_k_per_w: float
    sigma_lambda_nm_300k: float
    sigma_lambda_temp_slope_nm_per_k: float


def thermal_voltage(temp_k: float) -> float:
    """Return thermal voltage kT/q in volts."""
    return 8.617333262145e-5 * float(temp_k)


def _safe_exp(x: float) -> float:
    """Exponent with clipping for numerical stability."""
    return float(np.exp(np.clip(x, -120.0, 80.0)))


def solve_diode_current(voltage_v: float, temp_k: float, p: LEDParams, init_i: float = 0.0) -> float:
    """Solve Shockley + Rs + Rsh implicit equation using Newton iteration.

    Equation:
    I = I0 * (exp((V - I*Rs)/(n*Vt)) - 1) + (V - I*Rs)/Rsh
    """
    vt = thermal_voltage(temp_k)
    nvt = p.ideality * vt

    i_val = max(0.0, float(init_i))
    for _ in range(60):
        vj = float(voltage_v) - i_val * p.series_resistance_ohm
        exp_term = _safe_exp(vj / nvt)

        rhs = p.i0_a * (exp_term - 1.0) + vj / p.shunt_resistance_ohm
        f_val = i_val - rhs

        d_rhs_di = (
            p.i0_a * exp_term * (-p.series_resistance_ohm / nvt)
            - p.series_resistance_ohm / p.shunt_resistance_ohm
        )
        df_di = 1.0 - d_rhs_di

        step = f_val / df_di
        i_next = i_val - step
        if i_next < 0.0:
            i_next = 0.5 * i_val

        if abs(i_next - i_val) <= 1e-12 + 1e-9 * max(1.0, i_next):
            return float(i_next)
        i_val = float(i_next)

    raise RuntimeError(f"Diode current Newton iteration did not converge at V={voltage_v:.4f} V")


def solve_carrier_density_from_abc(current_density_a_m2: float, p: LEDParams) -> float:
    """Solve A*n + B*n^2 + C*n^3 = J/(q*d) for n >= 0 via Newton iteration."""
    target = float(current_density_a_m2) / (Q_E * p.active_thickness_m)
    if target <= 0.0:
        return 0.0

    n_val = max(target / max(p.a_srh, 1e-30), 1e14)
    for _ in range(80):
        f_val = p.a_srh * n_val + p.b_rad * n_val * n_val + p.c_auger * n_val**3 - target
        df_dn = p.a_srh + 2.0 * p.b_rad * n_val + 3.0 * p.c_auger * n_val * n_val
        n_next = n_val - f_val / df_dn
        if n_next <= 0.0:
            n_next = 0.5 * n_val
        if abs(n_next - n_val) <= 1e-12 + 1e-9 * max(1.0, n_next):
            return float(n_next)
        n_val = float(n_next)

    raise RuntimeError(f"ABC Newton iteration did not converge for J={current_density_a_m2:.6e} A/m^2")


def iqe_from_carrier_density(n_m3: float, p: LEDParams) -> float:
    """Return internal quantum efficiency from ABC recombination rates."""
    r_a = p.a_srh * n_m3
    r_b = p.b_rad * n_m3 * n_m3
    r_c = p.c_auger * n_m3**3
    denom = r_a + r_b + r_c
    if denom <= 0.0:
        return 0.0
    return float(r_b / denom)


def varshni_bandgap_ev(temp_k: float, p: LEDParams) -> float:
    """Return bandgap in eV via Varshni model."""
    t = float(temp_k)
    return float(p.eg0_ev - p.varshni_alpha_ev_per_k * t * t / (t + p.varshni_beta_k))


def peak_wavelength_nm_from_bandgap(eg_ev: float) -> float:
    """Return approximate EL peak wavelength from bandgap (nm)."""
    return float(1239.841984 / max(eg_ev, 1e-9))


def external_qe(iqe: float, p: LEDParams) -> float:
    """Return external quantum efficiency."""
    eqe = p.injection_efficiency * p.extraction_efficiency * iqe
    return float(np.clip(eqe, 0.0, 1.0))


def optical_power_w(current_a: float, eqe: float, photon_energy_ev: float) -> float:
    """Approximate optical power: P = (I/q) * EQE * (E_ph*q) = I * EQE * E_ph."""
    return float(current_a * eqe * photon_energy_ev)


def gaussian_spectrum(
    wavelength_nm: np.ndarray,
    power_w: float,
    lambda_peak_nm: float,
    sigma_nm: float,
) -> np.ndarray:
    """Return spectral power density (W/nm), normalized to integrate to power_w."""
    sigma = max(float(sigma_nm), 1e-6)
    shape = np.exp(-0.5 * ((wavelength_nm - float(lambda_peak_nm)) / sigma) ** 2)
    area = float(np.trapezoid(shape, wavelength_nm))
    if area <= 0.0:
        raise ValueError("Invalid wavelength grid for spectrum normalization")
    return float(power_w) * shape / area


def simulate_led_el_curve(p: LEDParams) -> Dict[str, np.ndarray]:
    """Run the end-to-end EL simulation over a voltage sweep."""
    voltage = np.linspace(2.2, 4.0, 160, dtype=np.float64)

    current = np.zeros_like(voltage)
    current_seed = 0.0
    for idx, v in enumerate(voltage):
        current[idx] = solve_diode_current(v, p.ambient_temp_k, p, init_i=current_seed)
        current_seed = current[idx]

    current_density = current / p.area_m2

    carrier_density = np.array(
        [solve_carrier_density_from_abc(j_val, p) for j_val in current_density],
        dtype=np.float64,
    )

    iqe = np.array([iqe_from_carrier_density(n_val, p) for n_val in carrier_density], dtype=np.float64)
    eqe_cold = np.array([external_qe(iqe_val, p) for iqe_val in iqe], dtype=np.float64)

    electric_power = voltage * current
    junction_temp = p.ambient_temp_k + p.thermal_resistance_k_per_w * (electric_power)

    bandgap_ev = np.array([varshni_bandgap_ev(tj, p) for tj in junction_temp], dtype=np.float64)
    peak_lambda_nm = np.array([peak_wavelength_nm_from_bandgap(e) for e in bandgap_ev], dtype=np.float64)

    optical_power = np.array(
        [optical_power_w(i_val, eqe_val, e_val) for i_val, eqe_val, e_val in zip(current, eqe_cold, bandgap_ev)],
        dtype=np.float64,
    )

    sigma_lambda_nm = p.sigma_lambda_nm_300k + p.sigma_lambda_temp_slope_nm_per_k * (junction_temp - 300.0)

    return {
        "voltage_v": voltage,
        "current_a": current,
        "current_density_a_m2": current_density,
        "carrier_density_m3": carrier_density,
        "iqe": iqe,
        "eqe": eqe_cold,
        "junction_temp_k": junction_temp,
        "bandgap_ev": bandgap_ev,
        "peak_lambda_nm": peak_lambda_nm,
        "optical_power_w": optical_power,
        "sigma_lambda_nm": sigma_lambda_nm,
    }


def build_spectrum_demo(curve: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float, float, float]]:
    """Generate EL spectra at low/mid/high bias and return summary tuples.

    Each tuple: (voltage, peak_lambda_nm, integrated_power_w, target_power_w)
    """
    wavelength_nm = np.linspace(380.0, 780.0, 1601, dtype=np.float64)
    indices = [25, 85, 145]

    summary: Dict[str, Tuple[float, float, float, float]] = {}
    for label, idx in zip(["low", "mid", "high"], indices):
        power = float(curve["optical_power_w"][idx])
        peak = float(curve["peak_lambda_nm"][idx])
        sigma = float(curve["sigma_lambda_nm"][idx])

        spd = gaussian_spectrum(wavelength_nm, power, peak, sigma)
        integrated = float(np.trapezoid(spd, wavelength_nm))

        summary[label] = (
            float(curve["voltage_v"][idx]),
            peak,
            integrated,
            power,
        )

    return summary


def run_validation(curve: Dict[str, np.ndarray], spectrum_report: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, float]:
    """Compute quality checks and enforce core physics-consistent assertions."""
    current = curve["current_a"]
    iqe = curve["iqe"]
    eqe = curve["eqe"]
    peak_lambda_nm = curve["peak_lambda_nm"]
    optical_power = curve["optical_power_w"]

    current_monotonic_min_step = float(np.min(np.diff(current)))
    iqe_peak_index = int(np.argmax(iqe))
    iqe_droop_ratio = float((iqe[iqe_peak_index] - iqe[-1]) / max(1e-12, iqe[iqe_peak_index]))
    red_shift_nm = float(peak_lambda_nm[-1] - peak_lambda_nm[0])
    eqe_peak = float(np.max(eqe))

    spectrum_rel_err = 0.0
    for _, (_, _, integrated, target) in spectrum_report.items():
        spectrum_rel_err = max(spectrum_rel_err, abs(integrated - target) / max(1e-12, target))

    optical_monotonic_min_step = float(np.min(np.diff(optical_power)))

    assert current_monotonic_min_step >= -1e-10, "Current should be non-decreasing with voltage"
    assert 1 < iqe_peak_index < (iqe.size - 2), "IQE peak should occur in interior bias range"
    assert iqe_droop_ratio > 0.02, "Expected noticeable high-injection IQE droop (>2%)"
    assert red_shift_nm > 0.2, "Expected thermal red-shift from low to high bias"
    assert eqe_peak <= 1.0 + 1e-12, "EQE must not exceed 1"
    assert spectrum_rel_err < 5e-4, "Spectrum integration should recover target optical power"
    assert optical_monotonic_min_step >= -1e-8, "Optical power should be non-decreasing with bias in this setup"

    return {
        "current_monotonic_min_step_a": current_monotonic_min_step,
        "iqe_peak": float(iqe[iqe_peak_index]),
        "iqe_peak_index": float(iqe_peak_index),
        "iqe_droop_ratio": iqe_droop_ratio,
        "eqe_peak": eqe_peak,
        "red_shift_nm": red_shift_nm,
        "spectrum_rel_err_max": spectrum_rel_err,
        "optical_power_last_w": float(optical_power[-1]),
    }


def main() -> None:
    params = LEDParams(
        area_m2=1.0e-6,
        active_thickness_m=8.0e-9,
        i0_a=8.0e-14,
        ideality=2.0,
        series_resistance_ohm=3.2,
        shunt_resistance_ohm=8.0e5,
        a_srh=9.0e7,
        b_rad=1.2e-16,
        c_auger=1.0e-40,
        extraction_efficiency=0.24,
        injection_efficiency=0.93,
        eg0_ev=2.84,
        varshni_alpha_ev_per_k=9.4e-4,
        varshni_beta_k=790.0,
        ambient_temp_k=300.0,
        thermal_resistance_k_per_w=42.0,
        sigma_lambda_nm_300k=12.5,
        sigma_lambda_temp_slope_nm_per_k=0.06,
    )

    curve = simulate_led_el_curve(params)
    spectrum_report = build_spectrum_demo(curve)
    metrics = run_validation(curve, spectrum_report)

    print("=== Electroluminescence MVP: Electrical -> Recombination -> Optical ===")
    print(f"Voltage sweep: {curve['voltage_v'][0]:.3f} V -> {curve['voltage_v'][-1]:.3f} V ({curve['voltage_v'].size} points)")
    print(f"Current range: {curve['current_a'][0]:.6e} A -> {curve['current_a'][-1]:.6e} A")
    print(f"Peak EQE: {np.max(curve['eqe']):.6f}")
    print(f"Peak IQE: {np.max(curve['iqe']):.6f}")
    print(f"Final optical power: {curve['optical_power_w'][-1]:.6e} W")
    print(f"Peak wavelength shift: {curve['peak_lambda_nm'][0]:.3f} nm -> {curve['peak_lambda_nm'][-1]:.3f} nm")

    print("\n=== Spectrum checks (integral should match optical power) ===")
    for label in ["low", "mid", "high"]:
        v, peak, integrated, target = spectrum_report[label]
        rel_err = abs(integrated - target) / max(1e-12, target)
        print(
            f"{label:>4s} bias | V={v:.3f} V | peak={peak:.2f} nm | "
            f"integral={integrated:.6e} W | target={target:.6e} W | rel_err={rel_err:.3e}"
        )

    print("\n=== Validation metrics ===")
    for key, value in metrics.items():
        print(f"{key:>28s}: {value:.6e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
