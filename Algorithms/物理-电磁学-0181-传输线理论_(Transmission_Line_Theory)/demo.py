"""Transmission Line Theory MVP (RLGC model, frequency domain).

This script provides a minimal but honest implementation of transmission-line
analysis using the telegrapher-equation closed forms.

- Uses per-unit-length RLGC parameters.
- Computes propagation constant gamma and characteristic impedance Z0.
- Computes input impedance for arbitrary load and line length.
- Produces a non-interactive sweep report.
- Includes two self-checks:
  1) matched load reflection ~= 0
  2) quarter-wave transformer relation Zin ~= Z0^2 / ZL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class RLGCLine:
    """Per-unit-length RLGC transmission line."""

    r: float  # ohm / m
    l: float  # henry / m
    g: float  # siemens / m
    c: float  # farad / m
    length: float  # m

    def _series_impedance(self, omega: np.ndarray) -> np.ndarray:
        return self.r + 1j * omega * self.l

    def _shunt_admittance(self, omega: np.ndarray) -> np.ndarray:
        return self.g + 1j * omega * self.c

    def propagation_constant(self, f_hz: np.ndarray | float) -> np.ndarray:
        """Return gamma(f) = alpha + j*beta, with alpha >= 0 branch selection."""

        f = np.asarray(f_hz, dtype=float)
        omega = 2.0 * np.pi * f
        gamma = np.sqrt(self._series_impedance(omega) * self._shunt_admittance(omega))
        return np.where(np.real(gamma) < 0.0, -gamma, gamma)

    def characteristic_impedance(self, f_hz: np.ndarray | float) -> np.ndarray:
        """Return Z0(f) from RLGC definition."""

        f = np.asarray(f_hz, dtype=float)
        omega = 2.0 * np.pi * f
        z0 = np.sqrt(self._series_impedance(omega) / self._shunt_admittance(omega))
        return np.where(np.real(z0) < 0.0, -z0, z0)


def reflection_coefficient(z_term: np.ndarray, z0: np.ndarray) -> np.ndarray:
    return (z_term - z0) / (z_term + z0)


def input_impedance(line: RLGCLine, f_hz: np.ndarray | float, z_load: complex) -> np.ndarray:
    """Input impedance seen at source end for a line of finite length."""

    gamma = line.propagation_constant(f_hz)
    z0 = line.characteristic_impedance(f_hz)
    t = np.tanh(gamma * line.length)
    return z0 * (z_load + z0 * t) / (z0 + z_load * t)


def voltage_current_profile(
    line: RLGCLine,
    f_hz: float,
    z_load: complex,
    v_load: complex = 1.0 + 0.0j,
    n_points: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return V(x), I(x), x where x=0 at load and x=length at source."""

    gamma = line.propagation_constant(float(f_hz)).item()
    z0 = line.characteristic_impedance(float(f_hz)).item()
    i_load = v_load / z_load

    x = np.linspace(0.0, line.length, n_points)
    cg = np.cosh(gamma * x)
    sg = np.sinh(gamma * x)

    v = v_load * cg + i_load * z0 * sg
    i = i_load * cg + (v_load / z0) * sg
    return x, v, i


def db_per_meter(alpha_np_per_m: np.ndarray) -> np.ndarray:
    return 8.685889638 * alpha_np_per_m


def run_frequency_sweep_demo() -> None:
    print("=== Scenario A: lossy 20 m line, load 100 ohm, sweep 1 MHz -> 1 GHz ===")

    line = RLGCLine(
        r=0.08,
        l=250e-9,
        g=2e-6,
        c=100e-12,
        length=20.0,
    )
    z_load = 100.0 + 0.0j
    freqs = np.logspace(6, 9, 12)

    gamma = line.propagation_constant(freqs)
    z0 = line.characteristic_impedance(freqs)
    zin = input_impedance(line, freqs, z_load)

    gamma_l = reflection_coefficient(np.full_like(z0, z_load), z0)
    gamma_in = gamma_l * np.exp(-2.0 * gamma * line.length)
    vswr_in = (1.0 + np.abs(gamma_in)) / np.maximum(1.0 - np.abs(gamma_in), EPS)

    print("f(MHz)  |Z0|(ohm)  |Zin|(ohm)  alpha(dB/m)  |Gamma_in|  VSWR_in")
    for idx, f in enumerate(freqs):
        alpha_db_m = db_per_meter(np.real(gamma[idx]))
        print(
            f"{f/1e6:7.2f}  {np.abs(z0[idx]):10.4f}  {np.abs(zin[idx]):10.4f}"
            f"  {alpha_db_m:11.6f}  {np.abs(gamma_in[idx]):10.6f}  {vswr_in[idx]:8.4f}"
        )


def validate_matched_load_case() -> None:
    print("\n=== Scenario B: matched-load sanity check ===")

    f = 100e6
    line = RLGCLine(r=0.0, l=250e-9, g=0.0, c=100e-12, length=10.0)
    z0 = line.characteristic_impedance(f).item()

    gamma_l = reflection_coefficient(np.array([z0]), np.array([z0])).item()
    zin = input_impedance(line, f, z0).item()

    rel_err = abs((zin - z0) / z0)
    print(f"Z0 = {z0:.6f} ohm")
    print(f"Gamma_L = {gamma_l:.6e}")
    print(f"relative error |Zin-Z0|/|Z0| = {rel_err:.3e}")

    assert abs(gamma_l) < 1e-12, "Matched load should have near-zero reflection coefficient."
    assert rel_err < 1e-12, "Matched load should keep Zin ~= Z0 for any length."


def validate_quarter_wave_transformer() -> None:
    print("\n=== Scenario C: quarter-wave transformer sanity check ===")

    f = 150e6
    vp = 2.0e8
    z0_target = np.sqrt(50.0 * 100.0)

    l_per_m = z0_target / vp
    c_per_m = 1.0 / (z0_target * vp)
    quarter_wave_len = vp / (4.0 * f)

    line = RLGCLine(r=0.0, l=l_per_m, g=0.0, c=c_per_m, length=quarter_wave_len)
    z_load = 100.0 + 0.0j

    zin = input_impedance(line, f, z_load).item()
    expected = (z0_target * z0_target) / z_load
    rel_err = abs((zin - expected) / expected)

    print(f"line length = {quarter_wave_len:.6f} m")
    print(f"Z0 = {z0_target:.6f} ohm, ZL = {z_load:.6f} ohm")
    print(f"computed Zin = {zin:.6f} ohm")
    print(f"expected Zin = {expected:.6f} ohm")
    print(f"relative error = {rel_err:.3e}")

    assert rel_err < 1e-9, "Quarter-wave transformer relation should hold in lossless case."


def run_profile_snapshot() -> None:
    print("\n=== Scenario D: voltage/current profile snapshot at 200 MHz ===")

    line = RLGCLine(r=0.08, l=250e-9, g=2e-6, c=100e-12, length=5.0)
    z_load = 25.0 + 0.0j

    x, v, i = voltage_current_profile(line, f_hz=200e6, z_load=z_load, v_load=1.0 + 0.0j, n_points=7)
    print("x(m)   |V(x)|(V)   |I(x)|(A)")
    for xi, vi, ii in zip(x, v, i):
        print(f"{xi:4.2f}   {abs(vi):9.6f}   {abs(ii):9.6f}")


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    run_frequency_sweep_demo()
    validate_matched_load_case()
    validate_quarter_wave_transformer()
    run_profile_snapshot()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
