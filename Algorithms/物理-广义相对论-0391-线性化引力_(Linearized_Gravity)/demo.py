"""Minimal runnable MVP for linearized gravity in TT gauge.

We use the weak-field metric:
    g_{mu nu} = eta_{mu nu} + h_{mu nu},  |h_{mu nu}| << 1

In vacuum and in Lorenz gauge, trace-reversed perturbations satisfy:
    □ \bar{h}_{mu nu} = 0

For a plane wave propagating along z in TT gauge, the two physical
polarizations obey 1D wave equations:
    d2 h_+ / dt2 - c^2 d2 h_+ / dz2 = 0
    d2 h_x / dt2 - c^2 d2 h_x / dz2 = 0

This demo solves those equations with a leapfrog finite-difference scheme,
compares with analytic right-moving Gaussian pulses, and prints diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c


@dataclass(frozen=True)
class PolarizationDiagnostics:
    name: str
    amplitude: float
    rmse: float
    max_abs_error: float
    relative_rmse: float
    relative_max_abs_error: float
    pde_residual_rms: float
    pde_residual_relative: float
    energy_relative_drift: float
    peak_abs_strain: float


@dataclass(frozen=True)
class SimulationResult:
    z: np.ndarray
    t: np.ndarray
    h_plus_hist: np.ndarray
    h_cross_hist: np.ndarray
    h_plus_exact_final: np.ndarray
    h_cross_exact_final: np.ndarray
    plus_diag: PolarizationDiagnostics
    cross_diag: PolarizationDiagnostics
    trace_max_abs: float
    detector_index: int
    detector_table: pd.DataFrame


def periodic_laplacian(field: np.ndarray, dz: float) -> np.ndarray:
    """Second spatial derivative with periodic boundary conditions."""
    return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (dz * dz)


def periodic_gradient(field: np.ndarray, dz: float) -> np.ndarray:
    """Central first derivative with periodic boundary conditions."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2.0 * dz)


def gaussian_on_ring(
    z: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
    domain_length: float,
) -> np.ndarray:
    """Periodic Gaussian pulse on [0, L)."""
    wrapped = ((z - center + 0.5 * domain_length) % domain_length) - 0.5 * domain_length
    return amplitude * np.exp(-((wrapped / width) ** 2))


def analytic_right_moving(
    z: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
    time_s: float,
    domain_length: float,
) -> np.ndarray:
    """Exact right-moving Gaussian pulse h(z,t)=f(z-ct) on a periodic domain."""
    shifted_center = (center + c * time_s) % domain_length
    return gaussian_on_ring(
        z=z,
        amplitude=amplitude,
        center=shifted_center,
        width=width,
        domain_length=domain_length,
    )


def initialize_previous_step(
    h0: np.ndarray,
    dhdt0: np.ndarray,
    dt: float,
    dz: float,
) -> np.ndarray:
    """Second-order initialization for leapfrog at t=-dt."""
    lap0 = periodic_laplacian(h0, dz)
    return h0 - dt * dhdt0 + 0.5 * ((c * dt) ** 2) * lap0


def energy_series(history: np.ndarray, dt: float, dz: float) -> np.ndarray:
    """Discrete wave energy E ~ ∫[(dh/dt)^2 + c^2(dh/dz)^2]/2 dz for each interior time."""
    n_steps = history.shape[0] - 1
    energies = np.empty(n_steps - 1, dtype=float)
    for n in range(1, n_steps):
        dhdt = (history[n + 1] - history[n - 1]) / (2.0 * dt)
        dhdz = periodic_gradient(history[n], dz)
        densities = 0.5 * (dhdt * dhdt + (c * dhdz) * (c * dhdz))
        energies[n - 1] = float(np.sum(densities) * dz)
    return energies


def simulate_single_polarization(
    name: str,
    z: np.ndarray,
    t: np.ndarray,
    domain_length: float,
    amplitude: float,
    center: float,
    width: float,
    dt: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, PolarizationDiagnostics]:
    """Evolve one polarization using leapfrog and compute diagnostics."""
    h0 = gaussian_on_ring(z, amplitude, center, width, domain_length)
    dhdt0 = -c * periodic_gradient(h0, dz)  # right-moving initial condition
    h_prev = initialize_previous_step(h0, dhdt0, dt, dz)
    h_curr = h0.copy()

    history = np.empty((len(t), len(z)), dtype=float)
    history[0] = h0

    for n in range(1, len(t)):
        h_next = 2.0 * h_curr - h_prev + ((c * dt) ** 2) * periodic_laplacian(h_curr, dz)
        history[n] = h_next
        h_prev, h_curr = h_curr, h_next

    final_exact = analytic_right_moving(
        z=z,
        amplitude=amplitude,
        center=center,
        width=width,
        time_s=float(t[-1]),
        domain_length=domain_length,
    )
    final_num = history[-1]
    err = final_num - final_exact
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs_error = float(np.max(np.abs(err)))
    amp_abs = abs(amplitude) + 1.0e-30

    # Residual from the discrete PDE at second last time slice.
    h_nm2 = history[-3]
    h_nm1 = history[-2]
    h_n = history[-1]
    d2h_dt2 = (h_n - 2.0 * h_nm1 + h_nm2) / (dt * dt)
    c2_lap = (c * c) * periodic_laplacian(h_nm1, dz)
    residual = d2h_dt2 - c2_lap
    pde_residual_rms = float(np.sqrt(np.mean(residual * residual)))
    denom = float(np.sqrt(np.mean(c2_lap * c2_lap))) + 1.0e-30
    pde_residual_relative = pde_residual_rms / denom

    energies = energy_series(history, dt, dz)
    energy_relative_drift = float((np.max(energies) - np.min(energies)) / np.mean(energies))

    diag = PolarizationDiagnostics(
        name=name,
        amplitude=amplitude,
        rmse=rmse,
        max_abs_error=max_abs_error,
        relative_rmse=rmse / amp_abs,
        relative_max_abs_error=max_abs_error / amp_abs,
        pde_residual_rms=pde_residual_rms,
        pde_residual_relative=pde_residual_relative,
        energy_relative_drift=energy_relative_drift,
        peak_abs_strain=float(np.max(np.abs(history))),
    )
    return history, final_exact, diag


def run_linearized_gravity_mvp(
    nz: int = 600,
    domain_length_m: float = 4.0e7,
    cfl: float = 0.80,
    n_steps: int = 220,
    amp_plus: float = 1.0e-4,
    amp_cross: float = 0.7e-4,
    width_m: float = 1.2e6,
) -> SimulationResult:
    """Run TT-gauge wave propagation for h_+ and h_x polarizations."""
    if nz < 32:
        raise ValueError("nz must be >= 32.")
    if domain_length_m <= 0.0:
        raise ValueError("domain_length_m must be positive.")
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must satisfy 0 < cfl <= 1.")
    if n_steps < 5:
        raise ValueError("n_steps must be >= 5.")
    if width_m <= 0.0:
        raise ValueError("width_m must be positive.")

    z = np.linspace(0.0, domain_length_m, nz, endpoint=False, dtype=float)
    dz = domain_length_m / nz
    dt = cfl * dz / c
    t = np.arange(n_steps + 1, dtype=float) * dt

    # Keep pulses far from boundaries to avoid wrap-around during simulation time.
    center_plus = 0.20 * domain_length_m
    center_cross = 0.28 * domain_length_m
    travel_distance = c * float(t[-1])
    if travel_distance > 0.30 * domain_length_m:
        raise ValueError("Simulation time too long; pulse wrap-around would pollute comparisons.")

    h_plus_hist, h_plus_exact_final, plus_diag = simulate_single_polarization(
        name="plus",
        z=z,
        t=t,
        domain_length=domain_length_m,
        amplitude=amp_plus,
        center=center_plus,
        width=width_m,
        dt=dt,
        dz=dz,
    )
    h_cross_hist, h_cross_exact_final, cross_diag = simulate_single_polarization(
        name="cross",
        z=z,
        t=t,
        domain_length=domain_length_m,
        amplitude=amp_cross,
        center=center_cross,
        width=width_m,
        dt=dt,
        dz=dz,
    )

    # In TT gauge for z-propagation: h_xx=+h_+, h_yy=-h_+, h_xy=h_x, so trace is zero.
    trace_hist = h_plus_hist + (-h_plus_hist)
    trace_max_abs = float(np.max(np.abs(trace_hist)))

    detector_index = int(0.65 * nz)
    sample_idx = np.linspace(0, n_steps, 10, dtype=int)
    detector_table = pd.DataFrame(
        {
            "time_s": t[sample_idx],
            "h_plus_numeric": h_plus_hist[sample_idx, detector_index],
            "h_cross_numeric": h_cross_hist[sample_idx, detector_index],
            "h_plus_exact": [
                analytic_right_moving(
                    z=np.array([z[detector_index]]),
                    amplitude=amp_plus,
                    center=center_plus,
                    width=width_m,
                    time_s=float(tt),
                    domain_length=domain_length_m,
                )[0]
                for tt in t[sample_idx]
            ],
            "h_cross_exact": [
                analytic_right_moving(
                    z=np.array([z[detector_index]]),
                    amplitude=amp_cross,
                    center=center_cross,
                    width=width_m,
                    time_s=float(tt),
                    domain_length=domain_length_m,
                )[0]
                for tt in t[sample_idx]
            ],
            "trace_hij": np.zeros_like(sample_idx, dtype=float),
        }
    )
    detector_table["plus_abs_err"] = np.abs(
        detector_table["h_plus_numeric"] - detector_table["h_plus_exact"]
    )
    detector_table["cross_abs_err"] = np.abs(
        detector_table["h_cross_numeric"] - detector_table["h_cross_exact"]
    )

    return SimulationResult(
        z=z,
        t=t,
        h_plus_hist=h_plus_hist,
        h_cross_hist=h_cross_hist,
        h_plus_exact_final=h_plus_exact_final,
        h_cross_exact_final=h_cross_exact_final,
        plus_diag=plus_diag,
        cross_diag=cross_diag,
        trace_max_abs=trace_max_abs,
        detector_index=detector_index,
        detector_table=detector_table,
    )


def run_checks(result: SimulationResult) -> None:
    """Fail fast if numerical quality is not acceptable for this MVP."""
    for d in (result.plus_diag, result.cross_diag):
        if d.relative_rmse > 2.5e-2:
            raise AssertionError(f"{d.name}: relative RMSE too high ({d.relative_rmse:.3e})")
        if d.relative_max_abs_error > 8.0e-2:
            raise AssertionError(
                f"{d.name}: relative max abs error too high ({d.relative_max_abs_error:.3e})"
            )
        if d.pde_residual_relative > 1.0e-10:
            raise AssertionError(
                f"{d.name}: discrete PDE residual too high ({d.pde_residual_relative:.3e})"
            )
        if d.energy_relative_drift > 7.0e-2:
            raise AssertionError(
                f"{d.name}: energy drift too high ({d.energy_relative_drift:.3e})"
            )
        if d.peak_abs_strain > 1.0e-2:
            raise AssertionError(
                f"{d.name}: peak strain too large for linear regime ({d.peak_abs_strain:.3e})"
            )
    if result.trace_max_abs > 1.0e-14:
        raise AssertionError(f"TT trace-free condition violated ({result.trace_max_abs:.3e})")


def main() -> None:
    result = run_linearized_gravity_mvp()
    run_checks(result)

    nz = len(result.z)
    n_steps = len(result.t) - 1
    domain_length = float(result.z[-1] + (result.z[1] - result.z[0]))
    dz = domain_length / nz
    dt = float(result.t[1] - result.t[0])
    cfl = c * dt / dz

    diag_table = pd.DataFrame(
        [
            {
                "polarization": result.plus_diag.name,
                "amplitude": result.plus_diag.amplitude,
                "rel_rmse": result.plus_diag.relative_rmse,
                "rel_max_abs_error": result.plus_diag.relative_max_abs_error,
                "residual_rel": result.plus_diag.pde_residual_relative,
                "energy_drift": result.plus_diag.energy_relative_drift,
                "peak_abs_strain": result.plus_diag.peak_abs_strain,
            },
            {
                "polarization": result.cross_diag.name,
                "amplitude": result.cross_diag.amplitude,
                "rel_rmse": result.cross_diag.relative_rmse,
                "rel_max_abs_error": result.cross_diag.relative_max_abs_error,
                "residual_rel": result.cross_diag.pde_residual_relative,
                "energy_drift": result.cross_diag.energy_relative_drift,
                "peak_abs_strain": result.cross_diag.peak_abs_strain,
            },
        ]
    )

    print("Linearized Gravity MVP (TT gauge, vacuum wave equation)")
    print("=" * 78)
    print(f"Grid points nz            : {nz}")
    print(f"Time steps                : {n_steps}")
    print(f"Domain length [m]         : {domain_length:.6e}")
    print(f"dz [m]                    : {dz:.6e}")
    print(f"dt [s]                    : {dt:.6e}")
    print(f"CFL = c*dt/dz             : {cfl:.6f}")
    print(f"Trace max |h_xx+h_yy|     : {result.trace_max_abs:.3e}")

    print("\nPolarization diagnostics:")
    with pd.option_context("display.precision", 10, "display.width", 180):
        print(diag_table.to_string(index=False))

    print("\nDetector sample (z index = {}):".format(result.detector_index))
    with pd.option_context("display.precision", 12, "display.width", 180):
        print(result.detector_table.to_string(index=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
